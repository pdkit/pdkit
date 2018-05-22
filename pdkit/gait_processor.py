# Copyright 2018 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author: Cosmin Stamate 

import sys
import traceback

import numpy as np
import pandas as pd

from scipy import interpolate, signal, fft
from scipy.fftpack import rfft
from pywt import wavedec

from pdkit.processor import Processor
from pdkit.gait_time_series import GaitTimeSeries

from scipy.integrate import cumtrapz


from pdkit.utils import (load_data,
                        numerical_integration, 
                        autocorrelation,
                        peakdet,
                        compute_interpeak,
                        butter_lowpass_filter,
                        crossings_nonzero_pos2neg,
                        autocorrelate)


class GaitProcessor(Processor):
    '''
       This is the main Gait Processor class. Once the data is loaded it will be
       accessible at data_frame, where it looks like:
       data_frame.x, data_frame.y, data_frame.z: x, y, z components of the acceleration
       data_frame.index is the datetime-like index
       
       This values are recommended by the author of the pilot study [1] and [3]
       
       step_size = 50.0
       start_end_offset = [100, 100]
       delta = 0.5
       loco_band = [0.5, 3]
       freeze_band = [3, 8]
       sampling_frequency = 100.0Hz
       cutoff_frequency = 2.0Hz
       filter_order = 2
       window = 256
       lower_frequency = 2.0Hz
       upper_frequency = 10.0Hz

       [1] Developing a tool for remote digital assessment of Parkinson s disease
            Kassavetis	P,	Saifee	TA,	Roussos	G,	Drougas	L,	Kojovic	M,	Rothwell	JC,	Edwards	MJ,	Bhatia	KP
            
       [2] The use of the fast Fourier transform for the estimation of power spectra: A method based 
            on time averaging over short, modified periodograms (IEEE Trans. Audio Electroacoust. 
            vol. 15, pp. 70-73, 1967)
            P. Welch

       [3] M. Bachlin et al., "Wearable Assistant for Parkinson’s Disease Patients With the Freezing of Gait Symptom,"
           in IEEE Transactions on Information Technology in Biomedicine, vol. 14, no. 2, pp. 436-446, March 2010.
    '''

    def __init__(self,
                 sampling_frequency=100.0,
                 cutoff_frequency=2.0,
                 filter_order=2,
                 window=256,
                 lower_frequency=2.0,
                 upper_frequency=10.0,
                 step_size=50.0,
                 start_end_offset=[100, 100],
                 delta=0.5,
                 loco_band=[0.5, 3],
                 freeze_band=[3, 8],
                 stride_fraction=1.0/8.0,
                 order=4,
                 threshold=0.5,
                 distance=90,
                 step_period=2,
                 stride_period=1,
                 pendulum_length=1):

        super().__init__(sampling_frequency,
                         cutoff_frequency,
                         filter_order,
                         window,
                         lower_frequency,
                         upper_frequency)

        self.step_size = step_size
        self.start_end_offset = start_end_offset
        self.delta = delta
        self.loco_band = loco_band
        self.freeze_band = freeze_band
        self.stride_fraction = stride_fraction
        self.order = order
        self.threshold = threshold
        self.distance = distance
        self.step_period = step_period
        self.stride_period = stride_period
        self.pendulum_length = pendulum_length


    def freeze_of_gait(self, data_frame):
        ''' 
            This method assess freeze of gait following [3].

            :param DataFrame data_frame: the data frame.

            :return list freeze_times: What times does freeze of gait occur.
            :return list freeze_indexes: What are the index in the dataframe when freeze of gait occurs.
            :return list locomotion_freezes: When does locomotion freeze happen,.
        '''
        
        # the sampling frequency was recommended by the author of the pilot study
        data = self.resample_signal(data_frame) 
        data = data.y.values

        f_res = self.sampling_frequency / self.window
        f_nr_LBs = int(self.loco_band[0] / f_res)
        f_nr_LBe = int(self.loco_band[1] / f_res)
        f_nr_FBs = int(self.freeze_band[0] / f_res)
        f_nr_FBe = int(self.freeze_band[1] / f_res)

        jPos = self.window + 1
        i = 0
        
        time = []
        sumLocoFreeze = []
        freezeIndex = []
        
        while jPos < len(data):
            
            jStart = jPos - self.window
            time.append(jPos)

            y = data[int(jStart):int(jPos)]
            y = y - np.mean(y)

            Y = np.fft.fft(y, int(self.window))
            Pyy = abs(Y*Y) / self.window #conjugate(Y) * Y / NFFT

            areaLocoBand = numerical_integration( Pyy[f_nr_LBs-1 : f_nr_LBe], self.sampling_frequency )
            areaFreezeBand = numerical_integration( Pyy[f_nr_FBs-1 : f_nr_FBe], self.sampling_frequency )

            sumLocoFreeze.append(areaFreezeBand + areaLocoBand)

            freezeIndex.append(areaFreezeBand / areaLocoBand)

            jPos = jPos + self.step_size
            i = i + 1

        freeze_times = time
        freeze_indexes = freezeIndex
        locomotion_freezes = sumLocoFreeze

        return freeze_times, freeze_indexes, locomotion_freezes


    def frequency_of_peaks(self, data_frame):
        ''' 
            This method assess the frequency of the peaks on the x-axis.

            :param DataFrame data_frame: the data frame.

            :return float frequency_of_peaks: The frequency of peaks on the x-axis.
        '''

        peaks_data = data_frame[self.start_end_offset[0]: -self.start_end_offset[1]].x.values
        maxtab, mintab = peakdet(peaks_data, self.delta)
        x = np.mean(peaks_data[maxtab[1:,0].astype(int)] - peaks_data[maxtab[:-1,0].astype(int)])
        frequency_of_peaks = 1/x

        return frequency_of_peaks
        

    def speed_of_gait(self, data_frame, wavelet_type='db3', wavelet_level=6):
        """
            This method assess the speed of gait following [2].
            It extracts the gait speed from the energies of the approximation coefficients of wavelet functions.

            :param DataFrame data_frame: the data frame.
            :param str wavelet_type: the type of wavelet to use. See https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html for a full list.
            :param int wavelet_level: the number of cycles the used wavelet should have. See https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html for a fill list.
            
            :return float gait_speed: The speed of gait.
        """

        coeffs = wavedec(data_frame.mag_sum_acc, wavelet=wavelet_type, level=wavelet_level)

        energy = [sum(coeffs[wavelet_level - i]**2) / len(coeffs[wavelet_level - i]) for i in range(wavelet_level)]

        WEd1 = energy[0] / (5 * np.sqrt(2))
        WEd2 = energy[1] / (4 * np.sqrt(2))
        WEd3 = energy[2] / (3 * np.sqrt(2))
        WEd4 = energy[3] / (2 * np.sqrt(2))
        WEd5 = energy[4] / np.sqrt(2)
        WEd6 = energy[5] / np.sqrt(2)

        gait_speed = 0.5 * np.sqrt(WEd1+(WEd2/2)+(WEd3/3)+(WEd4/4)+(WEd5/5))

        return gait_speed


    def walk_regularity_symmetry(self, data_frame):
        ''' 
            This method extracts the step and stride regularity and also walk symmetry.

            :param DataFrame data_frame: the data frame.

            :return list step_regularity: Regularity of steps on [x, y, z] coordinates.
            :return list stride_regularity: Regularity of stride on [x, y, z] coordinates.
            :return list walk_symmetry: Symmetry of walk on [x, y, z] coordinates.
        '''
        
        def _symmetry(v):
            maxtab, _ = peakdet(v, self.delta)
            return maxtab[1][1], maxtab[2][1]

        step_regularity_x, stride_regularity_x = _symmetry(autocorrelation(data_frame.x))
        step_regularity_y, stride_regularity_y = _symmetry(autocorrelation(data_frame.y))
        step_regularity_z, stride_regularity_z = _symmetry(autocorrelation(data_frame.z))

        symmetry_x = stride_regularity_x - step_regularity_x
        symmetry_y = stride_regularity_y - step_regularity_y
        symmetry_z = stride_regularity_z - step_regularity_z

        step_regularity = [step_regularity_x, step_regularity_y, step_regularity_z]
        stride_regularity = [stride_regularity_x, stride_regularity_y, stride_regularity_z]
        walk_symmetry = [symmetry_x, symmetry_y, symmetry_z]

        return step_regularity, stride_regularity, walk_symmetry


    def walk_direction_preheel(self, data_frame):
        '''
            Estimate local walk (not cardinal) direction with pre-heel strike phase.

            Inspired by Nirupam Roy's B.E. thesis: "WalkCompass:
            Finding Walking Direction Leveraging Smartphone's Inertial Sensors,".

            :param DataFrame data_frame: the data frame.

            :return array direction: Unit vector of local walk (not cardinal) direction
        '''


        # Sum of absolute values across accelerometer axes:
        data = data_frame.x.abs() + data_frame.y.abs() + data_frame.z.abs()

        # Find maximum peaks of smoothed data:
        dummy, ipeaks_smooth = self.heel_strikes(data)
        data = data.values

        # Compute number of samples between peaks using the real part of the FFT:
        interpeak = compute_interpeak(data, self.sampling_frequency)
        decel = np.int(np.round(self.stride_fraction * interpeak))

        # Find maximum peaks close to maximum peaks of smoothed data:
        ipeaks = []
        for ipeak_smooth in ipeaks_smooth:
            ipeak = np.argmax(data[ipeak_smooth - decel:ipeak_smooth + decel])
            ipeak += ipeak_smooth - decel
            ipeaks.append(ipeak)

        # Compute the average vector for each deceleration phase:
        vectors = []
        for ipeak in ipeaks:
            decel_vectors = np.asarray([[data_frame.x[i], data_frame.y[i], data_frame.z[i]]
                                        for i in range(ipeak - decel, ipeak)])
            vectors.append(np.mean(decel_vectors, axis=0))

        # Compute the average deceleration vector and take the opposite direction:
        direction = -1 * np.mean(vectors, axis=0)

        # Return the unit vector in this direction:
        direction /= np.sqrt(direction.dot(direction))

        return direction


    def heel_strikes(self, data_frame_axis):
        '''
            Estimate heel strike times between sign changes in accelerometer data.

            :param array data_frame_axis: Accelerometer data along one axis (preferably forward direction)
            
            :return array strikes: Heel strike timings
            :return list strike_indices: Heel strike timing indices.
        '''
        # Demean data:
        data = data_frame_axis.values
        data -= data.mean()

        # Low-pass filter the AP accelerometer data by the 4th order zero lag
        # Butterworth filter whose cut frequency is set to 5 Hz:
        filtered = butter_lowpass_filter(data, self.sampling_frequency, self.cutoff_frequency, self.order)

        # Find transitional positions where AP accelerometer changes from
        # positive to negative.
        transitions = crossings_nonzero_pos2neg(filtered)

        # Find the peaks of AP acceleration preceding the transitional positions,
        # and greater than the product of a threshold and the maximum value of
        # the AP acceleration:
        strike_indices_smooth = []
        filter_threshold = np.abs(self.threshold * np.max(filtered))
        for i in range(1, np.size(transitions)):
            segment = range(transitions[i-1], transitions[i])
            imax = np.argmax(filtered[segment])
            if filtered[segment[imax]] > filter_threshold:
                strike_indices_smooth.append(segment[imax])

        # Compute number of samples between peaks using the real part of the FFT:
        interpeak = compute_interpeak(data, self.sampling_frequency)
        decel = np.int(interpeak / 2)

        # Find maximum peaks close to maximum peaks of smoothed data:
        strike_indices = []
        for ismooth in strike_indices_smooth:
            istrike = np.argmax(data[ismooth - decel:ismooth + decel])
            istrike = istrike + ismooth - decel
            strike_indices.append(istrike)

        strikes = np.asarray(strike_indices)
        strikes -= strikes[0]
        strikes = strikes / self.sampling_frequency

        return strikes, strike_indices


    def gait_regularity_symmetry(self, data_frame_axis, unbias=1, normalize=2):
        '''
            Compute step and stride regularity and symmetry from accelerometer data.

            :param array data_frame_axis: Accelerometer data along one axis (preferably forward direction)
            :param int unbias: Unbiased autocorrelation: divide by range (1) or by weighted range (2).
            :param int normalize: Normalize: divide by 1st coefficient (1) or by maximum abs. value (2).

            :return float step_regularity: Step regularity measure along axis.
            :return float stride_regularity: Stride regularity measure along axis.
            :return symmetry: Symmetry measure along axis.
        '''

        coefficients, _ = autocorrelate(data_frame_axis, unbias=1, normalize=2)

        step_regularity = coefficients[self.step_period]
        stride_regularity = coefficients[self.stride_period]
        symmetry = np.abs(stride_regularity - step_regularity)

        return step_regularity, stride_regularity, symmetry

    def root_mean_square(self, data_frame):
        return 

    def gait(self, data_frame, axis='x'):
        '''
            Extract gait features from estimated heel strikes and accelerometer data.

            :param DataFrame data_frame: The data frame.
            :param string axis: The axis (preferably forward direction) from which to extract all gait features.

            :return int number_of_steps: Estimated number of steps based on heel strikes.
            :return float velocity: Velocity (if distance).
            :return float avg_step_length: Average step length (if distance).
            :return float avg_stride_length: Average stride length (if distance).
            :return float cadence: Number of steps divided by duration.
            :return array step_durations: Step duration.
            :return float avg_step_duration: Average step duration.
            :return float sd_step_durations: Standard deviation of step durations.
            :return list strides: Stride timings for each side.
            :return float avg_number_of_strides: Estimated number of strides based on alternating heel strikes.
            :return list stride_durations: Estimated stride durations.
            :return float avg_stride_duration: Average stride duration.
            :return float sd_step_durations: Standard deviation of stride durations.
            :return float step_regularity: Measure of step regularity along axis.
            :return float stride_regularity: Measure of stride regularity along axis.
            :return float symmetry: Measure of gait symmetry along axis.
        '''

        self.duration = data_frame.td[-1]
        data = data_frame[axis]
        
        strikes, strikes_ids = self.heel_strikes(data)

        step_durations = []
        for i in range(1, np.size(strikes)):
            step_durations.append(strikes[i] - strikes[i-1])

        avg_step_duration = np.mean(step_durations)
        sd_step_durations = np.std(step_durations)

        number_of_steps = np.size(strikes)

        # Cadence in steps / duration
        cadence = number_of_steps / self.duration

        strides1 = strikes[0::2]
        strides2 = strikes[1::2]

        stride_durations1 = []
        for i in range(1, np.size(strides1)):
            stride_durations1.append(strides1[i] - strides1[i-1])

        stride_durations2 = []
        for i in range(1, np.size(strides2)):
            stride_durations2.append(strides2[i] - strides2[i-1])

        strides = [strides1, strides2]
        stride_durations = [stride_durations1, stride_durations2]

        avg_number_of_strides = np.mean([np.size(strides1), np.size(strides2)])
        avg_stride_duration = np.mean((np.mean(stride_durations1),
                                    np.mean(stride_durations2)))
        sd_stride_durations = np.mean((np.std(stride_durations1),
                                    np.std(stride_durations2)))

        self.step_period = np.int(np.round(1 / avg_step_duration))
        self.stride_period = np.int(np.round(1 / avg_stride_duration))

        step_regularity, stride_regularity, symmetry = self.gait_regularity_symmetry(data)

        # Set distance-based measures to None if distance not set:
        if self.distance:
            velocity = self.distance / self.duration
            avg_step_length = number_of_steps / self.distance
            avg_stride_length = avg_number_of_strides / self.distance
        else:
            velocity = None
            avg_step_length = None
            avg_stride_length = None

        # Acceleration amplitude variability
        acceleration_amplitude_variability = [data[strikes_ids[i]: strikes_ids[i+1]].std() for i in range(len(strikes_ids) -1)]

        # Cycle frequency in Hertz
        cycle_frequency = self.sampling_frequency * rfft(data).argmax() / data.shape[0]

        # Foot symmetry
        walking = data[strikes_ids[0]: strikes_ids[-1]]
        total_steps = len(walking)

        steps = []

        for z in range(len(strikes_ids) - 1):
            steps.append([data[strikes_ids[z]: strikes_ids[z+1]]])

        first_foot= [len(a[0]) for a in steps[::2]]
        second_foot = [len(a[0])  for a in steps[1::2]]

        first_sym = np.sum([a / total_steps for a in first_foot])
        second_sym = np.sum([a / total_steps for a in second_foot])

        # Gait irregularity
        first_foot_gait_irreg = [a / self.sampling_frequency for a in first_foot]
        second_foot_gait_irreg = [a / self.sampling_frequency for a in second_foot]

        gait_irregularity = np.mean([first_foot_gait_irreg, second_foot_gait_irreg])

        # Gait variability

        # Harmonic ratio
        sig_fft = np.fft.fft(data)
        bins, _ = np.histogram(sig_fft, 100)

        harmonic_ratio = np.sum(bins[::2]) / np.sum(bins[1::2])

        # Root mean square
        root_mean_square = np.sqrt(np.mean(data_frame.mag_sum_acc))

        # Step timing variability
        step_timing_variability = np.std([len(a) for a in steps])

        # Strides
        first_strides = list(zip(first_foot[::2], first_foot[1::2]))
        second_strides = list(zip(second_foot[::2], second_foot[1::2]))

        # Stride duration
        first_str_duration = [np.sum(a) / self.sampling_frequency for a in first_strides]
        second_str_duration = [np.sum(a) / self.sampling_frequency for a in second_strides]

        # Stride frequency
        


        return number_of_steps, cadence, velocity, \
            avg_step_length, avg_stride_length, step_durations, \
            avg_step_duration, sd_step_durations, strides, stride_durations, \
            avg_number_of_strides, avg_stride_duration, sd_stride_durations, \
            step_regularity, stride_regularity, symmetry

    def micro_pace(self, data_frame, axis='x'):
        time = data_frame.td.values
        data = data_frame[axis].values
        # data = butter_lowpass_filter(data, self.sampling_frequency, cutoff=self.cutoff_frequency)

        velocity = cumtrapz(data, time)
        # velocity = butter_lowpass_filter(velocity, self.sampling_frequency, cutoff=self.cutoff_frequency)

        displacement = cumtrapz(velocity, time[0: -1])

        step_length = 2 * np.sqrt(2 * self.pendulum_length * displacement - displacement ** 2)
        step_velocity = step_length / time[1: -1]

        mean_step_length = np.mean(step_length)
        mean_step_velocity = np.mean(step_velocity)

        return  mean_step_velocity,\
                mean_step_length,\
                # swing_time_variability,\
                # stance_time_variability,\
                # step_time_variability 

    def micro_rythm(self, data_frame):
        strikes, indices = self.heel_strikes(data_frame)
        step_durations = [strikes[i] - strikes[i-1] for i in range(1, np.size(strikes))]



        mean_step_time = np.mean(step_durations)

        return  mean_step_time,\
                # mean_swing_time,\
                # mean_stance_time

    def micro_variability(self, data_frame, axis='x'):
        time = data_frame.td.values
        data = data_frame[axis].values
        # data = butter_lowpass_filter(data, self.sampling_frequency, cutoff=self.cutoff_frequency)

        velocity = cumtrapz(data, time)
        # velocity = butter_lowpass_filter(velocity, self.sampling_frequency, cutoff=self.cutoff_frequency)

        displacement = cumtrapz(velocity, time[0: -1])

        step_length = 2 * np.sqrt(2 * self.pendulum_length * displacement - displacement ** 2)
        step_velocity = step_length / time[1: -1]

        step_length_variability = np.std(step_length)
        step_velocity_variability = np.std(step_velocity)

        return  step_velocity_variability,\
                step_length_variability

    def micro_asymmetry(self):
        pass

        # return  swing_time_asymmetry,\
        #         step_time_asymmetry,\
        #         stance_time_asymmetry

    def micro_pastural_control(self):
        pass

        # return step_length_asymmetry

    def macro_volume(self, data_frame):
        total_walking_time = data_frame.td[-1]
        data_frame = data_frame['x']

        strikes, _ = self.heel_strikes(data_frame)
        total_steps = np.size(strikes)

        return  total_walking_time,\
                total_steps,\
                # total_bouts,\
                # mean_bout_length



if __name__ == '__main__':
    ts = GaitTimeSeries()
    ts = ts.load_data('../tests/data/cloudupdrs_gait.csv')

    gp = GaitProcessor()

# fix readthedocs!!!!! !!!

# put this list in pivotal
# list of implented vs lackcing gait features from paper (ieee-xplore)
# george will send wavelet for step length without distance :)
# identify the mid-section in walking and characterize:
    # count number of steps it takes to fully rotate body
    # rotational acceleration
    # use quaternions for this

# add units of measurement for all outputs
# add references to functions