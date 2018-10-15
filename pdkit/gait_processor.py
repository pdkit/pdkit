#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2018 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author: Cosmin Stamate 

import sys
import traceback

import numpy as np
import pandas as pd
import json

from scipy import interpolate, signal, fft
from scipy.fftpack import rfft
from pywt import wavedec

from pdkit.processor import Processor
from pdkit.gait_time_series import GaitTimeSeries

from scipy.integrate import cumtrapz
import matplotlib.pylab as plt
import matplotlib.patches as mpatches

from pdkit.utils import (load_data,
                        numerical_integration, 
                        autocorrelation,
                        peakdet,
                        compute_interpeak,
                        butter_lowpass_filter,
                        crossings_nonzero_pos2neg,
                        autocorrelate,
                        get_signal_peaks_and_prominences,
                        BellmanKSegment)

class GaitProcessor(Processor):
    """
        This is the main Gait Processor class. Once the data is loaded it will be
        accessible at data_frame, where it looks like:
        data_frame.x, data_frame.y, data_frame.z: x, y, z components of the acceleration
        data_frame.index is the datetime-like index
       
        These values are recommended by the author of the pilot study :cite:`g-Kassavetis2015, g-BachlinPRMHGT10`
       
        :param sampling_frequency: (optional) the sampling frequency in Hz (100.0 default)
        :type sampling_frequency: float
        :param cutoff_frequency: (optional) the cutoff frequency in Hz (2.0 default)
        :type cutoff_frequency: float
        :param filter_order: (optional) filter order (2 default)
        :type filter_order: int
        :param window: (optional) The size of the window used to traverse the time series (256 default)
        :type window: int
        :param lower_frequency: (optional) lower frequency in Hz (2.0 default)
        :type lower_frequency: float
        :param upper_frequency: (optional) upper frequency in Hz (10.0 default)
        :type upper_frequency: float
        :param loco_band: The ratio of the energy in the locomotion band, measured in Hz ([0.5, 3] default)
        :type loco_band: list
        :param freeze_band: The ration of energy in the freeze band, measured in Hz ([3, 8] default)
        :type freeze_band: list
        :param step_size: The average step size in centimeters (50.0 default).
        :type step_size: float
        :param delta: A point is considered a maximum peak if it has the maximal value, and was preceded (to the left) by a value lower by delta (0.5 default).
        :type delta: float
        :param duration: The duration in the the same time unit as the time series (None default). 
        :type duration: float
        :param distance: The total walking distance in meters (None default).
        :type distance: float

        :Examples:
         
        >>> import pdkit
        >>> gp = pdkit.GaitProcessor()
        >>> ts = pdkit.GaitTimeSeries().load(path_to_data)
        >>> rts = gp.resample(ts)
    """

    def __init__(self,
                 sampling_frequency=100.0,
                 cutoff_frequency=2.0,
                 filter_order=2,
                 window=256,
                 lower_frequency=2.0,
                 upper_frequency=10.0,
                 loco_band=[0.5, 3],
                 freeze_band=[3, 8],
                 step_size=50.0,
                 delta=0.5,
                 stride_fraction=1.0/8.0,
                 duration=None,
                 distance=None
                 ):

        super().__init__(sampling_frequency,
                         cutoff_frequency,
                         filter_order,
                         window,
                         lower_frequency,
                         upper_frequency)

        self.delta = delta
        self.step_size = step_size
        self.loco_band = loco_band
        self.freeze_band = freeze_band
        self.stride_fraction = stride_fraction
        self.duration=duration
        self.distance=distance
        
    def freeze_of_gait(self, x):
        """ 
            This method assess freeze of gait following :cite:`g-BachlinPRMHGT10`.

            :param x: The time series to assess freeze of gait on. This could be x, y, z or mag_sum_acc.
            :type x: pandas.Series
            :return freeze_time: What times do freeze of gait events occur. [measured in time (h:m:s)]
            :rtype freeze_time: numpy.ndarray
            :return freeze_indexe: Freeze Index is defined as the power in the “freeze” band [3–8 Hz] divided by the power in the “locomotor” band [0.5–3 Hz] [3]. [measured in Hz]
            :rtype freeze_indexe: numpy.ndarray
            :return list locomotor_freeze_index: Locomotor freeze index is the power in the “freeze” band [3–8 Hz] added to power in the “locomotor” band [0.5–3 Hz]. [measured in Hz]
            :rtype locomotor_freeze_index: numpy.ndarray
        """

        data = self.resample_signal(x).values

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
            Pyy = abs(Y*Y) / self.window

            areaLocoBand = numerical_integration( Pyy[f_nr_LBs-1 : f_nr_LBe], self.sampling_frequency )
            areaFreezeBand = numerical_integration( Pyy[f_nr_FBs-1 : f_nr_FBe], self.sampling_frequency )

            sumLocoFreeze.append(areaFreezeBand + areaLocoBand)

            freezeIndex.append(areaFreezeBand / areaLocoBand)

            jPos = jPos + self.step_size
            i = i + 1

        freeze_time = np.asarray(time, dtype=np.int32)
        freeze_index = np.asarray(freezeIndex, dtype=np.float32)
        locomotor_freeze_index = np.asarray(sumLocoFreeze, dtype=np.float32)

        return freeze_time, freeze_index, locomotor_freeze_index


    def frequency_of_peaks(self, x, start_offset=100, end_offset=100):
        """ 
            This method assess the frequency of the peaks on any given 1-dimensional time series.

            :param x: The time series to assess freeze of gait on. This could be x, y, z or mag_sum_acc.
            :type x: pandas.Series
            :param start_offset: Signal to leave out (of calculations) from the begining of the time series (100 default).
            :type start_offset: int
            :param end_offset: Signal to leave out (from calculations) from the end of the time series (100 default).
            :type end_offset: int
            :return frequency_of_peaks: The frequency of peaks on the provided time series [measured in Hz].
            :rtype frequency_of_peaks: float
        """

        peaks_data = x[start_offset: -end_offset].values
        maxtab, mintab = peakdet(peaks_data, self.delta)

        x = np.mean(peaks_data[maxtab[1:,0].astype(int)] - peaks_data[maxtab[:-1,0].astype(int)])

        frequency_of_peaks = abs(1/x)

        return frequency_of_peaks
        

    def speed_of_gait(self, x, wavelet_type='db3', wavelet_level=6):
        """ 
            This method assess the speed of gait following :cite:`g-MartinSB11`.

            It extracts the gait speed from the energies of the approximation coefficients of wavelet functions.
            Prefferably you should use the magnitude of x, y and z (mag_acc_sum) here, as the time series.

            :param x: The time series to assess freeze of gait on. This could be x, y, z or mag_sum_acc.
            :type x: pandas.Series
            :param wavelet_type: The type of wavelet to use. See https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html for a full list ('db3' default).
            :type wavelet_type: str
            :param wavelet_level: The number of cycles the used wavelet should have. See https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html for a fill list (6 default). 
            :type wavelet_level: int
            :return: The speed of gait [measured in meters/second].
            :rtype: float
        """

        coeffs = wavedec(x.values, wavelet=wavelet_type, level=wavelet_level)

        energy = [sum(coeffs[wavelet_level - i]**2) / len(coeffs[wavelet_level - i]) for i in range(wavelet_level)]

        WEd1 = energy[0] / (5 * np.sqrt(2))
        WEd2 = energy[1] / (4 * np.sqrt(2))
        WEd3 = energy[2] / (3 * np.sqrt(2))
        WEd4 = energy[3] / (2 * np.sqrt(2))
        WEd5 = energy[4] / np.sqrt(2)
        WEd6 = energy[5] / np.sqrt(2)

        gait_speed = 0.5 * np.sqrt(WEd1+(WEd2/2)+(WEd3/3)+(WEd4/4)+(WEd5/5)+(WEd6/6))

        return gait_speed


    def walk_regularity_symmetry(self, data_frame):
        """ 
            This method extracts the step and stride regularity and also walk symmetry.

            :param data_frame: The data frame. It should have x, y, and z columns.
            :type data_frame: pandas.DataFrame
            :return step_regularity: Regularity of steps on [x, y, z] coordinates, defined as the consistency of the step-to-step pattern.
            :rtype step_regularity: numpy.ndarray
            :return stride_regularity: Regularity of stride on [x, y, z] coordinates, defined as the consistency of the stride-to-stride pattern.
            :rtype stride_regularity: numpy.ndarray
            :return walk_symmetry: Symmetry of walk on [x, y, z] coordinates, defined as the difference between step and stride regularity.
            :rtype walk_symmetry: numpy.ndarray
        """
        
        def _symmetry(v):
            maxtab, _ = peakdet(v, self.delta)
            return maxtab[1][1], maxtab[2][1]

        step_regularity_x, stride_regularity_x = _symmetry(autocorrelation(data_frame.x))
        step_regularity_y, stride_regularity_y = _symmetry(autocorrelation(data_frame.y))
        step_regularity_z, stride_regularity_z = _symmetry(autocorrelation(data_frame.z))

        symmetry_x = step_regularity_x - stride_regularity_x
        symmetry_y = step_regularity_y - stride_regularity_y
        symmetry_z = step_regularity_z - stride_regularity_z

        step_regularity = np.array([step_regularity_x, step_regularity_y, step_regularity_z])
        stride_regularity = np.array([stride_regularity_x, stride_regularity_y, stride_regularity_z])
        walk_symmetry = np.array([symmetry_x, symmetry_y, symmetry_z])

        return step_regularity, stride_regularity, walk_symmetry


    def walk_direction_preheel(self, data_frame):
        """ 
            Estimate local walk (not cardinal) direction with pre-heel strike phase.

            Inspired by Nirupam Roy's B.E. thesis: "WalkCompass: Finding Walking Direction Leveraging Smartphone's Inertial Sensors"

            :param data_frame: The data frame. It should have x, y, and z columns.
            :type data_frame: pandas.DataFrame
            :return: Unit vector of local walk (not cardinal) direction.
            :rtype: numpy.ndarray
        """

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


    def heel_strikes(self, x):
        """ Estimate heel strike times between sign changes in accelerometer data.

            :param x: The time series to assess freeze of gait on. This could be x, y, z or mag_sum_acc.
            :type x: pandas.Series
            :return strikes: Heel strike timings measured in seconds.
            :rtype striles: numpy.ndarray
            :return strikes_idx: Heel strike timing indices of the time series.
            :rtype strikes_idx: numpy.ndarray
        """
        
        # Demean data:
        data = x.values
        data -= data.mean()
    
        # TODO: fix this
        # Low-pass filter the AP accelerometer data by the 4th order zero lag
        # Butterworth filter whose cut frequency is set to 5 Hz:
        filtered = butter_lowpass_filter(data, self.sampling_frequency, self.cutoff_frequency, self.filter_order)

        # Find transitional positions where AP accelerometer changes from
        # positive to negative.
        transitions = crossings_nonzero_pos2neg(filtered)

        # Find the peaks of AP acceleration preceding the transitional positions,
        # and greater than the product of a threshold and the maximum value of
        # the AP acceleration:
        strike_indices_smooth = []
        filter_threshold = np.abs(self.delta * np.max(filtered))
        for i in range(1, np.size(transitions)):
            segment = range(transitions[i-1], transitions[i])
            imax = np.argmax(filtered[segment])
            if filtered[segment[imax]] > filter_threshold:
                strike_indices_smooth.append(segment[imax])

        # Compute number of samples between peaks using the real part of the FFT:
        interpeak = compute_interpeak(data, self.sampling_frequency)
        decel = np.int(interpeak / 2)

        # Find maximum peaks close to maximum peaks of smoothed data:
        strikes_idx = []
        for ismooth in strike_indices_smooth:
            istrike = np.argmax(data[ismooth - decel:ismooth + decel])
            istrike = istrike + ismooth - decel
            strikes_idx.append(istrike)

        strikes = np.asarray(strikes_idx)
        strikes -= strikes[0]
        strikes = strikes / self.sampling_frequency

        return strikes, np.array(strikes_idx)

    def gait_regularity_symmetry(self, x, average_step_duration='autodetect', average_stride_duration='autodetect', unbias=1, normalize=2):
        """ 
            Compute step and stride regularity and symmetry from accelerometer data with the help of steps and strides.

            :param x: The time series to assess freeze of gait on. This could be x, y, z or mag_sum_acc.
            :type x: pandas.Series
            :param average_step_duration: Average duration of each step using the same time unit as the time series. If this is set to 'autodetect' it will infer this from the time series.
            :type average_step_duration: float
            :param average_stride_duration: Average duration of each stride using the same time unit as the time series. If this is set to 'autodetect' it will infer this from the time series.
            :type average_stride_duration: float
            :param unbias: Unbiased autocorrelation: divide by range (unbias=1) or by weighted range (unbias=2).
            :type unbias: int
            :param int normalize: Normalize: divide by 1st coefficient (normalize=1) or by maximum abs. value (normalize=2).
            :type normalize: int
            :return step_regularity: Step regularity measure along axis.
            :rtype step_regularity: float
            :return stride_regularity: Stride regularity measure along axis.
            :rtype stride_regularity: float
            :return symmetry: Symmetry measure along axis.
            :rtype symmetry: float
        """
        if (average_step_duration=='autodetect') or (average_stride_duration=='autodetect'):

            strikes, _ = self.heel_strikes(x)

            step_durations = []
            for i in range(1, np.size(strikes)):
                step_durations.append(strikes[i] - strikes[i-1])

            average_step_duration = np.mean(step_durations)

            number_of_steps = np.size(strikes)

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

            average_stride_duration = np.mean((np.mean(stride_durations1),
                                        np.mean(stride_durations2)))

            return self.gait_regularity_symmetry(x, average_step_duration, average_stride_duration)
        
        else:
            coefficients, _ = autocorrelate(x, unbias=1, normalize=2)

            step_period = np.int(np.round(1 / average_step_duration))
            stride_period = np.int(np.round(1 / average_stride_duration))

            step_regularity = coefficients[step_period]
            stride_regularity = coefficients[stride_period]
            symmetry = np.abs(stride_regularity - step_regularity)

            return step_regularity, stride_regularity, symmetry


    def gait(self, x):
        """ 
            Extract gait features from estimated heel strikes and accelerometer data.

            :param x: The time series to assess freeze of gait on. This could be x, y, z or mag_sum_acc.
            :type x: pandas.Series
            :return number_of_steps: Estimated number of steps based on heel strikes [number of steps].
            :rtype number_of_steps: int
            :return velocity: Velocity (if distance is provided) [meters/second].
            :rtype velocity: float
            :return avg_step_length: Average step length (if distance is provided) [meters].
            :rtype avg_step_length: float
            :return avg_stride_length: Average stride length (if distance is provided) [meters].
            :rtyoe avg_stride_length: float
            :return cadence: Number of steps divided by duration [steps/second].
            :rtype cadence: float
            :return array step_durations: Step duration [seconds].
            :rtype step_durations: np.ndarray
            :return float avg_step_duration: Average step duration [seconds].
            :rtype avg_step_duration: float
            :return float sd_step_durations: Standard deviation of step durations [seconds].
            :rtype sd_step_durations: np.ndarray
            :return list strides: Stride timings for each side [seconds].
            :rtype strides: numpy.ndarray
            :return float avg_number_of_strides: Estimated number of strides based on alternating heel strikes [number of strides].
            :rtype avg_number_of_strides: float
            :return list stride_durations: Estimated stride durations [seconds].
            :rtype stride_durations: numpy.ndarray
            :return float avg_stride_duration: Average stride duration [seconds].
            :rtype avg_stride_duration: float
            :return float sd_step_durations: Standard deviation of stride durations [seconds].
            :rtype sd_step_duration: float
            :return float step_regularity: Measure of step regularity along axis [percentage consistency of the step-to-step pattern].
            :rtype step_regularity: float
            :return float stride_regularity: Measure of stride regularity along axis [percentage consistency of the stride-to-stride pattern].
            :rtype stride_regularity: float
            :return float symmetry: Measure of gait symmetry along axis [difference between step and stride regularity].
            :rtype symmetry: float
        """
        
        data = x
        
        strikes, _ = self.heel_strikes(data)

        step_durations = []
        for i in range(1, np.size(strikes)):
            step_durations.append(strikes[i] - strikes[i-1])

        avg_step_duration = np.mean(step_durations)
        sd_step_durations = np.std(step_durations)

        number_of_steps = np.size(strikes)

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

        step_period = np.int(np.round(1 / avg_step_duration))
        stride_period = np.int(np.round(1 / avg_stride_duration))

        step_regularity, stride_regularity, symmetry = self.gait_regularity_symmetry(data,
                                                                                     average_step_duration=avg_step_duration, 
                                                                                     average_stride_duration=avg_stride_duration)

        cadence = None
        if self.duration:
            cadence = number_of_steps / self.duration
        

        velocity = None
        avg_step_length = None
        avg_stride_length = None
        
        if self.distance:
            velocity = self.distance / self.duration
            avg_step_length = number_of_steps / self.distance
            avg_stride_length = avg_number_of_strides / self.distance

        return [number_of_steps, cadence,
               velocity,
               avg_step_length,
               avg_stride_length,
               step_durations,
               avg_step_duration,
               sd_step_durations,
               strides,
               stride_durations,
               avg_number_of_strides,
               avg_stride_duration,
               sd_stride_durations,
               step_regularity,
               stride_regularity, 
               symmetry]
    
    def separate_into_sections(self, data_frame, labels_col='anno', labels_to_keep=[1,2], min_labels_in_sequence=100):
        """ Helper function to separate a time series into multiple sections based on a labeled column.
        
            :param data_frame: The data frame. It should have x, y, and z columns.
            :type data_frame: pandas.DataFrame
            :param labels_col: The column which has the labels we would like to separate the data_frame on on ('anno' default).
            :type labels_col: str
            :param labels_to_keep: The unique labele ids of the labels which we would like to keep, out of all the labels in the labels_col ([1, 2] default).
            :type labels_to_keep: list
            :param min_labels_in_sequence: The minimum number of samples which can make up a section (100 default).
            :type min_labels_in_sequence: int
            
            :return: A list of DataFrames, segmented accordingly.
            :rtype: list
        """
        sections = [[]]
        
        mask = data_frame[labels_col].apply(lambda x: x in labels_to_keep)

        for i,m in enumerate(mask):
            if m:
                sections[-1].append(i)
                
            if not m and len(sections[-1]) > min_labels_in_sequence:
                sections.append([])
        
        sections.pop()
        
        sections = [self.rebuild_indexes(data_frame.iloc[s]) for s in sections]
        
        return sections
    
    def bellman_segmentation(self, x, states):
        """ 
            Divide a univariate time-series, data_frame, into states contiguous segments, using Bellman k-segmentation algorithm on the peak prominences of the data.
        
            :param x: The time series to assess freeze of gait on. This could be x, y, z or mag_sum_acc.
            :type x: pandas.Series
            :param states: Number of contigous segments.
            :type states: int
            :return peaks: The peaks in our data_frame.
            :rtype peaks: list
            :return prominences: Peaks prominences.
            :rtype prominences: list
            :return bellman_idx: The indices of the segments.
            :rtype bellman_idx: list
        """
        peaks, prominences = get_signal_peaks_and_prominences(x)
        bellman_idx = BellmanKSegment(prominences, states)
        
        return peaks, prominences, bellman_idx
    
    def sklearn_segmentation(self, x, cluster_fn):
        """ 
            Divide a univariate time-series, data_frame, into states contiguous segments, using sk-learn clustering algorithms on the peak prominences of the data.
        
            :param x: The time series to assess freeze of gait on. This could be x, y, z or mag_sum_acc.
            :type x: pandas.Series
            :param cluster_fn: Any unsupervised learning algorithm from the sklearn library. It needs to have the `fit_predict` method.
            :param cluster_fn: sklearn.aglorithm
            :return peaks: The peaks in our data_frame.
            :rtype peaks: list
            :return prominences: Peaks prominences.
            :rtype prominences: list
            :return sklearn_idx: The indices of the segments.
            :rtype sklearn_idx: list
        """
        peaks, prominences = get_signal_peaks_and_prominences(x)
        
        # sklearn fix: reshape to (-1, 1)
        sklearn_idx = cluster_fn.fit_predict(prominences.reshape(-1, 1))
        
        return peaks, prominences, sklearn_idx

    def add_manual_segmentation_to_data_frame(self, data_frame, segmentation_dictionary):
        """ 
            Utility method to store manual segmentation of gait time series.

            :param data_frame: The data frame. It should have x, y, and z columns.
            :type data_frame: pandas.DataFrame
            :param segmentation_dictionary: A dictionary of the form {'signal_type': [(from, to), (from, to)], ..., 'signal_type': [(from, to), (from, to)]}. The from and to can either be of type numpy.datetime64 or int, depending on how you are segmenting the time series.
            :type segmentation_dictionary: dict
            :return: The data_frame with a new column named 'segmentation'.
            :rtype: pandas.DataFrame
        """
        
        # add some checks to see if dictionary is in the right format!
        
        data_frame['segmentation'] = 'unknown'
        
        for i, (k, v) in enumerate(segmentation_dictionary.items()):
            for start, end in v:                
                if type(start) != np.datetime64:
                    if start < 0: start = 0
                    if end > data_frame.size: end = data_frame.size
                    
                    start = data_frame.index.values[start]
                    end = data_frame.index.values[end]
                    
                data_frame.loc[start: end, 'segmentation'] = k
        
        return data_frame
            
    def plot_segmentation_dictionary(self, x, segmentation_dictionary, figsize=(10, 5)):
        """ 
            Utility method used to visualize how the segmentation dictionary interacts with the time series.

            :param data_frame: The data frame. It should have x, y, and z columns.
            :type data_frame: pandas.DataFrame
            :param segmentation_dictionary: A dictionary of the form {'signal_type': [(from, to), (from, to)], ..., 'signal_type': [(from, to), (from, to)]}.
            :type segmentation_dictionary: dict
            :param figsize: The size of the figure where we will plot the segmentation on top of the provided time series ((10, 5) default).
            :type figsize: tuple
        """
        data = x
        
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize[0], figsize[1])

        # fix this!!
        colors = 'bgrcmykwbgrcmykwbgrcmykw'

        data.plot(ax=ax)

        for i, (k, v) in enumerate(segmentation_dictionary.items()):
            for start, end in v:
                if type(start) != np.datetime64:
                    start = data.index.values[start]
                    end = data.index.values[end]

                plt.axvspan(start, end, color=colors[i], alpha=0.5)

        legend = [mpatches.Patch(color=colors[i], label="{}".format(k)) for i, k in enumerate(segmentation_dictionary.keys())]

        plt.legend(handles=legend)

        plt.show()
        
    def plot_segmentation_data_frame(self, segmented_data_frame, axis='mag_sum_acc', figsize=(10, 5)):
        """ 
            Utility method used to visualize how the segmentation dictionary interacts with the time series.

            :param segmented_data_frame: The segmented data frame. It should have x, y, z and segmentation columns.
            :type segmented_data_frame: pandas.DataFrame
            :param axis: The axis which we want to plot. We can choose from x, y, z and mag_sum_acc ('mag_acc_sum' default).
            :type axis: str
            :param figsize: The size of the figure where we will plot the segmentation on top of the provided time series ((10, 5) default).
            :type figsize: tuple
        """
        # fix this!!
        colors = 'bgrcmykwbgrcmykwbgrcmykw'
        
        keys = np.unique(segmented_data_frame['segmentation'])
        
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize[0], figsize[1])
        
        segmented_data_frame[axis].plot(ax=ax)

        for i, k in enumerate(keys):
            patch = segmented_data_frame['segmentation'].loc[segmented_data_frame['segmentation'] == k].index
            for p in patch:
                ax.axvline(p, color=colors[i], alpha=0.1)

        legend = [mpatches.Patch(color=colors[i], label="{}".format(k)) for i, k in enumerate(keys)]

        plt.legend(handles=legend)

        plt.show()
