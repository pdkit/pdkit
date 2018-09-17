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

       [2] Determination of a Patient's Speed and Stride Length Minimizing Hardware Requirements (Body Sensor Networks (BSN), 2011 International Conference on, 144-149)
            Martin, E. ; EECS Dept., Univ. of California, Berkeley, Berkeley, CA, USA ; Shia, V. ; Bajcsy, R.

       [3] M. Bachlin et al., "Wearable Assistant for Parkinson’s Disease Patients With the Freezing of Gait Symptom,"
           in IEEE Transactions on Information Technology in Biomedicine, vol. 14, no. 2, pp. 436-446, March 2010.

       [4] The use of the fast Fourier transform for the estimation of power spectra: A method based 
            on time averaging over short, modified periodograms (IEEE Trans. Audio Electroacoust. 
            vol. 15, pp. 70-73, 1967)
            P. Welch
    """

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
                 distance=0,
                 duration=0,
                 step_period=2,
                 stride_period=1,
                 pendulum_length=1,
                 ):

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
        self.duration = duration
        self.step_period = step_period
        self.stride_period = stride_period
        self.pendulum_length = pendulum_length
        
    def add_manual_segmentation_to_data_frame(self, data_frame, segmentation_dictionary):
        """ Utility method to store manual segmentation of gait time series.
            
            :param dict dictionary: A dictionary of the form {'signal_type': [(from, to), (from, to)], ..., 'signal_type': [(from, to), (from, to)]}
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
            
    def plot_segmentation_dictionary(self, data_frame, segmentation_dictionary, axis='mag_sum_acc', figsize=(10, 5)):
        data = data_frame[axis]
        
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize[0], figsize[1])

        colors = 'bgrcmykw'

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
        colors = 'bgrcmykw'
        
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
        
    def save_segmentation_as_json(self, file_path):
        with open('{}'.format(file_path), 'w') as fp:
            json.dump(self.segmentation, fp)
            
    def load_segmentation_from_json(self, file_name):
        with open('{}'.format(file_name), 'r') as fp:
            
            # do some checks ...
            
            self.segmentation = json.load(fp)

    def freeze_of_gait(self, data_frame_axis):
        """ This method assess freeze of gait following [3].

            :param Array data_frame_axis: A specific axis of the data frame. This could be x, y, z or mag_sum_acc.

            :return list freeze_time: What times do freeze of gait events occur. [measured in time (h:m:s)]
            :return list freeze_indexe: Freeze Index is defined as the power in the “freeze” band [3–8 Hz] di-vided by the power in the “locomotor” band [0.5–3 Hz] [3]. [measured in Hz]
            :return list locomotor_freeze_index: Locomotor freeze index is the power in the “freeze” band [3–8 Hz] added to power in the “locomotor” band [0.5–3 Hz]. [measured in Hz]
        """
        
        # the sampling frequency was recommended by the author of the pilot study
        data = self.resample_signal(data_frame_axis) 
        data = data.values

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


    def frequency_of_peaks(self, data_frame_axis):
        """ This method assess the frequency of the peaks on any given axis.

            :param Array data_frame_axis: A specific axis of the data frame. This could be x, y, z or mag_sum_acc.
            
            :return float frequency_of_peaks: The frequency of peaks on the x-axis. [measured in Hz]
        """

        peaks_data = data_frame_axis[self.start_end_offset[0]: -self.start_end_offset[1]].values
        maxtab, mintab = peakdet(peaks_data, self.delta)

        x = np.mean(peaks_data[maxtab[1:,0].astype(int)] - peaks_data[maxtab[:-1,0].astype(int)])

        frequency_of_peaks = abs(1/x)

        return frequency_of_peaks
        

    def speed_of_gait(self, data_frame_axis, wavelet_type='db3', wavelet_level=6):
        """ This method assess the speed of gait following [2].
            It extracts the gait speed from the energies of the approximation coefficients of wavelet functions.
            Prefferably you should use the magnitude of x, y and z (mag_acc_sum) here.

            :param Series data_frame_axis: A specific axis of the data frame. This could be x, y, z or mag_sum_acc.
            :param str wavelet_type: the type of wavelet to use. See https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html for a full list.
            :param int wavelet_level: the number of cycles the used wavelet should have. See https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html for a fill list.
            
            :return float gait_speed: The speed of gait. [measured in meters/second]
        """

        coeffs = wavedec(data_frame_axis.values, wavelet=wavelet_type, level=wavelet_level)

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
        """ This method extracts the step and stride regularity and also walk symmetry.

            :param DataFrame data_frame: The data frame. It should have x, y and z columns.

            :return list step_regularity: Regularity of steps on [x, y, z] coordinates. Defined as the consistency of the step-to-step pattern.
            :return list stride_regularity: Regularity of stride on [x, y, z] coordinates. Defined as the consistency of the stride-to-stride pattern.
            :return list walk_symmetry: Symmetry of walk on [x, y, z] coordinates. Defined as the difference between step and stride regularity.
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

        step_regularity = [step_regularity_x, step_regularity_y, step_regularity_z]
        stride_regularity = [stride_regularity_x, stride_regularity_y, stride_regularity_z]
        walk_symmetry = [symmetry_x, symmetry_y, symmetry_z]

        return step_regularity, stride_regularity, walk_symmetry


    def walk_direction_preheel(self, data_frame):
        """ Estimate local walk (not cardinal) direction with pre-heel strike phase.

            Inspired by Nirupam Roy's B.E. thesis: "WalkCompass:
            Finding Walking Direction Leveraging Smartphone's Inertial Sensors,".

            :param DataFrame data_frame: The data frame. It should have x, y and z columns.

            :return array direction: Unit vector of local walk (not cardinal) direction.
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


    def heel_strikes(self, data_frame_axis):
        """ Estimate heel strike times between sign changes in accelerometer data.

            :param Series data_frame_axis: A specific axis of the data frame. This could be x, y, z or mag_sum_acc.
            :param string axis: The axis (preferably forward direction) from which to extract all gait features.

            :return array strikes: Heel strike timings
            :return list strike_indices: Heel strike timing indices.
        """
        
        # Demean data:
        data = data_frame_axis.values
        data -= data.mean()
    
        # TODO: fix this
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
        """ Compute step and stride regularity and symmetry from accelerometer data.

            :param Series data_frame_axis: A specific axis of the data frame. This could be x, y, z or mag_sum_acc.
            :param string axis: The axis (preferably forward direction) from which to extract all gait features.
            :param int unbias: Unbiased autocorrelation: divide by range (1) or by weighted range (2).
            :param int normalize: Normalize: divide by 1st coefficient (1) or by maximum abs. value (2).

            :return float step_regularity: Step regularity measure along axis.
            :return float stride_regularity: Stride regularity measure along axis.
            :return symmetry: Symmetry measure along axis.
        """

        coefficients, _ = autocorrelate(data_frame_axis, unbias=1, normalize=2)

        step_regularity = coefficients[self.step_period]
        stride_regularity = coefficients[self.stride_period]
        symmetry = np.abs(stride_regularity - step_regularity)

        return step_regularity, stride_regularity, symmetry


    def gait(self, data_frame_axis):
        """ Extract gait features from estimated heel strikes and accelerometer data.

            :param DataFrame data_frame: The data frame.
            :param string axis: The axis (preferably forward direction) from which to extract all gait features.

            :return int number_of_steps: Estimated number of steps based on heel strikes. [number of steps]
            :return float velocity: Velocity (if distance). [meters/second]
            :return float avg_step_length: Average step length (if distance). [meters]
            :return float avg_stride_length: Average stride length (if distance). [meters]
            :return float cadence: Number of steps divided by duration. [steps / second]
            :return array step_durations: Step duration. [array of seconds]
            :return float avg_step_duration: Average step duration. [seconds]
            :return float sd_step_durations: Standard deviation of step durations. [array of seconds]
            :return list strides: Stride timings for each side. [array of seconds]
            :return float avg_number_of_strides: Estimated number of strides based on alternating heel strikes. [number of strides]
            :return list stride_durations: Estimated stride durations. [array of seconds]
            :return float avg_stride_duration: Average stride duration. [seconds]
            :return float sd_step_durations: Standard deviation of stride durations. [seconds]
            :return float step_regularity: Measure of step regularity along axis. [percentage consistency of the step-to-step pattern]
            :return float stride_regularity: Measure of stride regularity along axis. [percentage consistency of the stride-to-stride pattern]
            :return float symmetry: Measure of gait symmetry along axis. [difference between step and stride regularity]
        """
        
        data = data_frame_axis
        
        strikes, strikes_ids = self.heel_strikes(data)

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

        self.step_period = np.int(np.round(1 / avg_step_duration))
        self.stride_period = np.int(np.round(1 / avg_stride_duration))

        step_regularity, stride_regularity, symmetry = self.gait_regularity_symmetry(data)

        cadence = None
        if self.duration:
            cadence = number_of_steps / self.duration
        
        velocity = None
        avg_step_length = None
        avg_stride_length = None
        
        # Set distance-based measures to None if distance not set:
        if self.distance:
            velocity = self.distance / self.duration
            avg_step_length = number_of_steps / self.distance
            avg_stride_length = avg_number_of_strides / self.distance
            
        else:
            velocity = None
            avg_step_length = None
            avg_stride_length = None

        return number_of_steps, cadence, velocity, \
            avg_step_length, avg_stride_length, step_durations, \
            avg_step_duration, sd_step_durations, strides, stride_durations, \
            avg_number_of_strides, avg_stride_duration, sd_stride_durations, \
            step_regularity, stride_regularity, symmetry
    
    def separate_into_sections(self, data_frame, labels_col='anno', labels_to_keep=[1,2], min_labels_in_sequence=100):
        """ Helper function to separate a long signal into multiple sections based on a labeled column.
        
            :param DataFrame data_frame: The data frame.
            :param string labels_col: The column which has the labels we would like to separate on.
            :param list labels_to_keep: The id's of the labels which we would like to keep, out of all the labels in the labels_col.
            :param int min_labels_in_sequence: The minimum number of samples which can make up a section.
            
            :return list section: A list of DataFrames, segmented accordingly.
        
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
    
    def bellman_segmentation(self, data_frame, states):
        """ Divide a univariate time-series, data_frame, into states contiguous segments, using Bellman k-segmentation algorithm on the peak prominences of the data.
        
            :param DataFrame data_frame: The data frame.
            :param int states: Number of contigous segments.
            
            :return list peaks: The peaks in our data_frame.
            :return list primi: Peaks prominences.
            :return list bellman_idx: The indices of the segments.
        
        """
        peaks, prominences = get_signal_peaks_and_prominences(data_frame[self.axis])
        bellman_idx = BellmanKSegment(prominences, states)
        
        return peaks, prominences, bellman_idx
    
    def sklearn_segmentation(self, data_frame, cluster_fn):
        """ Divide a univariate time-series, data_frame, into states contiguous segments, using sk-learn clustering algorithms on the peak prominences of the data.
        
            :param DataFrame data_frame: The data frame.
            :param sklearn.aglorithm cluster_fn: Any unsupervised learning algorithm from the sklearn library. It needs to have the `fit_predict` method.
            
            :return list peaks: The peaks in our data_frame.
            :return list primi: Peaks prominences.
            :return list sklearn_idx: The indices of the segments.
        
        """
        peaks, prominences = get_signal_peaks_and_prominences(data_frame[self.axis])
        
        # sklearn fix: reshape to (-1, 1)
        sklearn_idx = cluster_fn.fit_predict(prominences.reshape(-1, 1))
        
        return peaks, prominences, sklearn_idx