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
from pywt import wavedec

from .utils import load_data, numerical_integration, autocorrelation, peakdet
from .processor import Processor
from .gait_time_series import GaitTimeSeries

class GaitProcessor(Processor):
    '''
       This is the main Gait Processor class. Once the data is loaded it will be
       accessible at data_frame, where it looks like:
       data_frame.x, data_frame.y, data_frame.z: x, y, z components of the acceleration
       data_frame.index is the datetime-like index
       
       This values are recommended by the author of the pilot study [1] and [3]
       
       step_size = 50.0
       start_offset = 100
       end_offset = 100
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

    def __init__(self, step_size=50.0, start_offset=100, end_offset=100, delta=0.5, loco_band=[0.5, 3], freeze_band=[3, 8]):
        super().__init__()

        self.step_size = step_size
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.delta = delta
        self.loco_band = loco_band
        self.freeze_band = freeze_band


    def freeze_of_gait(self, data_frame):
        ''' 
            This method assess freeze of gait following [3].

            :param DataFrame data_frame: the data frame.
            :return [list, list, list]: The returns are [freeze_times, freeze_indexes, locomotion_freezes].
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

        return [freeze_times, freeze_indexes, locomotion_freezes]

    def frequency_of_peaks(self, data_frame):
        ''' 
            This method assess the frequency of the peaks on the x-axis.

            :param DataFrame data_frame: the data frame.
            :return float: The frequency of peaks on the x-axis.
        '''

        peaks_data = data_frame[self.start_offset: -self.end_offset].x.values
        maxtab, mintab = peakdet(peaks_data, self.delta)
        x = np.mean(peaks_data[maxtab[1:,0].astype(int)] - peaks_data[maxtab[:-1,0].astype(int)])
        frequency_from_peaks = 1/x

        return frequency_of_peaks
        
    def speed_of_gait(self, data_frame, wavelet_type='db3', wavelet_level=6):
        ''' 
            This method assess the speed of gait following [2].
            It extracts the gait speed from the energies of the approximation coefficients of wavelet functions.

            :param DataFrame data_frame: the data frame.
            :param str wavelet_type: the type of wavelet to use. See https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html for a full list.
            :param int wavelet_level: the number of cycles the used wavelet should have. See https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html for a fill list.
            :return float: The speed of gait.
        '''

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
            :return [list, list, list]: The returns are [step_regularity, stride_regularity, walk_symmetry] and each list consists of [x, y, z].
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

        return [step_regularity, stride_regularity, walk_symmetry]

