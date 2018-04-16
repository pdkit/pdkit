#!/usr/bin/env python3
# Copyright 2018 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author(s): J.S. Pons

import sys
import logging

import numpy as np
import pandas as pd
from scipy import interpolate, signal, fft
from tsfresh.feature_extraction import feature_calculators

class TremorProcessor:
    '''
        This is the main Tremor Processor class. Once the data is loaded it will be
        accessible at data_frame, where it looks like:
        data_frame.x, data_frame.y, data_frame.z: x, y, z components of the acceleration
        data_frame.index is the datetime-like index
        
        These values are recommended by the author of the pilot study [1]
        
        sampling_frequency = 100.0Hz
        cutoff_frequency = 2.0Hz
        filter_order = 2
        window = 256
        lower_frequency = 2.0Hz
        upper_frequency = 10.0Hz

        :References:
       
        [1] Developing a tool for remote digital assessment of Parkinson s disease Kassavetis	P,	Saifee	TA,	Roussos	G,	Drougas	L,	Kojovic	M,	Rothwell	JC,	Edwards	MJ,	Bhatia	KP
            
        [2] The use of the fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms (IEEE Trans. Audio Electroacoust. vol. 15, pp. 70-73, 1967) P. Welch
            
        :Example:
         
        >>> import pdkit
        >>> tp = pdkit.TremorProcessor()
        >>> path = 'path/to/data.csv’
        >>> ts = TremorTimeSeries().load(path)
        >>> amplitude, frequency = tp.process(ts)
    '''

    def __init__(self, sampling_frequency=100.0, cutoff_frequency=2.0, filter_order=2,
                 window=256, lower_frequency=2.0, upper_frequency=10.0):
        try:
            self.amplitude = 0
            self.frequency = 0

            self.sampling_frequency = sampling_frequency
            self.cutoff_frequency = cutoff_frequency
            self.filter_order = filter_order
            self.window = window
            self.lower_frequency = lower_frequency
            self.upper_frequency = upper_frequency

            logging.debug("TremorProcessor init")

        except IOError as e:
            ierr = "({}): {}".format(e.errno, e.strerror)
            logging.error("TremorProcessor I/O error %s", ierr)

        except ValueError as verr:
            logging.error("TremorProcessor ValueError ->%s", verr.message)

        except:
            logging.error("Unexpected error on TremorProcessor init: %s", sys.exc_info()[0])

    def resample_signal(self, data_frame):
        '''
            Convenience method for frequency conversion and resampling of data frame. 
            Object must have a DatetimeIndex. After re-sampling, this methods interpolate the time magnitude sum 
            acceleration values and the x,y,z values of the data frame acceleration

            :param data_frame: the data frame to resample
            :param str sampling_frequency: the sampling frequency. Default is 100Hz, as recommended by the author of the pilot study [1]
            :return data_frame: data_frame.x, data_frame.y, data_frame.z: x, y, z components of the acceleration data_frame.index is the datetime-like index
        '''
        df_resampled = data_frame.resample(str(1 / self.sampling_frequency) + 'S').mean()

        f = interpolate.interp1d(data_frame.td, data_frame.mag_sum_acc)
        new_timestamp = np.arange(data_frame.td[0], data_frame.td[-1], 1.0 / self.sampling_frequency)
        df_resampled.mag_sum_acc = f(new_timestamp)

        logging.debug("resample signal")
        return df_resampled.interpolate(method='linear')

    def filter_signal(self, data_frame):
        '''
            This method filters a data frame signal as suggested in [1]. First step is to high pass filter the data
            frame using a [Butterworth]_ digital and analog filter. Then this method 
            filters the data frame along one-dimension using a [digital]_ filter. 

            :param data_frame: the data frame    
            :return dataframe: data_frame.x, data_frame.y, data_frame.z: x, y, z components of the acceleration data_frame.index is the datetime-like index
        '''
        b, a = signal.butter(self.filter_order, 2 * self.cutoff_frequency / self.sampling_frequency, 'high', analog=False)
        filtered_signal = signal.lfilter(b, a, data_frame.mag_sum_acc.values)
        data_frame['filtered_signal'] = filtered_signal

        logging.debug("filter signal")
        return data_frame

    def fft_signal(self, data_frame):
        '''
            This method perform Fast Fourier Transform on the data frame using a [hanning]_ window

            :param data_frame: the data frame    
            :param str window: hanning window size
            :return data_frame: data_frame.filtered_signal, data_frame.transformed_signal, data_frame.z: x, y, z components of the acceleration data_frame.index is the datetime-like index
        '''
        signal_length = len(data_frame.filtered_signal.values)
        ll = int ( signal_length / 2 - self.window / 2 )
        rr = int ( signal_length / 2 + self.window / 2 )
        msa = data_frame.filtered_signal[ll:rr].values
        hann_window = signal.hann(self.window)

        msa_window = (msa * hann_window)
        transformed_signal = fft(msa_window)

        data = {'filtered_signal': msa_window, 'transformed_signal': transformed_signal,
                'dt': data_frame.td[ll:rr].values}

        data_frame_fft = pd.DataFrame(data, index=data_frame.index[ll:rr],
                                      columns=['filtered_signal', 'transformed_signal', 'dt'])
        logging.debug("fft signal")
        return data_frame_fft

    def tremor_amplitude(self, data_frame):
        '''
            This methods extract the fft components and sum the ones from lower to upper freq as per [1]

            :param data_frame: the data frame    
            :param str lower_frequency: LOWER_FREQUENCY_TREMOR
            :param str upper_frequency: UPPER_FREQUENCY_TREMOR
            :return: amplitude is the the amplitude of the Tremor
            :return: frequency is the frequency of the Tremor
        '''
        signal_length = len(data_frame.filtered_signal)
        normalised_transformed_signal = data_frame.transformed_signal.values / signal_length

        k = np.arange(signal_length)
        T = signal_length / self.sampling_frequency
        frq = k / T  # two sides frequency range

        frq = frq[range(int(signal_length / 2))]  # one side frequency range
        ts = normalised_transformed_signal[range(int(signal_length / 2))]
        amplitude = sum(abs(ts[(frq > self.lower_frequency) & (frq < self.upper_frequency)]))
        frequency = frq[abs(ts).argmax(axis=0)]

        logging.debug("tremor amplitude calculated")

        return amplitude, frequency

    def tremor_amplitude_by_welch(self, data_frame):
        '''
            This methods uses the Welch method [2] to obtain the power spectral density, this is a robust 
            alternative to using fft_signal & calc_tremor_amplitude

            :param data_frame: the data frame    
            :param str lower_frequency: LOWER_FREQUENCY_TREMOR
            :param str upper_frequency: UPPER_FREQUENCY_TREMOR
            :return: amplitude is the the amplitude of the Tremor
            :return: frequency is the frequency of the Tremor
        '''
        frq, Pxx_den = signal.welch(data_frame.filtered_signal.values, self.sampling_frequency, nperseg=self.window)
        frequency = frq[Pxx_den.argmax(axis=0)]
        amplitude = sum(Pxx_den[(frq > self.lower_frequency) & (frq < self.upper_frequency)])

        logging.debug("tremor amplitude by welch calculated")

        return amplitude, frequency

    def spkt_welch_density(x, param = [{"coeff":0}]):
        '''
        
        :param param: 
        :return: 
        '''
        welch = feature_calculators.spkt_welch_density(x, param)
        logging.debug("tremor amplitude by tsfresh welch calculated")
        return list(welch)[0][1]

    def process(self, data_frame, method='fft'):
        '''
            This methods calculates the tremor amplitude of the data frame. It accepts two different methods,
            'fft' and 'welch'. First the signal gets re-sampled and then high pass filtered.

            :param data_frame: the data frame    
            :param str method: fft or welch.
            :return: amplitude is the the amplitude of the Tremor
            :return: frequency is the frequency of the Tremor
        '''
        try:
            data_frame_resampled = self.resample_signal(data_frame)
            data_frame_filtered = self.filter_signal(data_frame_resampled)

            if method == 'fft':
                data_frame_fft = self.fft_signal(data_frame_filtered)
                return self.tremor_amplitude(data_frame_fft)
            else:
                return self.tremor_amplitude_by_welch(data_frame_filtered)

        except ValueError as verr:
            logging.error("TremorProcessor ValueError ->%s", verr.message)
        except:
            logging.error("Unexpected error on TremorProcessor process: %s", sys.exc_info()[0])
