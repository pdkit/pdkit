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

    def number_peaks(self, x, n):
        """
            As in tsfresh [number_peaks]_

            Calculates the number of peaks of at least support n in the time series x. A peak of support n is defined as a
            subsequence of x where a value occurs, which is bigger than its n neighbours to the left and to the right.

            Hence in the sequence

            >>> x = [3, 0, 0, 4, 0, 0, 13]

            4 is a peak of support 1 and 2 because in the subsequences

            >>> [0, 4, 0]
            >>> [0, 0, 4, 0, 0]

            4 is still the highest value. Here, 4 is not a peak of support 3 because 13 is the 3th neighbour to the right of 4
            and its bigger than 4.

            :param x: the time series to calculate the feature of
            :type x: pandas.Series
            :param n: the support of the peak
            :type n: int
            :return: the value of this feature
            :return type: float
            """
        if n is None:
            n = 5
        peaks = feature_calculators.number_peaks(x, n)
        logging.debug("agg linear trend by tsfresh calculated")
        return peaks

    def agg_linear_trend(self, x, param = None):
        """
            As in tsfresh [spkt_welch_density]_
            
            Calculates a linear least-squares regression for values of the time series that were aggregated over chunks versus
            the sequence from 0 up to the number of chunks minus one.

            This feature assumes the signal to be uniformly sampled. It will not use the time stamps to fit the model.

            The parameters attr controls which of the characteristics are returned. Possible extracted attributes are "pvalue",
            "rvalue", "intercept", "slope", "stderr", see the documentation of linregress for more information.

            The chunksize is regulated by "chunk_len". It specifies how many time series values are in each chunk.

            Further, the aggregation function is controlled by "f_agg", which can use "max", "min" or , "mean", "median"

            :param x: the time series to calculate the feature of
            :type x: pandas.Series
            :param param: contains dictionaries {"attr": x, "chunk_len": l, "f_agg": f} with x, f an string and l an int
            :type param: list
            :return: the different feature values
            :return type: pandas.Series
        """
        if param is None:
            param = [{'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'min'},{'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'var'},{'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'min'}]
        agg = feature_calculators.agg_linear_trend(x, param)
        logging.debug("agg linear trend by tsfresh calculated")
        return list(agg)

    def spkt_welch_density(self, x, param = None):
        '''
            As in tsfresh [spkt_welch_density]_
            This feature calculator estimates the cross power spectral density of the time series x at different frequencies.
            To do so, the time series is first shifted from the time domain to the frequency domain.
            
            The feature calculators returns the power spectrum of the different frequencies.
            
    
            
            :param x: the time series to calculate the feature of
            :type x: pandas.Series
            :param param: contains dictionaries {"coeff": x} with x int
            :type param: list
            :return: the different feature values
            :return type: pandas.Series
        '''
        if param is None:
            param = [{'coeff': 2}, {'coeff': 5}, {'coeff': 8}]
        welch = feature_calculators.spkt_welch_density(x, param)
        logging.debug("spkt welch density by tsfresh calculated")
        return list(welch)

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
