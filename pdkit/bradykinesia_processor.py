#!/usr/bin/env python3
# Copyright 2018 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author(s): J.S. Pons

import sys
import logging

from numpy import genfromtxt, sqrt, arange, mean
import numpy as np
import pandas as pd
from scipy import interpolate, signal, fft

class BradykinesiaProcessor:
    '''
        This is the main Bradykinesia Processor class. Once the data is loaded it will be
        accessible at data_frame (pandas.DataFrame), where it looks like:
        data_frame.x, data_frame.y, data_frame.z: x, y, z components of the acceleration
        data_frame.index is the datetime-like index

        These values are recommended by the author of the pilot study :cite:`Kassavetis2015`

        sampling_frequency = 100.0Hz
        cutoff_frequency = 2.0Hz
        filter_order = 2
        window = 256
        lower_frequency = 2.0Hz
        upper_frequency = 10.0Hz

    '''

    def __init__(self, sampling_frequency=100.0, cutoff_frequency=2.0, filter_order=2,
                 window=256, lower_frequency=0.0, upper_frequency=4.0):
        try:
            self.amplitude = 0
            self.frequency = 0

            self.sampling_frequency = sampling_frequency
            self.cutoff_frequency = cutoff_frequency
            self.filter_order = filter_order
            self.window = window
            self.lower_frequency = lower_frequency
            self.upper_frequency = upper_frequency

            logging.debug("BradykinesiaProcessor init")

        except IOError as e:
            ierr = "({}): {}".format(e.errno, e.strerror)
            logging.error("BradykinesiaProcessor I/O error %s", ierr)

        except ValueError as verr:
            logging.error("BradykinesiaProcessor ValueError ->%s", verr.message)

        except:
            logging.error("Unexpected error on BradykinesiaProcessor init: %s", sys.exc_info()[0])

    def resample_signal(self, data_frame):
        '''
            Convenience method for frequency conversion and resampling of data frame.
            Object must have a DatetimeIndex. After re-sampling, this methods interpolate the time magnitude sum
            acceleration values and the x,y,z values of the data frame acceleration

            :param data_frame: the data frame to resample
            :type data_frame: pandas.DataFrame
            :return: the resampled data frame
            :rtype: pandas.DataFrame

        '''
        df_resampled = data_frame.resample(str(1 / self.sampling_frequency) + 'S').mean()

        f = interpolate.interp1d(data_frame.td, data_frame.mag_sum_acc)
        new_timestamp = np.arange(data_frame.td[0], data_frame.td[-1], 1.0 / self.sampling_frequency)
        df_resampled.mag_sum_acc = f(new_timestamp)

        logging.debug("resample signal")
        return df_resampled.interpolate(method='linear')

    def dc_remove_signal(self, data_frame):
        # first step is to remove the dc component of the signal as per [1]
        mean_signal = mean(data_frame.mag_sum_acc)
        data_frame['dc_mag_sum_acc'] = data_frame.mag_sum_acc - mean_signal
        logging.debug("dc remove signal")
        return data_frame

    def filter_signal(self, data_frame):
        '''
            This method filters a data frame signal as suggested in :cite:`Kassavetis2015`. First step is to high pass filter the data
            frame using a `Butterworth <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.butter.html>`_ digital and analog filter. Then this method
            filters the data frame along one-dimension using a `digital filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html>`_.

            :param data_frame: the input data frame
            :type data_frame: pandas.DataFrame
            :return data_frame: adds a column named 'filtered_signal' to the data frame
            :rtype data_frame: pandas.DataFrame
        '''
        b, a = signal.butter(self.filter_order, 2 * self.cutoff_frequency / self.sampling_frequency, 'high', analog=False)
        filtered_signal = signal.lfilter(b, a, data_frame.dc_mag_sum_acc.values)
        data_frame['filtered_signal'] = filtered_signal

        logging.debug("filter signal")
        return data_frame

    def fft_signal(self, data_frame):
        '''
            This method perform Fast Fourier Transform on the data frame using a `hanning window <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.hann.html>`_

            :param data_frame: the data frame
            :type data_frame: pandas.DataFrame
            :return: data frame with a 'filtered_singal', 'transformed_signal' and 'dt' columns
            :rtype: pandas.DataFrame
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

    def bradykinesia_amplitude_by_welch(self, data_frame):
        '''
            This methods uses the Welch method :cite:`Welch1967` to obtain the power spectral density, this is a robust
            alternative to using fft_signal & bradykinesia_amplitude

            :param data_frame: the data frame
            :type data_frame: pandas.DataFrame
            :return: amplitude is the the amplitude of the Bradykinesia
            :rtype amplitude: float
            :return: frequency is the frequency of the Bradykinesia
            :rtype frequency: float
        '''
        frq, Pxx_den = signal.welch(data_frame.filtered_signal.values, self.sampling_frequency, nperseg=self.window)
        frequency = frq[Pxx_den.argmax(axis=0)]
        amplitude = sum(Pxx_den[(frq > self.lower_frequency) & (frq < self.upper_frequency)])

        logging.debug("bradykinesia amplitude by welch calculated")

        return amplitude, frequency

    def bradykinesia_amplitude(self, data_frame):

        '''
            This methods extract the fft components and sum the ones from lower to upper freq as per :cite:`Kassavetis2015`

            :param data_frame: the data frame
            :type data_frame: pandas.DataFrame
            :return amplitude: the amplitude of the Bradykinesia
            :rtype amplitude: float
            :return frequency: the frequency of the Bradykinesia
            :rtype frequency: float
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

        logging.debug("Bradykinesia amplitude calculated")

        return amplitude, frequency

    def process(self, data_frame, method='fft'):
        '''
            This methods calculates the bradykinesia amplitude of the data frame. It accepts two different methods,
            'fft' and 'welch'. First the signal gets re-sampled, dc removed and high pass filtered.

            :param data_frame: the data frame
            :type data_frame: pandas.DataFrame
            :param method: fft or welch.
            :type method: str
            :return amplitude: the amplitude of the Bradykinesia
            :rtype amplitude: float
            :return frequency: the frequency of the Bradykinesia
            :rtype frequency: float

        '''
        try:
            data_frame_resampled = self.resample_signal(data_frame)
            data_frame_dc = self.dc_remove_signal(data_frame_resampled)
            data_frame_filtered = self.filter_signal(data_frame_dc)

            if method == 'fft':
                data_frame_fft = self.fft_signal(data_frame_filtered)
                return self.bradykinesia_amplitude(data_frame_fft)
            else:
                return self.bradykinesia_amplitude_by_welch(data_frame_filtered)

        except ValueError as verr:
            logging.error("BradykinesiaProcessor ValueError ->%s", verr.message)
        except:
            logging.error("Unexpected error on BradykinesiaProcessor process: %s", sys.exc_info()[0])