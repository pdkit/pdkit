# Copyright 2018 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author(s): Cosmin Stamate

import sys
import logging

import numpy as np
import pandas as pd

from scipy import interpolate, signal, fft

class Processor:
    '''
       This is the main Processor class. Once the data is loaded it will be
       accessible at data_frame, where it looks like:
       data_frame.x, data_frame.y, data_frame.z: x, y, z components of the acceleration
       data_frame.index is the datetime-like index
       
       This values are recommended by the author of the pilot study [1]
       
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
    '''

    def __init__(self, sampling_frequency=100.0, cutoff_frequency=2.0, filter_order=2,
                 window=256, lower_frequency=2.0, upper_frequency=10.0):
        try:
            self.sampling_frequency = sampling_frequency
            self.cutoff_frequency = cutoff_frequency
            self.filter_order = filter_order
            self.window = window
            self.lower_frequency = lower_frequency
            self.upper_frequency = upper_frequency

            logging.debug("Processor init")

        except IOError as e:
            ierr = "({}): {}".format(e.errno, e.strerror)
            logging.error("Processor I/O error %s", ierr)

        except ValueError as verr:
            logging.error("Processor ValueError ->%s", verr)

        except:
            logging.error("Unexpected error on Processor init: %s", sys.exc_info()[0])


    def resample_signal(self, data_frame):
        '''
            Convenience method for frequency conversion and resampling of data frame. 
            Object must have a DatetimeIndex. After re-sampling, this methods interpolate the time magnitude sum 
            acceleration values and the x,y,z values of the data frame acceleration

            :param data_frame: the data frame to resample
            :param str sampling_frequency: the sampling frequency. Default is 100Hz, as recommended by the author of the pilot study [1]
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
            frame using a butter Butterworth digital and analog filter 
            (https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.butter.html). Then the method 
            filter the data frame along one-dimension using a digital filter. 
            (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html)

            :param data_frame: the data frame    
            :param str cutoff_frequency: The path to load data from
            :param str filter_order: format of the file. Default is CloudUPDRS. Set to mpower for mpower data.
        '''
        b, a = signal.butter(self.filter_order, 2 * self.cutoff_frequency / self.sampling_frequency, 'high', analog=False)
        filtered_signal = signal.lfilter(b, a, data_frame.mag_sum_acc.values)
        data_frame['filtered_signal'] = filtered_signal

        logging.debug("filter signal")
        return data_frame


    def fft_signal(self, data_frame):
        '''
            This method perform Fast Fourier Transform on the data frame using a hanning window
            (https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.hann.html)

            :param data_frame: the data frame    
            :param str window: hanning window size
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
        
