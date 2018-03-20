#!/usr/bin/env python3
import sys
import logging

import numpy as np
import pandas as pd
from scipy import interpolate, signal, fft


class Processor:
    '''
        Processor Class
    '''

    def __init__(self):
        try:
            self.data_m = []
            self.date_times = []
            self.time_difference = []
            self.magnitude_sum_acceleration = []
            self.data_frame = pd.DataFrame()
            self.data_frame_fft = pd.DataFrame()
            self.transformed_signal = []
            self.amplitude = 0
            self.frequency = 0
            logging.debug("Processor init")

        except IOError as e:
            ierr = "({}): {}".format(e.errno, e.strerror)
            logging.error("Processor I/O error %s", ierr)
            # return 'IOError ' + ierr

        except ValueError as verr:
            logging.error("Processor ValueError ->%s", verr.message)
            # return 'TremorPRocessor ValueError'

        except:
            logging.error("Unexpected error on Processor init: %s", sys.exc_info()[0])
            # return 'Unexpected error on TremorProcessor init', sys.exc_info()[0]

    def load_cloudupdrs_data(self, filename):
        '''
           Load CloudUPDRS data

           :param str filename: The path to load data from
        '''
        # self.data_m = pd.read_table(filename, sep=',', header=None)
        self.data_m = np.genfromtxt(filename, delimiter=',', invalid_raise=False)
        self.date_times = pd.to_datetime((self.data_m[:, 0] - self.data_m[0, 0]))
        
        self.time_difference = (self.data_m[:, 0] - self.data_m[0, 0]) / NANOSEC_TO_SEC
        
        self.magnitude_sum_acceleration = \
            np.sqrt(self.data_m[:, 1] ** 2 + self.data_m[:, 2] ** 2 + self.data_m[:, 3] ** 2)

        data = {'dt': self.time_difference,
                'x': self.data_m[:, 1],
                'y': self.data_m[:, 2],
                'z': self.data_m[:, 3],
                'mag_sum_acc': self.magnitude_sum_acceleration}
        
        self.data_frame = pd.DataFrame(data, index=self.date_times, columns=['dt', 'x', 'y', 'z', 'mag_sum_acc'])
        logging.debug("cloudupdrs data loaded")

    def load_mpower_data(self, filename):
        '''
            Load MPower data

            :param str filename: The path to load data from
        '''
        raw_data = pd.read_json(filename)
        self.date_times = pd.to_datetime(raw_data.timestamp * NANOSEC_TO_SEC - raw_data.timestamp[0] * NANOSEC_TO_SEC)
        self.time_difference = (raw_data.timestamp - raw_data.timestamp[0])
        self.time_difference = self.time_difference.values
        self.magnitude_sum_acceleration = \
            np.sqrt(raw_data.x.values ** 2 + raw_data.y.values ** 2 + raw_data.z.values ** 2)

        data = {'dt': self.time_difference,
                'x': raw_data.x.values,
                'y': raw_data.y.values,
                'z': raw_data.z.values,
                'mag_sum_acc': self.magnitude_sum_acceleration}

        self.data_frame = pd.DataFrame(data, index=self.date_times, columns=['dt', 'x', 'y', 'z', 'mag_sum_acc'])
        logging.debug("mpower data loaded")

    def load_data(self, filename, formatfile='cloudupdrs'):
        '''
            General method to load data with different format file

            :param str filename: The path to load data from
            :param str formatfile: format of the file. Default is CloudUPDRS. Set to mpower for mpower data.
        '''
        if formatfile == 'mpower':
            self.load_mpower_data(filename),
        else:
            self.load_cloudupdrs_data(filename)

    def resample_signal(self, sampling_frequency=100.0):
        '''
            Resample signal
            We need to resample the signal as it is recorded with variable sampling rate

            :param str filename: The path to load data from
            :param str formatfile: format of the file. Default is CloudUPDRS. Set to mpower for mpower data.
        '''
        df_resampled = self.data_frame.resample(str(1 / sampling_frequency) + 'S').mean()
        # interpolate function
        f = interpolate.interp1d(self.time_difference, self.magnitude_sum_acceleration)
        # use arange to equally space the time difference
        new_timestamp = np.arange(self.time_difference[0], self.time_difference[-1], 1.0 / sampling_frequency)
        # interpolate the time magnitude sum acceleration values
        df_resampled.mag_sum_acc = f(new_timestamp)
        # interpolate the x,y,z values of the data frame
        self.data_frame = df_resampled.interpolate(method='linear')

        logging.debug("resample signal")

    def filter_signal(self, cutoff_frequency=2.0, filter_order=2, sampling_frequency=100.0):
        '''
            Filter signal. High pass filter the signal as per [1]
            [1] Developing a tool for remote digital assessment of Parkinson s disease
            Kassavetis	P,	Saifee	TA,	Roussos	G,	Drougas	L,	Kojovic	M,	Rothwell	JC,	Edwards	MJ,	Bhatia	KP

            :param str filename: The path to load data from
            :param str formatfile: format of the file. Default is CloudUPDRS. Set to mpower for mpower data.
        '''
        # first step is to high pass filter the signal as per [1]
        b, a = signal.butter(filter_order, 2 * cutoff_frequency / sampling_frequency, 'high', analog=False)
        filtered_signal = signal.lfilter(b, a, self.data_frame.mag_sum_acc.values)
        
        self.data_frame['filtered_signal'] = filtered_signal
        
        logging.debug("filter signal")

    def fft_signal(self, window=256):
        '''
            FFT signal. Perform fft on the signal using a hanning window

            :param str window: hanning window size
        '''
        signal_length = len(self.data_frame.filtered_signal.values)
        ll = int ( signal_length / 2 - window / 2 )
        rr = int ( signal_length / 2 + window / 2 )
        msa = self.data_frame.filtered_signal[ll:rr].values
        hann_window = signal.hann(window)

        msa_window = (msa * hann_window)
        self.transformed_signal = fft(msa_window)

        data = {'filtered_signal': msa_window, 'transformed_signal': self.transformed_signal,
                'dt': self.data_frame.dt[ll:rr].values}
        # fft signal is a new data frame
        self.data_frame_fft = pd.DataFrame(data, index=self.data_frame.index[ll:rr],columns=['filtered_signal', 'transformed_signal', 'dt'])
        logging.debug("fft signal")