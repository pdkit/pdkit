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


class FingerTappingProcessor:
    '''
            This is the main Finger Tapping Processor class. Once the data is loaded it will be accessible at data_frame (pandas.DataFrame), where it looks like: data_frame.x, data_frame.y: components of tapping position. data_frame.x_target, data_frame.y_target their target.

            These values are recommended by the author of the pilot study :cite:`Kassavetis2015`. Check reference for more details.

            window = 6 #seconds

            :Example:

            >>> import pdkit
            >>> ftp = pdkit.FingerTappingProcessor()
            >>> ts = pdkit.FingerTappingTimeSeries().load(path_to_data, 'ft_cloudupdrs')
            >>> frequency = ftp.frequency(ts)
        '''
    def __init__(self, window=6):
        try:
            self.window = window # secs

            logging.debug("TremorProcessor init")

        except IOError as e:
            ierr = "({}): {}".format(e.errno, e.strerror)
            logging.error("TremorProcessor I/O error %s", ierr)

        except ValueError as verr:
            logging.error("TremorProcessor ValueError ->%s", verr.message)

        except:
            logging.error("Unexpected error on TremorProcessor init: %s", sys.exc_info()[0])

    def frequency(self, data_frame):
        '''
            This method returns the number of #taps divided by the test duration

            :param data_frame: the data frame
            :type data_frame: pandas.DataFrame
            :return: frequency
            :rtype: float

        '''
        freq = sum(data_frame.action_type == 1) / data_frame.td[-1]
        return freq

    def moving_frequency(self, data_frame):
        '''
            This method returns moving frequency

            :param data_frame: the data frame
            :type data_frame: pandas.DataFrame
            :return: frequency
            :rtype: float

        '''
        f = []
        for i in range(0, (data_frame.td[-1].astype('int') - self.window)):
            f.append(sum(data_frame.action_type[(data_frame.td >= i) & (data_frame.td < (i + self.window))] == 1) / float(self.window))

        diff_mov_freq = (np.array(f[1:-1]) - np.array(f[0:-2])) / np.array(f[0:-2])

        return diff_mov_freq

    def continuous_frequency(self, data_frame):
        '''
            This method returns continuous frequency

            :param data_frame: the data frame
            :type data_frame: pandas.DataFrame
            :return: frequency
            :rtype: float

        '''
        tap_timestamps = data_frame.td[data_frame.action_type==1]
        cont_freq = 1.0/(np.array(tap_timestamps[1:-1])-np.array(tap_timestamps[0:-2]))

        return cont_freq

    def mean_moving_time(self, data_frame):
        '''
            This method calculates the mean time (ms) that the hand was moving from one target to the next

            :param data_frame: the data frame
            :type data_frame: pandas.DataFrame
            :return: the mean moving time in ms
            :rtype: float

        '''
        diff = data_frame.td[1:-1].values-data_frame.td[0:-2].values
        mmt = np.mean(diff[np.arange(1,len(diff),2)])

        # convert to ms
        return mmt * 1000.0

    def mean_alnt_target_distance(self, data_frame):
        '''
            This method calculates the distance (number of pixels) between alternate tapping

            :param data_frame: the data frame
            :type data_frame: pandas.DataFrame
            :return: the mean alternate target distance in pixels
            :rtype: float

        '''
        dist = np.sqrt((data_frame.x[1:-1].values-data_frame.x[0:-2].values)**2+(data_frame.y[1:-1].values-data_frame.y[0:-2].values)**2)
        matd = np.mean(dist[np.arange(1,len(dist),2)])

        return matd

    def kinesia_score_30(self, data_frame):
        '''
            This method calculates the mean number of key taps in 30 seconds (KS30)

            :param data_frame: the data frame
            :type data_frame: pandas.DataFrame
            :return: KS30
            :rtype: float

        '''
        tap_timestamps = data_frame.td[data_frame.action_type == 1]
        grouped = tap_timestamps.groupby(pd.TimeGrouper('30u'))
        return np.mean(grouped.size().values)