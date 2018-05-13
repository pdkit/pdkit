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
    def __init__(self):
        try:
            logging.debug("TremorProcessor init")

        except IOError as e:
            ierr = "({}): {}".format(e.errno, e.strerror)
            logging.error("TremorProcessor I/O error %s", ierr)

        except ValueError as verr:
            logging.error("TremorProcessor ValueError ->%s", verr.message)

        except:
            logging.error("Unexpected error on TremorProcessor init: %s", sys.exc_info()[0])

    def frequency(self, data_frame):
        # this is simply the #taps divided by the test duration
        freq = sum(data_frame.action_type == 1) / data_frame.td[-1]
        return freq

    def moving_frequency(self, data_frame):
        window = 6  # secs
        # self.t = self.timestamp / MILLISEC_TO_SEC
        f = []
        for i in range(0, (data_frame.td[-1].astype('int') - window)):
            f.append(sum(data_frame.action_type[(data_frame.td >= i) & (data_frame.td < (i + window))] == 1) / float(window))
        mov_freq = f
        diff_mov_freq = (np.array(f[1:-1]) - np.array(f[0:-2])) / np.array(f[0:-2])
        return diff_mov_freq

    def continuous_frequency(self, data_frame):
        tap_timestamps = data_frame.td[data_frame.action_type==1]
        cont_freq = 1.0/(np.array(tap_timestamps[1:-1])-np.array(tap_timestamps[0:-2]))
        return cont_freq

    def mean_moving_time(self, data_frame):
        # the mean time that the hand was moving from one target to the next
        diff = data_frame.td[1:-1].values-data_frame.td[0:-2].values
        mmt = np.mean(diff[np.arange(1,len(diff),2)])
        # convert to ms
        return mmt * 1000.0

    def mean_alnt_target_distance(self, data_frame):
        # the distance between alternate tappings (number of pixels)
        dist = np.sqrt((data_frame.x[1:-1].values-data_frame.x[0:-2].values)**2+(data_frame.y[1:-1].values-data_frame.y[0:-2].values)**2)
        matd = np.mean(dist[np.arange(1,len(dist),2)])
        return matd