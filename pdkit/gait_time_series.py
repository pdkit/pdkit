#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2018 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author: Cosmin Stamate 

import logging
from pdkit.utils import load_data
from pdkit.utils import get_sampling_rate_from_timestamp

class GaitTimeSeries:
    """
        This is a wrapper class to load the Gait Time Series data.
    """

    def __init__(self):
        logging.debug("GaitTimeSeries init")
   
    @staticmethod
    def load_data(filename, format_file='cloudupdrs'):
        """
            This is a general load data method where the format of data to load can be passed as a parameter,

            :param str filename: The path to load data from
            :param str format_file: format of the file. Default is CloudUPDRS ('cloudupdrs'). Set to 'mpower' for mpower data.

            :return DataFrame dataframe: data_frame.x, data_frame.y, data_frame.z: x, y, z components of the acceleration data_frame.index is the datetime-like index
        """
        
        logging.debug("{} data --> Loaded".format(format_file))
        
        data_frame = load_data(filename, format_file)
        data_frame.sampling_rate = get_sampling_rate_from_timestamp(data_frame)
        
        return data_frame