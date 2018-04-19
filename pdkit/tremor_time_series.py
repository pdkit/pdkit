#!/usr/bin/env python3
# Copyright 2018 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author(s): J.S. Pons

import logging
from pdkit.utils import load_data
import pandas_validator as pv


class CloudUPDRSDataFrameValidator(pv.DataFrameValidator):
    column_num = 5
    td = pv.FloatColumnValidator('td', min_value=-1, max_value=1000)
    x = pv.FloatColumnValidator('x', min_value=-100, max_value=100)
    y = pv.FloatColumnValidator('y', min_value=-100, max_value=100)
    z = pv.FloatColumnValidator('z', min_value=-100, max_value=100)
    mag_sum_acc = pv.FloatColumnValidator('mag_sum_acc', min_value=-100, max_value=100)


class TremorTimeSeries:
    '''
        This is a wrapper class to load the Tremor Time Series data.
    '''
    def __init__(self):
        logging.debug("TremorTimeSeries init")

    def load(self, filename, format_file='cloudupdrs'):
        '''
            This is a general load data method where the format of data to load can be passed as a parameter,

            :param str filename: The path to load data from
            :param str format_file: format of the file. Default is CloudUPDRS. Set to mpower for mpower data.
            :return dataframe: data_frame.x, data_frame.y, data_frame.z: x, y, z components of the acceleration data_frame.index is the datetime-like index
        '''
        try:
            ts = load_data(filename, format_file)
            validator = CloudUPDRSDataFrameValidator()

            if validator.is_valid(ts):
                return ts
            else:
                logging.error('Error loading data, wrong format.')
                return None
        except:
            logging.error('Error loading data, wrong format.')


