#!/usr/bin/env python3
# Copyright 2018 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author(s): J.S. Pons

import logging
from pdkit.utils import load_data
import pandas_validator as pv


class FTCloudUPDRSDataFrameValidator(pv.DataFrameValidator):
    column_num = 6
    td = pv.FloatColumnValidator('td', min_value=-1, max_value=10000)
    action_type = pv.FloatColumnValidator('action_type', min_value=0, max_value=1)
    x = pv.FloatColumnValidator('x', min_value=-10000, max_value=10000)
    y = pv.FloatColumnValidator('y', min_value=-10000, max_value=10000)
    x_target = pv.FloatColumnValidator('x_target', min_value=-10000, max_value=10000)
    y_target = pv.FloatColumnValidator('y_target', min_value=-10000, max_value=10000)


class FingerTappingTimeSeries:
    '''
        This is a wrapper class to load the Finger Tapping Time Series data.
    '''
    def __init__(self):
        logging.debug("FingerTappingTimeSeries init")

    def load(self, filename, format_file='ft_cloudupdrs'):
        '''
            This is a general load data method where the format of data to load can be passed as a parameter,

            :param str filename: The path to load data from
            :param str format_file: format of the file. Default is CloudUPDRS. Set to mpower for mpower data.
            :return dataframe: data_frame.x, data_frame.y: components of tapping position. data_frame.x_target,
            data_frame.y_target their target. data_frame.index is the datetime-like index
        '''
        try:
            ts = load_data(filename, format_file)
            validator = FTCloudUPDRSDataFrameValidator()

            if validator.is_valid(ts):
                return ts
            else:
                logging.error('Validator error loading data, wrong format.')
                return None
        except:
            logging.error('Validator error loading data, wrong format.')


