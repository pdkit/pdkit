#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2018 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author(s): J.S. Pons

import logging
import pdkit
import os
import sys
from os import listdir
from os.path import isfile, join
import pandas as pd
import re


class TestResultSet:
    '''
            This is the Test Result Set class. Its main functionality is to read all the files (measurements) within a given
            path and extract the features. It will return a data frame where the rows are the measurements and the columns
            correspond to the extracted features.

            :param folder_relative_path: (required) the relative folder path
            :type folder_relative_path: str

            :Example:

            >>> import pdkit
            >>> testResultSet = pdkit.TestResultSet(folderpath)
            >>> dataframe = testResultSet.process()

            where `folderpath` is the relative folder with the different measurements. For CloudUPDRS there are measurements
            in the following folder `./tests/data/S5`.

            :Example:

            >>> testResultSet.write_output(dataframe, name)

            To write the `data frame` to a output file (name)
        '''
    def __init__(self, folder_relative_path):
        try:
            self.folder_absolute_path = self.get_folder_absolute_path(folder_relative_path)
            self.folder_name = self.get_folder_name(folder_relative_path)
            self.files_list = self.get_files_list()
        except IOError as e:
            ierr = "({}): {}".format(e.errno, e.strerror)
            logging.error("TestResultSet I/O error %s", ierr)

        except ValueError as verr:
            logging.error("TestResultSet ValueError ->%s", verr.message)

        except:
            logging.error("Unexpected error on TestResultSet init: %s", sys.exc_info()[0])
        logging.debug("TestRestultSet init")

    def get_files_list(self):
        return [f for f in os.listdir(self.folder_absolute_path) if isfile(join(self.folder_absolute_path, f))]

    @staticmethod
    def get_session_id(filename):
        m = re.search(r"_(\d+).csv", filename, re.IGNORECASE)
        return m.group(1)

    @staticmethod
    def get_measurement_name(abr_measurement_type, filename):
        m = re.search(r"(?![%s])[a-zA-Z_\-]*" % abr_measurement_type, filename, re.IGNORECASE)
        return m.group(0)

    @staticmethod
    def get_folder_absolute_path(folder_relative_path):
        pwd = os.getcwd()
        if folder_relative_path.startswith('.'):
            return pwd + folder_relative_path[1:]
        else:
            return pwd + folder_relative_path

    @staticmethod
    def get_folder_name(folder_relative_path):
        if folder_relative_path.endswith('/'):
            return folder_relative_path.split('/')[-2]
        else:
            return folder_relative_path.split('/')[-1]

    def get_tremor_measurements(self, data_frame):
        abr_measurement_type = 'T_-_'
        tp = pdkit.TremorProcessor()

        for f in self.files_list:
            if f.startswith(abr_measurement_type):
                tts = pdkit.TremorTimeSeries().load(join(self.folder_absolute_path, f))
                features = tp.extract_features(tts, self.get_measurement_name(abr_measurement_type, f))
                data_frame = self.save_features_to_dataframe(features, data_frame, f)

        return data_frame

    def get_finger_tapping_measurements(self, data_frame):
        abr_measurement_type = 'FT_-_'
        ftp = pdkit.FingerTappingProcessor()

        for f in self.files_list:
            if f.startswith(abr_measurement_type):
                ftts = pdkit.FingerTappingTimeSeries().load(join(self.folder_absolute_path, f))
                features = ftp.extract_features(ftts, self.get_measurement_name(abr_measurement_type, f))
                data_frame = self.save_features_to_dataframe(features, data_frame, f)

        return data_frame

    def save_features_to_dataframe(self, features, data_frame, f):
        session_id = self.get_session_id(f)
        if data_frame.empty:
            data_frame = pd.DataFrame(features,columns=list(features.keys()),index=[0])
            data_frame.insert(0, 'id', session_id)
        else:
            found = False
            # iterate the dataframe, if it's the same session concat the data
            for index, row in data_frame.iterrows():
                if row['id'] == session_id:
                    data_frame = pd.concat(
                        [
                            data_frame,
                            pd.DataFrame(features, index=data_frame.index, columns=list(features.keys()))
                        ], axis=1
                    )
                    found = True
            # if it isn't the same session save in new row, if session data is in same folder this shouldn't happen
            if not found:
                features['id'] = session_id
                data_frame = data_frame.append(features, ignore_index=True)

        return data_frame

    def process(self):
        '''
            This method reads all the files (measurements) within a given path and extract the features. It will return a
            data frame where the rows are the measurements and the columns correspond to the extracted features. The data
            frame will have a column 'name' with the name of the measurement

            :return data_frame: the dataframe for the measurements placed in the folder
            :rtype data_frame: pandas.DataFrame
        '''

        features = pd.DataFrame()
        features = self.get_tremor_measurements(features)
        features = self.get_finger_tapping_measurements(features)
        # features = self.get_gait_measurements(features)
        return features

    def write_output(self, data_frame, filename=None, output_format='csv'):
        '''
            This method writes to a file the data frame received.

            :param data_frame: the dataframe to write
            :type data_frame: pandas.DataFrame
            :param filename: the name to give to the file
            :type filename: string
            :param output_format: the format of the file to write ('csv', 'json' or 'sql')
            :type output_format: string
        '''
        if filename is None:
            filename = self.folder_name

        filename = join(self.folder_absolute_path, filename) + '.' + output_format

        if output_format == 'json':
            data_frame.to_json(path_or_buf=filename, index=False)
        else:
            if output_format == 'sql':
                data_frame.to_sql(path_or_buf=filename, index=False)
            else:
                data_frame.to_csv(path_or_buf=filename, index=False)
