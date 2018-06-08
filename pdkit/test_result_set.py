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
            >>> dataframe = testResultSet.process(['tremor'])

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

    def process(self, params=None):
        '''
            This method reads all the files (measurements) within a given path and extract the features. It will return a
            data frame where the rows are the measurements and the columns correspond to the extracted features.

            :param params: the params ('tremor' for now)
            :type params: string
            :return data_frame: the dataframe for the measurements placed in the folder
            :rtype data_frame: pandas.DataFrame
        '''
        features_df = pd.DataFrame()

        if 'tremor' in params:
            abr_measurent_type='T'
            tp = pdkit.TremorProcessor()

            for f in self.files_list:
                if f.startswith(abr_measurent_type + ' - '):
                    tts = pdkit.TremorTimeSeries().load(join(self.folder_absolute_path, f))
                    features = tp.extract_features(tts)

                    if features_df.empty:
                        features_df = pd.DataFrame(features, columns=list(features.keys()), index=[0])
                        features_df.insert(0, 'name', f.split('.')[0])
                    else:
                        features['name'] = f.split('.')[0]
                        features_df = features_df.append(features, ignore_index=True)
        #
        # @TODO: should we join dataframes?
        #
        else:
            if 'finger_tapping' in params:
                abr_measurent_type = 'FT'
                ftp = pdkit.FingerTappingProcessor()

                for f in self.files_list:
                    if f.startswith(abr_measurent_type + ' - '):
                        ftts = pdkit.FingerTappingTimeSeries().load(join(self.folder_absolute_path, f))
                        features = ftp.extract_features(ftts)

                        if features_df.empty:
                            features_df = pd.DataFrame(features, columns=list(features.keys()), index=[0])
                            features_df.insert(0, 'name', f.split('.')[0])
                        else:
                            features['name'] = f.split('.')[0]
                            features_df = features_df.append(features, ignore_index=True)

        return features_df

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
