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
from os.path import isfile, join, isdir
import pandas as pd
import re
from tqdm import tqdm

class TestResultSet:
    """
        This is the Test Result Set class. Its main functionality is to read all the files (measurements) within a given
        path and extract the features. It will return a data frame where the rows are the measurements and the columns
        correspond to the extracted features.

        :param folder_relative_path: (required) the relative folder path
        :type folder_relative_path: str

        :Example:

        >>> import pdkit
        >>> testResultSet = pdkit.TestResultSet(folderpath)
        >>> dataframe = testResultSet.process

        where `folderpath` is the relative folder with the different measurements. For CloudUPDRS there are measurements
        in the following folder `./tests/data/S5`.

        :Example:

        >>> testResultSet.write_output(dataframe, name)

        To write the `data frame` to a output file (name)
    """
    def __init__(self, folder_relative_path):
        try:
            self.folder_relative_path = folder_relative_path
            self.folder_absolute_path = self.__get_folder_absolute_path(folder_relative_path)
            self.folder_name = self.__get_folder_name(folder_relative_path)
            self.dir_list = self.__get_dirs_list()
            self.features = pd.DataFrame()
        except IOError as e:
            ierr = "({}): {}".format(e.errno, e.strerror)
            logging.error("TestResultSet I/O error %s", ierr)

        except ValueError as verr:
            logging.error("TestResultSet ValueError ->%s", verr.message)

        except:
            logging.error("Unexpected error on TestResultSet init: %s", sys.exc_info()[0])
        logging.debug("TestRestultSet init")

    def __get_files_list(self, folder_absolute_path):
        return [f for f in os.listdir(folder_absolute_path) if isfile(join(folder_absolute_path, f))]

    def __get_dirs_list(self):
        return [f for f in os.listdir(self.folder_absolute_path) if (isdir(join(self.folder_absolute_path, f)) and not f.startswith('_'))]

    def __build_folder_path(self, folder_name):
        return join(self.folder_absolute_path, folder_name)

    @staticmethod
    def __get_session_id(filename):
        m = re.search(r"_(\d+).csv", filename, re.IGNORECASE)
        return m.group(1)

    @staticmethod
    def __get_measurement_name(abr_measurement_type, filename):
        m = re.search(r"(?![%s])[a-zA-Z_\-]*" % abr_measurement_type, filename, re.IGNORECASE).group(0)[3:].split('_')
        return ''.join([x[0] for x in m if len(x)>0])

    @staticmethod
    def __get_folder_absolute_path(folder_relative_path):
        pwd = os.getcwd()
        if folder_relative_path.startswith('.'):
            return pwd + folder_relative_path[1:]
        else:
            return pwd + folder_relative_path

    @staticmethod
    def __get_folder_name(folder_relative_path):
        if folder_relative_path.endswith('/'):
            return folder_relative_path.split('/')[-2]
        else:
            return folder_relative_path.split('/')[-1]

    def __get_tremor_measurements(self, data_frame, directory, files_list):
        """
            Convenience method that gets the finger tapping measurements

            :param data_frame: the dataframe where the features will be added
            :type data_frame: pandas.DataFrame
            :param directory: the directory name that contains the files
            :type features: str
            :param files_list: the list of files
            :type files_list: str
            :return data_frame: the dataframe
            :rtype data_frame: pandas.DataFrame
        """
        abr_measurement_type = 'T'
        tp = pdkit.TremorProcessor()

        for f in files_list:
            if f.startswith(abr_measurement_type):
                tts = pdkit.TremorTimeSeries().load(join(self.__build_folder_path(directory), f))
                features = tp.extract_features(tts, self.__get_measurement_name(abr_measurement_type, f))
                if features is not None:
                    data_frame = self.__save_features_to_dataframe(features, data_frame, f)
                else:
                    print('file error: '+f)

        return data_frame

    def __get_finger_tapping_measurements(self, data_frame, directory, files_list):
        """
            Convenience method that gets the finger tapping measurements

            :param data_frame: the dataframe where the features will be added
            :type data_frame: pandas.DataFrame
            :param directory: the directory name that contains the files
            :type features: str
            :param files_list: the list of files
            :type files_list: str
            :return data_frame: the dataframe
            :rtype data_frame: pandas.DataFrame
        """
        abr_measurement_type = 'FT'
        ftp = pdkit.FingerTappingProcessor()

        for f in files_list:
            if f.startswith(abr_measurement_type):
                ftts = pdkit.FingerTappingTimeSeries().load(join(self.__build_folder_path(directory), f))
                features = ftp.extract_features(ftts, self.__get_measurement_name(abr_measurement_type, f)+'-')
                if features is not None:
                    data_frame = self.__save_features_to_dataframe(features, data_frame, f)
                else:
                    print('file error: '+f)

        return data_frame

    def __save_features_to_dataframe(self, features, data_frame, file_name):
        """
            Convenience method that saves/add features to an existing dataframe

            :param features: the features to be added
            :type features: list
            :param data_frame: the dataframe where the features will be added
            :type data_frame: pandas.DataFrame
            :param file_name: a single file name
            :type file_name: str
            :return data_frame: the dataframe
            :rtype data_frame: pandas.DataFrame
        """
        session_id = self.__get_session_id(file_name)
        if data_frame.empty:
            data_frame = pd.DataFrame(features,columns=list(features.keys()),index=[0])
            data_frame.insert(0, 'id', session_id)
        else:
            # found = False
            # iterate the dataframe, if it's the same session concat the data
            for index, row in data_frame.iterrows():
                if row['id'] == session_id:
                    data_frame = pd.concat(
                        [
                            data_frame,
                            pd.DataFrame(features, index=data_frame.index, columns=list(features.keys()))
                        ], axis=1
                    )
                    # found = True
            # if it isn't the same session save in new row, if session data is in same folder this shouldn't happen
            # if not found:
            #     features['id'] = session_id
            #     data_frame = data_frame.append(features, ignore_index=True)

        return data_frame

    def process(self):
        """
            This method reads all the directories that contain files (measurements) within a given relative path and extracts
            the features. the resulting dataframe with all the features processed is saved in testResultSet.features
            Where features Dataframe the rows are the measurements and the columns correspond to
            the extracted features. The data frame will have a column 'id' with the name of the measurement.
        """

        features = pd.DataFrame()
        for d in tqdm(self.dir_list):
            if self.folder_relative_path.endswith('/'):
                files_list = self.__get_files_list(self.folder_relative_path+d)
            else:
                files_list = self.__get_files_list(join(self.folder_relative_path,d))
            features_tremor = self.__get_tremor_measurements(pd.DataFrame(), d, files_list)
            features_tremor_and_finger_tapping = self.__get_finger_tapping_measurements(features_tremor, d, files_list)
            if features.empty:
                features = features_tremor_and_finger_tapping
            else:
                try:
                    if features.loc[features['id'] == self.__get_session_id(files_list[0])].empty:
                        features = features.append(features_tremor_and_finger_tapping, ignore_index=True, sort=False)
                except:
                    print('directory error?: ' + d)

        self.features = features.fillna(0)

    def write_output(self, filename, output_format='csv'):
        """
            This method writes to a file the features data frame.

            :param filename: the name to give to the file
            :type filename: string
            :param output_format: the format of the file to write ('csv', 'json' or 'sql')
            :type output_format: string
        """
        try:
            filename = join(self.folder_absolute_path, filename) + '.' + output_format

            if output_format == 'json':
                self.features.to_json(path_or_buf=filename, index=False)
            else:
                if output_format == 'sql':
                    self.features.to_sql(path_or_buf=filename, index=False)
                else:
                    self.features.to_csv(path_or_buf=filename, index=False)
        except:
            logging.error("Unexpected error on writing output")