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
from os import listdir
from os.path import isfile, join
import pandas as pd


class TestResultSet:
    def __init__(self, folder_relative_path):
        self.folder_absolute_path = self.get_folder_absolute_path(folder_relative_path)
        self.folder_name = self.get_folder_name(folder_relative_path)
        self.files_list = self.get_files_list()

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


            # else:
            #     if 'finger_tapping' in params:
            #         abr_measurent_type = 'FT'
            #
            #         for f in self.files_list:
            #             if f.startswith(abr_measurent_type + ' - '):
            #                 ts = pdkit.FingerTappingTimeSeries().load(join(self.folder_absolute_path, f))

        return features_df

    def write_output(self, data_frame, filename, output_format='csv'):
        filename = join(self.folder_absolute_path, filename) + '.' + output_format
        # filename = self.folder_absolute_path + filename + '.' + output_format

        if output_format == 'json':
            data_frame.to_json(path_or_buf=filename, index=False)
        else:
            if output_format == 'sql':
                data_frame.to_sql(path_or_buf=filename, index=False)
            else:
                data_frame.to_csv(path_or_buf=filename, index=False)
