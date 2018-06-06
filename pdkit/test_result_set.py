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

    def process(self, params=None, output=True, output_format='csv'):
        features_df = pd.DataFrame()

        if 'tremor' in params:
            abr_measurent_type='T'
            tp = pdkit.TremorProcessor()

            for f in self.files_list:
                if f.startswith(abr_measurent_type + ' - '):
                    ts = pdkit.TremorTimeSeries().load(join(self.folder_absolute_path, f))
                    features = tp.extract_features(ts)

                    if features_df.empty:
                        features_df = pd.DataFrame(features, columns=list(features.keys()), index=[0])
                    else:
                        features_df = features_df.append(features, ignore_index=True)

        return features_df
