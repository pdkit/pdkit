#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2018 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author(s): J.S. Pons

import logging
import pdkit
# from _pdkit import pdkit as pdk
from scipy.cluster.vq import kmeans, whiten
import numpy as np
import pandas as pd
from numpy import array
from scipy.spatial.distance import euclidean
from os.path import join
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,f1_score,precision_recall_fscore_support



class Clinical_UPDRS:
    """
            Supervised Learning for UPDRS using the evaluation of the clinical staff.

            :param data_frame: testResultSet
            :type data_frame: pandas.DataFrame
            :param data_frame_file_path: the path to read the data frame from
            :type data_frame_file_path: pandas.DataFrame
        """

    def __init__(self, labels_file_path, data_frame=None, data_frame_file_path=None):
        # try:
            # read the labelled data
        self.labels = pd.read_csv(labels_file_path).sort_values(by=['id'])
        self.knns = np.array([])

        if data_frame_file_path is not None:
            data_frame = pd.read_csv(data_frame_file_path).fillna(0)
            self.data_frame = data_frame.sort_values(by=['id'])
        else:
            self.data_frame = data_frame.sort_values(by=['id'])

        self.clusters = np.array([])
        self.observations = [
            "OT-RH",
            "OT-LH",
            "TT-RH",
            "TT-LH",
            "PS-RH",
            "PS-LH",
            "LA-RL",
            "LA-LL",
            "PTOTH-RH",
            "PTOTH-LH",
            "KTOH-RH",
            "KTOH-LH",
            "RTA-RH",
            "RTA-LH",
            "RTA-RL",
            "RTA-LL",
        ]
        self.__train()

        # except IOError as e:
        #     ierr = "({}): {}".format(e.errno, e.strerror)
        #     logging.error("Clinical UPDRS I/O error %s", ierr)
        #
        # except ValueError as verr:
        #     logging.error("Clinical UPDRS ValueError ->%s", verr.message)
        #
        # except:
        #     logging.error("Unexpected error on Clinical UPDRS init: %s", sys.exc_info()[0])

    def __train(self):
        for obs in self.observations:
            # obs = "KTOH-RH"
            features, ids = self.__get_features_for_observation(observation=obs, skip_id=3497, last_column_is_id=True)
            normalised_data = whiten(features)

            x = pd.DataFrame(normalised_data)
            y = self.labels[obs].values

            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(x, y)

            if not self.knns:
                self.knns = [[obs, knn]]
            else:
                self.knns.append([obs, knn])

    def __get_features_for_observation(self, data_frame=None, observation='LA-LL',
                                       skip_id=None, last_column_is_id=False):
        """
            Extract the features for a given observation from a data frame

            :param data_frame: data frame to get features from
            :type data_frame: pandas.DataFrame
            :param observation: observation name
            :type observation: string
            :param skip_id: skip any test with a given id (optional)
            :type skip_id: int
            :param last_column_is_id: skip the last column of the data frame (useful when id is last column - optional)
            :type last_column_is_id: bool
            :return features: the features
            :rtype features: np.array
        """
        try:
            features = np.array([])

            if data_frame is None:
                data_frame = self.data_frame

            for index, row in data_frame.iterrows():
                if not skip_id == row['id']:
                    features_row = np.nan_to_num(row[row.keys().str.contains(observation)].values)
                    features_row = np.append(features_row, row['id'])
                    features = np.vstack([features, features_row]) if features.size else features_row

            # not the same when getting a single point
            if last_column_is_id:
                if np.ndim(features) > 1:
                    to_return = features[:,:-1]
                else:
                    to_return = features[:-1]
            else:
                to_return = features

            return to_return, data_frame['id'].values
        except:
            logging.error(" observation not found in data frame")

    def __get_knn_by_observation(self, observation):
        for (obs, knn) in self.knns:
            if obs == observation:
                return knn

    def predict(self, measurement, output_format='array'):
        scores = np.array([])
        for obs in self.observations:
            knn = self.__get_knn_by_observation(obs)
            p, ids = self.__get_features_for_observation(data_frame=measurement, observation=obs,
                                                         skip_id=3497, last_column_is_id=True)

            score = knn.predict(pd.DataFrame(p).T)
            scores = np.append(scores, score, axis=0)

        if output_format == 'array':
            return scores.astype(int)
        else:
            return np.array_str(scores.astype(int))

