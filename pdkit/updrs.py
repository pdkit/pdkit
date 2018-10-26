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


class UPDRS:
    """
        This class calculates the `UPDRS score Part 3 <https://en.wikipedia.org/wiki/\
        Unified_Parkinson%27s_disease_rating_scale>`_ for a given testResultSet. UPDRS performs \
        `k-means <https://docs.scipy.org/doc/scipy-0.7.x/reference/cluster.vq.html#scipy.cluster.vq.kmeans>`_ \
        on a set of observation vectors forming k clusters.

        :Example:

        >>> import pdkit
        >>> updrs = pdkit.UPDRS(data_frame)

        The UPDRS scores can be written to a file. You can pass the name of a `filename` and the `output_format`

        >>> updrs.write_model(filename='scores', output_format='csv')

        To score a new measurement against the trained knn clusters.

        >>> updrs.score(measurement)

        To read the testResultSet data from a file. See TestResultSet class for more details.

        >>> updrs = pdkit.UPDRS(data_frame_file_path=file_path_to_testResultSet_file)

        :param data_frame: testResultSet
        :type data_frame: pandas.DataFrame
        :param data_frame_file_path: the path to read the data frame from
        :type data_frame_file_path: string
    """

    def __init__(self, data_frame=None, data_frame_file_path=None):
        try:
            if data_frame_file_path is not None:
                data_frame = pd.read_csv(data_frame_file_path)
                data_frame = data_frame.fillna(data_frame.mean())
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

        except IOError as e:
            ierr = "({}): {}".format(e.errno, e.strerror)
            logging.error("UPDRS load data, file not found, I/O error %s", ierr)

        except ValueError as verr:
            logging.error("UPDRS ValueError ->%s", verr.message)

        except:
            logging.error("Unexpected error on UPDRS init: %s", sys.exc_info()[0])

    def __train(self, n_clusters=4):
        """
            Calculate cluster's centroids and standard deviations. If there are at least the number of threshold rows \
            then:

            * Observations will be normalised.

            * Standard deviations will be returned.

            * Clusters will be returned.

            * Centroids are ordered based on their distance from an arbitrary -100, -100 point.

            If there are not enough Observations, then centroids and standard deviations will be set to the empty list.

            General strategy: Use numpy.array for calculations. Keep everything in float. Convert arrays back to lists \
            at the end.

            :param n_clusters: the number of clusters
            :type n_clusters: int
        """

        try:
            for obs in self.observations:
                features, ids = self.__get_features_for_observation(observation=obs, last_column_is_id=True)
                # the last column is the observation id
                normalised_data = whiten(features)

                # skip any rows that contain just zero values... they create nans
                first_safe_row = pdkit.utils.non_zero_index(normalised_data)
                observation_ids = features.tolist()
                sd = features[first_safe_row] / normalised_data[first_safe_row]

                # Calculate centroids and sort result
                centroids_array, _ = kmeans(normalised_data, n_clusters)
                sorted_centroids = pdkit.utils.centroid_sort(centroids_array)

                if not self.clusters:
                    self.clusters = [[obs, sd.tolist(), sorted_centroids.tolist()]]
                else:
                    self.clusters.append([obs, sd.tolist(),sorted_centroids.tolist()])
        except IOError as e:
            ierr = "({}): {}".format(e.errno, e.strerror)
            logging.error("Error training UPDRS, file not found, I/O error %s", ierr)

        except ValueError as verr:
            logging.error("Error training UPDRS ValueError ->%s", verr.message)

        except:
            logging.error("Unexpected error on training UPDRS init: %s", sys.exc_info()[0])

    def __get_centroids_by_observation(self, observation):
        for (obs, sd, cen) in self.clusters:
            if obs == observation:
                return cen

    def __get_sd_by_observation(self, observation):
        for (obs, sd, cen) in self.clusters:
            if obs == observation:
                return sd

    def __get_centroids_sd(self, observation):
        for (obs, sd, cen) in self.clusters:
            if obs == observation:
                return cen, sd

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

    def get_single_score(self, point, centroids=None, sd=None):
        """
            Get a single score is a wrapper around the result of classifying a Point against a group of centroids. \
            Attributes:

            observation_score (dict): Original received point and normalised point.

            :Example:

            >>> { "original": [0.40369016, 0.65217912], "normalised": [1.65915104, 3.03896181]}

            nearest_cluster (int): Index of the nearest cluster. If distances match, then lowest numbered cluster \
            wins.

            distances (list (float)): List of distances from the Point to each cluster centroid. E.g:

            >>> [2.38086238, 0.12382605, 2.0362993, 1.43195021]

            centroids (list (list (float))): A list of the current centroidswhen queried. E.g:

            >>> [ [0.23944831, 1.12769265], [1.75621978, 3.11584191], [2.65884563, 1.26494783], \
            [0.39421099, 2.36783733] ]

            :param point: the point to classify
            :type point: pandas.DataFrame
            :param centroids: the centroids
            :type centroids: np.array
            :param sd: the standard deviation
            :type sd: np.array
            :return score: the score for a given observation
            :rtype score: int
        """
        normalised_point = array(point) / array(sd)

        observation_score = {
            'original': point,
            'normalised': normalised_point.tolist(),
        }

        distances = [
            euclidean(normalised_point, centroid)
            for centroid in centroids
        ]

        return int(distances.index(min(distances)))

    def write_model(self, filename='scores', filepath='', output_format='csv'):
        """
            This method calculates the scores and writes them to a file the data frame received. If the output format
            is other than 'csv' it will print the scores.

            :param filename: the name to give to the file
            :type filename: string
            :param filepath: the path to save the file
            :type filepath: string
            :param output_format: the format of the file to write ('csv')
            :type output_format: string
        """
        scores_array = np.array([])
        for obs in self.observations:
            c, sd = self.__get_centroids_sd(obs)
            points, ids = self.__get_features_for_observation(observation=obs, last_column_is_id=True)

            b = np.array([])
            for p in points:
                b = np.append(b, [self.get_single_score(p, centroids=c, sd=sd)])

            scores_array = np.vstack([scores_array, b]) if scores_array.size else b

        scores_array = np.concatenate((ids[:, np.newaxis], scores_array.transpose()), axis=1)
        header = 'id,'+','.join(self.observations)

        try:
            if output_format == 'csv':
                filename = join(filepath, filename) + '.' + output_format
                np.savetxt(filename, scores_array, delimiter=",", fmt='%i', header=header,comments='')
            else:
                print(scores_array)
        except:
            logging.error("Unexpected error on writing output")

    def score(self, measurement, output_format='array'):
        """
            Method to score/classify a measurement against the trained knn clusters

            :param measurement: the point to classify
            :type measurement: pandas.DataFrame
            :param output_format: the format to return the scores ('array' or 'str')
            :type output_format: string
            :return scores: the scores for a given test/point
            :rtype scores: np.array
        """
        scores = np.array([])
        for obs in self.observations:
            c, sd = self.__get_centroids_sd(obs)
            p, ids = self.__get_features_for_observation(data_frame = measurement, observation=obs,
                                                         last_column_is_id=True)

            scores = np.append(scores, [self.get_single_score(p, centroids=c, sd=sd)], axis=0)

        if output_format == 'array':
            return scores.astype(int)
        else:
            return np.array_str(scores.astype(int))

