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


class ClusterProcessor:
    def __init__(self, data_frame):
        """
        The data frame is the testResultSet
        :param data_frame: testResultSet
        :type data_frame: pandas.DataFrame
        """
        self.data_frame = data_frame.sort_values(by=['id'])
        self.clusters = np.array([])
        self.observations = [
            "OT-LH",
            "OT-RH",
            "TT-LH",
            "TT-RH",
            "KTOH-LH",
            "KTOH-RH",
            "LA-LL",
            "LA-RL",
            "PTOTH-LH",
            "PTOTH-RH",
            "PS-LH",
            "PS-RH",
            "RTA-LH",
            "RTA-RH",
            "RTA-LL",
            "RTA-RL"
        ]

    def process(self):
        """
        Calculate cluster's centroids and standard deviations.
        If there are at least the number of threshold rows then:
        * Observations will be normalised.
        * Standard deviations will be stored.
        * Clusters will be calculated.
        * Centroids are ordered based on their distance from an arbitrary -100,
            -100 point.
        * Ordered centroids are written to the Obscluster.
        * Writes the updated Obscluster instance to the database.

        If there are not enough Observations, then centroids and standard
        deviations will be set to the empty list. This means that this function
        can be used to update an Obscluster after data deletion.

        General strategy: Use numpy.array for calculations. Keep everything in
        float. Convert arrays back to lists at the end ready for saving /
        output.
        """

        for obs in self.observations:
            # obs = "LA-LL"
            features = self.get_features_for_observation(observation=obs)
            # the last column is the observation id
            normalised_data = whiten(features[:,:-1])

            # skip any rows that contain just zero values... they create nans
            first_safe_row = pdkit.utils.non_zero_index(normalised_data)
            # first_safe_row = pdk.utils.non_zero_index(normalised_data)
            observation_ids = features[:,-1].tolist()
            sd = features[:,:-1][first_safe_row] / normalised_data[first_safe_row]

            # Calculate centroids and sort result
            centroids_array, _ = kmeans(normalised_data, 4)
            sorted_centroids = pdkit.utils.centroid_sort(centroids_array)
            # sorted_centroids = pdk.utils.centroid_sort(centroids_array)

            if not self.clusters:
                # self.standard_deviations = [{'obs': obs, 'sd': sd.tolist(),'ids':observation_ids}]
                self.clusters = [[obs, sd.tolist(), sorted_centroids.tolist()]]
            else:
                # self.standard_deviations.append({'id': obs, 'sd': sd.tolist(),'ids':observation_ids})
                self.clusters.append([obs, sd.tolist(),sorted_centroids.tolist()])

    def get_centroids_by_observation(self, observation):
        for (obs, sd, cen) in self.clusters:
            if obs == observation:
                return cen

    def get_sd_by_observation(self, observation):
        for (obs, sd, cen) in self.clusters:
            if obs == observation:
                return sd

    def get_features_for_observation(self, observation='LA-LL', skip=None):
        try:
            a = np.array([])
            for index, row in self.data_frame.iterrows():
                if not skip == row['id']:
                    b = np.nan_to_num(row[row.keys().str.contains(observation)].values)
                    b = np.append(b, row['id'])
                    a = np.vstack([a, b]) if a.size else b

            return a
        except:
            logging.error("observation not found in dataframe")