#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2018 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author(s): J.S. Pons

from numpy import array
from scipy.spatial.distance import euclidean


class Classification:
    """
        Classification is a wrapper around the result of classifying a Point
        against a group of Obscluster centroids.
        Attributes:
            observation_score (dict): Original recieved point and normalised point.
                E.g.
                {
                    "original": [0.40369016, 0.65217912],
                    "normalised": [1.65915104, 3.03896181],
                }
            nearest_cluster (int): Index of the nearest cluster. If distances
                match, then lowest numbered cluster wins.
            distances (list (float)): List of distances from the Point to each
                cluster centroid. E.g.
                [2.38086238, 0.12382605, 2.0362993, 1.43195021]
            centroids (list (list (float))): A list of the current centroids
                of the Obscluster when queried. E.g.
                [
                    [0.23944831, 1.12769265],
                    [1.75621978, 3.11584191],
                    [2.65884563, 1.26494783],
                    [0.39421099, 2.36783733],
                ]
    """

    def __init__(self, centroids, standard_deviations):
        assert len(centroids[0]) == len(standard_deviations), (
            'Dimension mismatch: centroids and standard deviations'
        )
        # assert len(standard_deviations) == len(point), (
        #     'Dimension mismatch: standard deviations and point'
        # )
        self.centroids = centroids
        # self.point = point
        self.standard_deviations = standard_deviations

    def classify(self, point):
        """
        Run the classification calculation
        """
        normalised_point = array(point) / array(self.standard_deviations)

        self.observation_score = {
            'original': point,
            'normalised': normalised_point.tolist(),
        }

        self.distances = [
            euclidean(normalised_point, centroid)
            for centroid in self.centroids
        ]

        self.nearest_cluster = self.distances.index(min(self.distances))
