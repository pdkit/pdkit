#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2019 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author: Cosmin Stamate 

import logging
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

class QuickTest:
    """
        Supervised Learning method for top-N feature importance classification to clinical observations.
        It uses `Random Forest classifier from <http://scikit-learn.org/stable/modules/\
        generated/sklearn.neighbors.KNeighborsClassifier.html>`_

        :Example:

        >>> import pdkit
        >>> test_restults_set = pdkit.TestResultSet(data_folder)
        >>> scores = pd.read_csv(path_to_scores_csv)
        >>> quick_test = pdkit.QuickTest(trs, scores)

        where the `data_folder` is the path to the UPDRS results files and  `path_to_scores_csv` is \
        the path to the UPDRS scores corresponding to the results.

        To get top-3 observations based on most variance explained.

        >>> quick_test.top_observations(3)

        :param test_results_set: (required) A data frame where the rows are the measurements and the columns correspond to the extracted features
        :type test_results_set: pdkit.TestResultSet
        :param scores: (required) The corresponding scores for each row of features.
        :type scores: pd.DataFrame
        :param n_estimators: (optional) The number of estimators for the RandomForestClassifier.
        :type scores: int
        :param random_state: (optional) The random state for the RandomForestClassifier. This helps with reproducibility.
        :type scores: int
    """
    
    def __init__(self, test_results_set, scores, n_estimators=100, random_state=123):
        
        self.features = test_results_set
        self.scores = scores
        self.n_estimators = n_estimators
        self.random_state = random_state
        
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
                "RTA-LL"]

        try:
            # train random forest and extract importances
            self.__get_feature_importance()

        except ValueError as verr:
            logging.error("QuickTest ValueError ->%s", verr.message)

        except:
            logging.error("Unexpected error on QuickTest init: %s", sys.exc_info()[0])
    
    def __get_feature_importance(self):
        """
            This will train the RandomForestClassifier on all the features and extract all the features importances.
        """
        # Build a forest and compute the feature importances
        self._feature_classifier = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)

        # fit it on the data and scores
        self._feature_classifier.fit(self.features.features.values, self.scores.values)

        # get the importances and order them
        self.feature_importances = self._feature_classifier.feature_importances_

    
    def top_observations(self, top=3):
        """
            Aggregates all the features into the corresponding observations and returns the top-N.

            :param top: The number of top observations to return
            :type top: int
            :return features: The top-N observations and the corresponding variance explained for each.
            :rtype features: pd.DataFrame
        """
        # get all the observations
        self.observations = {k: 0 for k in self.observations}
        sorted_indices = np.argsort(self.feature_importances)[::-1]

        # iterate over feature importance with the features indices
        for i, (imp, ind) in enumerate(zip(self.feature_importances, sorted_indices)):
                
                # get the feature name from the indice
                feature_name = self.features.features.columns[[ind]][0]

                # iterate over all the ovservations
                for k, _ in self.observations.items():

                    # sum the feature importance per observation if the feature name starts with the observation name
                    if k in feature_name:
                        self.observations[k] += imp

                        
        # we can wrap the dictionary in a dataframe for easy plotting
        # we need to transpose it to get the features as rows
        self.observations = pd.DataFrame([self.observations]).T

        # rename the only column, so we get a nice legend when we plot
        self.observations.columns = ['feature_importance']
        
        return self.observations.sort_values(by='feature_importance', ascending=False)[:top]
