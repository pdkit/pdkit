#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2019 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author: Cosmin Stamate
import numpy as np

from keras.optimizers import adam

from pdkit.models import VOICE
from pdkit.utils import window_features


class VoiceProcessor(object):
    """
        Generic Convolutional neural network used for voice processing.
        Initializing this processor will instantiate a model and the model uses the keras api: https://keras.io.

        :param input_shape: (optional) Shape of the input data, this has to be in 2d, without the minibatch size
        :type input_shape: tuple
        :param labels: (optional) Number of classes, this should be 1 for binary classification
        :type labels: int
        :param output_activation: (optional) The activation function to use on the output data. For binary classification use 'sigmoid', for multiclass use 'softmax'
        :type output_activation: str


        :Examples:
         
        >>> import pdkit
        >>> qoi = pdkit.VoiceProcessor()
        >>> qoi.model.fit(X, y)
    """
    def __init__(self,
                 input_shape=(150, 4),
                 labels=1,
                 output_activation='sigmoid'):
        
        self.model = VOICE( input_shape=input_shape,
                            conv_layers=[[(3, 3), (3, 1), 0, 0], [(9, 9), (3, 1), 0, 0], [(18, 9), (3, 1), 0, 0]],
                            dense_layers=[],
                            padding='same',
                            optimizer=adam(lr=0.001),
                            output_layer=[labels, output_activation])
        
    

    def window_data(self, x, y=None, window_size=100, overlap=10):
        
        idx = window_features(np.arange(x.shape[0]), window_size, overlap)
        
        features = x[idx]
        if y:
            labels = [y] * idx.shape[0]
            return features, labels
        
        return features