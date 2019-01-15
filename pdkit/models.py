"""Summary
"""
import inspect

from keras.models import Model
from keras.layers import Input
from keras.layers.core import (Dense,
                               Dropout,
                               Flatten,
                               Activation)

from keras.layers.convolutional import Conv1D
from keras.layers.pooling import AveragePooling1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add

from keras.layers.advanced_activations import LeakyReLU


def time_steps(input, conv_layer, time_conv_layer, padding):    
    conv = Conv1D(conv_layer[0][0], conv_layer[0][1], padding=padding)(input)
    batch = BatchNormalization()(conv)
    act = LeakyReLU(alpha=conv_layer[2])(batch)

    inner = shared_weights_steps(input=act,
                                 outer_conv = conv,
                                 time_conv_layer=time_conv_layer,
                                 padding=padding,
                                )

    pool = MaxPooling1D(pool_size=conv_layer[1][0], strides=conv_layer[1][1], padding=padding)(inner)
    drop = Dropout(conv_layer[3])(pool)
        
    return drop

def shared_weights_steps(input, outer_conv, time_conv_layer, padding):
    for i in range(time_conv_layer[4]):
        if i == 0:
            _c2 = Conv1D(time_conv_layer[0][0], time_conv_layer[0][1], padding=padding)
    
        c2a = _c2(input)
        s2a = add([outer_conv, c2a])
        b2a = BatchNormalization()(s2a)
        a2a = LeakyReLU(alpha=time_conv_layer[2])(b2a)
        d2a = Dropout(time_conv_layer[3])(a2a)
        
        input = d2a
    
    return input


def RCL(input_shape,
        rec_conv_layers, 
        dense_layers,
        output_layer=[1, 'sigmoid'],
        padding='same', 
        optimizer='adam',
        loss='binary_crossentropy'):
    """Summary
    
    Args:
        input_shape (tuple): The shape of the input layer.
        output_nodes (int): Number of nodes in the output layer. It depends on the loss function used.
        rec_conv_layers (list): RCL descriptor 
                                [
                                  [
                                    [(filter, kernel), (pool_size, stride), leak, drop],
                                    [(filter, kernel), (pool_size, stride), leak, drop],
                                    [(filter, kernel), (pool_size, stride), leak, drop, timesteps],
                                  ],
                                  ...
                                  [
                                    [],[],[]
                                  ]
                                ]
        dense_layers (TYPE): Dense layer descriptor [[fully_connected, leak, drop], ... []]
        padding (str, optional): Type of padding for conv and pooling layers
        optimizer (str or object optional): Keras optimizer as string or keras optimizer
    
    Returns:
        model: The compiled Kears model, ready for training.
    """
    
    inputs = Input(shape=input_shape)
    
    for i, c in enumerate(rec_conv_layers):
        
        conv = Conv1D(c[0][0][0], c[0][0][1], padding=padding)(inputs)
        batch = BatchNormalization()(conv)
        act = LeakyReLU(alpha=c[0][2])(batch)
        pool = MaxPooling1D(pool_size=c[0][1][0], strides=c[0][1][1], padding=padding)(act)
        d1 = Dropout(c[0][3])(pool)
        
        inner = time_steps( input=d1,
                            conv_layer=c[1],
                            time_conv_layer=c[2],
                            padding=padding)
    
    drop = Flatten()(inner)

    for i, d in enumerate(dense_layers):
        dense = Dense(d[0], activation='relu')(drop)
        bn = BatchNormalization()(dense)
        act = LeakyReLU(alpha=d[1])(bn)
        drop = Dropout(d[2])(act)
    
    output = Dense(output_layer[0], activation=output_layer[1])(drop)
    
    model = Model(inputs=inputs, outputs=output)
        
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    
    return model


def VOICE(input_shape,
           conv_layers,
           dense_layers,
           output_layer=[1, 'sigmoid'],
           padding='same', 
           optimizer='adam',
           loss='binary_crossentropy'):
    """Conv1D CNN used primarily for voice data.
    
    Args:
        input_shape (tuple): The shape of the input layer
        targets (int): Number of targets
        conv_layers (list): Conv layer descriptor [[(filter, kernel), (pool_size, stride), leak, drop], ... []]
        dense_layers (TYPE): Dense layer descriptor [[fully_connected, leak, drop]]
        padding (str, optional): Type of padding for conv and pooling layers
        optimizer (str or object optional): Keras optimizer as string or keras optimizer
    
    Returns:
        TYPE: model, build_arguments
    """

    inputs = Input(shape=input_shape)
    
    for i, c in enumerate(conv_layers):
        if i == 0:
            conv = Conv1D(c[0][0], c[0][1], padding=padding)(inputs)
        else:
            conv = Conv1D(c[0][0], c[0][1], padding=padding)(drop)
        bn = BatchNormalization()(conv)
        act = LeakyReLU(alpha=c[2])(bn)
        pool = MaxPooling1D(pool_size=c[1][0], strides=c[1][1], padding=padding)(act)
        drop = Dropout(c[3])(pool)
    
    drop = Flatten()(drop)

    for i, d in enumerate(dense_layers):
        dense = Dense(d[0], activation='relu')(drop)
        bn = BatchNormalization()(dense)
        act = LeakyReLU(alpha=d[1])(bn)
        drop = Dropout(d[2])(act)
    
    output = Dense(output_layer[0], activation=output_layer[1])(drop)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    
    return model


def DNN(input_shape,
           dense_layers,
           output_layer=[1, 'sigmoid'],
           optimizer='adam',
           loss='binary_crossentropy'):
    """Summary
    
    Args:
        input_shape (list): The shape of the input layer
        targets (int): Number of targets
        dense_layers (list): Dense layer descriptor [fully_connected]
        optimizer (str or object optional): Keras optimizer as string or keras optimizer
    
    Returns:
        TYPE: model, build_arguments
    """

    inputs = Input(shape=input_shape)
    
    dense = inputs

    for i, d in enumerate(dense_layers):
        dense = Dense(d, activation='relu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.3)(dense)
    
    output = Dense(output_layer[0], activation=output_layer[1])(dense)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    
    return model
