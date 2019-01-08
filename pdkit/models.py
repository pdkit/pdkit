"""Summary
"""
import inspect

from keras.models import Sequential, Model
from keras.layers import Input, Activation
from keras.layers.core import (Dense,
                               Dropout,
                               Flatten,
                               ActivityRegularization,
                               Activation,
                               Lambda,
                               Reshape)

from keras.layers.convolutional import Conv1D, Convolution2D
from keras.layers.pooling import AveragePooling1D, MaxPooling1D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add

from keras.initializers import glorot_uniform

from keras.layers.advanced_activations import LeakyReLU

def save_keras_model(path, model, compress=True):
    """Summary
    
    Args:
        path (TYPE): Description
        model (TYPE): Description
        compress (bool, optional): Description
    """
    import os
    import shutil

    json_path = "{}.json".format(path)
    h5_path = "{}.h5".format(path)
    # serialize model to JSON
    model_json = model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(h5_path)
    if compress:
        import tarfile
        tar_path = '{}.tar.gz'.format(path)
        tar = tarfile.open(tar_path, 'w:gz')
        tar.add(json_path, arcname=json_path.split("/")[-1])
        tar.add(h5_path, arcname=h5_path.split("/")[-1])
        tar.close()
        os.remove(json_path)
        os.remove(h5_path)
        print(("Saved model to {}".format(tar_path)))
    else:
        print(("Saved model to {}".format(path)))


def load_keras_model(path):
    """Summary
    
    Args:
        path (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    import tarfile, os
    from keras.models import model_from_json
    is_tar = False
    if tarfile.is_tarfile(path):
        is_tar = True
        tar = tarfile.open(path, "r:gz")
        tar.extractall(os.path.dirname(path))
        tar.close()
        path = path.split(".")[0]

    json_path = "{}.model.json".format(path)
    h5_path = "{}.model.h5".format(path)

    # load json and create model
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(h5_path)

    if is_tar:
        os.remove(json_path)
        os.remove(h5_path)

    print(("Loaded model from {}".format(path)))
    return loaded_model

def res_unit(inputs, out, length, leak=0.0):
    """Summary
    
    Args:
        inputs (TYPE): Description
        out (TYPE): Description
        length (TYPE): Description
        leak (float, optional): Description
    
    Returns:
        TYPE: Description
    """
    if inputs._keras_shape[2] != out:
        inputs = Conv1D(out, length, border_mode='same')(inputs)

    x = BatchNormalization()(inputs)
    x = Activation(LeakyReLU(alpha=leak))(x)
    x = Conv1D(out, length, border_mode='same')(x)

    x = BatchNormalization()(x)
    x = Activation(LeakyReLU(alpha=leak))(x)
    x = Conv1D(out, length, border_mode='same')(x)

    x = add([inputs, x], mode='sum')

    return x

def new_resnet(inputs, outputs, layers):
    """Summary
    
    Args:
        inputs (TYPE): Description
        outputs (TYPE): Description
        layers (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    inputs = Input(shape=inputs)
    
    print(inputs._keras_shape)
    
    x = Conv1D(nb_filter=64, filter_length=7, border_mode='same', subsample_length=2)(inputs)
    
    x = BatchNormalization()(x)
    x = Activation(LeakyReLU(alpha=0.0))(x)
    
    print(x._keras_shape)
    
    #x = MaxPooling1D(pool_length=4, strides=2, border_mode='same')(x)
    
    print(x._keras_shape)
    
    nb_filters = 64
    for i in range(layers):
        x = res_unit(x, nb_filters, 1)
        
        print(x._keras_shape)
        
        nb_filters *= 2
    
    print(x._keras_shape)
    
    x = AveragePooling1D(pool_length=x._keras_shape[1], strides=32, border_mode='same')(x)
    x = Flatten()(x)
    
    print(x._keras_shape)
    
    output = Dense(outputs, activation='softmax')(x)
    model = Model(inputs, output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
    
    
def res_block_1d(inputs, kernels, leak=0.0, shortcut=False):
    """Summary
    
    Args:
        inputs (TYPE): Description
        kernels (TYPE): Description
        leak (float, optional): Description
        shortcut (bool, optional): Description
    
    Returns:
        TYPE: Description
    """
    for i, k in enumerate(kernels):
        if i == 0:
            x = Conv1D(k, 1, border_mode='same')(inputs)
        else:
            x = Conv1D(k, 1, border_mode='same')(x)
            
        x = BatchNormalization()(x)
        
        if i == len(kernels):
            if shortcut:
                short = Conv1D(k, 1, border_mode='same')
                short = BatchNormalization(short)
                x = add([x, short])
            else:
                x = add([x, input])
                
        x = Activation(LeakyReLU(alpha=leak))(x)
    return x

def ResNet(input_shape, output_shape, optimizer='adam', leak=0.0):
    """Summary
    
    Args:
        input_shape (TYPE): Description
        output_shape (TYPE): Description
        optimizer (str, optional): Description
        leak (float, optional): Description
    
    Returns:
        TYPE: Description
    """
    inputs = Input(shape=input_shape)
    
    x = Conv1D(64, 3, border_mode='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = MaxPooling1D(pool_length=2, border_mode='same')(x)
    
    x = res_block_1d(x, [64, 64, 256], leak=leak, shortcut=True)
    x = res_block_1d(x, [64, 64, 256], leak=leak, shortcut=False)
    x = res_block_1d(x, [64, 64, 256], leak=leak, shortcut=False)
    
    x = res_block_1d(x, [128, 128, 512], leak=leak, shortcut=True)
    x = res_block_1d(x, [128, 128, 512], leak=leak, shortcut=False)
    x = res_block_1d(x, [128, 128, 512], leak=leak, shortcut=False)
    x = res_block_1d(x, [128, 128, 512], leak=leak, shortcut=False)
    
    x = res_block_1d(x, [256, 256, 1024], leak=leak, shortcut=True)
    x = res_block_1d(x, [256, 256, 1024], leak=leak, shortcut=False)
    x = res_block_1d(x, [256, 256, 1024], leak=leak, shortcut=False)
    x = res_block_1d(x, [256, 256, 1024], leak=leak, shortcut=False)
#     x = res_block_1d(x, [256, 256, 1024], leak=leak, shortcut=False)
#     x = res_block_1d(x, [256, 256, 1024], leak=leak, shortcut=False)
    
#     x = res_block_1d(x, [512, 512, 2048], leak=leak, shortcut=True)
#     x = res_block_1d(x, [512, 512, 2048], leak=leak, shortcut=False)
#     x = res_block_1d(x, [512, 512, 2048], leak=leak, shortcut=False)
    
    x = AveragePooling1D(pool_length=4)(x)
    
    x = Flatten()(x)
    outputs = Dense(output_shape, activation='softmax')(x)
    
    model = Model(inputs, outputs)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model
    

def residual_time_steps(steps, conv_shape, timesteps, input, pool=2, leak=0.3, drop=0.5, time_leak=0.2, time_drop=0.0):
    """Summary
    
    Args:
        steps (TYPE): Description
        conv_shape (TYPE): Description
        timesteps (TYPE): Description
        input (TYPE): Description
        pool (int, optional): Description
        leak (float, optional): Description
        drop (float, optional): Description
        time_leak (float, optional): Description
        time_drop (float, optional): Description
    
    Returns:
        TYPE: Description
    """
    for i in range(steps):
        
        c2 = Conv1D(conv_shape[0], conv_shape[1], padding='same')(input)
        b2 = BatchNormalization()(c2)
        a2 = LeakyReLU(alpha=leak)(b2)
        
        recurrent = residual_shared_weights_steps(steps=timesteps,
                                                 conv_shape=conv_shape,
                                                 input=a2, 
                                                 conv=c2,
                                                 leak=time_leak,
                                                 drop=time_drop)


        merge = add([recurrent, input])
        b3 = BatchNormalization()(merge)
        a3 = LeakyReLU(alpha=leak)(b3)
        p3 = MaxPooling1D(pool_size=pool, strides=None, padding='same')(a3)
        d3 = Dropout(drop)(p3)
        input = d3
        
    return input

def residual_shared_weights_steps(steps, conv_shape, input, conv, leak, drop):
    """Summary
    
    Args:
        steps (TYPE): Description
        conv_shape (TYPE): Description
        input (TYPE): Description
        conv (TYPE): Description
        leak (TYPE): Description
        drop (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    inputs = []
    
    for i in range(steps):
        if i == 0:
            _c2 = Conv1D(conv_shape[0], conv_shape[1], padding='same')
            c2a = _c2(input)
        else:
            c2a = _c2(d2a)
        inputs.append(c2a)
        if (i > 0) & (i % 2 == 0):
            s2a = add([conv, inputs[-1], inputs[-1]])
        else:
            s2a = add([conv, inputs[-1]])
            
        b2a = BatchNormalization()(s2a)
        a2a = LeakyReLU(alpha=leak)(b2a)
        d2a = Dropout(drop)(a2a)
        
    out = add([conv, d2a])
    out = BatchNormalization()(out)
    out = LeakyReLU(alpha=leak)(out)
    out = Dropout(drop)(out)
    return out


def time_steps(input, conv_layer, time_conv_layer, padding):
    """Summary
    
    Args:
        input (TYPE): Description
        conv_layer (TYPE): Description
        time_conv_layer (TYPE): Description
        padding (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    
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
    """Summary
    
    Args:
        input (TYPE): Description
        outer_conv (TYPE): Description
        time_conv_layer (TYPE): Description
        padding (TYPE): Description
    
    Returns:
        TYPE: Description
    """
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


def ResCL(input_shape=(1024, 124),
        convs=[(128, 3), (256, 6), (512, 9)],
        residual_steps=2,
        recurrent_steps=3,
        targets=5, 
        optimizer='adam',
        pool=2,
        leak=0.3,
        drop=0.5,
        time_leak=0.2,
        time_drop=0.0):
    """Summary
    
    Args:
        input_shape (tuple, optional): Description
        convs (list, optional): Description
        residual_steps (int, optional): Description
        recurrent_steps (int, optional): Description
        targets (int, optional): Description
        optimizer (str, optional): Description
        pool (int, optional): Description
        leak (float, optional): Description
        drop (float, optional): Description
        time_leak (float, optional): Description
        time_drop (float, optional): Description
    
    Returns:
        TYPE: Description
    """

    inputs = Input(shape=input_shape)
    
    for i, c in enumerate(convs):
        if i == 0:
            c1 = Conv1D(c[0], c[1], padding='same')(inputs)
        else:
            c1 = Conv1D(c[0], c[1], padding='same')(core)
        b1 = BatchNormalization()(c1)
        a1 = LeakyReLU(alpha=leak)(b1)
        p1 = MaxPooling1D(pool_size=pool, strides=None, padding='same')(a1)
        d1 = Dropout(drop)(p1)

        core = residual_time_steps( steps=residual_steps,
                                    conv_shape=(c[0], c[1]),
                                    timesteps=recurrent_steps,
                                    input=d1,
                                    leak=leak,
                                    drop=drop,
                                    time_leak=time_leak,
                                    time_drop=0.0)
    
    flat = Flatten()(core)
    output = Dense(targets, activation='softmax')(flat)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model


def SIMPLE(input_shape,
           
           conv_layers,
           flatten_dropout,
           dense_layers,
           output_layer=[1, 'sigmoid'],
           padding='same', 
           optimizer='adam',
           loss='binary_crossentropy'):
    """Summary
    
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
    drop = Dropout(flatten_dropout)(drop)

    for i, d in enumerate(dense_layers):
        dense = Dense(d[0], activation='relu')(drop)
        bn = BatchNormalization()(dense)
        act = LeakyReLU(alpha=d[1])(bn)
        drop = Dropout(d[2])(act)
    
    output = Dense(output_layer[0], activation=output_layer[1])(drop)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    
    return model


def PERCOM(input_shape,
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
