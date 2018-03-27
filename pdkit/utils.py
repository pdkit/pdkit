#!/usr/bin/env python3
import pandas as pd
import numpy as np

NANOSEC_TO_SEC = 1000000000.0


def load_cloudupdrs_data(filename):
    '''
       This method loads data in the cloudupdrs format
       
       Usually the data will be saved in a csv file and it should look like this:
       
       timestamp0, x0, y0, z0
       timestamp1, x1, y1, z1
       timestamp0, x2, y2, z2
       .
       .
       .
       timestampn, xn, yn, zn
       
       where x, y, z are the components of the acceleration

       :param str filename: The path to load data from
    '''
    # data_m = pd.read_table(filename, sep=',', header=None)
    data_m = np.genfromtxt(filename, delimiter=',', invalid_raise=False)
    date_times = pd.to_datetime((data_m[:, 0] - data_m[0, 0]))
    time_difference = (data_m[:, 0] - data_m[0, 0]) / NANOSEC_TO_SEC
    magnitude_sum_acceleration = \
        np.sqrt(data_m[:, 1] ** 2 + data_m[:, 2] ** 2 + data_m[:, 3] ** 2)
    data = {'td': time_difference, 'x': data_m[:, 1], 'y': data_m[:, 2], 'z': data_m[:, 3],
            'mag_sum_acc': magnitude_sum_acceleration}
    data_frame = pd.DataFrame(data, index=date_times, columns=['td', 'x', 'y', 'z', 'mag_sum_acc'])
    return data_frame


def load_mpower_data(filename):
    '''
        This method loads data in the mpower format
       
        https://www.synapse.org/#!Synapse:syn4993293/wiki/247859
        
        The format is like:
        
        [ 
            {
                "timestamp":19298.67999479167,
                "x": ... ,
                "y": ...,
                "z": ...,
            }, {...}, {...}
        ]
        

        :param str filename: The path to load data from
    '''
    raw_data = pd.read_json(filename)
    date_times = pd.to_datetime(raw_data.timestamp * NANOSEC_TO_SEC - raw_data.timestamp[0] * NANOSEC_TO_SEC)
    time_difference = (raw_data.timestamp - raw_data.timestamp[0])
    time_difference = time_difference.values
    magnitude_sum_acceleration = \
        np.sqrt(raw_data.x.values ** 2 + raw_data.y.values ** 2 + raw_data.z.values ** 2)
    data = {'td': time_difference, 'x': raw_data.x.values, 'y': raw_data.y.values,
            'z': raw_data.z.values, 'mag_sum_acc': magnitude_sum_acceleration}
    data_frame = pd.DataFrame(data, index=date_times, columns=['td', 'x', 'y', 'z', 'mag_sum_acc'])
    return data_frame


def load_data(filename, format_file='cloudupdrs'):
    '''
        This is a general load data method where the format of data to load can be passed as a parameter,

        :param str filename: The path to load data from
        :param str format_file: format of the file. Default is CloudUPDRS. Set to mpower for mpower data.
    '''
    if format_file == 'mpower':
        return load_mpower_data(filename)
    else:
        return load_cloudupdrs_data(filename)
