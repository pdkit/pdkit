#!/usr/bin/env python3
# Copyright 2018 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author(s): J.S. Pons, Cosmin Stamate

import pandas as pd
import numpy as np

import sys
import functools

def load_cloudupdrs_data(filename, time_difference=1000000000.0):
    '''
       This method loads data in the cloudupdrs format
       
       Usually the data will be saved in a csv file and it should look like this:
       
      .. code-block:: json
      
         timestamp0, x0, y0, z0
         timestamp1, x1, y1, z1
         timestamp0, x2, y2, z2
         .
         .
         .
         timestampn, xn, yn, zn
       
      where x, y, z are the components of the acceleration

      :param str filename: The path to load data from
      :param float time_difference: Convert times. The default is from from nanoseconds to seconds.
    '''
    # data_m = pd.read_table(filename, sep=',', header=None)
    data_m = np.genfromtxt(filename, delimiter=',', invalid_raise=False)
    date_times = pd.to_datetime((data_m[:, 0] - data_m[0, 0]))
    time_difference = (data_m[:, 0] - data_m[0, 0]) / time_difference
    magnitude_sum_acceleration = \
        np.sqrt(data_m[:, 1] ** 2 + data_m[:, 2] ** 2 + data_m[:, 3] ** 2)
    data = {'td': time_difference, 'x': data_m[:, 1], 'y': data_m[:, 2], 'z': data_m[:, 3],
            'mag_sum_acc': magnitude_sum_acceleration}
    data_frame = pd.DataFrame(data, index=date_times, columns=['td', 'x', 'y', 'z', 'mag_sum_acc'])
    return data_frame


def load_mpower_data(filename, time_difference=1000000000.0):
    '''
        This method loads data in the [mpower]_ format \
        
        The format is like: 
       .. code-block:: json
            
            [
               {
                  "timestamp":19298.67999479167,
                  "x": ... ,
                  "y": ...,
                  "z": ...,
               },
               {...},
               {...}
            ]

       :param str filename: The path to load data from
       :param float time_difference: Convert times. The default is from from nanoseconds to seconds.
    '''
    raw_data = pd.read_json(filename)
    date_times = pd.to_datetime(raw_data.timestamp * time_difference - raw_data.timestamp[0] * time_difference)
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


def numerical_integration(signal, sampling_frequency):
    '''
        Numerically integrate a signal with it's sampling frequency.

        :param array signal: A 1-dimensional array or list (the signal).
        :param float sampling_frequency: The sampling frequency for the signal.
    '''
        
    integrate = sum(signal[1:]) / sampling_frequency + sum(signal[:-1])
    integrate /= sampling_frequency * 2
    
    return integrate

def autocorrelation(signal):
    ''' 
        The correlation of a signal with a delayed copy of itself.
        More info here: https://en.wikipedia.org/wiki/Autocorrelation#Estimation

        :param array signal: A 1-dimensional array or list (the signal).
    '''

    signal = np.array(signal)
    n = len(signal)
    variance = signal.var()
    signal -= signal.mean()
    
    r = np.correlate(signal, signal, mode = 'full')[-n:]
    result = r / (variance * (np.arange(n, 0, -1)))
    
    return result


def peakdet(signal, delta, x = None):
    '''
        Find the local maxima and minima ("peaks") in a 1-dimensional signal.
        Converted from MATLAB script at http://billauer.co.il/peakdet.html

        :param array signal: A 1-dimensional array or list (the signal).
        :param float delta: The peak threashold. A point is considered a maximum peak if it has the maximal value, and was preceded (to the left) by a value lower by delta.
        :param array x: indices in local maxima and minima are replaced with the corresponding values in x.
    '''
    
    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(signal))

    v = np.asarray(signal)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.inf, -np.inf
    mnpos, mxpos = np.nan, np.nan

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)

