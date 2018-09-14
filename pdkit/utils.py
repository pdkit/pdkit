#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2018 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author(s): J.S. Pons, Cosmin Stamate

import sys
import re
import math

import pandas as pd
import numpy as np

from scipy.fftpack import rfft, fftfreq
from scipy.signal import butter, lfilter, correlate
import logging

from numpy import array
from scipy.spatial.distance import euclidean


def load_cloudupdrs_data(filename, convert_times=1000000000.0):
    """
       This method loads data in the cloudupdrs format
       
       Usually the data will be saved in a csv file and it should look like this:
       
      .. code-block:: json
      
         timestamp_0, x_0, y_0, z_0
         timestamp_1, x_1, y_1, z_1
         timestamp_2, x_2, y_2, z_2
         .
         .
         .
         timestamp_n, x_n, y_n, z_n
       
      where x, y, z are the components of the acceleration

      :param filename: The path to load data from
      :type filename: string
      :param convert_times: Convert times. The default is from from nanoseconds to seconds.
      :type convert_times: float
    """
    # data_m = pd.read_table(filename, sep=',', header=None)
    data_m = np.genfromtxt(filename, delimiter=',', invalid_raise=False)
    date_times = pd.to_datetime((data_m[:, 0] - data_m[0, 0]))
    time_difference = (data_m[:, 0] - data_m[0, 0]) / convert_times
    magnitude_sum_acceleration = \
        np.sqrt(data_m[:, 1] ** 2 + data_m[:, 2] ** 2 + data_m[:, 3] ** 2)
    data = {'td': time_difference, 'x': data_m[:, 1], 'y': data_m[:, 2], 'z': data_m[:, 3],
            'mag_sum_acc': magnitude_sum_acceleration}
    data_frame = pd.DataFrame(data, index=date_times, columns=['td', 'x', 'y', 'z', 'mag_sum_acc'])
    return data_frame


def load_mpower_data(filename, convert_times=1000000000.0):
    """
        This method loads data in the `mpower <https://www.synapse.org/#!Synapse:syn4993293/wiki/247859>`_ format
        
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

        :param filename: The path to load data from
        :type filename: string
        :param convert_times: Convert times. The default is from from nanoseconds to seconds.
        :type convert_times: float
    """
    raw_data = pd.read_json(filename)
    date_times = pd.to_datetime(raw_data.timestamp * convert_times - raw_data.timestamp[0] * convert_times)
    time_difference = (raw_data.timestamp - raw_data.timestamp[0])
    time_difference = time_difference.values
    magnitude_sum_acceleration = \
        np.sqrt(raw_data.x.values ** 2 + raw_data.y.values ** 2 + raw_data.z.values ** 2)
    data = {'td': time_difference, 'x': raw_data.x.values, 'y': raw_data.y.values,
            'z': raw_data.z.values, 'mag_sum_acc': magnitude_sum_acceleration}
    data_frame = pd.DataFrame(data, index=date_times, columns=['td', 'x', 'y', 'z', 'mag_sum_acc'])
    return data_frame


def load_finger_tapping_cloudupdrs_data(filename, convert_times=1000.0):
    """
        This method loads data in the cloudupdrs format for the finger tapping processor

        Usually the data will be saved in a csv file and it should look like this:

        .. code-block:: json

         timestamp_0, . , action_type_0, x_0, y_0, . , . , x_target_0, y_target_0
         timestamp_1, . , action_type_1, x_1, y_1, . , . , x_target_1, y_target_1
         timestamp_2, . , action_type_2, x_2, y_2, . , . , x_target_2, y_target_2
         .
         .
         .
         timestamp_n, . , action_type_n, x_n, y_n, . , . , x_target_n, y_target_n

        where data_frame.x, data_frame.y: components of tapping position. data_frame.x_target,
        data_frame.y_target their target.

        :param filename: The path to load data from
        :type filename: string
        :param convert_times: Convert times. The default is from from milliseconds to seconds.
        :type convert_times: float

    """
    data_m = np.genfromtxt(filename, delimiter=',', invalid_raise=False, skip_footer=1)
    date_times = pd.to_datetime((data_m[:, 0] - data_m[0, 0]))
    time_difference = (data_m[:, 0] - data_m[0, 0]) / convert_times
    data = {'td': time_difference, 'action_type': data_m[:, 2],'x': data_m[:, 3], 'y': data_m[:, 4],
            'x_target': data_m[:, 7], 'y_target': data_m[:, 8]}
    data_frame = pd.DataFrame(data, index=date_times, columns=['td', 'action_type','x', 'y', 'x_target', 'y_target'])
    return data_frame


def load_finger_tapping_mpower_data(filename, button_left_rect, button_right_rect, convert_times=1000.0):
    """
        This method loads data in the `mpower <https://www.synapse.org/#!Synapse:syn4993293/wiki/247859>`_ format
    """
    raw_data = pd.read_json(filename)
    date_times = pd.to_datetime(raw_data.TapTimeStamp * convert_times - raw_data.TapTimeStamp[0] * convert_times)
    time_difference = (raw_data.TapTimeStamp - raw_data.TapTimeStamp[0])
    time_difference = time_difference.values
    x = []
    y = []
    x_target = []
    y_target = []

    x_left, y_left, width_left, height_left = re.findall(r'-?\d+\.?\d*', button_left_rect)
    x_right, y_right, width_right, height_right = re.findall(r'-?\d+\.?\d*', button_right_rect)

    x_left_target = float(x_left) + ( float(width_left) / 2.0 )
    y_left_target = float(y_left) + ( float(height_left) / 2.0 )

    x_right_target = float(x_right) + ( float(width_right) / 2.0 )
    y_right_target = float(y_right) + ( float(height_right) / 2.0 )

    for row_index, row in raw_data.iterrows():
        x_coord, y_coord = re.findall(r'-?\d+\.?\d*', row.TapCoordinate)
        x.append(float(x_coord))
        y.append(float(y_coord))
        if row.TappedButtonId == 'TappedButtonLeft':
            x_target.append(x_left_target)
            y_target.append(y_left_target)
        else:
            x_target.append(x_right_target)
            y_target.append(y_right_target)

    data = {'td': time_difference, 'action_type': 1.0, 'x': x, 'y': y, 'x_target': x_target, 'y_target': y_target}
    data_frame = pd.DataFrame(data, index=date_times, columns=['td', 'action_type', 'x', 'y', 'x_target', 'y_target'])
    data_frame.index.name = 'timestamp'
    return data_frame


def load_data(filename, format_file='cloudupdrs', button_left_rect=None, button_right_rect=None):
    """
        This is a general load data method where the format of data to load can be passed as a parameter,

        :param filename: The path to load data from
        :type filename: str
        :param format_file: format of the file. Default is CloudUPDRS. Set to mpower for mpower data.
        :type format_file: str
        :param button_left_rect: mpower param
        :type button_left_rect: str
        :param button_right_rect: mpower param
        :type button_right_rect: str

    """
    if format_file == 'mpower':
        return load_mpower_data(filename)
    else:
        if format_file == 'ft_cloudupdrs':
            return load_finger_tapping_cloudupdrs_data(filename)
        else:
            if format_file == 'ft_mpower':
                if button_left_rect is not None and button_right_rect is not None:
                    return load_finger_tapping_mpower_data(filename, button_left_rect, button_right_rect)
            else:
                return load_cloudupdrs_data(filename)


def numerical_integration(signal, sampling_frequency):
    """
        Numerically integrate a signal with it's sampling frequency.

        :param signal: A 1-dimensional array or list (the signal).
        :type signal: array
        :param sampling_frequency: The sampling frequency for the signal.
        :type sampling_frequency: float
    """
        
    integrate = sum(signal[1:]) / sampling_frequency + sum(signal[:-1])
    integrate /= sampling_frequency * 2
    
    return integrate


def autocorrelation(signal):
    """ 
        The `correlation <https://en.wikipedia.org/wiki/Autocorrelation#Estimation>`_ of a signal with a delayed copy of itself.

        :param signal: A 1-dimensional array or list (the signal).
        :type signal: array

    """

    signal = np.array(signal)
    n = len(signal)
    variance = signal.var()
    signal -= signal.mean()
    
    r = np.correlate(signal, signal, mode = 'full')[-n:]
    result = r / (variance * (np.arange(n, 0, -1)))
    
    return result


def peakdet(signal, delta, x=None):
    """
        Find the local maxima and minima ("peaks") in a 1-dimensional signal.
        Converted from `MATLAB script <http://billauer.co.il/peakdet.html>`_ 

        :param signal: A 1-dimensional array or list (the signal).
        :type signal: array
        :param delta: The peak threashold. A point is considered a maximum peak if it has the maximal value, and was preceded (to the left) by a value lower by delta.
        :type delta: float
        :param x: indices in local maxima and minima are replaced with the corresponding values in x.
        :type x: array
        :return maxtab, mintab
        :rtype delta: np.array

    """
    
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


def compute_interpeak(data, sample_rate):
    """
        Compute number of samples between signal peaks using the real part of FFT.

        :param data: list or numpy array
        :type data: time series
        :param sample_rate: sample rate of accelerometer reading (Hz)
        :type sample_rate: float
        :return interpeak: number of samples between peaks
        :rtype interpeak: integer


        :Examples:

        >>> import numpy as np
        >>> from mhealthx.signals import compute_interpeak
        >>> data = np.random.random(10000)
        >>> sample_rate = 100
        >>> interpeak = compute_interpeak(data, sample_rate)
    
    """

    # Real part of FFT:
    freqs = fftfreq(data.size, d=1.0/sample_rate)
    f_signal = rfft(data)

    # Maximum non-zero frequency:
    imax_freq = np.argsort(f_signal)[-2]
    freq = np.abs(freqs[imax_freq])

    # Inter-peak samples:
    interpeak = np.int(np.round(sample_rate / freq))

    return interpeak


def butter_lowpass_filter(data, sample_rate, cutoff=10, order=4):
    """
        `Low-pass filter <http://stackoverflow.com/questions/25191620/
        creating-lowpass-filter-in-scipy-understanding-methods-and-units>`_ data by the [order]th order zero lag Butterworth filter
        whose cut frequency is set to [cutoff] Hz.

        :param data: time-series data,
        :type data: numpy array of floats
        :param: sample_rate: data sample rate
        :type sample_rate: integer
        :param cutoff: filter cutoff
        :type cutoff: float
        :param order: order
        :type order: integer
        :return y: low-pass-filtered data
        :rtype y: numpy array of floats

        :Examples:

        >>> from mhealthx.signals import butter_lowpass_filter
        >>> data = np.random.random(100)
        >>> sample_rate = 10
        >>> cutoff = 5
        >>> order = 4
        >>> y = butter_lowpass_filter(data, sample_rate, cutoff, order)
    
    """
    

    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    y = lfilter(b, a, data)

    return y


def crossings_nonzero_pos2neg(data):
    """
        Find `indices of zero crossings from positive to negative values <http://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python>`_.

        :param data: numpy array of floats
        :type data: numpy array of floats
        :return crossings: crossing indices to data
        :rtype crossings: numpy array of integers

        :Examples:

        >>> import numpy as np
        >>> from mhealthx.signals import crossings_nonzero_pos2neg
        >>> data = np.random.random(100)
        >>> crossings = crossings_nonzero_pos2neg(data)
    
    """
    import numpy as np

    if isinstance(data, np.ndarray):
        pass
    elif isinstance(data, list):
        data = np.asarray(data)
    else:
        raise IOError('data should be a numpy array')

    pos = data > 0

    crossings = (pos[:-1] & ~pos[1:]).nonzero()[0]

    return crossings


def autocorrelate(data, unbias=2, normalize=2):
    """
        Compute the autocorrelation coefficients for time series data.
        Here we use scipy.signal.correlate, but the results are the same as in
        Yang, et al., 2012 for unbias=1:


        "The autocorrelation coefficient refers to the correlation of a time
        series with its own past or future values. iGAIT uses unbiased
        autocorrelation coefficients of acceleration data to scale the regularity
        and symmetry of gait.
        The autocorrelation coefficients are divided by :math:`fc(0)`,
        so that the autocorrelation coefficient is equal to :math:`1` when :math:`t=0`:

        .. math::

            NFC(t) = \\frac{fc(t)}{fc(0)}


        Here :math:`NFC(t)` is the normalised autocorrelation coefficient, and :math:`fc(t)` are
        autocorrelation coefficients."

        :param data: time series data
        :type data: numpy array
        :param unbias: autocorrelation, divide by range (1) or by weighted range (2)
        :type unbias: integer or None
        :param normalize: divide by 1st coefficient (1) or by maximum abs. value (2)
        :type normalize: integer or None
        :return coefficients: autocorrelation coefficients [normalized, unbiased]
        :rtype coefficients: numpy array
        :return N: number of coefficients
        :rtype N: integer


        :Examples:

        >>> import numpy as np
        >>> from mhealthx.signals import autocorrelate
        >>> data = np.random.random(100)
        >>> unbias = 2
        >>> normalize = 2
        >>> plot_test = True
        >>> coefficients, N = autocorrelate(data, unbias, normalize, plot_test)
    """

    # Autocorrelation:
    coefficients = correlate(data, data, 'full')
    size = np.int(coefficients.size/2)
    coefficients = coefficients[size:]
    N = coefficients.size

    # Unbiased:
    if unbias:
        if unbias == 1:
            coefficients /= (N - np.arange(N))
        elif unbias == 2:
            coefficient_ratio = coefficients[0]/coefficients[-1]
            coefficients /= np.linspace(coefficient_ratio, 1, N)
        else:
            raise IOError("unbias should be set to 1, 2, or None")

    # Normalize:
    if normalize:
        if normalize == 1:
            coefficients /= np.abs(coefficients[0])
        elif normalize == 2:
            coefficients /= np.max(np.abs(coefficients))
        else:
            raise IOError("normalize should be set to 1, 2, or None")

    return coefficients, N


def centroid_sort(centroids):
    """
        Sort centroids. This is required so that the same cluster centroid is
        always the 0th one. It should also be the most negative. Order defined by
        the Euclidean distance between the centroid and an arbitrary "small" point
        [-100, -100] (in each dimension) to account for possible negatives. Cluster
        0 is the closest to that point, etc.

        0.  Set up
        >>> from numpy.testing import assert_array_equal
        1.  Single centroids just return themselves.
        >>> centroid_sort(array([[1.1, 2.2]]))
        array([[ 1.1,  2.2]])
        >>> centroid_sort(array([[1.1, 2.2, 3.3]]))
        array([[ 1.1,  2.2,  3.3]])
        2.  Positive 2d centroids are ordered.
        >>> centroids = array([
        ...     [5.34443858, 0.63266844],  # 3
        ...     [2.69156877, 0.76448578],  # 1
        ...     [4.74784197, 1.0815235 ],  # 2
        ...     [1.02330015, 0.16788118],  # 0
        ... ])
        >>> expected_sorted_centroids = array([
        ...     [1.02330015, 0.16788118],  # 0
        ...     [2.69156877, 0.76448578],  # 1
        ...     [4.74784197, 1.0815235 ],  # 2
        ...     [5.34443858, 0.63266844],  # 3
        ... ])
        >>> result = centroid_sort(centroids)
        >>> assert_array_equal(result, expected_sorted_centroids)
        3.  3d centroids spanning the origin are ordered.
        >>> centroids = array([
        ...     [ 3,   3,  4  ],  # 3
        ...     [ 1.5, 2,  3  ],  # 2
        ...     [-1,  -1, -1 ],   # 0
        ...     [ 0,   1,  0.5],  # 1
        ... ])
        >>> expected_sorted_centroids = array([
        ...     [-1,  -1, -1 ],   # 0
        ...     [ 0,   1,  0.5],  # 1
        ...     [ 1.5, 2,  3  ],  # 2
        ...     [ 3,   3,  4  ],  # 3
        ... ])
        >>> result = centroid_sort(centroids)
        >>> assert_array_equal(result, expected_sorted_centroids)

        :param centroids: array centroids
        :type centroids: numpy array
        :return centroids: array centroids
        :rtype centroids: numpy array

    """
    dimensions = len(centroids[0])
    negative_base_point = array(dimensions*[-100])

    decorated = [
        (euclidean(centroid, negative_base_point), centroid)
        for centroid in centroids
    ]
    decorated.sort()

    return array([centroid for dist, centroid in decorated])


def non_zero_index(arr):
    """
        Raises:
            ValueError: If no-non-zero rows can be found.
        0.  Empty array raises.
        >>> arr = array([])
        >>> non_zero_index(arr)
        Traceback (most recent call last):
        ...
        ValueError: No non-zero values
        1.  Array with zero values raises.
        >>> arr = array([
        ...     [0, 0],
        ...     [0, 0],
        ...     [0, 0, 0],
        ... ])
        >>> non_zero_index(arr)
        Traceback (most recent call last):
        ...
        ValueError: No non-zero values
        2.  Array with a non-zero value will have that index returned.
        >>> arr = array([
        ...     [0, 0],
        ...     [0, 0, 0],
        ...     [1, 0, 0],  # Still has zeros
        ...     [1, 1, 0],
        ...     [0, 1, 1],
        ...     [-1, 0, 0],
        ...     [-1, 2, 3],  # First non-zero array
        ...     [1, 2, 3],
        ... ])
        >>> non_zero_index(arr)
        6

        :param arr: array
        :type arr: numpy array
        :return index: Index of first non-zero entry in an array.
        :rtype index: int
    """
    for index, row in enumerate(arr):
        if non_zero_row(row):
            return index
    raise ValueError('No non-zero values')


def non_zero_row(arr):
    """
        0.  Empty row returns False.
        >>> arr = array([])
        >>> non_zero_row(arr)
        False
        1.  Row with a zero returns False.
        >>> arr = array([1, 4, 3, 0, 5, -1, -2])
        >>> non_zero_row(arr)
        False
        2.  Row with no zeros returns True.
        >>> arr = array([-1, -0.1, 0.001, 2])
        >>> non_zero_row(arr)
        True

        :param arr: array
        :type arr: numpy array
        :return empty: If row is completely free of zeros
        :rtype empty: bool
    """
    if len(arr) == 0:
        return False

    for item in arr:
        if item == 0:
            return False

    return True
