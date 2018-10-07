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
from scipy.signal import butter, lfilter, correlate, freqz

import matplotlib.pylab as plt

import scipy.signal as sig
from scipy.cluster.vq import kmeans, vq, kmeans2


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

      :param str filename: The path to load data from
      :param float convert_times: Convert times. The default is from from nanoseconds to seconds.
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

def get_sampling_rate_from_timestamp(d):
    # group on minutes as pandas gives us the same second number
    # for seconds belonging to different minutes
    minutes = d.groupby(d.index.minute)

    # get the first minute (0) since we normalised the time above
    sampling_rate = d.iloc[minutes.indices[0]].index.second.value_counts().mean()
    print('Sampling rate is {} Hz'.format(sampling_rate))
    
    return sampling_rate

def load_segmented_data(filename):
    """
        Helper function to load segmented gait time series data.

        :param filename: The full path of the file that contais our data. This should be a comma separated value (csv file).
        :type filename: str

        :return: The gait time series segmented data, with a x, y, z, mag_acc_sum and segmented columns.
        :rtype: pandas.DataFrame
    """
    data = pd.read_csv(filename, index_col=0)
    data.index = data.index.astype(np.datetime64)
    
    return data

def load_freeze_data(filename):
    data = pd.read_csv(filename, delimiter=' ', header=None,)
    data.columns = ['td', 'ankle_f', 'ankle_v', 'ankle_l', 'leg_f', 'leg_v', 'leg_l', 'x', 'y', 'z', 'anno']
    data.td = data.td - data.td[0]
    
    # the dataset specified it uses ms
    date_time = pd.to_datetime(data.td, unit='ms')
    
    mag_acc_sum = np.sqrt(data.x ** 2 + data.y ** 2 + data.z ** 2)

    data['mag_sum_acc'] = mag_acc_sum
    data.index = date_time
    
    del data.index.name
    
    sampling_rate = get_sampling_rate_from_timestamp(data)
    
    return data

def load_huga_data(filepath):
    data = pd.read_csv(filepath, delimiter='\t', comment='#')
    
    # this dataset does not have timestamps so we had to infer the sampling rate from the description
    # we used 1 because sample each second
    # 58.82 because that's 679073 samples divided by 11544 seconds
    # and we used 1000 because milliseconds to seconds

    freq = int((1 / 58.82) * 1000)
    
    # this will make that nice date index that we know and love ...
    data.index = pd.date_range(start='1970-01-01', periods=data.shape[0], freq='{}ms'.format(freq))
    
    # this hardcoded as we don't need all that data...
    keep = ['acc_lt_x', 'acc_lt_y', 'acc_lt_z']#, 'act']
    
    drop = [c for c in data.columns if c not in keep]
    data = data.drop(columns=drop)
    
    # just keep the last letter (x, y and z)
    data = data.rename(lambda x: x[-1], axis=1)
    
    mag_acc_sum = np.sqrt(data.x ** 2 + data.y ** 2 + data.z ** 2)

    data['mag_sum_acc'] = mag_acc_sum
    
    data['td'] = data.index - data.index[0]
    
    sampling_rate = get_sampling_rate_from_timestamp(data)
    
    return data

def load_physics_data(filename):
    dd = pd.read_csv(filename)
    dd['mag_sum_acc'] = np.sqrt(dd.x ** 2 + dd.y ** 2 + dd.z ** 2)
    dd.index = pd.to_datetime(dd.time, unit='s')
    
    del dd.index.name
    dd = dd.drop(columns=['time'])
    
    sampling_rate = get_sampling_rate_from_timestamp(dd)
    
    return dd

def load_accapp_data(filename, convert_times=1000.0):
    df = pd.read_csv(filename, sep='\t', header=None)
    df.drop(columns=[0, 5], inplace=True)
    df.columns = ['td', 'x', 'y', 'z']
    df.td = (df.td - df.td[0])
    df.index = pd.to_datetime(df.td * convert_times * 1000)
    df.td = df.td / convert_times
    del df.index.name
    df['mag_sum_acc'] = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)
    return df

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

       :param str filename: The path to load data from
       :param float time_difference: Convert times. The default is from from nanoseconds to seconds.
       
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

          :param str filename: The path to load data from
          :param float convert_times: Convert times. The default is from from milliseconds to seconds.
    """
    data_m = np.genfromtxt(filename, delimiter=',', invalid_raise=False, skip_footer=1)
    date_times = pd.to_datetime((data_m[:, 0] - data_m[0, 0]))
    time_difference = (data_m[:, 0] - data_m[0, 0]) / convert_times
    data = {'td': time_difference, 'action_type': data_m[:, 2],'x': data_m[:, 3], 'y': data_m[:, 4],
            'x_target': data_m[:, 7], 'y_target': data_m[:, 8]}
    data_frame = pd.DataFrame(data, index=date_times, columns=['td', 'action_type','x', 'y', 'x_target', 'y_target'])
    return data_frame


def load_finger_tapping_mpower_data(filename, button_left_rect, button_right_rect, convert_times=1000.0):
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

        :param filename: The full path of the data file to load.
        :type filename: str
        :param format_file: format of the file. Default is CloudUPDRS. Set to mpower for mpower data.
        :type format_file: str
    """
    if format_file == 'mpower':
        return load_mpower_data(filename)
    
    elif format_file == 'segmented':
        return load_segmented_data(filename)

    elif format_file == 'accapp':
        return load_accapp_data(filename)

    elif format_file == 'physics':
        return load_physics_data(filename)
    
    elif format_file == 'freeze':
        return load_freeze_data(filename)
    
    elif format_file == 'huga':
        return load_huga_data(filename)

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
        :return: The integrated signal.
        :rtype: numpy.ndarray
    """
        
    integrate = sum(signal[1:]) / sampling_frequency + sum(signal[:-1])
    integrate /= sampling_frequency * 2
    
    return np.array(integrate)


def autocorrelation(signal):
    """ 
        The `correlation <https://en.wikipedia.org/wiki/Autocorrelation#Estimation>`_ of a signal with a delayed copy of itself.

        :param signal: A 1-dimensional array or list (the signal).
        :type signal: array
        :return: The autocorrelated signal.
        :rtype: numpy.ndarray
    """

    signal = np.array(signal)
    n = len(signal)
    variance = signal.var()
    signal -= signal.mean()
    
    r = np.correlate(signal, signal, mode = 'full')[-n:]
    result = r / (variance * (np.arange(n, 0, -1)))
    
    return np.array(result)


def peakdet(signal, delta, x =None):
    """
        Find the local maxima and minima (peaks) in a 1-dimensional signal.
        Converted from MATLAB script <http://billauer.co.il/peakdet.html>

        :param array signal: A 1-dimensional array or list (the signal).
        :type signal: array
        :param delta: The peak threashold. A point is considered a maximum peak if it has the maximal value, and was preceded (to the left) by a value lower by delta.
        :type delta: float
        :param x: Indices in local maxima and minima are replaced with the corresponding values in x (None default).
        :type x: array
        :return np.array(maxtab), np.array(mintab)
        :return maxtab: The highest peaks.
        :rtype maxtab: numpy.ndarray
        :return mintab: The lowest peaks.
        :rtype mintab: numpy.ndarray
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

        :param data: 1-dimensional time series data.
        :type data: array
        :param sample_rate: Sample rate of accelerometer reading (Hz)
        :type sample_rate: float
        :return interpeak: Number of samples between peaks 
        :rtype interpeak: int
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


def butter_lowpass_filter(data, sample_rate, cutoff=10, order=4, plot=False):
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
    
    if plot:
        w, h = freqz(b, a, worN=8000)
        plt.subplot(2, 1, 1)
        plt.plot(0.5*sample_rate*w/np.pi, np.abs(h), 'b')
        plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
        plt.axvline(cutoff, color='k')
        plt.xlim(0, 0.5*sample_rate)
        plt.title("Lowpass Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.grid()
        plt.show()

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


def get_signal_peaks_and_prominences(data):
    """ Get the signal peaks and peak prominences.
        
        :param data array: One-dimensional array.
        
        :return peaks array: The peaks of our signal.
        :return prominences array: The prominences of the peaks.
    """
    peaks, _ = sig.find_peaks(data)
    prominences = sig.peak_prominences(data, peaks)[0]
        
    return peaks, prominences

def smoothing_window(data, window=[1, 1, 1]):
    """ This is a smoothing functionality so we can fix misclassifications.
        It will run a sliding window of form [border, smoothing, border] on the
        signal and if the border elements are the same it will change the 
        smooth elements to match the border. An example would be for a window
        of [2, 1, 2] we have the following elements [1, 1, 0, 1, 1], this will
        transform it into [1, 1, 1, 1, 1]. So if the border elements match it
        will transform the middle (smoothing) into the same as the border.
        
        :param data array: One-dimensional array.
        :param window array: Used to define the [border, smoothing, border]
                             regions.
                             
        :return data array: The smoothed version of the original data.
    """
    
    for i in range(len(data) - sum(window)):
        
        start_window_from = i
        start_window_to = i+window[0]

        end_window_from = start_window_to + window[1]
        end_window_to = end_window_from + window[2]

        if np.all(data[start_window_from: start_window_to] == data[end_window_from: end_window_to]):
            data[start_window_from: end_window_to] = data[start_window_from]
            
    return data


def BellmanKSegment(x,k):
    # Divide a univariate time-series, x, into k contiguous segments
    # Cost is the sum of the squared residuals from the mean of each segment
    # Returns array containg the index for the endpoint of each segment in ascending order
    
    n = x.size
    cost = np.matrix(np.ones(shape=(k,n))*np.inf)
    startLoc = np.zeros(shape=(k,n), dtype=int)

    #Calculate residuals for all possible segments O(n^2)
    res = np.zeros(shape=(n,n)) # Each segment begins at index i and ends at index j inclusive.
    for i in range(n-1):
        mu = x[i]
        r = 0.0
        for j in range(i+1,n):

            r = r + ((j-i)/(j-i+1))*(x[j] - mu)*(x[j] - mu) #incrementally update squared residual
            mu = (x[j] + (j-i)*mu)/(j-i+1) #incrementally update mean
            res[i,j] = r #equivalent to res[i,j] = np.var(x[i:(j+1)])*(1+j-i) 
           

    #Determine optimal segmentation O(kn^2)
    segment = 0
    for j in range(n):
        cost[segment,j] = res[0,j]
        startLoc[segment, j] = 0

    for segment in range(1,k):
        for i in range(segment,n-1):   
            for j in range(i+1,n):
                tmpcost = res[i,j] + cost[segment-1,i-1]
                if cost[segment,j] > tmpcost: #break ties with smallest j                   
                    cost[segment,j]= tmpcost
                    startLoc[segment, j] = i
   

    #Backtrack to determine endpoints of each segment for the optimal partition
    endPoint = np.zeros(shape=(k,1))
    v = n
    for segment in range(k-1,-1,-1):
        endPoint[segment] = v-1         
        v = startLoc[segment,v-1]       

    return ExpandSegmentIndicies(endPoint)

def ExpandSegmentIndicies(endPoint):
    startPoint = -1
    lbls = np.array([])
    for segment in range(endPoint.size):
        lbls = np.append( arr=lbls ,values=np.repeat(segment, np.int(endPoint[segment]-startPoint)) )
        startPoint = endPoint[segment]
    return lbls


def plot_segmentation(data, peaks, segment_indexes):
    """ Will plot the data and segmentation based on the peaks and segment indexes.
    
        :param 1d-array data: The orginal axis of the data that was segmented into sections.
        :param 1d-array peaks: Peaks of the data.
        :param 1d-array segment_indexes: These are the different classes, corresponding to each peak.
        
        Will not return anything, instead it will plot the data and peaks with different colors for each class.
    
    """
    
    plt.plot(data);
    
    for segment in np.unique(segment_indexes):
        plt.plot(peaks[np.where(segment_indexes == segment)[0]], data[peaks][np.where(segment_indexes == segment)[0]], 'o')
        
    plt.show()


def DisplayBellmanK(data, ix):
    plt.plot(data);
    for segment in np.unique(ix):
        plt.plot(np.where(ix == segment)[0],data[np.where(ix == segment)[0]],'o')
    plt.show()