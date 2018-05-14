#!/usr/bin/env python3
# Copyright 2018 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author(s): J.S. Pons

import sys
import logging

import numpy as np
import pandas as pd
from scipy import interpolate, signal, fft
from tsfresh.feature_extraction import feature_calculators

class TremorProcessor:
    '''
        This is the main Tremor Processor class. Once the data is loaded it will be
        accessible at data_frame (pandas.DataFrame), where it looks like:
        data_frame.x, data_frame.y, data_frame.z: x, y, z components of the acceleration
        data_frame.index is the datetime-like index
        
        These values are recommended by the author of the pilot study :cite:`Kassavetis2015`
        
        sampling_frequency = 100.0Hz
        cutoff_frequency = 2.0Hz
        filter_order = 2
        window = 256
        lower_frequency = 2.0Hz
        upper_frequency = 10.0Hz
            
        :Example:
         
        >>> import pdkit
        >>> tp = pdkit.TremorProcessor()
        >>> ts = pdkit.TremorTimeSeries().load(path_to_data)
        >>> amplitude, frequency = tp.process(ts)
    '''

    def __init__(self, sampling_frequency=100.0, cutoff_frequency=2.0, filter_order=2,
                 window=256, lower_frequency=2.0, upper_frequency=10.0):
        try:
            self.amplitude = 0
            self.frequency = 0

            self.sampling_frequency = sampling_frequency
            self.cutoff_frequency = cutoff_frequency
            self.filter_order = filter_order
            self.window = window
            self.lower_frequency = lower_frequency
            self.upper_frequency = upper_frequency

            logging.debug("TremorProcessor init")

        except IOError as e:
            ierr = "({}): {}".format(e.errno, e.strerror)
            logging.error("TremorProcessor I/O error %s", ierr)

        except ValueError as verr:
            logging.error("TremorProcessor ValueError ->%s", verr.message)

        except:
            logging.error("Unexpected error on TremorProcessor init: %s", sys.exc_info()[0])

    def resample_signal(self, data_frame):
        '''
            Convenience method for frequency conversion and resampling of data frame. 
            Object must have a DatetimeIndex. After re-sampling, this methods interpolate the time magnitude sum 
            acceleration values and the x,y,z values of the data frame acceleration

            :param data_frame: the data frame to resample
            :type data_frame: pandas.DataFrame
            :return: the resampled data frame
            :rtype: pandas.DataFrame

        '''
        df_resampled = data_frame.resample(str(1 / self.sampling_frequency) + 'S').mean()

        f = interpolate.interp1d(data_frame.td, data_frame.mag_sum_acc)
        new_timestamp = np.arange(data_frame.td[0], data_frame.td[-1], 1.0 / self.sampling_frequency)
        df_resampled.mag_sum_acc = f(new_timestamp)

        logging.debug("resample signal")
        return df_resampled.interpolate(method='linear')

    def filter_signal(self, data_frame, ts = 'mag_sum_acc'):
        '''
            This method filters a data frame signal as suggested in :cite:`Kassavetis2015`. First step is to high pass filter the data
            frame using a `Butterworth <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.butter.html>`_ digital and analog filter. Then this method 
            filters the data frame along one-dimension using a `digital filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html>`_. 

            :param data_frame: the input data frame
            :type data_frame: pandas.DataFrame
            :param ts: time series name of data frame to filter
            :type ts: string
            :return data_frame: adds a column named 'filtered_signal' to the data frame
            :rtype data_frame: pandas.DataFrame
        '''
        b, a = signal.butter(self.filter_order, 2 * self.cutoff_frequency / self.sampling_frequency, 'high', analog=False)
        filtered_signal = signal.lfilter(b, a, data_frame[ts].values)
        data_frame['filtered_signal'] = filtered_signal

        logging.debug("filter signal")
        return data_frame

    def fft_signal(self, data_frame):
        '''
            This method perform Fast Fourier Transform on the data frame using a `hanning window <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.hann.html>`_

            :param data_frame: the data frame
            :type data_frame: pandas.DataFrame
            :return: data frame with a 'filtered_singal', 'transformed_signal' and 'dt' columns
            :rtype: pandas.DataFrame
        '''
        signal_length = len(data_frame.filtered_signal.values)
        ll = int ( signal_length / 2 - self.window / 2 )
        rr = int ( signal_length / 2 + self.window / 2 )
        msa = data_frame.filtered_signal[ll:rr].values
        hann_window = signal.hann(self.window)

        msa_window = (msa * hann_window)
        transformed_signal = fft(msa_window)

        data = {'filtered_signal': msa_window, 'transformed_signal': transformed_signal,
                'dt': data_frame.td[ll:rr].values}

        data_frame_fft = pd.DataFrame(data, index=data_frame.index[ll:rr],
                                      columns=['filtered_signal', 'transformed_signal', 'dt'])
        logging.debug("fft signal")
        return data_frame_fft

    def calculate_amplitude(self, data_frame):
        '''
            This methods extract the fft components and sum the ones from lower to upper freq as per :cite:`Kassavetis2015`

            :param data_frame: the data frame
            :type data_frame: pandas.DataFrame
            :return amplitude: the amplitude
            :rtype amplitude: float
            :return frequency: the frequency
            :rtype frequency: float
        '''
        signal_length = len(data_frame.filtered_signal)
        normalised_transformed_signal = data_frame.transformed_signal.values / signal_length

        k = np.arange(signal_length)
        T = signal_length / self.sampling_frequency
        frq = k / T  # two sides frequency range

        frq = frq[range(int(signal_length / 2))]  # one side frequency range
        ts = normalised_transformed_signal[range(int(signal_length / 2))]
        amplitude = sum(abs(ts[(frq > self.lower_frequency) & (frq < self.upper_frequency)]))
        frequency = frq[abs(ts).argmax(axis=0)]

        logging.debug("tremor amplitude calculated")

        return amplitude, frequency

    def calculate_amplitude_by_welch(self, data_frame):
        '''
            This methods uses the Welch method :cite:`Welch1967` to obtain the power spectral density, this is a robust 
            alternative to using fft_signal & calculate_amplitude

            :param data_frame: the data frame
            :type data_frame: pandas.DataFrame
            :return: the amplitude
            :rtype amplitude: float
            :return: the frequency
            :rtype frequency: float
        '''
        frq, Pxx_den = signal.welch(data_frame.filtered_signal.values, self.sampling_frequency, nperseg=self.window)
        frequency = frq[Pxx_den.argmax(axis=0)]
        amplitude = sum(Pxx_den[(frq > self.lower_frequency) & (frq < self.upper_frequency)])

        logging.debug("tremor amplitude by welch calculated")

        return amplitude, frequency

    def approximate_entropy(self, x, m=None, r=None):
        """
        As in tsfresh `approximate_entropy <https://github.com/blue-yonder/tsfresh/blob/master/tsfresh/feature_extraction/feature_calculators.py#L1601>`_

        Implements a `vectorized approximate entropy algorithm <https://en.wikipedia.org/wiki/Approximate_entropy>`_ 
        For short time-series this method is highly dependent on the parameters,
        but should be stable for N > 2000, see :cite:`Yentes2013`. Other shortcomings and alternatives discussed in :cite:`Richman2000`
        
        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param m: Length of compared run of data
        :type m: int
        :param r: Filtering level, must be positive
        :type r: float
        :return: Approximate entropy
        :rtype: float
        """
        if m is None or r is None:
            m = 2.0
            r = 0.3
        entropy = feature_calculators.approximate_entropy(x, m, r)
        logging.debug("approximate entropy by tsfresh calculated")
        return entropy

    def autocorrelation(self, x, lag):
        """
        As in tsfresh `autocorrelation <https://github.com/blue-yonder/tsfresh/blob/master/tsfresh/feature_extraction/feature_calculators.py#L1457>`_

        Calculates the autocorrelation of the specified lag, according to the `formula <https://en.wikipedia.org/wiki/Autocorrelation#Estimation>`_:
        
        .. math::
            
            \\frac{1}{(n-l)\sigma^{2}} \\sum_{t=1}^{n-l}(X_{t}-\\mu )(X_{t+l}-\\mu)
        
        where :math:`n` is the length of the time series :math:`X_i`, :math:`\sigma^2` its variance and :math:`\mu` its
        mean. `l` denotes the lag.

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param lag: the lag
        :type lag: int
        :return: the value of this feature
        :rtype: float
        """
        # This is important: If a series is passed, the product below is calculated
        # based on the index, which corresponds to squaring the series.
        if lag is None:
            lag=0
        _autoc = feature_calculators.autocorrelation(x, lag)
        logging.debug("autocorrelation by tsfresh calculated")
        return _autoc

    def partial_autocorrelation(self, x, param):
        """
        As in tsfresh `partial_autocorrelation <https://github.com/blue-yonder/tsfresh/blob/master/tsfresh/feature_extraction/feature_calculators.py#L308>`_

        Calculates the value of the partial autocorrelation function at the given lag. The lag `k` partial autocorrelation
        of a time series :math:`\\lbrace x_t, t = 1 \\ldots T \\rbrace` equals the partial correlation of :math:`x_t` and
        :math:`x_{t-k}`, adjusted for the intermediate variables
        :math:`\\lbrace x_{t-1}, \\ldots, x_{t-k+1} \\rbrace` (:cite:`Wilson2015`).
        Following `this notes <https://onlinecourses.science.psu.edu/stat510/node/62>`_, it can be defined as
        
        
        .. math::
        
            \\alpha_k = \\frac{ Cov(x_t, x_{t-k} | x_{t-1}, \\ldots, x_{t-k+1})}
            {\\sqrt{ Var(x_t | x_{t-1}, \\ldots, x_{t-k+1}) Var(x_{t-k} | x_{t-1}, \\ldots, x_{t-k+1} )}}
        
        with (a) :math:`x_t = f(x_{t-1}, \\ldots, x_{t-k+1})` and (b) :math:`x_{t-k} = f(x_{t-1}, \\ldots, x_{t-k+1})`
        being AR(k-1) models that can be fitted by OLS. Be aware that in (a), the regression is done on past values to
        predict :math:`x_t` whereas in (b), future values are used to calculate the past value :math:`x_{t-k}`.
        It is said in :cite:`Wilson2015` that "for an AR(p), the partial autocorrelations [ :math:`\\alpha_k` ] will be nonzero for `k<=p`
        and zero for `k>p`."
        With this property, it is used to determine the lag of an AR-Process.
        
        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param param: contains dictionaries {"lag": val} with int val indicating the lag to be returned
        :type param: list
        :return: the value of this feature
        :rtype: float
        """
        if param is None:
            param = [{'lag': 3}, {'lag': 5}, {'lag': 6}]
        _partialc = feature_calculators.partial_autocorrelation(x, param)
        logging.debug("partial autocorrelation by tsfresh calculated")
        return _partialc

    def minimum(self, x):
        """
        Calculates the lowest value of the time series x.
        
        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :rtype: float
        """
        return np.min(x)

    def mean(self, x):
        """
            Returns the mean of x
            :param x: the time series to calculate the feature of
            :type x: pandas.Series
            :return: the value of this feature
            :rtype: float
            """
        logging.debug("mean calculated")

        return np.mean(x)

    def ratio_value_number_to_time_series_length(self, x):
        """
            As in tsfresh `ratio_value_number_to_time_series_length <https://github.com/blue-yonder/tsfresh/blob/master/tsfresh/feature_extraction/feature_calculators.py#L830>`_

            Returns a factor which is 1 if all values in the time series occur only once,
            and below one if this is not the case.
            In principle, it just returns: # unique values / # values
            
            :param x: the time series to calculate the feature of
            :type x: pandas.Series
            :return: the value of this feature
            :rtype: float
        """
        ratio = feature_calculators.ratio_value_number_to_time_series_length(x)
        logging.debug("ratio value number to time series length by tsfresh calculated")
        return list(ratio)

    def change_quantiles(self, x, ql=None, qh=None, isabs=None, f_agg=None):
        """
            As in tsfresh `change_quantiles <https://github.com/blue-yonder/tsfresh/blob/master/tsfresh/feature_extraction/feature_calculators.py#L1248>`_

            First fixes a corridor given by the quantiles ql and qh of the distribution of x.
            Then calculates the average, absolute value of consecutive changes of the series x inside this corridor.
            Think about selecting a corridor on the
            y-Axis and only calculating the mean of the absolute change of the time series inside this corridor.
            
            
            :param x: the time series to calculate the feature of
            :type x: pandas.Series
            :param ql: the lower quantile of the corridor
            :type ql: float
            :param qh: the higher quantile of the corridor
            :type qh: float
            :param isabs: should the absolute differences be taken?
            :type isabs: bool
            :param f_agg: the aggregator function that is applied to the differences in the bin
            :type f_agg: str, name of a numpy function (e.g. mean, var, std, median)
            :return: the value of this feature
            :rtype: float
        """
        if ql is None or qh is None or isabs is None or f_agg is None:
            f_agg = 'mean'
            isabs = True
            qh = 0.2
            ql = 0.0
        quantile = feature_calculators.change_quantiles(x, ql, qh, isabs, f_agg)
        logging.debug("change_quantiles by tsfresh calculated")
        return quantile

    def number_peaks(self, x, n = None):
        """
            As in tsfresh `number_peaks <https://github.com/blue-yonder/tsfresh/blob/master/tsfresh/feature_extraction/feature_calculators.py#L1003>`_
            
            Calculates the number of peaks of at least support n in the time series x. A peak of support n is defined as a
            subsequence of x where a value occurs, which is bigger than its n neighbours to the left and to the right.

            Hence in the sequence

            >>> x = [3, 0, 0, 4, 0, 0, 13]

            4 is a peak of support 1 and 2 because in the subsequences

            >>> [0, 4, 0]
            >>> [0, 0, 4, 0, 0]

            4 is still the highest value. Here, 4 is not a peak of support 3 because 13 is the 3th neighbour to the right of 4
            and its bigger than 4.

            :param x: the time series to calculate the feature of
            :type x: pandas.Series
            :param n: the support of the peak
            :type n: int
            :return: the value of this feature
            :rtype: float
            """
        if n is None:
            n = 5
        peaks = feature_calculators.number_peaks(x, n)
        logging.debug("agg linear trend by tsfresh calculated")
        return peaks

    def agg_linear_trend(self, x, param = None):
        """
            As in tsfresh `agg_inear_trend <https://github.com/blue-yonder/tsfresh/blob/master/tsfresh/feature_extraction/feature_calculators.py#L1727>`_
            
            Calculates a linear least-squares regression for values of the time series that were aggregated over chunks versus
            the sequence from 0 up to the number of chunks minus one.

            This feature assumes the signal to be uniformly sampled. It will not use the time stamps to fit the model.

            The parameters attr controls which of the characteristics are returned. Possible extracted attributes are "pvalue",
            "rvalue", "intercept", "slope", "stderr", see the documentation of linregress for more information.

            The chunksize is regulated by "chunk_len". It specifies how many time series values are in each chunk.

            Further, the aggregation function is controlled by "f_agg", which can use "max", "min" or , "mean", "median"

            :param x: the time series to calculate the feature of
            :type x: pandas.Series
            :param param: contains dictionaries {"attr": x, "chunk_len": l, "f_agg": f} with x, f an string and l an int
            :type param: list
            :return: the different feature values
            :rtype: pandas.Series
        """
        if param is None:
            param = [{'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'min'},{'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'var'},{'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'min'}]
        agg = feature_calculators.agg_linear_trend(x, param)
        logging.debug("agg linear trend by tsfresh calculated")
        return list(agg)

    def spkt_welch_density(self, x, param = None):
        '''
            As in tsfresh `spkt_welch_density <https://github.com/blue-yonder/tsfresh/blob/master/tsfresh/feature_extraction/feature_calculators.py#L1162>`_
            This feature calculator estimates the cross power spectral density of the time series x at different frequencies.
            To do so, the time series is first shifted from the time domain to the frequency domain.
            
            The feature calculators returns the power spectrum of the different frequencies.
            
            :param x: the time series to calculate the feature of
            :type x: pandas.Series
            :param param: contains dictionaries {"coeff": x} with x int
            :type param: list
            :return: the different feature values
            :rtype: pandas.Series
        '''
        if param is None:
            param = [{'coeff': 2}, {'coeff': 5}, {'coeff': 8}]
        welch = feature_calculators.spkt_welch_density(x, param)
        logging.debug("spkt welch density by tsfresh calculated")
        return list(welch)

    def percentage_of_reoccurring_datapoints_to_all_datapoints(self, x):
        """
        As in tsfresh `percentage_of_reoccurring_datapoints_to_all_datapoints <https://github.com/blue-yonder/tsfresh/blob/master/tsfresh/feature_extraction/feature_calculators.py#L739>`_

        Returns the percentage of unique values, that are present in the time series
        more than once.
        
        
        len(different values occurring more than once) / len(different values)
            
            
        This means the percentage is normalized to the number of unique values,
        in contrast to the percentage_of_reoccurring_values_to_all_values.
        
        
        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :rtype: float
        """
        _perc = feature_calculators.percentage_of_reoccurring_datapoints_to_all_datapoints(x)
        logging.debug("percentage of reoccurring datapoints to all datapoints by tsfresh calculated")
        return _perc

    def abs_energy(self, x):
        """
        As in tsfresh `abs_energy <https://github.com/blue-yonder/tsfresh/blob/master/tsfresh/feature_extraction/feature_calculators.py#L390>`_

        Returns the absolute energy of the time series which is the sum over the squared values
        
        
        .. math:: 
        
            E=\\sum_{i=1,\ldots, n}x_i^2
        
        
        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :rtype: float
        """
        _energy = feature_calculators.abs_energy(x)
        logging.debug("abs energy by tsfresh calculated")
        return _energy

    def fft_aggregated(self, x, param):
        """
        As in tsfresh `fft_aggregated <https://github.com/blue-yonder/tsfresh/blob/master/tsfresh/feature_extraction/feature_calculators.py#L896>`_

        Returns the spectral centroid (mean), variance, skew, and kurtosis of the absolute fourier transform spectrum.
        
        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param param: contains dictionaries {"aggtype": s} where s str and in ["centroid", "variance",
            "skew", "kurtosis"]
        :type param: list
        :return: the different feature values
        :rtype: pandas.Series
        """
        if param is None:
            param = [{'aggtype': 'centroid'}]
        _fft_agg = feature_calculators.fft_aggregated(x, param)
        logging.debug("fft aggregated by tsfresh calculated")
        return list(_fft_agg)

    def fft_coefficient(self, x, param):
        """
        As in tsfresh `fft_coefficient <https://github.com/blue-yonder/tsfresh/blob/master/tsfresh/feature_extraction/feature_calculators.py#L852>`_

        Calculates the fourier coefficients of the one-dimensional discrete Fourier Transform for real input by fast
        fourier transformation algorithm
        
        
        .. math:: 
        
            A_k =  \\sum_{m=0}^{n-1} a_m \\exp \\left \\{ -2 \\pi i \\frac{m k}{n} \\right \\}, \\qquad k = 0, \\ldots , n-1.


        The resulting coefficients will be complex, this feature calculator can return the real part (attr=="real"),
        the imaginary part (attr=="imag), the absolute value (attr=""abs) and the angle in degrees (attr=="angle).
        
        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param param: contains dictionaries {"coeff": x, "attr": s} with x int and x >= 0, s str and in ["real", "imag",
            "abs", "angle"]
        :type param: list
        :return: the different feature values
        :rtype: pandas.Series
        """
        if param is None:
            param = [{'attr': 'abs', 'coeff': 44},{'attr': 'abs', 'coeff': 63},{'attr': 'abs', 'coeff': 0},{'attr': 'real', 'coeff': 0},{'attr': 'real', 'coeff': 23}]
        _fft_coef = feature_calculators.fft_coefficient(x, param)
        logging.debug("fft coefficient by tsfresh calculated")
        return list(_fft_coef)

    def sum_values(self, x):
        """
        Calculates the sum over the time series values
        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :rtype: bool
        """
        if len(x) == 0:
            return 0

        return np.sum(x)

    def dc_remove_signal(self, data_frame):
        '''
            Removes the dc component of the signal as per :cite:`Kassavetis2015`

            :param data_frame: the data frame
            :type data_frame: pandas.DataFrame
            :return: the data frame with dc remove signal field
            :rtype: pandas.DataFrame
        '''

        mean_signal = np.mean(data_frame.mag_sum_acc)
        data_frame['dc_mag_sum_acc'] = data_frame.mag_sum_acc - mean_signal
        logging.debug("dc remove signal")
        return data_frame

    def bradykinesia(self, data_frame, method = 'fft'):
        '''
            This method calculates the bradykinesia amplitude of the data frame. It accepts two different methods,
            'fft' and 'welch'. First the signal gets re-sampled, dc removed and then high pass filtered.

            :param data_frame: the data frame
            :type data_frame: pandas.DataFrame
            :param method: fft or welch.
            :type method: str
            :return amplitude: the amplitude of the Bradykinesia
            :rtype amplitude: float
            :return frequency: the frequency of the Bradykinesia
            :rtype frequency: float

        '''
        try:
            data_frame_resampled = self.resample_signal(data_frame)
            data_frame_dc = self.dc_remove_signal(data_frame_resampled)
            data_frame_filtered = self.filter_signal(data_frame_dc, 'dc_mag_sum_acc')

            if method == 'fft':
                data_frame_fft = self.fft_signal(data_frame_filtered)
                return self.calculate_amplitude(data_frame_fft)
            else:
                return self.calculate_amplitude_by_welch(data_frame_filtered)
        except ValueError as verr:
            logging.error("TremorProcessor bradykinesia ValueError ->%s", verr.message)
        except:
            logging.error("Unexpected error on TemorProcessor bradykinesia: %s", sys.exc_info()[0])

    def process(self, data_frame, method='fft'):
        '''
            This method calculates the tremor amplitude of the data frame. It accepts two different methods,
            'fft' and 'welch'. First the signal gets re-sampled and then high pass filtered.

            :param data_frame: the data frame
            :type data_frame: pandas.DataFrame
            :param method: fft or welch.
            :type method: str
            :return amplitude: the amplitude of the Tremor
            :rtype amplitude: float
            :return frequency: the frequency of the Tremor
            :rtype frequency: float

        '''
        try:
            data_frame_resampled = self.resample_signal(data_frame)
            data_frame_filtered = self.filter_signal(data_frame_resampled)

            if method == 'fft':
                data_frame_fft = self.fft_signal(data_frame_filtered)
                return self.calculate_amplitude(data_frame_fft)
            else:
                return self.calculate_amplitude_by_welch(data_frame_filtered)

        except ValueError as verr:
            logging.error("TremorProcessor ValueError ->%s", verr.message)
        except:
            logging.error("Unexpected error on TremorProcessor process: %s", sys.exc_info()[0])
