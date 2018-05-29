#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2018 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author(s): J.S. Pons

import unittest
import pandas_validator as pv
import pdkit


class CloudUPDRSDataFrameValidator(pv.DataFrameValidator):
    column_num = 5
    row_num = 5
    dt = pv.FloatColumnValidator('dt', min_value=-10, max_value=10)
    x = pv.FloatColumnValidator('x', min_value=-10, max_value=10)
    y = pv.FloatColumnValidator('y', min_value=-10, max_value=10)
    z = pv.FloatColumnValidator('z', min_value=-10, max_value=10)
    mag_sum_acc = pv.FloatColumnValidator('mag_sum_acc', min_value=-10, max_value=10)


class TremorProcessingTest(unittest.TestCase):
    def setUp(self):
        self.tp = pdkit.TremorProcessor()
        self.filename_cloudupdrs = './tests/data/kinetic_tremor_left_hand.csv'
        self.filename_mpower = './tests/data/mpower_tremor.json'

    def tearDown(self):
        self.tp = None

    def test_cloudupdrs_data(self):
        self.wrong_data = './tests/data/kinetic_tremor_wrong_format.csv'
        ts = pdkit.TremorTimeSeries().load(self.wrong_data, 'cloudupdrs')
        validator = CloudUPDRSDataFrameValidator()
        # print('---> ', validator.is_valid(self.tp.data_frame))
        # print(self.tp.data_frame)
        self.assertEqual(False, validator.is_valid(ts))

    def test_tremor_amplitude_cloudupdrs(self):
        ts = pdkit.TremorTimeSeries().load(self.filename_cloudupdrs, 'cloudupdrs')
        amplitude, frequency = self.tp.amplitude(ts)
        self.assertEqual(float("{0:.14f}".format(amplitude)), float("{0:.14f}".format(2.390463750531757)))

    def test_tremor_freq_cloudupdrs(self):
        ts = pdkit.TremorTimeSeries().load(self.filename_cloudupdrs, 'cloudupdrs')
        amplitude, frequency = self.tp.amplitude(ts)
        self.assertEqual(float("{0:.5f}".format(frequency)), float("{0:.5f}".format(2.34375)))

    def test_tremor_amplitude_mpower(self):
        ts = pdkit.TremorTimeSeries().load(self.filename_mpower, 'mpower')
        amplitude, frequency = self.tp.amplitude(ts)
        self.assertEqual(float("{0:.14f}".format(amplitude)), float("{0:.14f}".format(0.4186992556201507)))

    def test_tremor_freq_mpower(self):
        ts = pdkit.TremorTimeSeries().load(self.filename_mpower, 'mpower')
        amplitude, frequency = self.tp.amplitude(ts)
        self.assertEqual(float("{0:.5f}".format(frequency)), float("{0:.5f}".format(7.421875)))

    def test_tremor_amplitude_welch_cloudupdrs(self):
        ts = pdkit.TremorTimeSeries().load(self.filename_cloudupdrs, 'cloudupdrs')
        amplitude, frequency = self.tp.amplitude(ts, 'welch')
        self.assertEqual(float("{0:.14f}".format(amplitude)), float("{0:.14f}".format(6.39553002855188)))

    def test_tremor_freq_welch_cloudupdrs(self):
        ts = pdkit.TremorTimeSeries().load(self.filename_cloudupdrs, 'cloudupdrs')
        amplitude, frequency = self.tp.amplitude(ts, 'welch')
        self.assertEqual(float("{0:.3f}".format(frequency)), float("{0:.3f}".format(3.125)))

    def test_tremor_amplitude_welch_mpower(self):
        ts = pdkit.TremorTimeSeries().load(self.filename_mpower, 'mpower')
        amplitude, frequency = self.tp.amplitude(ts, 'welch')
        self.assertEqual(float("{0:.14f}".format(amplitude)), float("{0:.14f}".format(0.16300804916508932)))

    def test_tremor_freq_welch_mpower(self):
        ts = pdkit.TremorTimeSeries().load(self.filename_mpower, 'mpower')
        amplitude, frequency = self.tp.amplitude(ts, 'welch')
        self.assertEqual(float("{0:.5f}".format(frequency)), float("{0:.5f}".format(5.859375)))


# if __name__ == '__main__':
#     unittest.main()

suite = unittest.TestLoader().loadTestsFromTestCase(TremorProcessingTest)
unittest.TextTestRunner(verbosity=2).run(suite)