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
    td = pv.FloatColumnValidator('td', min_value=-1, max_value=1000)
    x = pv.FloatColumnValidator('x', min_value=-100, max_value=100)
    y = pv.FloatColumnValidator('y', min_value=-100, max_value=100)
    z = pv.FloatColumnValidator('z', min_value=-100, max_value=100)
    mag_sum_acc = pv.FloatColumnValidator('mag_sum_acc', min_value=-100, max_value=100)


class BradykinesiaProcessingTest(unittest.TestCase):
    def setUp(self):
        self.tp = pdkit.TremorProcessor(lower_frequency=0.0, upper_frequency=4.0)
        self.filename_cloudupdrs = './tests/data/pronation_supination_left_hand.csv'

    def tearDown(self):
        self.tp = None

    def test_cloudupdrs_data(self):
        self.wrong_data = './tests/data/pronation_supination_wrong_format.csv'
        ts = pdkit.TremorTimeSeries().load(self.wrong_data, 'cloudupdrs')
        validator = CloudUPDRSDataFrameValidator()
        self.assertEqual(False, validator.is_valid(ts))

    def test_bradykinesia_amplitude_cloudupdrs(self):
        ts = pdkit.TremorTimeSeries().load(self.filename_cloudupdrs, 'cloudupdrs')
        amplitude, frequency = self.tp.bradykinesia(ts)
        self.assertEqual(float("{0:.14f}".format(amplitude)), float("{0:.14f}".format(1.525984985866312)))

    def test_tremor_freq_cloudupdrs(self):
        ts = pdkit.TremorTimeSeries().load(self.filename_cloudupdrs, 'cloudupdrs')
        amplitude, frequency = self.tp.bradykinesia(ts)
        self.assertEqual(float("{0:.5f}".format(frequency)), float("{0:.5f}".format(1.953125)))

    def test_tremor_amplitude_welch_cloudupdrs(self):
        ts = pdkit.TremorTimeSeries().load(self.filename_cloudupdrs, 'cloudupdrs')
        amplitude, frequency = self.tp.bradykinesia(ts, 'welch')
        self.assertEqual(float("{0:.14f}".format(amplitude)), float("{0:.14f}".format(5.0055189177887085)))

    def test_tremor_freq_welch_cloudupdrs(self):
        ts = pdkit.TremorTimeSeries().load(self.filename_cloudupdrs, 'cloudupdrs')
        amplitude, frequency = self.tp.bradykinesia(ts, 'welch')
        self.assertEqual(float("{0:.5f}".format(frequency)), float("{0:.5f}".format(2.34375)))


suite = unittest.TestLoader().loadTestsFromTestCase(BradykinesiaProcessingTest)
unittest.TextTestRunner(verbosity=2).run(suite)