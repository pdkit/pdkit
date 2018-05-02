#!/usr/bin/env python3
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


class BradykinesiaProcessingTest(unittest.TestCase):
    def setUp(self):
        self.bp = pdkit.BradykinesiaProcessor()
        self.filename_cloudupdrs = './tests/data/pronation_supination_left_hand.csv'

    def tearDown(self):
        self.bp = None

    def test_cloudupdrs_data(self):
        self.wrong_data = './tests/data/pronation_supination_wrong_format.csv'
        ts = pdkit.BradykinesiaTimeSeries().load(self.wrong_data, 'cloudupdrs')
        validator = CloudUPDRSDataFrameValidator()
        # print('---> ', validator.is_valid(self.tp.data_frame))
        # print(self.tp.data_frame)
        self.assertEqual(False, validator.is_valid(ts))

    def test_bradykinesia_amplitude_cloudupdrs(self):
        ts = pdkit.BradykinesiaTimeSeries().load(self.filename_cloudupdrs, 'cloudupdrs')
        amplitude, frequency = self.bp.process(ts)
        self.assertEqual(float("{0:.14f}".format(amplitude)), float("{0:.14f}".format(1.525984985866312)))

    def test_tremor_freq_cloudupdrs(self):
        ts = pdkit.BradykinesiaTimeSeries().load(self.filename_cloudupdrs, 'cloudupdrs')
        amplitude, frequency = self.bp.process(ts)
        self.assertEqual(float("{0:.6f}".format(frequency)), float("{0:.6f}".format(1.953125)))

    # def test_tremor_amplitude_welch_cloudupdrs(self):
    #     ts = pdkit.TremorTimeSeries().load(self.filename_cloudupdrs, 'cloudupdrs')
    #     amplitude, frequency = self.tp.process(ts, 'welch')
    #     self.assertEqual(float("{0:.14f}".format(amplitude)), float("{0:.14f}".format(6.39553002855188)))
    #
    # def test_tremor_freq_welch_cloudupdrs(self):
    #     ts = pdkit.TremorTimeSeries().load(self.filename_cloudupdrs, 'cloudupdrs')
    #     amplitude, frequency = self.tp.process(ts, 'welch')
    #     self.assertEqual(float("{0:.3f}".format(frequency)), float("{0:.3f}".format(3.125)))


# if __name__ == '__main__':
#     unittest.main()

suite = unittest.TestLoader().loadTestsFromTestCase(BradykinesiaProcessingTest)
unittest.TextTestRunner(verbosity=2).run(suite)