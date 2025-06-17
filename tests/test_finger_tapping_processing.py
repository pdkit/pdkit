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


class FTCloudUPDRSDataFrameValidator(pv.DataFrameValidator):
    column_num = 6
    row_num = 6
    td = pv.FloatColumnValidator('td', min_value=-1, max_value=1000)
    action_type = pv.FloatColumnValidator('action_type', min_value=0, max_value=1)
    x = pv.FloatColumnValidator('x', min_value=-1000, max_value=1000)
    y = pv.FloatColumnValidator('y', min_value=-1000, max_value=1000)
    x_target = pv.FloatColumnValidator('x_target', min_value=-1000, max_value=1000)
    y_target = pv.FloatColumnValidator('y_target', min_value=-1000, max_value=1000)


class FingerTappingProcessingTest(unittest.TestCase):
    def setUp(self):
        self.ftp = pdkit.FingerTappingProcessor()
        self.filename_cloudupdrs = './tests/data/finger_tapping_two_target_right_hand.csv'

    def tearDown(self):
        self.ftp = None

    def test_finger_tapping_cloudupdrs_data(self):
        self.wrong_data = './tests/data/finger_tapping_wrong_format.csv'
        ts = pdkit.FingerTappingTimeSeries().load(self.wrong_data, 'ft_cloudupdrs')
        validator = FTCloudUPDRSDataFrameValidator()
        self.assertEqual(False, validator.is_valid(ts))

    def test_finger_tapping_frequency(self):
        ts = pdkit.FingerTappingTimeSeries().load(self.filename_cloudupdrs, 'ft_cloudupdrs')
        freq, duration = self.ftp.frequency(ts)
        self.assertEqual(float("{0:.14f}".format(freq)), float("{0:.14f}".format(3.4455385342949314)))

    def test_finger_tapping_mean_moving_time(self):
        ts = pdkit.FingerTappingTimeSeries().load(self.filename_cloudupdrs, 'ft_cloudupdrs')
        mmt, duration = self.ftp.mean_moving_time(ts)
        self.assertEqual(float("{0:.14f}".format(mmt)), float("{0:.14f}".format(136.75369458128074)))

    def test_finger_tapping_mean_alnt_target_distance(self):
        ts = pdkit.FingerTappingTimeSeries().load(self.filename_cloudupdrs, 'ft_cloudupdrs')
        matd, duration = self.ftp.mean_alnt_target_distance(ts)
        self.assertEqual(float("{0:.14f}".format(matd)), float("{0:.14f}".format(480.04403524710443)))

    def test_finger_tapping_incoordination_score(self):
        ts = pdkit.FingerTappingTimeSeries().load(self.filename_cloudupdrs, 'ft_cloudupdrs')
        inc_s, duration = self.ftp.incoordination_score(ts)
        self.assertEqual(float("{0:.14f}".format(inc_s)), float("{0:.14f}".format(0.7649147030988328)))

    def test_finger_tapping_kinesia_scores(self):
        ts = pdkit.FingerTappingTimeSeries().load(self.filename_cloudupdrs, 'ft_cloudupdrs')
        ks, duration = self.ftp.kinesia_scores(ts)
        self.assertEqual( ks , 204)

    def test_finger_tapping_akinesia_times(self):
        ts = pdkit.FingerTappingTimeSeries().load(self.filename_cloudupdrs, 'ft_cloudupdrs')
        at, duration = self.ftp.akinesia_times(ts)
        self.assertEqual(float("{0:.14f}".format(at)), float("{0:.14f}".format(0.15414705882352947)))

    def test_finger_tapping_dysmetria_score(self):
        ts = pdkit.FingerTappingTimeSeries().load(self.filename_cloudupdrs, 'ft_cloudupdrs')
        ds, duration = self.ftp.dysmetria_score(ts)
        self.assertEqual(float("{0:.14f}".format(ds)), float("{0:.14f}".format(58.2339485875795)))


suite = unittest.TestLoader().loadTestsFromTestCase(FingerTappingProcessingTest)
unittest.TextTestRunner(verbosity=2).run(suite)