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


class TestResultSetCloudUPDRSDataFrameValidator(pv.DataFrameValidator):
    column_num = 437
    row_num = 2
    # td = pv.FloatColumnValidator('td', min_value=-1, max_value=1000)
    # action_type = pv.FloatColumnValidator('action_type', min_value=0, max_value=1)
    # x = pv.FloatColumnValidator('x', min_value=-1000, max_value=1000)
    # y = pv.FloatColumnValidator('y', min_value=-1000, max_value=1000)
    # x_target = pv.FloatColumnValidator('x_target', min_value=-1000, max_value=1000)
    # y_target = pv.FloatColumnValidator('y_target', min_value=-1000, max_value=1000)


class TestResultSetTest(unittest.TestCase):
    def setUp(self):
        self.trs = pdkit.TestResultSet(self.folderpath)
        self.folderpath = './tests/data'

    def tearDown(self):
        self.trs = None

    def test_result_set_test_cloudupdrs_data(self):
        df = self.trs.process()
        validator = TestResultSetCloudUPDRSDataFrameValidator()
        self.assertEqual(True, validator.is_valid(df))


suite = unittest.TestLoader().loadTestsFromTestCase(TestResultSetTest)
unittest.TextTestRunner(verbosity=2).run(suite)