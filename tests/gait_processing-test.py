# Copyright 2018 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author: Cosmin Stamate 

import unittest

import numpy as np
import pandas_validator as pv
# import pdkit

# import sys
# import os

# sys.path.append(os.path.abspath('../pdkit'))
from pdkit.gait_processor import GaitProcessor
from pdkit.gait_time_series import GaitTimeSeries

def round_to_two(lst):
    return list(np.around(np.array(lst), 2).astype(np.float32))

class CloudUPDRSDataFrameValidator(pv.DataFrameValidator):
    column_num = 5
    row_num = 5
    dt = pv.FloatColumnValidator('dt', min_value=-10, max_value=10)
    x = pv.FloatColumnValidator('x', min_value=-10, max_value=10)
    y = pv.FloatColumnValidator('y', min_value=-10, max_value=10)
    z = pv.FloatColumnValidator('z', min_value=-10, max_value=10)
    mag_sum_acc = pv.FloatColumnValidator('mag_sum_acc', min_value=-10, max_value=10)

class GaitProcessingTest(unittest.TestCase):
    def setUp(self):
        self.gp = GaitProcessor(duration=120, distance=90)
        self.filename_cloudupdrs = './tests/data/cloudupdrs_gait.csv'
        self.filename_mpower = './tests/data/mpower_gait.json'

        # Data to test freeze of gait

        self.freeze_times = [257,  307,  357,  407,  457,  507,  557,  607,  657,  707,  757,
                             807,  857,  907,  957, 1007, 1057, 1107, 1157, 1207, 1257, 1307,
                             1357, 1407, 1457, 1507, 1557, 1607, 1657, 1707, 1757, 1807, 1857,
                             1907, 1957, 2007, 2057, 2107, 2157, 2207, 2257, 2307]
        
        self.freeze_indexes = round_to_two([0.66275233, 0.48494184, 0.5249604 , 0.49955592, 0.43975925,
                                            0.38096038, 0.42747506, 0.35568315, 0.29876724, 0.3028857 ,
                                            0.29365826, 0.29416978, 0.2934438 , 0.31991965, 0.3044721 ,
                                            0.48086074, 0.6303387 , 0.64325756, 0.7543462 , 0.51056   ,
                                            0.82259506, 0.58685344, 0.60540056, 0.33504343, 0.39846754,
                                            0.32807222, 0.29868916, 0.30436054, 0.3181052 , 0.38775647,
                                            0.46554214, 0.4651025 , 0.5032366 , 0.48541462, 0.5435754 ,
                                            0.4435356 , 0.56342155, 0.33841345, 0.7144568 , 0.92833215,
                                            3.0611734 , 2.1126976 ])
        
        self.locomotion_freezes = round_to_two([8.9955121e-01, 2.0360003e+00, 2.8336084e+00, 4.1309290e+00,
                                                4.8003669e+00, 5.7900286e+00, 5.6657634e+00, 5.9355884e+00,
                                                5.9315796e+00, 6.2054639e+00, 6.1077957e+00, 6.1818423e+00,
                                                5.8416581e+00, 5.7142901e+00, 5.2229681e+00, 4.2915435e+00,
                                                3.5569930e+00, 2.8637218e+00, 2.1925688e+00, 2.1188903e+00,
                                                2.5527506e+00, 3.1371610e+00, 3.9612195e+00, 4.4408898e+00,
                                                4.4523473e+00, 4.4430523e+00, 4.3213620e+00, 4.2383804e+00,
                                                4.2852478e+00, 4.4170160e+00, 4.5098805e+00, 4.5359721e+00,
                                                4.7111130e+00, 3.8320572e+00, 3.7665429e+00, 2.7708292e+00,
                                                2.1944942e+00, 1.0180814e+00, 7.6819503e-01, 3.2823542e-01,
                                                3.1221139e-03, 5.2019866e-04])

        # Results for walk symmetry
        self.xyz_step_regularity = round_to_two([0.44895787870999526, 0.27681789027394327, 0.42835833214314395])
        self.xyz_stride_regularity = round_to_two([0.73, 0.639939599082686, 0.33527723619800587])
        self.xyz_walk_symmetry = round_to_two([-0.2758341299374629, -0.3631217088087428, 0.09308109594513808])
        
        # Results for direction
        self.direction = round_to_two([-0.99266317, 0.12314679,  0.06250603])

        # Results for heel strikes
        self.strikes =  [0.0,0.52,1.01,1.53,2.03,2.25,2.52,2.75,3.03,3.53,4.09,4.78,5.05,5.52,5.76,6.01,6.24,6.51,7.01,7.24,7.51,7.75,8.02,8.54,8.81]
        self.strike_indices =  [83,135,184,236,286,308,335,358,386,436,492,561,588,635,659,684,707,734,784,807,834,858,885,937,964]

        # Results for gait features
        self.number_of_steps = 22
        self.cadence = 0.18
        self.velocity = 0.75
        self.avg_step_length = 0.24
        self.avg_stride_length = 0.12
        self.step_durations = round_to_two([0.5, 0.51, 0.51, 0.21999999999999997, 0.2699999999999998, 0.23000000000000043, 0.2799999999999998, 0.5, 0.56, 0.96, 0.46999999999999975, 0.2400000000000002, 0.25, 0.23000000000000043, 0.2699999999999996, 0.25, 0.25, 0.23000000000000043, 0.2699999999999996, 0.2400000000000002, 0.2699999999999996])
        self.avg_step_duration = 0.36
        self.sd_step_durations = 0.18

        strides_1 = [0., 1.01, 1.74, 2.24, 3.02, 4.54, 5.25, 5.73, 6.25, 6.73, 7.24]
        strides_2 = [0.5, 1.52, 2.01, 2.52, 3.58, 5.01, 5.5,  6., 6.5, 7., 7.51]
        self.strides = [round_to_two(strides_1), round_to_two(strides_2)]

        stride_durations_1 = round_to_two([1.01, 0.73, 0.5000000000000002, 0.7799999999999998, 1.52, 0.71, 0.4800000000000004, 0.5199999999999996, 0.4800000000000004, 0.5099999999999998])
        stride_durations_2 = round_to_two([1.02, 0.48999999999999977, 0.5100000000000002, 1.06, 1.4299999999999997, 0.4900000000000002, 0.5, 0.5, 0.5, 0.5099999999999998])
        self.stride_durations = [round_to_two(stride_durations_1), round_to_two(stride_durations_2)]
        
        self.avg_number_of_strides = 11.0
        self.avg_stride_duration = 0.71
        self.sd_stride_durations = 0.32
        self.step_regularity = 0.39
        self.stride_regularity = 0.82
        self.symmetry = 0.43
        
    def tearDown(self):
        self.gp = None

    @staticmethod
    def load_data(filename, format_file):
        df = GaitTimeSeries()
        return df.load_data(filename, format_file)

    def test_cloudupdrs_data(self):
        self.wrong_data = './tests/data/kinetic_tremor_wrong_format.csv'
        df = self.load_data(self.wrong_data, 'cloudupdrs')
        validator = CloudUPDRSDataFrameValidator()
        
        self.assertEqual(False, validator.is_valid(df))
    
    def test_freeze_of_gait(self):
        df = self.load_data(self.filename_cloudupdrs, 'cloudupdrs')
        freeze_times, freeze_indexes, locomotion_freezes = self.gp.freeze_of_gait(df.x)
        
        freeze_times = list(freeze_times)
        freeze_indexes = round_to_two(freeze_indexes)
        locomotion_freezes = round_to_two(locomotion_freezes)

        self.assertCountEqual(freeze_times, self.freeze_times)
        self.assertCountEqual(freeze_indexes, self.freeze_indexes)
        self.assertCountEqual(locomotion_freezes, self.locomotion_freezes)
    
    def test_frequency_of_peaks(self):
        df = self.load_data(self.filename_cloudupdrs, 'cloudupdrs')
        frequency_of_peaks = self.gp.frequency_of_peaks(df.x)
        
        self.assertEqual(frequency_of_peaks, 192.72632494759722)
    
    def test_speed_of_gait(self):
        df = self.load_data(self.filename_cloudupdrs, 'cloudupdrs')
        gait_speed = self.gp.speed_of_gait(df.mag_sum_acc, wavelet_level=6)
        
        self.assertEqual(gait_speed, 1.4608160129012553)
    
    def test_walk_regularity_symmetry(self):
        df = self.load_data(self.filename_cloudupdrs, 'cloudupdrs')
        [step_regularity, stride_regularity, walk_symmetry] = self.gp.walk_regularity_symmetry(df)

        self.assertEqual(round_to_two(step_regularity), self.xyz_step_regularity)
        self.assertEqual(round_to_two(stride_regularity), self.xyz_stride_regularity)
        self.assertEqual(round_to_two(walk_symmetry), self.xyz_walk_symmetry)

    def test_walk_direction_preheel(self):
        df = self.load_data(self.filename_cloudupdrs, 'cloudupdrs')
        direction = self.gp.walk_direction_preheel(df)

        self.assertEqual(round_to_two(direction), self.direction)

    def test_heel_strikes(self):
        df = self.load_data(self.filename_cloudupdrs, 'cloudupdrs')
        strikes, strike_indices = self.gp.heel_strikes(df.x)

        strikes = list(strikes)
        strike_indices = list(strike_indices)

        # print('strikes = ', "".join([f'{s},' for s in strikes]))
        # print('strike_indices = ', "".join([f'{i},' for i in strike_indices]))

        # for i, s in enumerate(strikes):
        #     if self.strikes[i] != s:
        #         print(i, s, self.strikes[i])


        self.assertEqual(strikes, self.strikes)
        self.assertEqual(strike_indices, self.strike_indices)


    def test_gait_regularity_symmetry(self):
        df = self.load_data(self.filename_cloudupdrs, 'cloudupdrs')
        step_regularity, stride_regularity, symmetry = round_to_two(self.gp.gait_regularity_symmetry(df.x))

        self.assertEqual(step_regularity, np.float32(0.35))
        self.assertEqual(stride_regularity, np.float32(0.82))
        self.assertEqual(symmetry, np.float32(0.47))

    def test_gait(self):
        df = self.load_data(self.filename_cloudupdrs, 'cloudupdrs')
        the_gait = self.gp.gait(df.mag_sum_acc)

        print(the_gait[9][0])
        print(the_gait[9][1])

        self.assertEqual(self.number_of_steps, np.round(the_gait[0], 2))
        self.assertEqual(self.cadence, np.round(the_gait[1], 2))
        self.assertEqual(self.velocity, np.round(the_gait[2], 2))
        self.assertEqual(self.avg_step_length, np.round(the_gait[3], 2))
        self.assertEqual(self.avg_stride_length, np.round(the_gait[4], 2))
        self.assertEqual(self.step_durations, round_to_two(the_gait[5]))
        self.assertEqual(self.avg_step_duration, np.round(the_gait[6], 2))
        self.assertEqual(self.sd_step_durations, np.round(the_gait[7], 2))
        self.assertEqual(self.strides, [round_to_two(the_gait[8][0]), round_to_two(the_gait[8][1])])
        self.assertEqual(self.stride_durations, [round_to_two(the_gait[9][0]), round_to_two(the_gait[9][1])])
        self.assertEqual(self.avg_number_of_strides, np.round(the_gait[10], 2))
        self.assertEqual(self.avg_stride_duration, np.round(the_gait[11], 2))
        self.assertEqual(self.sd_stride_durations, np.round(the_gait[12], 2))
        self.assertEqual(self.step_regularity, np.round(the_gait[13], 2))
        self.assertEqual(self.stride_regularity, np.round(the_gait[14], 2))
        self.assertEqual(self.symmetry, np.round(the_gait[15], 2))
    

if __name__ == '__main__':
    pass
    unittest.main()