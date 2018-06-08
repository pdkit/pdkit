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
    return list(np.around(np.array(lst), 2))

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
        self.gp = GaitProcessor()
        self.filename_cloudupdrs = './tests/data/cloudupdrs_gait.csv'
        self.filename_mpower = './tests/data/mpower_gait.json'

        # Data to test freeze of gait
        self.freeze_indexes = round_to_two([2.022959744620578, 1.8162459981882777, 1.9943703659825391, 2.096233538564874, 1.9553966149187563, 3.1868342578699087, 2.948316901072047, 2.9327461208382224, 2.262528079399512, 2.6985282781187725, 2.107541421891744, 2.494609553495476, 2.122866676170536, 2.8001839396170576, 1.9906767580361675, 1.8571521952756058, 1.4185542488031768, 1.0104766238419907, 0.9048487004860442, 1.1023259851453922, 1.6972693614341943, 1.7625628061842689, 1.6416391833505406, 1.8550812039972588, 1.7884858929012508, 1.8577990398601534, 1.5667343489452676, 1.7624403198755907, 1.6724902078400952, 1.659990892288312, 1.5989449947502568, 2.191193205987883, 2.3212753593098805, 2.6171579167920562, 2.1832651986781695, 1.2896940154267278, 0.7944237870303018, 0.5745238180956264, 0.4859739599565407, 0.691029394669036, 0.5339579242940563, 0.1133032593454634])
        self.freeze_times = [int(i) for i in [257, 307.0, 357.0, 407.0, 457.0, 507.0, 557.0, 607.0, 657.0, 707.0, 757.0, 807.0, 857.0, 907.0, 957.0, 1007.0, 1057.0, 1107.0, 1157.0, 1207.0, 1257.0, 1307.0, 1357.0, 1407.0, 1457.0, 1507.0, 1557.0, 1607.0, 1657.0, 1707.0, 1757.0, 1807.0, 1857.0, 1907.0, 1957.0, 2007.0, 2057.0, 2107.0, 2157.0, 2207.0, 2257.0, 2307.0]]
        self.locomotion_freezes = round_to_two([1.4314870442454457, 1.7688607061770032, 2.3598814904715217, 2.9963933398955147, 3.1017761271497553, 2.5416038341739355, 2.577758162381949, 2.4412606522321436, 2.479067897574038, 2.7208879709765887, 2.6771147421165793, 2.7143505246206656, 2.7511343501294707, 2.32686845108453, 2.0152236967591524, 1.9319062318254518, 1.8461812833963096, 1.7591522147517984, 1.892095192649026, 1.8863765481984829, 2.1013498626586, 2.176976789819269, 2.573575818698249, 2.6000901002259464, 3.0028488966851006, 2.652355051071823, 2.723775057317436, 2.4067572093696867, 2.508245455614576, 2.190300920249192, 2.3259506621328243, 2.084716046227962, 1.9585484584504567, 1.5830157829571523, 1.300323810077697, 1.1945823768448263, 0.9564055468797126, 0.679827763525884, 0.5262856772915747, 0.16153435974694574, 0.041345519151655255, 0.012900217303049619])

        # Results for walk symmetry
        self.xyz_step_regularity = round_to_two([0.44895787870999526, 0.27681789027394327, 0.42835833214314395])
        self.xyz_stride_regularity = round_to_two([0.73, 0.639939599082686, 0.33527723619800587])
        self.xyz_walk_symmetry = round_to_two([0.2758341299374629, 0.3631217088087428, -0.09308109594513808])

        # Results for direction
        self.direction = round_to_two([-0.97766317, -0.16314679,  0.13250603])

        # Results for heel strikes
        self.strikes = [0., 0.49, 1., 1.51, 2.22, 2.73, 3.74, 4.06, 5.01, 5.72, 6.22, 7.21, 7.72, 8.24, 8.75]
        self.strike_indices = [103, 152, 203, 254, 325, 376, 477, 509, 604, 675, 725, 824, 875, 927, 978]

        # Results for gait features
        self.number_of_steps = 15
        self.cadence = 0.64
        self.velocity = 3.87
        self.avg_step_length = 0.17
        self.avg_stride_length = 0.08
        self.step_durations = round_to_two([0.49, 0.51, 0.51, 0.7100000000000002, 0.5099999999999998, 1.0100000000000002, 0.3199999999999994, 0.9500000000000002, 0.71, 0.5, 0.9900000000000002, 0.5099999999999998, 0.5200000000000005, 0.5099999999999998])
        self.avg_step_duration = 0.62
        self.sd_step_durations = 0.21
        self.strides = [round_to_two([0.  , 1.  , 2.22, 3.74, 5.01, 6.22, 7.72, 8.75]), round_to_two([0.49, 1.51, 2.73, 4.06, 5.72, 7.21, 8.24])]
        self.stride_durations = [round_to_two([1.0, 1.2200000000000002, 1.52, 1.2699999999999996, 1.21, 1.5, 1.0300000000000002]), round_to_two([1.02, 1.22, 1.3299999999999996, 1.6600000000000001, 1.4900000000000002, 1.0300000000000002])]
        self.avg_number_of_strides = 7.50
        self.avg_stride_duration = 1.27
        self.sd_stride_durations = 0.21
        self.step_regularity = 0.51
        self.stride_regularity = 0.82
        self.symmetry = 0.31
        
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
        [freeze_times, freeze_indexes, locomotion_freezes] = self.gp.freeze_of_gait(df)
        
        self.assertEqual(list(freeze_times), list(self.freeze_times))
        self.assertEqual(round_to_two(freeze_indexes), self.freeze_indexes)
        self.assertEqual(round_to_two(locomotion_freezes), self.locomotion_freezes)
    
    def test_frequency_of_peaks(self):
        df = self.load_data(self.filename_cloudupdrs, 'cloudupdrs')
        frequency_of_peaks = self.gp.frequency_of_peaks(df)
        
        self.assertEqual(frequency_of_peaks, -192.72632494759722)
    
    def test_speed_of_gait(self):
        df = self.load_data(self.filename_cloudupdrs, 'cloudupdrs')
        gait_speed = self.gp.speed_of_gait(df)
        
        self.assertEqual(gait_speed, 1.4426881267136054)
    
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

        self.assertEqual(round_to_two(strikes), self.strikes)
        self.assertEqual(strike_indices, self.strike_indices)


    def test_gait_regularity_symmetry(self):
        df = self.load_data(self.filename_cloudupdrs, 'cloudupdrs')
        step_regularity, stride_regularity, symmetry = round_to_two(self.gp.gait_regularity_symmetry(df.x))

        self.assertEqual(step_regularity, 0.97)
        self.assertEqual(stride_regularity, 0.99)
        self.assertEqual(symmetry, 0.02)

    def test_gait(self):
        df = self.load_data(self.filename_cloudupdrs, 'cloudupdrs')
        the_gait = self.gp.gait(df, axis='x')

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
    unittest.main()