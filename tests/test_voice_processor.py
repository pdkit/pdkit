#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2018 Birkbeck College. All rights reserved.
#
# Licensed under the MIT license. See file LICENSE for details.
#
# Author(s): Alex Noble

import unittest
import os
import pandas as pd
import numpy as np
from pdkit import ComprehensiveVoiceProcessor

class VoiceProcessorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), 'data', 'audio')
    
    def test_module_import(self):
            """Test that voice features module can be imported"""
            try:
                from pdkit import voice_features
                self.assertIsNotNone(voice_features)
            except ImportError:
                self.skipTest("Voice features module requires librosa")
        
    def test_voice_processor_exists(self):
        """Test that VoiceProcessor class exists"""
        try:
            from pdkit.voice_processor import ComprehensiveVoiceProcessor, VoiceProcessor
            self.assertTrue(hasattr(VoiceProcessor, 'extract_features'))
            self.assertTrue(hasattr(ComprehensiveVoiceProcessor, 'extract_voice_analysis_features'))
        except ImportError:
            self.skipTest("VoiceProcessor requires additional dependencies")

    def _test_single_audio_file(self, audio_file, expected_csv):
        """Helper method to test a single audio file against expected features"""
        vp = ComprehensiveVoiceProcessor(audio_file)
        measures, names, f0 = vp.extract_voice_analysis_features()
        
        self.assertGreater(len(measures), 100, f"Should extract 100+ features for {audio_file}")
        self.assertEqual(len(names), len(measures), "Feature names should match measures")
        self.assertIsInstance(f0, np.ndarray, "F0 should be numpy array")
        
        # Load expected results
        expected_df = pd.read_csv(expected_csv)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame([measures], columns=names)
        
        # Compare each feature
        mismatches = []
        for col in expected_df.columns:
            if col in results_df.columns:
                expected_val = expected_df[col].values[0]
                actual_val = results_df[col].values[0]
                
                # Skip NaN comparisons
                if pd.isna(expected_val) and pd.isna(actual_val):
                    continue
                
                # Check within 1% relative tolerance (or 1e-6 absolute for near-zero values)
                if not pd.isna(expected_val):
                    if abs(expected_val) < 1e-6:
                        # Use absolute tolerance for near-zero values
                        abs_diff = abs(actual_val - expected_val)
                        if abs_diff > 1e-6:
                            mismatches.append(
                                f"{col}: expected {expected_val:.6e}, got {actual_val:.6e} "
                                f"(abs diff: {abs_diff:.6e})"
                            )
                    else:
                        # Use relative tolerance for normal values
                        rel_diff = abs(actual_val - expected_val) / abs(expected_val)
                        if rel_diff > 0.01:
                            mismatches.append(
                                f"{col}: expected {expected_val:.6f}, got {actual_val:.6f} "
                                f"(rel diff: {rel_diff*100:.2f}%)"
                            )
        
        if mismatches:
            self.fail(
                f"Feature mismatches in {os.path.basename(audio_file)}:\n" + 
                "\n".join(mismatches[:10])  # Show first 10
            )
        
        return len(measures)
    
    @unittest.skipUnless(
        os.path.exists(os.path.join(os.path.dirname(__file__), 'data', 'audio', 'hc', 'AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5.wav')),
        "HC sample 1 not available"
    )
    def test_hc_sample_1(self):
        """Test feature extraction on healthy control sample 1"""
        audio_file = os.path.join(self.test_data_dir, 'hc', 'AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5.wav')
        expected_csv = os.path.join(self.test_data_dir, 'results', 'hc', 'AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5.csv')
        n_features = self._test_single_audio_file(audio_file, expected_csv)
        print(f"HC sample 1: {n_features} features validated")
    
    @unittest.skipUnless(
        os.path.exists(os.path.join(os.path.dirname(__file__), 'data', 'audio', 'hc', 'AH_114S_A89F3548-0B61-4770-B800-2E26AB3908B6.wav')),
        "HC sample 2 not available"
    )
    def test_hc_sample_2(self):
        """Test feature extraction on healthy control sample 2"""
        audio_file = os.path.join(self.test_data_dir, 'hc', 'AH_114S_A89F3548-0B61-4770-B800-2E26AB3908B6.wav')
        expected_csv = os.path.join(self.test_data_dir, 'results', 'hc', 'AH_114S_A89F3548-0B61-4770-B800-2E26AB3908B6.csv')
        n_features = self._test_single_audio_file(audio_file, expected_csv)
        print(f"HC sample 2: {n_features} features validated")

    @unittest.skipUnless(
        os.path.exists(os.path.join(os.path.dirname(__file__), 'data', 'audio', 'pd', 'AH_545616858-3A749CBC-3FEB-4D35-820E-E45C3E5B9B6A.wav')),
        "PD sample 1 not available"
    )
    def test_pd_sample_1(self):
        """Test feature extraction on Parkinson's disease sample 1"""
        audio_file = os.path.join(self.test_data_dir, 'pd', 'AH_545616858-3A749CBC-3FEB-4D35-820E-E45C3E5B9B6A.wav')
        expected_csv = os.path.join(self.test_data_dir, 'results', 'pd', 'AH_545616858-3A749CBC-3FEB-4D35-820E-E45C3E5B9B6A.csv')
        n_features = self._test_single_audio_file(audio_file, expected_csv)
        print(f"PD sample 1: {n_features} features validated")

    @unittest.skipUnless(
        os.path.exists(os.path.join(os.path.dirname(__file__), 'data', 'audio', 'pd', 'AH_545622717-461DFFFE-54AF-42AF-BA78-528BD505D624.wav')),
        "PD sample 2 not available"
    )
    def test_pd_sample_2(self):
        """Test feature extraction on Parkinson's disease sample 2"""
        audio_file = os.path.join(self.test_data_dir, 'pd', 'AH_545622717-461DFFFE-54AF-42AF-BA78-528BD505D624.wav')
        expected_csv = os.path.join(self.test_data_dir, 'results', 'pd', 'AH_545622717-461DFFFE-54AF-42AF-BA78-528BD505D624.csv')
        n_features = self._test_single_audio_file(audio_file, expected_csv)
        print(f"PD sample 2: {n_features} features validated")

    def test_all_samples_consistency(self):
        """Test that all samples extract the same number of features"""
        test_files = [
            ('hc', 'AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5.wav', 'AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5.csv'),
            ('hc', 'AH_114S_A89F3548-0B61-4770-B800-2E26AB3908B6.wav', 'AH_114S_A89F3548-0B61-4770-B800-2E26AB3908B6.csv'),
            ('pd', 'AH_545616858-3A749CBC-3FEB-4D35-820E-E45C3E5B9B6A.wav', 'AH_545616858-3A749CBC-3FEB-4D35-820E-E45C3E5B9B6A.csv'),
            ('pd', 'AH_545622717-461DFFFE-54AF-42AF-BA78-528BD505D624.wav', 'AH_545622717-461DFFFE-54AF-42AF-BA78-528BD505D624.csv'),
        ]
        
        feature_counts = []
        for type, audio_name, csv_name in test_files:
            audio_path = os.path.join(self.test_data_dir, type, audio_name)
            csv_path = os.path.join(self.test_data_dir, 'results', type, csv_name)
            
            if os.path.exists(audio_path) and os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                feature_counts.append((audio_name, len(df.columns)))
        
        if len(feature_counts) > 1:
            # All samples should have same number of features
            first_count = feature_counts[0][1]
            for name, count in feature_counts:
                self.assertEqual(
                    count, first_count,
                    f"{name} has {count} features, expected {first_count}"
                )
            print(f"All {len(feature_counts)} samples have consistent {first_count} features")


if __name__ == '__main__':
    unittest.main()