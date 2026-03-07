
import unittest
import numpy as np
import os
import sys
import joblib
from unittest.mock import MagicMock

# Adjust path to find modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # Correctly points to d:\renhangxi_tiktok_bysj
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock Django settings BEFORE importing predict_service
sys.modules['django.conf'] = MagicMock()
sys.modules['django.conf'].settings = MagicMock()
sys.modules['django.conf'].settings.configure = MagicMock()

from ml_pipeline.theme_baseline_engine import ThemeBaselineCalculator
from services.predict_service import DiggPredictionService

class TestTopicAwarePrediction(unittest.TestCase):
    
    def setUp(self):
        # Setup mock global stats
        self.global_stats = {
            'mean': 10000,
            'std': 20000,
            'p25': 2000,
            'p50': 5000, 
            'p75': 15000,
            'max': 100000,
            'sample_count': 10000
        }
        self.calculator = ThemeBaselineCalculator()
        
    def test_bayesian_smoothing_zero_samples(self):
        """Test Case A: N=0 (Should return Global)"""
        local_data = [] # Empty
        result = self.calculator.calculate(local_data, global_stats=self.global_stats, smoothing_factor=10)
        
        print(f"\n[Test A] Zero Samples -> Result: {result['mean']}")
        self.assertEqual(result['mean'], self.global_stats['mean'])
        self.assertEqual(result['p50'], self.global_stats['p50'])
        
    def test_bayesian_smoothing_large_samples(self):
        """Test Case B: N=1000 (Should allow Local to dominate)"""
        # Create 1000 samples with mean ~200 (far from global 10000)
        local_data = [{'digg_count': 200} for _ in range(1000)]
        
        result = self.calculator.calculate(local_data, global_stats=self.global_stats, smoothing_factor=10)
        
        # Formula: (1000*200 + 10*10000) / 1010 = (200000+100000)/1010 = 297
        expected_mean = (1000 * 200 + 10 * 10000) / 1010
        print(f"\n[Test B] Large Samples (N=1000, Local=200, Global=10000) -> Result: {result['mean']:.2f} (Expected: {expected_mean:.2f})")
        
        self.assertAlmostEqual(result['mean'], expected_mean, delta=1.0)
        # Should be much closer to 200 than 10000
        self.assertTrue(result['mean'] < 500)
        
    def test_bayesian_smoothing_small_samples(self):
        """Test Case C: N=Small (Should be weighted average)"""
        # Create 5 samples with mean 200
        local_data = [{'digg_count': 200} for _ in range(5)]
        
        result = self.calculator.calculate(local_data, global_stats=self.global_stats, smoothing_factor=10)
        
        # Formula: (5*200 + 10*10000) / 15 = (1000 + 100000) / 15 = 101000/15 = 6733
        expected_mean = (5 * 200 + 10 * 10000) / 15
        print(f"\n[Test C] Small Samples (N=5, Local=200, Global=10000) -> Result: {result['mean']:.2f} (Expected: {expected_mean:.2f})")
        
        self.assertAlmostEqual(result['mean'], expected_mean, delta=1.0)
        # Should be pulled significantly towards global
        self.assertTrue(result['mean'] > 6000)

    def test_prediction_service_integration(self):
        """Test that the prediction service accepts theme info without crashing"""
        service = DiggPredictionService()
        
        # Mock data with a new theme name
        video_data = {
            'follower_count': 5000,
            'publish_hour': 18,
            'duration_sec': 15,
            'theme_label': 'TestNewTheme'
        }
        
        # Run prediction
        try:
            result = service.predict_digg_count(video_data, theme_baseline=self.global_stats)
            print(f"\n[Test D] Prediction Service Result: {result}")
            self.assertTrue('predicted_digg' in result)
            self.assertTrue('quality_score' in result)
        except Exception as e:
            self.fail(f"Prediction service raised exception: {e}")

if __name__ == '__main__':
    unittest.main()
