"""
Comprehensive Verification and Test Suite
Predictive Pavement Management System (Granica x IIT Guwahati Hackathon)
"""

import os
import unittest
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import cv2

from pipeline import (
    RoadVisionEngine,
    AccelerometerSignalEngine,
    PredictivePavementAnalytics,
    GranicaParquetExporter,
    GRANICA_PARQUET_SCHEMA,
    generate_granica_hackathon_dataset
)
from export_pdf import generate_work_order_pdf

class TestPredictivePavementSystem(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.vision_engine = RoadVisionEngine()
        cls.signal_engine = AccelerometerSignalEngine(sampling_rate_hz=100.0)
        
    def test_01_schema_compliance(self):
        """Test strict 12-field PyArrow schema compliance."""
        expected_fields = [
            "observation_id", "timestamp", "data_source", "route_zone",
            "gps_lat_long", "pothole_area_sqm", "z_accel_peak_g",
            "traffic_volume_index", "road_damage_index", "days_since_paved",
            "predictive_degradation_score", "synthetic_flag"
        ]
        schema_names = [f.name for f in GRANICA_PARQUET_SCHEMA]
        self.assertEqual(schema_names, expected_fields)
        self.assertEqual(len(schema_names), 12)

    def test_02_parquet_dataset_generation_and_metadata(self):
        """Verify Granica Parquet export, embedded metadata, and 75/20/5 split."""
        test_parquet = "test_output.parquet"
        df, export_res = generate_granica_hackathon_dataset(100, test_parquet)
        
        self.assertTrue(os.path.exists(test_parquet))
        self.assertEqual(len(df), 100)
        
        # Read back table and verify metadata
        summary = GranicaParquetExporter.read_dataset_summary(test_parquet)
        meta = summary["metadata"]
        
        self.assertIn("observation_source", meta)
        self.assertIn("collection_window", meta)
        self.assertIn("data_split", meta)
        self.assertEqual(meta["data_split"], "75% Observed, 20% Inferred, 5% Synthetic")
        self.assertEqual(meta["granica_compliance"], "TRUE")
        
        # Check synthetic count
        synthetic_count = df["synthetic_flag"].sum()
        self.assertEqual(synthetic_count, 5) # 5% of 100
        
        if os.path.exists(test_parquet):
            os.remove(test_parquet)

    def test_03_rdi_and_pdi_calculation(self):
        """Verify RDI and PDI equations under controlled conditions."""
        # Case 1: Moderate pothole, high peak shock, moderate traffic
        # RDI = 0.5 * (area/4.0) + 0.3 * ((z - 1.0)/3.5) + 0.2 * (traffic/200)
        area = 2.0 # norm = 0.5
        z_peak = 2.75 # norm = 1.75/3.5 = 0.5
        traffic = 100.0 # norm = 0.5
        # Expected: 0.5*(0.5) + 0.3*(0.5) + 0.2*(0.5) = 0.25 + 0.15 + 0.10 = 0.50
        rdi = PredictivePavementAnalytics.calculate_rdi(area, z_peak, traffic)
        self.assertAlmostEqual(rdi, 0.50, places=2)
        
        # PDI = RDI * (1 + days / 365)
        # days = 365 -> factor = 2.0 -> PDI = 1.00
        pdi = PredictivePavementAnalytics.calculate_pdi(rdi, 365)
        self.assertAlmostEqual(pdi, 1.00, places=2)

    def test_04_scipy_signal_peak_detection(self):
        """Verify Butterworth filtering and SciPy find_peaks for vertical shocks > 1.8g."""
        df_accel = self.signal_engine.generate_synthetic_mobile_run(
            duration_sec=30, pothole_spikes=3
        )
        res = self.signal_engine.detect_impact_peaks(df_accel, threshold_g=1.8)
        
        self.assertGreaterEqual(res["peak_count"], 2)
        self.assertGreater(res["max_peak_g"], 1.8)
        self.assertEqual(len(res["filtered_z"]), len(df_accel))

    def test_05_road_spam_verification(self):
        """Verify AI spam filter accurately blocks non-road images and passes road images."""
        spam_img_path = os.path.join("sample_data", "sample_spam_indoor.jpg")
        road_img_path = os.path.join("sample_data", "sample_road_pothole.jpg")
        
        if os.path.exists(spam_img_path):
            img_spam = cv2.imread(spam_img_path)
            is_road, reason, _ = self.vision_engine.verify_road_surface(img_spam)
            self.assertFalse(is_road, f"Spam image should be rejected: {reason}")
            
        if os.path.exists(road_img_path):
            img_road = cv2.imread(road_img_path)
            is_road, reason, _ = self.vision_engine.verify_road_surface(img_road)
            self.assertTrue(is_road, f"Real road image should be accepted: {reason}")

    def test_06_spatial_dbscan_clustering(self):
        """Verify 100m spatial clustering and Full Resurface vs Spot Repair decision engine."""
        # Create 4 defects tightly grouped within 50 meters
        coords = [
            (26.19120, 91.69250),
            (26.19125, 91.69255),
            (26.19130, 91.69260),
            (26.19135, 91.69265),
            # And 1 isolated defect 1.5 km away
            (26.18000, 91.68000)
        ]
        records = []
        for i, (lat, lon) in enumerate(coords):
            records.append({
                "observation_id": f"TEST_OBS_{i}",
                "timestamp": pd.Timestamp.now(tz="UTC"),
                "data_source": "CCTV",
                "route_zone": "Core_V_North_Loop" if i < 4 else "Sports_Complex_Corridor",
                "gps_lat_long": f"{lat:.5f}, {lon:.5f}",
                "latitude": lat,
                "longitude": lon,
                "pothole_area_sqm": 1.2,
                "z_accel_peak_g": 2.2,
                "traffic_volume_index": 80.0,
                "road_damage_index": 0.55,
                "days_since_paved": 400,
                "predictive_degradation_score": 1.15 if i < 4 else 0.45,
                "synthetic_flag": False
            })
        df_test = pd.DataFrame(records)
        df_clustered, work_orders = PredictivePavementAnalytics.spatial_dbscan_clustering(
            df_test, eps_meters=100.0, density_threshold_count=3, density_threshold_pdi=0.85
        )
        
        # We expect 2 clusters (the 4 grouped items and the 1 isolated item)
        self.assertEqual(len(work_orders), 2)
        
        # Dense cluster should trigger "Full Segment Resurfacing"
        dense_wo = next(wo for wo in work_orders if wo["defect_count"] == 4)
        self.assertEqual(dense_wo["action"], "Full Segment Resurfacing")
        self.assertGreater(dense_wo["projected_cost_savings_pct"], 35.0)

    def test_07_pdf_work_order_generation(self):
        """Verify PDF Work Order generation."""
        sample_wo = {
            "work_order_id": "WO-IITG-EST-2026-TEST",
            "route_zone": "Core_V_North_Loop",
            "center_gps": "26.19120, 91.69250",
            "action": "Full Segment Resurfacing",
            "defect_count": 4,
            "mean_rdi": 0.65,
            "mean_pdi": 1.35,
            "total_area_sqm": 7.5,
            "spot_repair_cost_inr": 30000,
            "full_resurface_cost_inr": 34000,
            "projected_cost_savings_pct": 54.7
        }
        test_pdf = "test_work_order.pdf"
        out_file = generate_work_order_pdf(sample_wo, [], test_pdf)
        self.assertTrue(os.path.exists(out_file))
        self.assertGreater(os.path.getsize(out_file), 1000) # Valid non-empty PDF
        if os.path.exists(out_file):
            os.remove(out_file)

if __name__ == "__main__":
    unittest.main()
