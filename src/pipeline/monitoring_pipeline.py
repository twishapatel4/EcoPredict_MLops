import pandas as pd
import os
import sys

# 1. USE THE LEGACY REPORT CLASS (This matches the presets)
from evidently.legacy.report.report import Report

# 2. USE THE LEGACY PRESETS
from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset

from src.exception import CustomException
from src.logger import logging

class ModelMonitoring:
    def __init__(self):
        # Ensure path points to the correct location in the root
        self.reference_data_path = os.path.join("data", "raw_data.csv")
        self.report_path = os.path.join("artifacts", "evidently_report.html")

    def initiate_monitoring(self, current_df):
        try:
            logging.info("Starting Evidently AI Monitoring Report generation")
            
            # Load Reference Data
            if not os.path.exists(self.reference_data_path):
                raise Exception(f"Reference data not found at {self.reference_data_path}")
                
            ref_df = pd.read_csv(self.reference_data_path)
            
            # Select columns
            relevant_cols = ['year', 'energy_per_capita', 'gdp_per_capita', 'co2_per_capita']
            ref_df = ref_df[relevant_cols]
            
            # 3. Define the Report (This now uses the Legacy Engine)
            report = Report(metrics=[
                # DataDriftPreset(num_stattest_threshold=0.14),
                DataDriftPreset(),
                DataQualityPreset(),
                TargetDriftPreset()
            ])

            logging.info("Running drift detection...")
            report.run(reference_data=ref_df, current_data=current_df)

            # Save the result
            report.save_html(self.report_path)
            return self.report_path

        except Exception as e:
            raise CustomException(e, sys)

# ... rest of your script stays the same ...

if __name__ == "__main__":
    # Test logic: We compare the training data against a "modified" version of itself 
    # to simulate real-world drift.
    monitor = ModelMonitoring()
    df = pd.read_csv("data/raw_data.csv")
    # train_data = pd.read_csv("data/raw_data.csv")
    train_df = df[df['year'] < 2019].copy()
    test_df = df[df['year'] >= 2019].copy()
    
    # Simulate "Current" data by adding some noise to GDP
    # current_simulation = test_df.sample(200,random_state=42).copy()
    current_simulation = test_df.sample(min(200, len(test_df)), random_state=42)
    current_simulation['gdp_per_capita'] = current_simulation['gdp_per_capita'] * 1.2 
    
    report_file = monitor.initiate_monitoring(current_simulation)
    print(f"Monitoring test complete. Open this file in your browser: {report_file}")