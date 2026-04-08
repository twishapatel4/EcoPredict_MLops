import pandas as pd
import os
import sys
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from src.exception import CustomException
from src.logger import logging

class ModelMonitoring:
    def __init__(self):
        self.reference_data_path = os.path.join("data", "raw_data", "raw_data.csv")
        self.report_path = os.path.join("artifacts", "evidently_report.html")

    def initiate_monitoring(self, current_df):
        """
        Compares reference data with current user input data
        """
        try:
            logging.info("Starting Evidently AI Monitoring Report generation")
            
            # 1. Load Reference Data
            ref_df = pd.read_csv(self.reference_data_path)
            
            # 2. Keep only the columns the model actually uses
            relevant_cols = ['year', 'energy_per_capita', 'gdp_per_capita', 'co2_per_capita']
            ref_df = ref_df[relevant_cols]
            
            # Ensure current_df has same columns (if prediction isn't available, we check features)
            # For this demo, we assume current_df is a batch of new data
            
            # 3. Define the Report
            report = Report(metrics=[
                DataDriftPreset(), # Detects if GDP/Energy distributions have shifted
                DataQualityPreset(), # Detects missing values or outliers
                TargetDriftPreset() # Detects if the predicted CO2 is behaving weirdly
            ])

            # 4. Run the Audit
            logging.info("Running drift detection algorithms...")
            report.run(reference_data=ref_df, current_data=current_df)

            # 5. Save as an Interactive HTML Dashboard
            report.save_html(self.report_path)
            logging.info(f"Evidently Report saved successfully at {self.report_path}")
            
            return self.report_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Test logic: We compare the training data against a "modified" version of itself 
    # to simulate real-world drift.
    monitor = ModelMonitoring()
    train_data = pd.read_csv("data/raw_data/raw_data.csv")
    
    # Simulate "Current" data by adding some noise to GDP
    current_simulation = train_data.sample(200).copy()
    current_simulation['gdp_per_capita'] = current_simulation['gdp_per_capita'] * 1.2 
    
    report_file = monitor.initiate_monitoring(current_simulation)
    print(f"Monitoring test complete. Open this file in your browser: {report_file}")