import pandas as pd
import os
import sys

# Using the Legacy Report and Presets as established in your environment
from evidently.legacy.report.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset

from src.exception import CustomException
from src.logger import logging

class ModelMonitoring:
    def __init__(self):
        """
        Initializes paths to artifacts generated during Data Transformation.
        This ensures the 'Reference' is exactly what the model was trained on.
        """
        self.reference_data_path = os.path.join("artifacts", "reference_data.csv")
        self.current_data_path = os.path.join("artifacts", "current_data.csv")
        self.report_path = os.path.join("artifacts", "evidently_report.html")

    def initiate_monitoring(self, current_df=None):
        """
        Generates the Evidently AI report comparing Reference (Train) and Current (Test/Prod).
        
        Args:
            current_df (pd.DataFrame, optional): A specific dataframe to check. 
                                               If None, it loads 'current_data.csv'.
        """
        try:
            logging.info("Starting Monitoring: Loading Reference and Current Data")
            
            # 1. Load Reference Data (The exact training data)
            if not os.path.exists(self.reference_data_path):
                raise Exception(f"Reference data not found at {self.reference_data_path}. Please run Data Transformation first.")
            
            ref_df = pd.read_csv(self.reference_data_path)
            
            # 2. Load Current Data (The exact test data or passed dataframe)
            if current_df is None:
                if not os.path.exists(self.current_data_path):
                    raise Exception(f"Current data not found at {self.current_data_path}")
                current_df = pd.read_csv(self.current_data_path)

            # 3. Align Columns
            # We must monitor the same columns used in the trainer
            relevant_cols = ['energy_per_capita', 'gdp_per_capita', 'co2_per_capita']
            
            # Ensure columns exist in both dataframes
            ref_df = ref_df[relevant_cols]
            current_df = current_df[relevant_cols]
            
            logging.info(f"Comparing Reference ({len(ref_df)} rows) vs Current ({len(current_df)} rows)")

            # 4. Define the Report Configuration
            # We use 0.2 as a threshold to stabilize alerts for economic data
            report = Report(metrics=[
                DataDriftPreset(num_stattest_threshold=0.2), 
                DataQualityPreset(),
                TargetDriftPreset()
            ])

            # 5. Run Audit
            logging.info("Running Evidently Drift Audit...")
            report.run(reference_data=ref_df, current_data=current_df)

            # 6. Save Report
            os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
            report.save_html(self.report_path)
            
            logging.info(f"Monitoring Report successfully saved to: {self.report_path}")
            return self.report_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # --- STANDALONE TEST EXECUTION ---
    # This block runs when you execute this file directly: python src/components/model_monitoring.py
    try:
        monitor = ModelMonitoring()
        
        # We try to run it using the saved artifacts
        report_file = monitor.initiate_monitoring()
        
        print("\n" + "="*30)
        print("MONITORING PIPELINE SUCCESS")
        print(f"Report Location: {report_file}")
        print("="*30)

    except Exception as e:
        print(f"Monitoring failed. Did you run Data Transformation first?\nError: {e}")