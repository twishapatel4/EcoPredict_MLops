import sys
import os
import pandas as pd
import numpy as np
import shap
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.raw_data_path = os.path.join('data', 'raw_data3.csv')

    def _add_trajectory_features(self, input_df):
        """
        Internal helper to calculate the 3 Trajectory features 
        required by the new model.
        """
        try:
            # 1. Load historical data to find the 'Previous Year' values
            hist_df = pd.read_csv(self.raw_data_path)
            iso = input_df['iso_code'].iloc[0]
            
            # Get historical trend for this specific country
            country_hist = hist_df[hist_df['iso_code'] == iso].sort_values('year')

            # 2. Calculate Economic Efficiency (Snapshot)
            input_df['gdp_energy_ratio'] = input_df['gdp_per_capita'] / (input_df['energy_per_capita'] + 1e-6)

            if not country_hist.empty:
                # 3. Calculate Velocity (Current Input Energy - Last Known Energy)
                last_known_energy = country_hist['energy_per_capita'].iloc[-1]
                input_df['energy_velocity'] = input_df['energy_per_capita'] - last_known_energy

                # 4. Calculate Momentum (Historical 3-year average velocity)
                # We use the historical trend to tell the model the country's behavior
                hist_velocity = country_hist['energy_per_capita'].diff().tail(3).mean()
                input_df['energy_momentum'] = hist_velocity if not np.isnan(hist_velocity) else 0
            else:
                # Defaults for brand new countries not in our database
                input_df['energy_velocity'] = 0
                input_df['energy_momentum'] = 0

            return input_df
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            logging.info("Starting prediction pipeline")
            
            # Add the required trajectory features
            features = self._add_trajectory_features(features)

            preprocessor_path = os.path.join('artifacts', "preprocessor4.pkl")
            model_path = os.path.join('artifacts', "model4.pkl")

            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)

            logging.info(f"Transforming features. Columns: {features.columns.tolist()}")
            data_scaled = preprocessor.transform(features)

            logging.info("Making predictions")
            preds = model.predict(data_scaled)

            return preds
        except Exception as e:
            raise CustomException(e, sys)
        
    def explain(self, features):
        try:
            # Add trajectory features so SHAP can see the "Trend" impact
            features = self._add_trajectory_features(features)

            model_path = os.path.join("artifacts", "model4.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor4.pkl")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_transformed = preprocessor.transform(features)
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data_transformed)
            feature_names = preprocessor.get_feature_names_out()

            return shap_values, data_transformed, feature_names

        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self, year: int, energy_per_capita: float, gdp_per_capita: float, iso_code: str):
        self.year = year
        self.energy_per_capita = energy_per_capita
        self.gdp_per_capita = gdp_per_capita
        self.iso_code = iso_code

    def get_data_as_dataframe(self):
        try:
            # Note: We only create the 4 user columns here. 
            # The PredictPipeline will add the other 3 automatically.
            data_dict = {
                "year": [self.year],
                "energy_per_capita": [self.energy_per_capita],
                "gdp_per_capita": [self.gdp_per_capita],
                "iso_code": [self.iso_code]
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys)