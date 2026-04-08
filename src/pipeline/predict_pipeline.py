import sys
import os
import pandas as pd
import shap
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info("Starting prediction pipeline")
            # Load the preprocessor and model objects
            preprocessor_path = os.path.join('artifacts', "preprocessor.pkl")
            model_path = os.path.join('artifacts', "model.pkl")

            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)

            # Transform the input features using the preprocessor
            logging.info("Transforming input features")
            data_scaled = preprocessor.transform(features)

            # Make predictions using the trained model
            logging.info("Making predictions")
            preds = model.predict(data_scaled)

            return preds
        except Exception as e:
            raise CustomException(e, sys)
        
    def explain(self, features):
        """
        Calculates SHAP values to explain the prediction
        """
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # 1. We must transform the features exactly like we did for prediction
            data_transformed = preprocessor.transform(features)
            
            # 2. Initialize the Explainer
            # TreeExplainer is specialized for XGBoost/RandomForest
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data_transformed)

            # 3. Get Feature Names (This is tricky with ColumnTransformer)
            # We need to know which column name belongs to which number
            feature_names = preprocessor.get_feature_names_out()

            return shap_values, data_transformed, feature_names

        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    """
    Responsible for mapping user input from the web form to a DataFrame that can be processed by our model.
    """
    def __init__(self,year: int, energy_per_capita: float, gdp_per_capita: float, iso_code: str):
        self.year = year
        self.energy_per_capita = energy_per_capita
        self.gdp_per_capita = gdp_per_capita
        self.iso_code = iso_code

    def get_data_as_dataframe(self):
        try:
            logging.info("Converting user input to DataFrame")
            data_dict = {
                "year": [self.year],
                "energy_per_capita": [self.energy_per_capita],
                "gdp_per_capita": [self.gdp_per_capita],
                "iso_code": [self.iso_code]
            }
            df = pd.DataFrame(data_dict)
            return df
        except Exception as e:
            raise CustomException(e, sys)
