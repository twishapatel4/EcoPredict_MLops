import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # Path where the preprocessor pickle file will be saved
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function creates the preprocessing pipeline for both 
        numerical and categorical data.
        """
        try:
            # We use these features to predict CO2 emissions
            numerical_columns = ["year", "energy_per_capita", "gdp_per_capita"]
            categorical_columns = ["iso_code"]

            # Numerical Pipeline: Handling missing values and scaling
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical Pipeline: Handling missing names and encoding
            # We set sparse_output=False to avoid the concatenation error
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combining both pipelines
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, raw_path):
        """
        This function reads the raw data, performs chronological splitting,
        applies transformations, and saves the preprocessor object.
        """
        try:
            df = pd.read_csv(raw_path)
            logging.info("Read raw data successfully")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "co2_per_capita"
            
            # 1. Chronological Split (Time-Series requirement for 2026 AI Standard)
            # We train on past data and test on future data
            logging.info("Splitting data into train and test based on Year (2019 threshold)")
            train_df = df[df['year'] < 2019]
            test_df = df[df['year'] >= 2019]

            # 2. Separate Input Features (X) and Target (y)
            # We drop 'country' because it's redundant with 'iso_code'
            input_feature_train_df = train_df.drop(columns=[target_column_name, 'country'], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name, 'country'], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # 3. Apply Transformation
            logging.info("Applying preprocessing object on training and testing dataframes")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # --- DENSE ARRAY CONVERSION (The Fix for the ValueError) ---
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()

            # 4. Combine X and y back into final arrays for the Model Trainer
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            # 5. Save the 'Math Recipe' (preprocessor.pkl)
            logging.info("Saving preprocessing object to artifacts folder.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

# --- STANDALONE EXECUTION BLOCK ---
if __name__ == "__main__":
    try:
        # UPDATED PATH: Looking for raw_data in data/raw_data/
        raw_data_file = os.path.join('data', 'raw_data.csv')
        
        if not os.path.exists(raw_data_file):
            print(f"Error: {raw_data_file} not found. Please run Data Ingestion first.")
        else:
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(raw_data_file)
            
            print("--- Transformation Success ---")
            print(f"Preprocessor Object saved at: {preprocessor_path}")
            print(f"Train Array shape: {train_arr.shape}")
            print(f"Test Array shape: {test_arr.shape}")

    except Exception as e:
        # Capture error and print detail
        print(f"Capture Detail: {e}")