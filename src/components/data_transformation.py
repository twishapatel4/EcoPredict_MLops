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
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor3.pkl")

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
            
            # --- FINAL OPTIMIZED CHRONOLOGICAL SPLIT ---
            # Train: 2000-2021 (maximize training data for better patterns)
            # Test: 2022 (single year for validation)
            logging.info("Applying 2-way split: Train (2000-2021), Test (2022)")
            
            train_df = df[df['year'] < 2022]
            test_df  = df[df['year'] >= 2022]
            
            # For the 3-return value compatibility, val_df = test_df
            val_df = test_df

            if test_df.empty:
                logging.warning("Test dataframe is empty! Check if your CSV contains 2022 data.")

            # Inside initiate_data_transformation...
            train_df.to_csv(os.path.join("artifacts", "reference_data.csv"), index=False)
            test_df.to_csv(os.path.join("artifacts", "current_data.csv"), index=False)
            logging.info("Saved reference and current dataframes to artifacts")
            # Separate Features and Target
            def split_input_target(data):
                X = data.drop(columns=[target_column_name, 'country'], axis=1)
                y = data[target_column_name]
                return X, y

            input_train_df, target_train_df = split_input_target(train_df)
            input_val_df, target_val_df     = split_input_target(val_df)
            input_test_df, target_test_df   = split_input_target(test_df)

            # Apply Transformation
            logging.info("Fitting preprocessor on Train and transforming Val/Test")
            
            train_arr_features = preprocessing_obj.fit_transform(input_train_df)
            val_arr_features   = preprocessing_obj.transform(input_val_df)
            test_arr_features  = preprocessing_obj.transform(input_test_df)

            # Ensure dense arrays
            def ensure_dense(arr):
                return arr.toarray() if hasattr(arr, "toarray") else arr

            train_arr_features = ensure_dense(train_arr_features)
            val_arr_features   = ensure_dense(val_arr_features)
            test_arr_features  = ensure_dense(test_arr_features)

            # Combine features and targets
            train_arr = np.c_[train_arr_features, np.array(target_train_df)]
            val_arr   = np.c_[val_arr_features,   np.array(target_val_df)]
            test_arr  = np.c_[test_arr_features,  np.array(target_test_df)]

            logging.info("Saving preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                val_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    raw_data_file = os.path.join('data', 'raw_data3.csv')
    if os.path.exists(raw_data_file):
        dt = DataTransformation()
        train, val, test, obj_path = dt.initiate_data_transformation(raw_data_file)
        print(f"Splits complete: Train {train.shape}, Val {val.shape}, Test {test.shape}")