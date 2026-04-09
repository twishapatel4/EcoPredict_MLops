import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

import mlflow
import mlflow.sklearn
import dagshub

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model4.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            # The last column is our target (co2_per_capita)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBoost": XGBRegressor(),
            }

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models
            )
            
            logging.info(f"Model Evaluation Report: {model_report}")

            # 2. Get the best model score and name
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]  
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable accuracy (R2 < 0.6)")

            logging.info(f"Best found model: {best_model_name}")

            y_test_pred = best_model.predict(X_test)
            mae = mean_absolute_error(y_test, y_test_pred)
            mse = mean_squared_error(y_test, y_test_pred)

            # 4. Log to MLflow with all metrics
            self.log_to_mlflow(best_model, best_model_name, best_model_score, mae, mse)

            # 5. Save model locally
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)

    def log_to_mlflow(self, best_model, best_model_name, r2, mae, mse):
        try:
            # Initialize DagsHub for MLflow tracking
            dagshub.init(repo_owner='twishapatel4', repo_name='EcoPredict_MLops', mlflow=True)
            mlflow.set_experiment("EcoPredict_Global_Energy")

            with mlflow.start_run():
                # Log Parameters
                mlflow.log_param("algorithm", best_model_name)
                
                # Log Metrics
                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("mse", mse)

                # Log the model object
                mlflow.sklearn.log_model(best_model, "champion_model")
                
                logging.info(f"MLflow: Logged {best_model_name} (R2: {r2:.4f}, MAE: {mae:.4f})")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    from src.components.data_transformation import DataTransformation
    
    # 1. Data Loading and Transformation
    raw_data_file = os.path.join('data', 'raw_data3.csv')
    if not os.path.exists(raw_data_file):
        print(f"Error: {raw_data_file} not found.")
    else:
        data_transformation = DataTransformation()
        train_arr, val_arr, test_arr, _ = data_transformation.initiate_data_transformation(raw_data_file)
        
        # 2. Run Trainer
        model_trainer = ModelTrainer()
        score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(f"Model Training Complete. Best Model R2 Score: {score:.4f}")