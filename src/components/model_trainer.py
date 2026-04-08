import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
# from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

import mlflow
import mlflow.sklearn
import dagshub
# from urllib.parse import urlparse

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

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
            print(model_report)
            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
                

            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable accuracy")
            
            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")

            self.log_to_mlflow(best_model, best_model_name, best_model_score,X_test, y_test)

            # Save locally as well
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)

    def log_to_mlflow(self, best_model, best_model_name, best_model_score, X_test, y_test):
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            mlruns_path = os.path.join(project_root, "mlruns")
        
            # mlflow.set_tracking_uri(f"file:.//{mlruns_path}") 
            dagshub.init(repo_owner='twishapatel4', repo_name='GlobalEnergy', mlflow=True)
            # mlflow.set_tracking_uri("sqlite:///mlflow.db") 

            mlflow.set_experiment("EcoPredict_Global_Energy")
            # 1. Start a "Run" (A recorded attempt)
            with mlflow.start_run():
                # 2. Log "How" you trained it (Parameters)
                mlflow.log_param("algorithm", best_model_name)
                
                # 3. Log "How well" it did (Metrics)
                mlflow.log_metric("r2_score", best_model_score)

                # 4. Save the actual model file to MLflow
                mlflow.sklearn.log_model(best_model, "champion_model")
                
                logging.info(f"MLflow: Logged {best_model_name} with score {best_model_score}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    from src.components.data_transformation import DataTransformation
    import os
    
    # 1. We need the data from Phase 2
    raw_data_file = os.path.join('data', 'raw_data.csv')
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(raw_data_file)
    
    # 2. Run Trainer
    model_trainer = ModelTrainer()
    score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Model Training Complete. Best Model R2 Score: {score}")