import os
import sys

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trainer_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting Independent and dependent features from Train and Test Data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "LinearRegression" : LinearRegression(),
                "Ridge" : Ridge(),
                "Lasso" : Lasso(),
                "ElasticNet" : ElasticNet()
            }

            model_report : dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            logging.info(f"Model Report : {model_report}")

            # To get the best score from the dict
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name
            best_model_name = list(models.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name : {best_model_name}, R2 Score : {best_model_score}")
            logging.info(f"Best Model Found, Model Name : {best_model_name}, R2 Score : {best_model_score}")

            save_object(file_path=self.model_trainer_config.trainer_model_file_path, 
                        obj = best_model)


        except Exception as e:
            logging.info("Error Occured in Model Training")
            raise CustomException(e, sys)