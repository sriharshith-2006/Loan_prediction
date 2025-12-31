import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing arrays")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "KNeighbors": KNeighborsClassifier(),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
                "CatBoost": CatBoostClassifier(verbose=False)
            }

            params = {
                "Logistic Regression": {},
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 5, 10]
                },
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10]
                },
                "KNeighbors": {
                    "n_neighbors": [5, 7, 9]
                },
                "XGBoost": {
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5],
                    "n_estimators": [100, 200]
                },
                "CatBoost": {
                    "depth": [4, 6],
                    "learning_rate": [0.01, 0.1],
                    "iterations": [100, 200]
                }
            }

            best_model_score = 0
            best_model_name = None
            best_model_obj = None

            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")

                if params[model_name]:
                    search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=params[model_name],
                        n_iter=10,
                        cv=3,
                        scoring="accuracy",
                        n_jobs=-1,
                        random_state=42
                    )
                    search.fit(X_train, y_train)
                    trained_model = search.best_estimator_
                else:
                    model.fit(X_train, y_train)
                    trained_model = model

                y_pred = trained_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                logging.info(f"{model_name} Accuracy: {accuracy}")

                if accuracy > best_model_score:
                    best_model_score = accuracy
                    best_model_name = model_name
                    best_model_obj = trained_model

            logging.info(
                f"Best Model: {best_model_name} with Accuracy: {best_model_score}"
            )

            if best_model_score < 0.7:
                raise CustomException(
                    "No model achieved acceptable accuracy", sys
                )

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model_obj
            )

            logging.info("Best classification model saved successfully")

            return best_model_score

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)
