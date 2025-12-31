import sys
import os
import pandas as pd
import numpy as np
import pickle

from src.exception import CustomException
from src.logger import logging

class CustomData:
    """
    This class is used to take raw user inputs and convert them into a DataFrame
    for model prediction.
    """
    def __init__(self, person_age, person_gender, person_education,
                 person_income, person_emp_exp, person_home_ownership,
                 loan_amnt, loan_intent, loan_int_rate,
                 loan_percent_income, cb_person_cred_hist_length,
                 credit_score, previous_loan_defaults_on_file):
        self.person_age = person_age
        self.person_gender = person_gender
        self.person_education = person_education
        self.person_income = person_income
        self.person_emp_exp = person_emp_exp
        self.person_home_ownership = person_home_ownership
        self.loan_amnt = loan_amnt
        self.loan_intent = loan_intent
        self.loan_int_rate = loan_int_rate
        self.loan_percent_income = loan_percent_income
        self.cb_person_cred_hist_length = cb_person_cred_hist_length
        self.credit_score = credit_score
        self.previous_loan_defaults_on_file = previous_loan_defaults_on_file

    def get_data_as_dataframe(self):
        try:
            data = {
                "person_age": [self.person_age],
                "person_gender": [self.person_gender],
                "person_education": [self.person_education],
                "person_income": [self.person_income],
                "person_emp_exp": [self.person_emp_exp],
                "person_home_ownership": [self.person_home_ownership],
                "loan_amnt": [self.loan_amnt],
                "loan_intent": [self.loan_intent],
                "loan_int_rate": [self.loan_int_rate],
                "loan_percent_income": [self.loan_percent_income],
                "cb_person_cred_hist_length": [self.cb_person_cred_hist_length],
                "credit_score": [self.credit_score],
                "previous_loan_defaults_on_file": [self.previous_loan_defaults_on_file]
            }
            df = pd.DataFrame(data)
            logging.info("Input data converted to DataFrame")
            return df
        except Exception as e:
            logging.error("Error in converting input to DataFrame")
            raise CustomException(e, sys)


class PredictPipeline:
    """
    This class handles loading the trained model and preprocessor, and makes predictions.
    """
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features):
        try:
            # Load model and preprocessor
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)

            with open(self.preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)

            # Transform features
            data_scaled = preprocessor.transform(features)

            # Make predictions
            preds = model.predict(data_scaled)
            logging.info("Prediction completed successfully")
            return preds
        except Exception as e:
            logging.error("Error during prediction")
            raise CustomException(e, sys)
