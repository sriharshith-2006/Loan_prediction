import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_path: str = os.path.join("artifacts", "preprocessor.pkl")
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        ##This function is responsible for data transformation
        try:
            logging.info("Creating data transformer object")
            numerical_features = [
                                    'person_age',
                                    'person_income',
                                    'person_emp_exp',
                                    'loan_amnt',
                                    'loan_int_rate',
                                    'loan_percent_income',
                                    'cb_person_cred_hist_length',
                                    'credit_score'
                                ]
            categorical_features = [
                                    'person_gender',
                                    'person_education',
                                    'person_home_ownership',
                                    'loan_intent',
                                    'previous_loan_defaults_on_file'
                                ]
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),##mode
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical and categorical pipelines created")
            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_features),
                    ('cat_pipeline',cat_pipeline,categorical_features)
                ]
            )
            return preprocessor
        except Exception as e:
            logging.error("Error occurred while creating data transformer object")
            raise CustomException(e, sys) from e
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Starting data transformation process")
            # Read the training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Training and testing data read successfully")

            # Separate input features and target variable
            target_column_name = 'loan_status'
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Get the preprocessor object
            preprocessor_obj = self.get_data_transformer_object()

            # Fit and transform the training data, transform the testing data
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Data transformation completed successfully")

            # Combine transformed features with target variable
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessor_obj
            )  
            return (
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_obj_path
            )
        except Exception as e:
            logging.error("Error occurred during data transformation")
            raise CustomException(e, sys) from e