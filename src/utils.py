import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save an object to a file using dill serialization.

    Args:
        file_path (str): Path to the file where the object will be saved.
        obj: Object to be saved.

    Raises:
        CustomException: If an error occurs during the save process.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluate regression models using R-squared score.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Testing features.
        y_test: Testing target.
        models: Dictionary of regression models to evaluate.

    Returns:
        report: Dictionary containing model names and their R-squared scores.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_params = params[list(models.keys())[i]]
            gs = GridSearchCV(model, model_params, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_pred=y_train_pred, y_true=y_train)
            test_model_score = r2_score(y_pred=y_test_pred, y_true=y_test)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
    
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)  