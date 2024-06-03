import os
import sys
import numpy as np
import pandas as pd
import dill

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
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)