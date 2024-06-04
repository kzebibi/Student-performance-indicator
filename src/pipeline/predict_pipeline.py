import sys  # Import sys module
import pandas as pd  # Import pandas library
from src.exception import CustomException  # Import CustomException class
from src.utils import load_object  # Import load_object function
import os  # Import os module


class PredictPipeline:
    """Class for making predictions"""

    def __init__(self) -> None:
        pass

    def predict(self, features):
        """Predict method to make predictions based on input features"""
        try:
            model_path = os.path.join("artifacts", "model.pkl")  # Define model path
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')  # Define preprocessor path
            print("Before Loading")  # Print message
            model = load_object(file_path=model_path)  # Load model object
            preprocessor = load_object(file_path=preprocessor_path)  # Load preprocessor object
            print("After Loading")  # Print message
            data_scaled = preprocessor.transform(features)  # Transform features
            preds = model.predict(data_scaled)  # Make predictions
            return preds  # Return predictions

        except Exception as e:
            raise CustomException(e, sys)  # Raise CustomException with error message

class CustomData:
    """Class for handling custom data"""

    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education, lunch: str,
                 test_preparation_course: str, reading_score: int, writing_score: int):
        """Initialize CustomData object with input data"""
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """Method to convert custom data to a DataFrame"""
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)  # Return DataFrame with custom data

        except Exception as e:
            raise CustomException(e, sys)  # Raise CustomException with error message