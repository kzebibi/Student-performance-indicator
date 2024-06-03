# # For CSV
# df_csv = pd.read_csv("data.csv")

# # For JSON
# df_json = pd.read_json("data.json")

# # For HTML
# df_html = pd.read_html("data.html")[0]  # Assuming the data is in the first table

# # For Excel
# df_excel = pd.read_excel("data.xlsx")

# # For MongoDB
# import pymongo
# client = pymongo.MongoClient("mongodb://localhost:27017/")
# db = client["mydatabase"]
# collection = db["mycollection"]
# df_mongo = pd.DataFrame(list(collection.find()))

# # For MySQL
# import mysql.connector
# mydb = mysql.connector.connect(
#   host="localhost",
#   user="yourusername",
#   password="yourpassword",
#   database="mydatabase"
# )
# mycursor = mydb.cursor()
# mycursor.execute("SELECT * FROM yourtable")
# df_mysql = pd.DataFrame(mycursor.fetchall())

# # For SQLite
# import sqlite3
# conn = sqlite3.connect('example.db')
# df_sqlite = pd.read_sql_query("SELECT * FROM table_name", conn)

import pickle
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation, DataTransformationConfig
@dataclass
class DataIngestionConfig:
    """Data Ingestion Configuration class."""
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    """Class for data ingestion."""
    
    def __init__(self):
        """Initialize DataIngestion class."""
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        """Method to initiate data ingestion."""
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset
            df = pd.read_csv(r"notebooks\data\stud.csv")
            logging.info("Read the dataset as dataframe")
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train test split initiated")
            # Perform train test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=53)
            
            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)
