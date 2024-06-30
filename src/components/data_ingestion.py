import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        # Initializing the data config paths/variables
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        # Starting the Data Ingestion Process
        logging.info("Entered the data ingestion component")
        try:
            # Reading the data
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Read the dataset as a dataframe")
            # Saving the raw data in the artifacts folder
            os.makedirs(os.path.dirname(
                self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,
                      index=False, header=True)

            # Splitting the data into training and testing set
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(
                df, random_state=42, test_size=0.2)
            # Saving the train data in the artifacts folder
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True)
            # Saving the test data in the artifacts folder
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
