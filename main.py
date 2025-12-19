import os
import sys
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
from src.exception import CustomException
from src.logging import logging
from src.custom.data_ingestion import DataIngestion
from src.custom.data_transformation import DataTransformation
from src.custom.model_trainer import ModelTrainer
# Load environment variables
load_dotenv()

password = os.getenv('POSTGRES_PASSWORD')
username = os.getenv('POSTGRES_USER')
host = os.getenv('POSTGRES_HOST')
port = os.getenv('POSTGRES_PORT')
name = os.getenv('POSTGRES_DB')


if __name__ == '__main__':
    try:
        print("==== Data Ingestion Begins ====")
        data_ingestion = DataIngestion()
        neo_df = data_ingestion.initiate_data_ingestion(
            password=password,
            username=username,
            host=host,
            port=port,
            name=name
        )
        logging.info("Data ingestion completed successfully.")

        print("==== Data Transformation Begins ====")
        data_transformation = DataTransformation()
        x_resample, x_test_encoded, y_resample, y_test_encoded = data_transformation.initiate_data_transformation(neo_df)
        logging.info("Data transformation completed successfully.")

        print("==== Model Training Begins ====")
        model_trainer = ModelTrainer()
        model_trainer.model_training_with_mlflow(
            x_train=x_resample,
            y_train=y_resample,
            x_test=x_test_encoded,
            y_test=y_test_encoded
        )
        logging.info("Model training and MLflow logging completed successfully.")

    except Exception as e:
        logging.error("Pipeline execution failed.")
        raise CustomException(e, sys)
