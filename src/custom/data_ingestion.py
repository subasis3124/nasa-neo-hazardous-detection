import os
import sys
from src.exception import CustomException, error_message_details
from src.logging import logging
from src.constants.config_entity import DataIngestionConfig
from src.utils.utils import read_data_from_pg


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
        self.table_name = self.config.data_table_name
        self.data_ingestion_path = self.config.data_ingestion_config
        self.data_ingestion_csv_path = os.path.join(
            self.data_ingestion_path, self.config.data_csv_config
        )
        # Ensure directory exists
        os.makedirs(self.data_ingestion_path, exist_ok=True)

    def initiate_data_ingestion(self, password, username, host, port, name):
        try:
            logging.info("=" * 50)
            logging.info("INITIATED DATA INGESTION")
            logging.info("-" * 50)
            logging.info("-- Starting reading the data present in the database")
            logging.info("---- Reading Neo Table")

            neo_df = read_data_from_pg(
                username,
                password,
                host,
                port,
                name,
                self.table_name
            )
            logging.info(f"---- Shape of the Neo data is {neo_df.shape}")

            logging.info("-- Storing the data in located directory")
            neo_df.to_csv(self.data_ingestion_csv_path, index=False)
            logging.info(f"---- Neo Data Stored at {self.data_ingestion_csv_path}")
            return neo_df

        except Exception as e:
            logging.error(error_message_details(e, sys))
            raise CustomException(e, sys)
