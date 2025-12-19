import os
import sys
import pandas as pd
import numpy as np
import joblib
import base64
from io import BytesIO
from src.logging import logging
from src.utils.utils import create_engine_for_database
from src.exception import CustomException
from src.constants.config_entity import DataTransformationConfig
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()


password = os.getenv('POSTGRES_PASSWORD')
username = os.getenv('POSTGRES_USER')
host = os.getenv('POSTGRES_HOST')
port = os.getenv('POSTGRES_PORT')
name = os.getenv('POSTGRES_DB')


class DataTransformation:
    def __init__(self):
        config = DataTransformationConfig()
        self.preprocessor_table_name = config.preprocessing_table_name
        self.label_encoder_table_name = config.label_encoder_table_name
        self.features = config.features
        self.target = config.target
        self.drop_features = config.drop_columns
        self.num_cols = config.num_cols
        self.cat_cols = config.cat_cols

    def gathering_required_data(self, neo_data):
        try:
            logging.info(f'Before Required Gathering Columns: {neo_data.columns.tolist()}')
            neo_data['diameter_range'] = neo_data['max_diameter_km'] - neo_data['min_diameter_km']
            df = neo_data.drop(columns=self.drop_features, axis=1)
            logging.info(f'After Required Gathering Columns: {df.columns.tolist()}')
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def training_and_testing_split(self, req_data):
        try:
            x = req_data[self.features]
            y = req_data[self.target]

            x['diameter_range'] = np.log(x['diameter_range'])
            x['relative_velocity_kps'] = np.log(x['relative_velocity_kps'])
            x['miss_distance_km'] = np.sqrt(x['miss_distance_km'])

            logging.info(f"""Independent Features: {self.features}
                         and dependent feature {self.target}""")
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42, stratify=y
            )
            logging.info(f"""Shapes x_train: {x_train.shape},
                         x_test: {x_test.shape}, y_train: {y_train.shape},
                         y_test: {y_test.shape}""")
            return x_train, x_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)

    def preprocess_data(self, x_train, y_train, x_test, y_test):
        try:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), self.num_cols),
                    ('cat', OneHotEncoder(), self.cat_cols)
                ]
            )

            x_train_encoded = preprocessor.fit_transform(x_train)
            x_test_encoded = preprocessor.transform(x_test)

            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)

            # Create database engine
            engine = create_engine_for_database(
                user_name=username, password=password, host=host, port=port, database_name=name
            )

            # Serialize artifacts with joblib + base64
            def serialize_artifact(obj):
                bytes_buffer = BytesIO()
                joblib.dump(obj, bytes_buffer)
                return base64.b64encode(bytes_buffer.getvalue()).decode('utf-8')

            preprocessor_base64 = serialize_artifact(preprocessor)
            le_base64 = serialize_artifact(le)

            # Save to database
            preprocessor_df = pd.DataFrame([{
                "name": "preprocessor",
                "artifact": preprocessor_base64,
                "created_at": pd.Timestamp.now()
            }])
            label_encoder_df = pd.DataFrame([{
                "name": "label_encoder",
                "artifact": le_base64,
                "created_at": pd.Timestamp.now()
            }])

            preprocessor_df.to_sql(self.preprocessor_table_name,
                                   engine, if_exists='append', index=False)
            label_encoder_df.to_sql(self.label_encoder_table_name,
                                    engine, if_exists='append', index=False)

            return x_train_encoded, y_train_encoded, x_test_encoded, y_test_encoded
        except Exception as e:
            raise CustomException(e, sys)

    def treating_imbalanced_data(self, x_train_encoded, y_train_encoded):
        try:
            smote = BorderlineSMOTE()
            x_resample, y_resample = smote.fit_resample(x_train_encoded, y_train_encoded)
            return x_resample, y_resample
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, data):
        try:
            logging.info("=" * 50)
            logging.info("INITIATED DATA TRANSFORMATION")
            logging.info("-" * 50)

            df = self.gathering_required_data(data)
            x_train, x_test, y_train, y_test = self.training_and_testing_split(df)
            x_train_encoded, y_train_encoded, x_test_encoded, y_test_encoded = self.preprocess_data(
                x_train, y_train, x_test, y_test
            )
            x_resample, y_resample = self.treating_imbalanced_data(x_train_encoded, y_train_encoded)

            return x_resample, x_test_encoded, y_resample, y_test_encoded

        except Exception as e:
            raise CustomException(e, sys)

    # ------------------ Loader functions ------------------ #
    @staticmethod
    def load_artifact_from_db(table_name: str, engine):
        """
        Load the latest artifact (preprocessor or label encoder) from database safely.
        """
        import pandas as pd
        df = pd.read_sql(f"""SELECT artifact FROM
                         {table_name}
                         ORDER BY created_at DESC
                         LIMIT 1""",
                         engine)
        if df.empty:
            return None
        artifact_base64 = df['artifact'].values[0]
        artifact_bytes = base64.b64decode(artifact_base64)
        artifact = joblib.load(BytesIO(artifact_bytes))
        return artifact
