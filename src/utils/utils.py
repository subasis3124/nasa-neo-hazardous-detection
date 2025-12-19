import requests
import os
import tempfile
import joblib
import logging
import sys
import mlflow
import base64
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from io import BytesIO
import pandas as pd
from src.exception import CustomException
from catboost import CatBoostClassifier
from xgboost import Booster
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def extract_best_model():
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("""SELECT * FROM model_training_logs
                     ORDER BY training_date DESC
                     LIMIT 1""", engine)
    artifact_uri = df["artifact_uri"].iloc[0]
    model_name = df["model_name"].iloc[0]
    # Path to model artifact in MLflow
    model_artifact_path = f"{artifact_uri}/models/{model_name}/{model_name}.model"
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download artifact
        local_path = mlflow.artifacts.download_artifacts(model_artifact_path, dst_path=tmpdir)
        # Load based on extension/type
        if "CatBoost" in model_name:
            model = CatBoostClassifier()
            model.load_model(local_path)
        elif "XGB" in model_name:
            model = Booster()
            model.load_model(local_path)
        else:
            model = joblib.load(local_path)
    return model


def load_artifact_from_db(table_name: str, engine):
    df = pd.read_sql(f"SELECT artifact FROM {table_name} ORDER BY created_at DESC LIMIT 1", engine)
    if df.empty:
        return None
    artifact_base64 = df['artifact'].values[0]
    artifact_bytes = base64.b64decode(artifact_base64)
    artifact = joblib.load(BytesIO(artifact_bytes))
    return artifact


def create_engine_for_database(user_name, password, host, port, database_name):
    engine = create_engine(
        f"postgresql+psycopg2://{user_name}:{password}@{host}:{port}/{database_name}",
        connect_args={
                "sslmode": "require",
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5
            }
    )
    return engine


def read_data_from_pg(user_name, password, host, port, database_name, table_name):
    try:
        logging.info("Creating Engine")
        engine = create_engine(
            f"postgresql+psycopg2://{user_name}:{password}@{host}:{int(port)}/{database_name}",
            connect_args={
                "sslmode": "require",
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5
            }
        )
        logging.info("Engine Created")
        query = f'SELECT * FROM "{table_name}"'
        logging.info(f"Executing query: {query}")
        df = pd.read_sql(query, engine)
        logging.info(f"Data read from the database with table name {table_name} having shape: {df.shape}")
        return df

    except Exception as e:
        logging.error(f"Error while reading the database with {table_name}: {e}")
        raise CustomException(e, sys)


def fetch_data(start_date, end_date, api):
    url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={start_date}&end_date={end_date}&api_key={api}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print('Unable to connect')


def get_mlflow_metrics(actual, predicted):
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted)
    recall = recall_score(actual, predicted)
    f1 = f1_score(actual, predicted)
    return accuracy, precision, recall, f1
