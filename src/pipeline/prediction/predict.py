import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv

import mlflow
import dagshub

from src.utils.utils import extract_best_model, fetch_data, load_artifact_from_db
from src.constants.config_entity import PredictionConfig, DataTransformationConfig

# Load environment variables
load_dotenv()
nasa_api = os.getenv('NASA_API_KEY')
database_url = os.getenv('DATABASE_URL')
engine = create_engine(url=database_url)

REPO_OWNER_NAME = os.getenv("DAGS_REPO_OWNER_NAME")
REPO_NAME = os.getenv("DAGS_REPO_NAME")
MLFLOW_REMOTE_TRACKING_URL = os.getenv("MLFLOW_REMOTE_TRACKING_URL")

# Initialize DAGHub + MLflow
dagshub.init(repo_owner=REPO_OWNER_NAME, repo_name=REPO_NAME, mlflow=True)
if MLFLOW_REMOTE_TRACKING_URL:
    mlflow.set_tracking_uri(MLFLOW_REMOTE_TRACKING_URL)


class Predicting:
    def __init__(self):
        prediction_config = PredictionConfig()
        transform_config = DataTransformationConfig()

        self.drop_columns = transform_config.drop_columns
        self.features = transform_config.features
        self.target = transform_config.target
        self.preprocessor_table = transform_config.preprocessing_table_name
        self.label_encoder_table = transform_config.label_encoder_table_name
        self.pred_log_table = prediction_config.prediction_log_table

        # Load best model and artifacts
        self.best_model = extract_best_model()
        self.preprocessor = load_artifact_from_db(self.preprocessor_table, engine)
        self.label_encoder = load_artifact_from_db(self.label_encoder_table, engine)

    def gather_data(self, start_date, end_date):
        neo_list = []
        today = datetime.today()

        while start_date < today:
            if end_date >= today:
                end_date = today

            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            print("-" * 50)
            print(f"Fetching data from [{start_str}] to [{end_str}]")
            print("-" * 50)

            data = fetch_data(start_str, end_str, nasa_api)

            if data and "near_earth_objects" in data:
                for date, asteroids in data["near_earth_objects"].items():
                    for asteroid in asteroids:
                        for ca_data in asteroid.get("close_approach_data", []):
                            neo_list.append({
                                "id": asteroid.get("id"),
                                "name": asteroid.get("name"),
                                "absolute_magnitude_h": asteroid.get("absolute_magnitude_h"),
                                "min_diameter_km": asteroid.get("estimated_diameter", {}).get("kilometers", {}).get("estimated_diameter_min"),
                                "max_diameter_km": asteroid.get("estimated_diameter", {}).get("kilometers", {}).get("estimated_diameter_max"),
                                "close_approach_date": ca_data.get("close_approach_date"),
                                "close_approach_date_full": ca_data.get("close_approach_date_full"),
                                "epoch_date_close_approach": ca_data.get("epoch_date_close_approach"),
                                "relative_velocity_kps": float(ca_data.get("relative_velocity", {}).get("kilometers_per_second", 0.0)),
                                "miss_distance_km": float(ca_data.get("miss_distance", {}).get("kilometers", 0.0)),
                                "orbiting_body": ca_data.get("orbiting_body"),
                                "is_potentially_hazardous": asteroid.get("is_potentially_hazardous_asteroid"),
                                "is_sentry_object": asteroid.get("is_sentry_object"),
                                "nasa_jpl_url": asteroid.get("nasa_jpl_url"),
                            })

            start_date = end_date + timedelta(days=1)
            end_date = start_date + timedelta(days=6)

        data = pd.DataFrame(neo_list)
        print(f"Data gathered with shape: {data.shape}")
        return data

    def transforming_data(self, data):
        id_series = data['id']
        print(f"Number of records: {len(id_series)}")

        data['diameter_range'] = data['max_diameter_km'] - data['min_diameter_km']
        df = data.drop(columns=self.drop_columns, axis=1)
        print(f"Shape after dropping columns: {df.shape}")

        x = df[self.features].copy()
        y = df[self.target].copy()
        print(f"Feature matrix shape: {x.shape}, Target vector shape: {y.shape}")

        # Transform skewed features
        x['diameter_range'] = np.log(x['diameter_range'])
        x['relative_velocity_kps'] = np.log(x['relative_velocity_kps'])
        x['miss_distance_km'] = np.sqrt(x['miss_distance_km'])
        print("Skewness corrected for features")

        x_encoded = self.preprocessor.transform(x)
        print(f"Encoded features shape: {x_encoded.shape}")

        return id_series, y, x_encoded

    def predict_data(self, id_series, y, x_encoded):
        # Model prediction
        encoded_prediction = self.best_model.predict(x_encoded)
        prediction = self.label_encoder.inverse_transform(encoded_prediction)
        pred_proba = self.best_model.predict_proba(x_encoded)
        prob_of_false = pred_proba[:, 0]
        prob_of_true = pred_proba[:, 1]

        # Fetch training model date
        query = "SELECT training_date FROM model_training_logs ORDER BY training_date DESC LIMIT 1;"
        trained_model_date_df = pd.read_sql(query, engine)
        if trained_model_date_df.empty:
            trained_model_date = None
        else:
            trained_model_date = trained_model_date_df.iloc[0, 0]

        # Prepare prediction dataframe
        prediction_dict = {
            "id": id_series.values,
            "actual": y.values.ravel(),
            "model_prediction": prediction,
            "probability_being_falsely_hazardous": prob_of_false,
            "probability_being_truely_hazardous": prob_of_true,
            "trained_model_date_used": [trained_model_date] * len(id_series)
        }

        prediction_df = pd.DataFrame(prediction_dict)
        return prediction_df


if __name__ == "__main__":
    end_date = datetime.today()
    start_date = end_date - timedelta(days=6)

    predicting = Predicting()
    data = predicting.gather_data(start_date=start_date, end_date=end_date)
    print("[INFO] Gathering Data Completed")

    id_series, y, encoded_data = predicting.transforming_data(data)
    print("[INFO] Data Transformation Completed")

    prediction_df = predicting.predict_data(id_series, y, encoded_data)
    print("[INFO] Prediction Completed")

    # Push to database
    prediction_df.to_sql(predicting.pred_log_table, engine, if_exists='append', index=False)
    print("[INFO] Prediction Info Pushed to Database")
