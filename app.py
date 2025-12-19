from flask import Flask, request, jsonify
from sqlalchemy import create_engine
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import os
import mlflow
# import dagshub
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from src.utils.utils import extract_best_model, load_artifact_from_db
from src.constants.config_entity import DataTransformationConfig
from src.constants.params import PARAMS
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

REPO_OWNER_NAME = os.getenv("DAGS_REPO_OWNER_NAME")
REPO_NAME = os.getenv("DAGS_REPO_NAME")
DAGS_USER_TOKEN = os.getenv("DAGS_USER_TOKEN")

# Set MLflow tracking URI directly with token (no interactive auth)
MLFLOW_TRACKING_URI = f"https://{DAGS_USER_TOKEN}@dagshub.com/{REPO_OWNER_NAME}/{REPO_NAME}.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Now you can use MLflow as usual
# e.g., mlflow.list_experiments()

app = Flask(__name__)

parsed = urlparse(DATABASE_URL)
query = parse_qs(parsed.query)

if "channel_binding" in query:
    del query["channel_binding"]

new_query = urlencode(query, doseq=True)
cleaned_url = urlunparse(parsed._replace(query=new_query))
# DB connection
engine = create_engine(cleaned_url, echo=True, future=True)
best_model = extract_best_model()
preprocessor = load_artifact_from_db("preprocessing_table", engine)
label_encoder = load_artifact_from_db("label_encoder_table", engine)
model_metrics_query = """SELECT 
                         recall * 100 AS recall_pct,
                         accuracy * 100 AS accuracy_pct,
                         precision * 100 AS precision_pct,
                         f1_score * 100 AS f1_score_pct
                         FROM model_training_logs
                         ORDER BY training_date DESC
                         LIMIT 1;"""
model_metrics = pd.read_sql(model_metrics_query, engine)

web_url = "http://localhost:5000/"

@app.route("/")
def index():
    try:
        best_model_query = "SELECT model_name FROM model_training_logs ORDER BY training_date DESC LIMIT 1;"
        last_trained_date_query = "SELECT training_date FROM model_training_logs ORDER BY training_date DESC LIMIT 1;"

        best_model_df = pd.read_sql(best_model_query, engine)
        if best_model_df.empty:
            best_model_name = "No model found"
        else:
            best_model_name = best_model_df.iloc[0, 0]

        date_trained_df = pd.read_sql(last_trained_date_query, engine)
        if date_trained_df.empty:
            date_trained = "N/A"
        else:
            date_trained = date_trained_df.iloc[0, 0]

        return jsonify({
            "1. Name": "Nasa Near Earth Hazardous Object Detection",
            "2. Job": "Traditional Machine Learning Approach to detect whether a approaching asteroid is hazardous or not.",
            "3. Data Used for training": "https://api.nasa.gov/",
            "4. Best Model": best_model_name,
            "5. Last Trained on": str(date_trained),
            "6. GitHub Link": "https://github.com/Subrat1920/Nasa-Near-Earth-Hazardous-Detection.git",
            "7. DagsHub Link (Data Version Control)": "https://dagshub.com/Subrat1920/Nasa-Near-Earth-Hazardous-Detection.mlflow",
            "8. How to use the model as an API": f"{web_url}call_docs",
            "9. Documentation": f"{web_url}docs"
        })
        
    except Exception as e:
        print("Index route error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/docs")
def documentation():
    return jsonify({
        "a. project": "NASA NEO Hazard Classification API",
        "b. description": [
            "This API predicts whether a Near-Earth Object (NEO) is hazardous using NASA's asteroid dataset. "
            "It is powered by a machine learning pipeline with automated data preprocessing, feature "
            "engineering, model training, experiment tracking, and evaluation."
        ],
        "c. pipeline_procedure": [
            "1. Data Collection: NASA Near-Earth Object dataset",
            "2. Data Preprocessing: Handling missing values, feature scaling, encoding categorical variables",
            "3. Feature Engineering: Derived features like velocity ratios, size ranges, etc.",
            "4. Model Training: Multiple algorithms tested (Logistic Regression, Random Forest, XGBoost, RNN)",
            "5. Experiment Tracking: MLflow + DAGsHub",
            "6. Model Selection: Best model selected based on F1 score",
            "7. Deployment: Flask API with PostgreSQL logging"
        ],
        "d. algorithms_evaluated": [name for name, model, params in PARAMS] if PARAMS else [],
        "e. metric_priority": {
            "1. primary_metric": "Recall",
            "2. reason": "In hazardous object detection, missing a dangerous asteroid (false negative) is far riskier than "
                         "raising a false alarm (false positive). Therefore, recall is prioritized to ensure that as many "
                         "true hazardous objects as possible are identified, even if it means sometimes predicting safe "
                         "objects as hazardous.",
            "3. tradeoff": "A higher recall may slightly reduce precision, but it ensures safety by minimizing the chance of "
                           "overlooking a truly hazardous object."
                                },

        "f. performance_metrics": {
            "recall": model_metrics['recall_pct'][0],
            "accuracy": model_metrics['accuracy_pct'][0],
            "precision": model_metrics['precision_pct'][0],
            "f1_score": model_metrics['f1_score_pct'][0]
        },
        "g. features": {
            "prediction": "Submit asteroid parameters and get hazard classification result",
            "logging": "All predictions stored in PostgreSQL database",
            "tracking": "Experiment logs stored with MLflow and DAGsHub"
        },
        "h. input_example": {
            "id": 2085990,
            "name": "85990 (1999 JV6)",
            "absolute_magnitude_h": 20.27,
            "min_diameter_km": 0.234723,
            "max_diameter_km": 0.524856,
            "close_approach_date": "2015-01-05",
            "close_approach_date_full": "2015-Jan-05 11:16",
            "epoch_date_close_approach": 1420456560000,
            "relative_velocity_kps": 7.694743,
            "miss_distance_km": 1.246329e+07,
            "orbiting_body": "Earth",
            "is_sentry_object": False,
            "nasa_jpl_url": "https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=2085990"
        },
        "i. output_example": {
            "hazardous": True,
            "probability_of_false": 0.4470,
            "probability_of_true": 0.5530,
            "used_model_trained_on": "2025-09-01"
        },
        "j. author": ["Subrat Mishra", "Subasis Mishra"],
        "k. repository": "https://github.com/Subrat1920/nasa-neo-hazard-classification",
        "l. portfolio": "https://mishra-subrat.framer.website/"
    })

@app.route("/call_docs")
def api_documentation():
    return jsonify({
        "project": "NASA NEO Hazard Prediction API",
        "description": (
            "This API predicts whether a Near-Earth Object (NEO) is hazardous "
            "based on NASA's open dataset. "
            "The model emphasizes Recall to minimize false negatives, "
            "ensuring potentially hazardous asteroids are not missed."
        ),
        "endpoints": {
            "/predict": "POST - Submit asteroid data for hazard prediction",
            "/docs": "GET - Column and payload documentation",
            "/call_docs": "GET - Example API usage"
        },
        "examples": {
            "1. PowerShell": [
                f"Invoke-WebRequest -Uri \"{web_url}predict\" `",
                "  -Method POST `",
                "  -ContentType \"application/json\" `",
                "  -Body '{",
                "    \"id\": 2085990,",
                "    \"name\": \"85990 (1999 JV6)\",",
                "    \"absolute_magnitude_h\": 20.27,",
                "    \"min_diameter_km\": 0.234723,",
                "    \"max_diameter_km\": 0.524856,",
                "    \"close_approach_date\": \"2015-01-05\",",
                "    \"close_approach_date_full\": \"2015-Jan-05 11:16\",",
                "    \"epoch_date_close_approach\": 1420456560000,",
                "    \"relative_velocity_kps\": 7.694743,",
                "    \"miss_distance_km\": 12463290,",
                "    \"orbiting_body\": \"Earth\",",
                "    \"is_sentry_object\": false,",
                "    \"nasa_jpl_url\": \"https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=2085990\"",
                "}'"
            ],
            "2. Python (requests)": [
                "import requests",
                "",
                f"url = \"{web_url}predict\"",
                "data = {",
                "    \"id\": 2085990,",
                "    \"name\": \"85990 (1999 JV6)\",",
                "    \"absolute_magnitude_h\": 20.27,",
                "    \"min_diameter_km\": 0.234723,",
                "    \"max_diameter_km\": 0.524856,",
                "    \"close_approach_date\": \"2015-01-05\",",
                "    \"close_approach_date_full\": \"2015-Jan-05 11:16\",",
                "    \"epoch_date_close_approach\": 1420456560000,",
                "    \"relative_velocity_kps\": 7.694743,",
                "    \"miss_distance_km\": 12463290,",
                "    \"orbiting_body\": \"Earth\",",
                "    \"is_sentry_object\": False,",
                "    \"nasa_jpl_url\": \"https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=2085990\"",
                "}",
                "",
                "response = requests.post(url, json=data)",
                "print(response.json())"
            ]
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        config = DataTransformationConfig()
        drop_columns = config.drop_columns
        features = config.features

        data = request.get_json()
        t_df = pd.DataFrame([data])

        # Feature engineering
        t_df["diameter_range"] = t_df["max_diameter_km"] - t_df["min_diameter_km"]
        df = t_df.drop(columns=drop_columns, errors="ignore")

        x = df[features].copy()
        x["diameter_range"] = np.log(x["diameter_range"])
        x["relative_velocity_kps"] = np.log(x["relative_velocity_kps"])
        x["miss_distance_km"] = np.sqrt(x["miss_distance_km"])

        # Transform + predict
        x_encoded = preprocessor.transform(x)
        encoded_prediction = best_model.predict(x_encoded)
        prediction = label_encoder.inverse_transform(encoded_prediction)

        pred_proba = best_model.predict_proba(x_encoded)
        probability_of_false = pred_proba[:, 0].tolist()
        probability_of_true = pred_proba[:, 1].tolist()

        query = "SELECT training_date FROM model_training_logs ORDER BY training_date DESC LIMIT 1;"
        trained_model_date_df = pd.read_sql(query, engine)
        trained_date = trained_model_date_df["training_date"].iloc[0]

        return jsonify(
            {
                "prediction": str(prediction[0]),
                "probability_of_true": probability_of_true,
                "probability_of_false": probability_of_false,
                "used_model_trained_on": str(trained_date)
            }
        ), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
