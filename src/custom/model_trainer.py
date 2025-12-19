import os
import sys
import tempfile
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import mlflow
from mlflow.models import infer_signature
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from catboost import CatBoostClassifier
import dagshub
from dotenv import load_dotenv
from sqlalchemy import create_engine
from src.constants.config_entity import ModelTrainerConfig
from src.exception import CustomException
from src.logging import logging
from src.utils.utils import get_mlflow_metrics

load_dotenv()

REPO_OWNER_NAME = os.getenv("DAGS_REPO_OWNER_NAME")
REPO_NAME = os.getenv("DAGS_REPO_NAME")
MLFLOW_REMOTE_TRACKING_URL = os.getenv("MLFLOW_REMOTE_TRACKING_URL")
DATABASE_URL = os.getenv("DATABASE_URL")  # SQLAlchemy connection string

dagshub.init(repo_owner=REPO_OWNER_NAME, repo_name=REPO_NAME, mlflow=True)
if MLFLOW_REMOTE_TRACKING_URL:
    mlflow.set_tracking_uri(MLFLOW_REMOTE_TRACKING_URL)

ml_run_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def _save_model_to_file(model, filepath):
    """Save model depending on type (CatBoost, XGBoost, or sklearn)."""
    try:
        if isinstance(model, CatBoostClassifier):
            model.save_model(filepath)
            return filepath
    except Exception:
        pass

    try:
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            booster.save_model(filepath)
            return filepath
    except Exception:
        pass

    joblib.dump(model, filepath)
    return filepath


def log_training_to_db(model_name, metrics, run_id, artifact_uri, engine, table_name="model_training_logs"):
    """Append best model info to database table, including MLflow URIs."""
    training_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_log = pd.DataFrame([{
        "model_name": model_name,
        "training_date": training_date,
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1_score": metrics.get("f1_score"),
        "run_id": run_id,
        "artifact_uri": artifact_uri
    }])
    df_log.to_sql(table_name, engine, if_exists="append", index=False)


class ModelTrainer:
    def __init__(self):
        config = ModelTrainerConfig()
        self.models_with_params = config.model_params
        self.hyp_parameter_scores = config.parameter_scoring
        self.engine = create_engine(DATABASE_URL)

    def model_training_with_mlflow(self, x_train, y_train, x_test, y_test):
        try:
            mlflow.set_experiment(experiment_name=f"NASA NEO Training on {ml_run_name}")

            best_model_info = {
                "recall": -np.inf,
                "model": None,
                "name": None,
                "params": None,
                "metrics": None,
                "y_pred_test": None,
                "run_id": None,
                "artifact_uri": None,
            }

            for model_name, model, param_grid in self.models_with_params:
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    scoring=self.hyp_parameter_scores,
                    refit='average_precision',
                    n_jobs=-1,
                    verbose=1,
                    n_iter=10,
                    random_state=42,
                )
                search.fit(x_train, y_train)

                tuned_model = search.best_estimator_
                tuned_params = search.best_params_
                y_pred_test = tuned_model.predict(x_test)
                acc, prec, recall, f1 = get_mlflow_metrics(y_test, y_pred_test)

                # Log all tuned models to MLflow
                with mlflow.start_run(run_name=f"Tuned | {model_name}") as run:
                    mlflow.log_params(tuned_params)
                    mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": recall, "f1_score": f1})

                    with tempfile.TemporaryDirectory() as tmpdir:
                        model_file = os.path.join(tmpdir, f"{model_name.replace(' ', '_')}.model")
                        _save_model_to_file(tuned_model, model_file)
                        mlflow.log_artifact(model_file, artifact_path=f"models/{model_name.replace(' ', '_')}")

                        try:
                            input_example = np.array(x_train[:5])
                            input_path = os.path.join(tmpdir, "input_example.npy")
                            np.save(input_path, input_example)
                            mlflow.log_artifact(input_path, artifact_path=f"models/{model_name.replace(' ', '_')}")
                        except Exception:
                            pass

                        try:
                            sig = infer_signature(x_train[:50], tuned_model.predict(x_train[:50]))
                            sig_path = os.path.join(tmpdir, "signature.txt")
                            with open(sig_path, "w") as f:
                                f.write(str(sig))
                            mlflow.log_artifact(sig_path, artifact_path=f"models/{model_name.replace(' ', '_')}")
                        except Exception:
                            pass

                # Update best model by recall
                if recall > best_model_info["recall"]:
                    best_model_info.update({
                        "recall": recall,
                        "model": tuned_model,
                        "name": model_name,
                        "params": tuned_params,
                        "metrics": {"accuracy": acc, "precision": prec, "recall": recall, "f1_score": f1},
                        "y_pred_test": y_pred_test,
                        "run_id": run.info.run_id,
                        "artifact_uri": run.info.artifact_uri,
                    })

            # Log best model metrics to MLflow & database
            if best_model_info["model"] is not None:
                best_model = best_model_info["model"]
                y_pred_best = best_model_info["y_pred_test"]
                acc, prec, recall, f1 = get_mlflow_metrics(y_test, y_pred_best)
                cm = confusion_matrix(y_test, y_pred_best)
                cr = classification_report(y_test, y_pred_best)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                best_model_name = f"best_{best_model_info['name'].replace(' ', '_')}_{timestamp}"

                with mlflow.start_run(run_name=f"Best Model | {best_model_info['name']}") as best_run:
                    mlflow.log_params(best_model_info["params"] if best_model_info["params"] else {})
                    mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": recall, "f1_score": f1})

                    with tempfile.TemporaryDirectory() as tmpdir:
                        model_file = os.path.join(tmpdir, f"{best_model_name}.model")
                        _save_model_to_file(best_model, model_file)
                        mlflow.log_artifact(model_file, artifact_path=f"models/{best_model_name}")

                        cm_path = os.path.join(tmpdir, "confusion_matrix.txt")
                        with open(cm_path, "w") as f:
                            f.write(str(cm))
                        mlflow.log_artifact(cm_path, artifact_path=f"models/{best_model_name}")

                        cr_path = os.path.join(tmpdir, "classification_report.txt")
                        with open(cr_path, "w") as f:
                            f.write(str(cr))
                        mlflow.log_artifact(cr_path, artifact_path=f"models/{best_model_name}")

                        try:
                            input_example = np.array(x_train[:5])
                            input_example_path = os.path.join(tmpdir, "input_example.npy")
                            np.save(input_example_path, input_example)
                            mlflow.log_artifact(input_example_path, artifact_path=f"models/{best_model_name}")
                        except Exception:
                            pass

                        try:
                            signature = infer_signature(x_train[:50], best_model.predict(x_train[:50]))
                            sig_path = os.path.join(tmpdir, "signature.txt")
                            with open(sig_path, "w") as f:
                                f.write(str(signature))
                            mlflow.log_artifact(sig_path, artifact_path=f"models/{best_model_name}")
                        except Exception:
                            pass

                logging.info(f"Best model '{best_model_info['name']}' logged in MLflow as '{best_model_name}'.")

                # Log best model metrics + run info to DB
                log_training_to_db(
                    model_name=best_model_name,
                    metrics={"accuracy": acc, "precision": prec, "recall": recall, "f1_score": f1},
                    run_id=best_run.info.run_id,
                    artifact_uri=best_run.info.artifact_uri,
                    engine=self.engine
                )

        except Exception as e:
            logging.error("Got an error while training the model with MLflow")
            raise CustomException(e, sys)
