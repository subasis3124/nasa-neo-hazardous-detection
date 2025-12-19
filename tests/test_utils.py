from unittest.mock import patch, MagicMock
import pandas as pd
import base64
import pickle
# Import functions to test
from src.utils import utils


# ----------------- TEST create_engine_for_database -----------------
def test_create_engine_for_database_returns_engine():
    engine = utils.create_engine_for_database("user", "pass", "host", "5432", "db")
    from sqlalchemy.engine.base import Engine
    assert isinstance(engine, Engine)


# ----------------- TEST read_data_from_pg -----------------
@patch("pandas.read_sql")
@patch("src.utils.utils.create_engine")
def test_read_data_from_pg(mock_create_engine, mock_read_sql):
    mock_create_engine.return_value = MagicMock()
    mock_df = pd.DataFrame({"col1": [1, 2]})
    mock_read_sql.return_value = mock_df

    df = utils.read_data_from_pg("user", "pass", "host", "5432", "db", "table")
    assert isinstance(df, pd.DataFrame)
    assert df.equals(mock_df)


# ----------------- TEST fetch_data -----------------
@patch("requests.get")
def test_fetch_data_success(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": "ok"}
    mock_get.return_value = mock_response

    result = utils.fetch_data("2025-09-01", "2025-09-02", "fake_api")
    assert result == {"data": "ok"}


@patch("requests.get")
def test_fetch_data_failure(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_get.return_value = mock_response

    result = utils.fetch_data("2025-09-01", "2025-09-02", "fake_api")
    # It prints instead of raising, so should return None
    assert result is None


# ----------------- TEST get_mlflow_metrics -----------------
def test_get_mlflow_metrics():
    actual = [1, 0, 1, 1]
    predicted = [1, 0, 0, 1]
    acc, prec, rec, f1 = utils.get_mlflow_metrics(actual, predicted)
    assert 0 <= acc <= 1
    assert 0 <= prec <= 1
    assert 0 <= rec <= 1
    assert 0 <= f1 <= 1


# ----------------- TEST load_artifact_from_db -----------------
@patch("joblib.load")
@patch("pandas.read_sql")
def test_load_artifact_from_db(mock_read_sql, mock_joblib_load):
    # Create a fake base64 artifact
    dummy_obj = {"a": 1}
    serialized = base64.b64encode(pickle.dumps(dummy_obj)).decode('utf-8')
    mock_df = pd.DataFrame({"artifact": [serialized]})
    mock_read_sql.return_value = mock_df
    mock_joblib_load.return_value = dummy_obj

    result = utils.load_artifact_from_db("table", MagicMock())
    assert result == dummy_obj


# ----------------- TEST extract_best_model -----------------
@patch("src.utils.utils.create_engine")
@patch("src.utils.utils.mlflow.artifacts.download_artifacts")
@patch("src.utils.utils.pd.read_sql")
@patch("src.utils.utils.joblib.load")
def test_extract_best_model(mock_joblib_load, mock_read_sql, mock_mlflow_download, mock_create_engine):
    # Mock engine to avoid real DB call
    mock_create_engine.return_value = MagicMock()

    # Mock dataframe returned from SQL
    df = pd.DataFrame({
        "artifact_uri": ["/fake/path"],
        "model_name": ["DummyModel"]
    })
    mock_read_sql.return_value = df

    mock_mlflow_download.return_value = "/tmp/fake.model"
    mock_joblib_load.return_value = "model_object"

    result = utils.extract_best_model()
    assert result == "model_object"
