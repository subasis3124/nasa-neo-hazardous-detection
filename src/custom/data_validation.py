import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timezone
from src.constants.entity import DataTransformationEntity
from dotenv import load_dotenv
import uuid
import os
load_dotenv()


# Optional: chi-square p-value if SciPy exists; otherwise we still compute chi2 statistic.
try:
    from scipy.stats import chisquare  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


class DataDriftDetector:
    """
    Drift detection using:
      - PSI for numeric (quantile bins from BASELINE only)
      - PSI for categorical (category union)
      - Chi-square test for categorical (if SciPy available)
    """
    def __init__(self, baseline_df: pd.DataFrame, new_df: pd.DataFrame, db_uri: str):
        self.base = baseline_df.copy()
        self.new = new_df.copy()
        self.db_uri = db_uri
        self.results = []
        self.run_id = str(uuid.uuid4())
        self.created_at = datetime.now(timezone.utc)

    # -------------------------- Public API --------------------------

    def detect_drift(
        self,
        numeric_buckets: int = 10,
        psi_warn: float = 0.10,
        psi_alert: float = 0.20,
        chi2_alpha: float = 0.05,
        include: list | None = None,
        exclude: list | None = None
    ) -> pd.DataFrame:
        base, new = self._align_columns(include, exclude)
        num_cols = base.select_dtypes(include=[np.number]).columns
        cat_cols = base.select_dtypes(include=["object", "category", "bool"]).columns

        # Numeric features: PSI with bins defined on BASELINE quantiles
        for col in num_cols:
            b = base[col].dropna()
            n = new[col].dropna()
            if len(b) == 0 or len(n) == 0:
                continue

            psi = self._psi_numeric(b, n, buckets=numeric_buckets)
            status = self._status_from_psi(psi, psi_warn, psi_alert)

            self._append_result(
                feature=col,
                feature_type="numeric",
                psi=psi,
                chi2_stat=None,
                chi2_p_value=None,
                chi2_alpha=chi2_alpha,
                drift_status=status,
                baseline_count=len(b),
                new_count=len(n)
            )

        # Categorical features: PSI on category proportions; optional chi-square
        for col in cat_cols:
            b = base[col].astype("object").dropna()
            n = new[col].astype("object").dropna()
            if len(b) == 0 or len(n) == 0:
                continue

            psi_cat = self._psi_categorical(b, n)
            chi2_stat, chi2_p = self._chi2_categorical(b, n) if _HAS_SCIPY else (self._chi2_stat_only(b, n), None)

            status = self._status_from_psi(psi_cat, psi_warn, psi_alert)
            # If chi2 is available and significant, escalate the status
            if chi2_p is not None and chi2_p < chi2_alpha and status != "High Drift":
                status = "High Drift (chi2)"

            self._append_result(
                feature=col,
                feature_type="categorical",
                psi=psi_cat,
                chi2_stat=chi2_stat,
                chi2_p_value=chi2_p,
                chi2_alpha=chi2_alpha,
                drift_status=status,
                baseline_count=len(b),
                new_count=len(n)
            )

        return pd.DataFrame(self.results)

    def push_to_db(self, table_name: str = "neo_data_drift") -> None:
        df = pd.DataFrame(self.results)
        if df.empty:
            print("No drift results to push.")
            return
        engine = create_engine(self.db_uri)
        df.to_sql(table_name, engine, if_exists="append", index=False)
        print(f"Drift results pushed to table: {table_name}")

    # -------------------------- Internals --------------------------

    def _align_columns(self, include, exclude):
        base = self.base.copy()
        new = self.new.copy()
        common = base.columns.intersection(new.columns)

        if include is not None:
            common = pd.Index([c for c in common if c in include])
        if exclude is not None:
            common = pd.Index([c for c in common if c not in exclude])

        return base[common], new[common]

    @staticmethod
    def _status_from_psi(psi: float, warn: float, alert: float) -> str:
        if psi > alert:
            return "High Drift"
        if psi > warn:
            return "Moderate Drift"
        return "No Drift"

    def _append_result(
        self,
        feature: str,
        feature_type: str,
        psi: float | None,
        chi2_stat: float | None,
        chi2_p_value: float | None,
        chi2_alpha: float | None,
        drift_status: str,
        baseline_count: int,
        new_count: int
    ):
        self.results.append({
            "run_id": self.run_id,
            "feature": feature,
            "feature_type": feature_type,
            "method": "psi" if feature_type == "numeric" else "psi+chi2" if _HAS_SCIPY else "psi",
            "psi": float(psi) if psi is not None else None,
            "chi2_stat": float(chi2_stat) if chi2_stat is not None else None,
            "chi2_p_value": float(chi2_p_value) if chi2_p_value is not None else None,
            "chi2_alpha": float(chi2_alpha) if chi2_alpha is not None else None,
            "drift_status": drift_status,
            "baseline_count": int(baseline_count),
            "new_count": int(new_count),
            "created_at": self.created_at,  # UTC timestamp
        })

    @staticmethod
    def _psi_numeric(base: pd.Series, new: pd.Series, buckets: int = 10) -> float:
        """
        PSI for numeric features using BASELINE quantile bins.
        """
        eps = 1e-12
        # Quantile edges on baseline only
        quantiles = np.linspace(0, 1, buckets + 1)
        edges = np.unique(np.quantile(base, quantiles))
        # guard if many duplicates in quantiles -> fallback to min/max
        if len(edges) < 3:
            edges = np.array([base.min(), np.median(base), base.max()])  # 2 bins fallback

        base_counts, _ = np.histogram(base, bins=edges)
        new_counts, _ = np.histogram(new, bins=edges)

        base_props = base_counts / max(base_counts.sum(), 1)
        new_props = new_counts / max(new_counts.sum(), 1)

        base_props = np.clip(base_props, eps, None)
        new_props = np.clip(new_props, eps, None)

        return float(np.sum((base_props - new_props) * np.log(base_props / new_props)))

    @staticmethod
    def _psi_categorical(base: pd.Series, new: pd.Series) -> float:
        """
        PSI for categorical features using category union.
        """
        eps = 1e-12
        cats = sorted(set(base.unique()).union(set(new.unique())))
        b_counts = base.value_counts().reindex(cats, fill_value=0).astype(float)
        n_counts = new.value_counts().reindex(cats, fill_value=0).astype(float)

        b_props = b_counts / max(b_counts.sum(), 1.0)
        n_props = n_counts / max(n_counts.sum(), 1.0)

        b_props = np.clip(b_props.values, eps, None)
        n_props = np.clip(n_props.values, eps, None)

        return float(np.sum((b_props - n_props) * np.log(b_props / n_props)))

    @staticmethod
    def _chi2_stat_only(base: pd.Series, new: pd.Series) -> float:
        """
        Chi-square statistic without p-value (no SciPy).
        """
        cats = sorted(set(base.unique()).union(set(new.unique())))
        b = base.value_counts().reindex(cats, fill_value=0).astype(float)
        n = new.value_counts().reindex(cats, fill_value=0).astype(float)

        total_new = n.sum()
        # Expected under H0: proportions from baseline applied to new's total
        expected = (b / max(b.sum(), 1.0)) * total_new
        # Avoid division by zero
        mask = expected > 0
        chi2 = float((((n[mask] - expected[mask]) ** 2) / expected[mask]).sum())
        return chi2

    @staticmethod
    def _chi2_categorical(base: pd.Series, new: pd.Series) -> tuple[float, float]:
        """
        Chi-square goodness-of-fit: compare NEW to BASELINE proportions.
        Requires SciPy; returns (statistic, p_value).
        """
        cats = sorted(set(base.unique()).union(set(new.unique())))
        b = base.value_counts().reindex(cats, fill_value=0).astype(float)
        n = new.value_counts().reindex(cats, fill_value=0).astype(float)

        total_new = n.sum()
        expected = (b / max(b.sum(), 1.0)) * total_new
        # chisquare expects observed and expected arrays
        stat, p = chisquare(f_obs=n.values, f_exp=np.clip(expected.values, 1e-12, None))
        return float(stat), float(p)


# -------------------------- Example usage --------------------------
if __name__ == "__main__":
    print("[INFO] Main function started")

    # --------------------- Load drop columns ---------------------
    data_transform_entity = DataTransformationEntity()
    drop_columns = data_transform_entity.DROP_COLUMNS
    print(f"[INFO] Drop columns loaded: {drop_columns}")

    # --------------------- Create DB engine ---------------------
    db_uri = os.getenv("DATABASE_URL")
    engine = create_engine(url=db_uri)
    print("[INFO] Database engine created")

    # --------------------- Queries ---------------------
    # Baseline: data before the last model training
    query_for_baseline = """
        WITH last_trained_model_date AS (
            SELECT MAX(training_date::date) AS model_last_trained
            FROM model_training_logs
        )
        SELECT *
        FROM train_neo tn
        CROSS JOIN last_trained_model_date ltmd
        WHERE tn.close_approach_date::date < ltmd.model_last_trained;
    """

    # New/current: only data after last training
    query_for_new_data = """
        WITH last_trained_model_date AS (
            SELECT MAX(training_date::date) AS model_last_trained
            FROM model_training_logs
        )
        SELECT *
        FROM train_neo tn
        CROSS JOIN last_trained_model_date ltmd
        WHERE tn.close_approach_date::date >= ltmd.model_last_trained;
    """

    # --------------------- Load data ---------------------
    baseline_df = pd.read_sql(query_for_baseline, engine)
    print(f"[INFO] Baseline dataframe loaded with shape: {baseline_df.shape}")

    new_df = pd.read_sql(query_for_new_data, engine)
    print(f"[INFO] New dataframe loaded with shape: {new_df.shape}")

    # --------------------- Feature Engineering ---------------------
    print("[INFO] Performing feature engineering: diameter_range")
    for df in [baseline_df, new_df]:
        df['diameter_range'] = df['max_diameter_km'] - df['min_diameter_km']

    # --------------------- Drop unnecessary columns ---------------------
    print(f"[INFO] Dropping not required columns: {drop_columns}")
    baseline_df.drop(columns=drop_columns, inplace=True, errors='ignore')
    new_df.drop(columns=drop_columns, inplace=True, errors='ignore')

    # --------------------- Initialize drift detector ---------------------
    detector = DataDriftDetector(baseline_df=baseline_df, new_df=new_df, db_uri=db_uri)
    print("[INFO] DataDriftDetector initialized")

    # --------------------- Detect drift ---------------------
    print("[INFO] Running drift detection")
    report = detector.detect_drift()
    print("[INFO] Drift report generated:")
    print(report.head())  # print first few rows for quick logging

    # --------------------- Push results to database ---------------------
    detector.push_to_db(table_name="neo_data_drift")
    print("[INFO] Drift results pushed to database table 'neo_data_drift'")
