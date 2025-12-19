import os
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from src.constants.entity import AsteroidRiskAnalysisEntity
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

TODAY = datetime.today().date()
DB_URL = os.getenv("DATABASE_URL")
EPS = 1e-12

QUERY = """WITH target_date AS (
    SELECT created_at
    FROM neo_data_drift
    ORDER BY created_at DESC
    LIMIT 1
)
SELECT
    tn.id,
    tn.min_diameter_km,
    tn.max_diameter_km,
    tn.relative_velocity_kps,
    tn.miss_distance_km,
    pt.probability_being_truely_hazardous,
    tn.is_potentially_hazardous
FROM train_neo tn
JOIN prediction_table pt
    ON pt.id::bigint = tn.id
WHERE close_approach_date::date >= (
        SELECT created_at::date - INTERVAL '7 days'
        FROM target_date
    )
  AND close_approach_date::date <= (
        SELECT created_at::date
        FROM target_date
    );
"""

engine = create_engine(url=DB_URL)


class AsteroidRiskAnalysis:
    def __init__(self):
        self.risk_entity = AsteroidRiskAnalysisEntity()
        self.risk_analysis_table_name = self.risk_entity.RISK_ANALYSIS_TABLE_NAME
        self.cols_show = self.risk_entity.COLS_SHOWS
        self.manual_weights = self.risk_entity.MANUAL_WEIGHTS

    def compute_physical_metrics(
        self,
        df: pd.DataFrame,
        density: float = 2600.0,
        diameter_unit: str = "m",
        velocity_unit: str = "m/s",
    ) -> pd.DataFrame:
        """
        Compute avg diameter (davg), mass (kg), and impact energy (J).
        - dmin, dmax expected in supplied diameter_unit (default meters)
        - rel_velocity expected in velocity_unit (default m/s)
        Adds columns: 'davg_m', 'mass_kg', 'impact_energy_j'
        """
        df = df.copy()

        # convert input units to meters and m/s
        if diameter_unit == "km":
            conv_d = 1000.0
        elif diameter_unit == "m":
            conv_d = 1.0
        else:
            raise ValueError("diameter_unit must be 'm' or 'km'")

        if velocity_unit == "km/s":
            conv_v = 1000.0
        elif velocity_unit == "m/s":
            conv_v = 1.0
        else:
            raise ValueError("velocity_unit must be 'm/s' or 'km/s'")

        dmin_m = df["min_diameter_km"].astype(float) * conv_d
        dmax_m = df["max_diameter_km"].astype(float) * conv_d
        davg_m = (dmin_m + dmax_m) / 2.0

        # volume of sphere, radius = davg/2
        radius = davg_m / 2.0
        volume = (4.0 / 3.0) * np.pi * radius**3
        mass_kg = volume * density

        # velocity (m/s)
        v = df["relative_velocity_kps"].astype(float) * conv_v

        # kinetic energy (J)
        impact_energy_j = 0.5 * mass_kg * v**2

        df["davg_m"] = davg_m
        df["mass_kg"] = mass_kg
        df["impact_energy_j"] = impact_energy_j

        return df

    def normalize_energy_and_moid(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, MinMaxScaler]:
        """
        Normalize impact energy to [0,1] and transform miss_distance_km into an
        inverse MOID risk measure (higher => more risky) and normalize to [0,1].
        Returns modified df and the fitted scaler for reproducibility.
        """
        df = df.copy()

        # Energy normalization
        energy = (
            df["impact_energy_j"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
            .values.reshape(-1, 1)
        )
        scaler_energy = MinMaxScaler(feature_range=(0, 1))
        Enorm = scaler_energy.fit_transform(energy).reshape(-1)

        # MOID inverse: smaller distance => higher risk
        miss_km = df["miss_distance_km"].astype(float).replace([np.inf, -np.inf], np.nan)
        miss_km = miss_km.fillna(miss_km.max())
        inv_moid = 1.0 / (miss_km.values + EPS)
        inv_moid = inv_moid.reshape(-1, 1)
        scaler_moid = MinMaxScaler(feature_range=(0, 1))
        Rmoid_norm = scaler_moid.fit_transform(inv_moid).reshape(-1)

        df["Enorm"] = Enorm
        df["Rmoid_norm"] = Rmoid_norm

        return df, (scaler_energy, scaler_moid)

    def prepare_probabilities(
        self,
        df: pd.DataFrame,
        classifier: Optional[object] = None,
        feature_cols: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Ensure df contains a 'Phazardous' column (0..1).
        - If already present, leave as is (clamped to [0,1]).
        - Else if classifier is provided, predict_proba using feature_cols.
        """
        df = df.copy()

        if "probability_being_truely_hazardous" in df.columns:
            df["Phazardous"] = (
                df["probability_being_truely_hazardous"]
                .astype(float)
                .clip(0.0, 1.0)
            )
            return df

        if classifier is None or feature_cols is None:
            raise ValueError(
                "Either provide 'Phazardous' column in df or pass classifier and feature_cols"
            )

        X = df[feature_cols].astype(float).values
        probs = classifier.predict_proba(X)[:, 1]
        df["Phazardous"] = np.clip(probs, 0.0, 1.0)

        return df

    def compute_weights(
        self,
        df_norm: pd.DataFrame,
        method: str = "manual",
        manual_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
        label_col: str = "is_hazardous",
    ) -> Tuple[float, float, float]:
        """
        Compute weights (w1, w2, w3) for Enorm, Rmoid_norm, Phazardous.
        - method='manual' : uses manual_weights (normalized automatically)
        - method='data-driven' : uses logistic regression coefficients trained to predict label_col.
        Returns normalized absolute coefficients as weights (sum to 1).
        """
        if method == "manual":
            w = np.array(manual_weights, dtype=float)
            if np.any(w < 0):
                raise ValueError("manual_weights must be non-negative")

            w_sum = w.sum()
            if w_sum == 0:
                raise ValueError("manual_weights cannot sum to zero")

            return tuple((w / w_sum).tolist())

        if method == "data-driven":
            if label_col not in df_norm.columns:
                raise ValueError(
                    f"label_col '{label_col}' not found in dataframe for data-driven weighting"
                )

            X = (
                df_norm[["Enorm", "Rmoid_norm", "Phazardous"]]
                .astype(float)
                .fillna(0.0)
                .values
            )
            y = df_norm[label_col].astype(int).values

            clf = LogisticRegression(solver="lbfgs", max_iter=1000)
            clf.fit(X, y)

            coefs = clf.coef_.reshape(-1)
            abs_coefs = np.abs(coefs)

            if abs_coefs.sum() == 0:
                return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

            weights = abs_coefs / abs_coefs.sum()
            return tuple(weights.tolist())

        raise ValueError("method must be 'manual' or 'data-driven'")

    @staticmethod
    def categorize_risk(score: float) -> str:
        """Map normalized risk score to category."""
        if score < 0:
            score = 0.0
        if score <= 0.2:
            return "Very Low Risk"
        elif score <= 0.5:
            return "Low Risk"
        elif score <= 0.8:
            return "Medium Risk"
        return "High Risk"

    def compute_composite_score(
        self,
        df: pd.DataFrame,
        weights: Tuple[float, float, float],
    ) -> pd.DataFrame:
        """
        Apply weights to normalized features and compute composite Risk Score,
        then normalize to [0,1]. Adds columns: 'RiskScore_raw', 'RiskScorenorm', 'RiskCategory'.
        """
        df = df.copy()
        w1, w2, w3 = weights
        df["RiskScore_raw"] = (
            w1 * df["Enorm"] + w2 * df["Rmoid_norm"] + w3 * df["Phazardous"]
        )

        rs = df["RiskScore_raw"].astype(float)
        min_r, max_r = rs.min(), rs.max()

        if np.isclose(max_r, min_r):
            df["RiskScorenorm"] = 0.0
        else:
            df["RiskScorenorm"] = (rs - min_r) / (max_r - min_r)

        df["RiskCategory"] = df["RiskScorenorm"].apply(self.categorize_risk)

        return df

    def initiate_asteroid_risk_analysis(self, data: pd.DataFrame):
        data_def_1 = self.compute_physical_metrics(
            df=data, density=2600.0, diameter_unit="km", velocity_unit="km/s"
        )
        data_def_2, _ = self.normalize_energy_and_moid(df=data_def_1)
        data_def_3 = self.prepare_probabilities(data_def_2)

        manual_w = self.compute_weights(
            df_norm=data_def_3, method="manual", manual_weights=self.manual_weights
        )
        dd_w = self.compute_weights(
            df_norm=data_def_3,
            method="data-driven",
            label_col="is_potentially_hazardous",
        )

        manual_data = self.compute_composite_score(df=data_def_3, weights=manual_w)
        data_dd = self.compute_composite_score(df=data_def_3, weights=dd_w)

        return manual_data[self.cols_show], data_dd[self.cols_show]

    def merge_asteroid_manual_dd_data(
        self, manual_data: pd.DataFrame, data_dd: pd.DataFrame
    ) -> pd.DataFrame:
        fin_data = manual_data[self.cols_show].copy()
        fin_data.rename(
            columns={
                "RiskScorenorm": "RiskScorenormManual",
                "RiskCategory": "RiskCategoryManual",
            },
            inplace=True,
        )
        fin_data["RiskScorenormDDriven"] = data_dd["RiskScorenorm"]
        fin_data["RiskCategoryDDriven"] = data_dd["RiskCategory"]
        fin_data["risk_analysed_on_date"] = TODAY
        return fin_data

    def push_final_data_to_db(self, fin_data: pd.DataFrame) -> None:
        if fin_data.empty:
            print("The final data is empty...")
            return
        fin_data.to_sql(
            self.risk_analysis_table_name, engine, if_exists="append", index=False
        )
        print(
            f"[INFO] Final dataframe pushed to DB: {self.risk_analysis_table_name}"
        )


if __name__ == "__main__":
    risk_analysis = AsteroidRiskAnalysis()
    df = pd.read_sql(QUERY, engine)
    print(f"[INFO] The shape of the dataframe is {df.shape}")

    manual_w_data, dd_w_data = risk_analysis.initiate_asteroid_risk_analysis(data=df)
    print(
        f"[INFO] Shape of manually weighted data {manual_w_data.shape} "
        f"and data-driven weighted data {dd_w_data.shape}"
    )

    final_data = risk_analysis.merge_asteroid_manual_dd_data(
        manual_data=manual_w_data, data_dd=dd_w_data
    )
    print(f"[INFO] Final dataframe shape: {final_data.shape}")
    print(f"[INFO] Final dataframe columns: {final_data.columns}")

    risk_analysis.push_final_data_to_db(fin_data=final_data)
    print("[INFO] Script execution completed")
