import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# --- Models ---
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

# Try CatBoost; skip if not installed
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CatBoostRegressor = None  # type: ignore
    CATBOOST_AVAILABLE = False


# ========================
# CONFIG
# ========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "enhanced_crop_dataset.csv")
TARGET = "yield_per_hectare"
NUMERIC_FEATURES = [
    "temp_max", "temp_min", "rainfall", "humidity", "ph", "altitude", "crop_year"
]
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_OUT_PATH = "crop_yield_recommender_ensemble.pkl"


@dataclass
class TrainedModels:
    xgb: Optional[XGBRegressor]
    rf: Optional[RandomForestRegressor]
    et: Optional[ExtraTreesRegressor]
    cat: Optional["CatBoostRegressor"] # type: ignore
    feature_names: List[str]
    crop_name: str
    metrics: Dict[str, float]
    weights: Dict[str, float]


class CropRecommenderEnsemble:
    def __init__(self, csv_path: str):
        print("[1/10] Loading dataset...")
        self.df = pd.read_csv(csv_path, low_memory=False)
        print(f"   ✅ Dataset shape: {self.df.shape}")

        print("[2/10] Cleaning columns and types...")
        self.df.columns = (
            self.df.columns.str.strip().str.lower().str.replace(" ", "_")
        )

        # Fill categorical safely
        for col in ["state", "season", "crop", "district_name"]:
            if col in self.df.columns:
                self.df.loc[:, col] = self.df[col].astype(str).fillna("Unknown")

        # Fill numeric with median
        numeric_cols = [
            "temp_max", "temp_min", "rainfall", "humidity", "ph", "altitude",
            "production_qty", "area", "yield_ratio", "crop_year", "area_",
            "production_", "yield_per_hectare"
        ]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df.loc[:, col] = pd.to_numeric(self.df[col], errors="coerce")
                med = self.df[col].median()
                self.df.loc[:, col] = self.df[col].fillna(med)

        before = self.df.shape[0]
        self.df.drop_duplicates(inplace=True)
        print(f"   ✅ Dropped duplicates: {before - self.df.shape[0]} rows removed.")
        print(f"   ✅ Cleaned dataset shape: {self.df.shape}")

        self.models: Optional[TrainedModels] = None

    # ---------- Helpers ----------
    def _split(self, df_crop: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        df_crop = df_crop.dropna(subset=[TARGET]).copy()
        X = df_crop[NUMERIC_FEATURES].copy()
        y = df_crop[TARGET].astype(float).values

        # Ensure all features exist and fill NA with column medians
        for c in NUMERIC_FEATURES:
            if c not in X.columns:
                X[c] = np.nan
        X = X[NUMERIC_FEATURES].fillna(X.median(numeric_only=True))

        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        return X_train, X_test, y_train, y_test

    @staticmethod
    def _rmse(y_true, y_pred) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def _clip_nonnegative(arr: np.ndarray) -> np.ndarray:
        return np.maximum(arr, 0.0)

    @staticmethod
    def _weights_from_rmses(rmse_dict: Dict[str, float]) -> Dict[str, float]:
        # inverse-rmse weighting; skip models with nan/inf
        inv = {k: (1.0 / v) for k, v in rmse_dict.items() if np.isfinite(v) and v > 0}
        s = sum(inv.values())
        if s == 0:
            # fallback to uniform
            n = len(inv) if len(inv) > 0 else 1
            return {k: 1.0 / n for k in inv}
        return {k: v / s for k, v in inv.items()}

    # ---------- Training ----------
    def train(self, crop_name: str) -> TrainedModels:
        print("[3/10] Filtering data by crop...")
        df_crop = self.df[self.df["crop"].str.lower() == crop_name.lower()]
        print(f"   ✅ Crop '{crop_name}': {df_crop.shape[0]} rows found.")
        if df_crop.empty:
            raise ValueError(f"No rows found for crop '{crop_name}'.")

        # Ensure required columns exist
        missing_features = [c for c in NUMERIC_FEATURES if c not in df_crop.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        print("[4/10] Splitting train/test...")
        X_train, X_test, y_train, y_test = self._split(df_crop)
        print(f"   ✅ Train: {X_train.shape}, Test: {X_test.shape}")

        metrics: Dict[str, float] = {}

        # ---------------------------
        # XGBoost
        # ---------------------------
        print("[5/10] Training XGBoost...")
        xgb = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            tree_method="hist",
        )
        xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        xgb_pred = xgb.predict(X_test)
        xgb_pred = self._clip_nonnegative(xgb_pred)
        rmse_xgb = self._rmse(y_test, xgb_pred)
        metrics["rmse_xgb"] = rmse_xgb
        print(f"   ✅ XGBoost RMSE: {rmse_xgb:.3f}")

        # ---------------------------
        # Random Forest
        # ---------------------------
        print("[6/10] Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_pred = self._clip_nonnegative(rf_pred)
        rmse_rf = self._rmse(y_test, rf_pred)
        metrics["rmse_rf"] = rmse_rf
        print(f"   ✅ Random Forest RMSE: {rmse_rf:.3f}")

        # ---------------------------
        # Extra Trees
        # ---------------------------
        print("[7/10] Training Extra Trees...")
        et = ExtraTreesRegressor(
            n_estimators=600,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        et.fit(X_train, y_train)
        et_pred = et.predict(X_test)
        et_pred = self._clip_nonnegative(et_pred)
        rmse_et = self._rmse(y_test, et_pred)
        metrics["rmse_extratrees"] = rmse_et
        print(f"   ✅ Extra Trees RMSE: {rmse_et:.3f}")

        # ---------------------------
        # CatBoost (optional)
        # ---------------------------
        if CATBOOST_AVAILABLE:
            print("[8/10] Training CatBoost...")
            cat = CatBoostRegressor(
                depth=8,
                learning_rate=0.05,
                n_estimators=600,
                loss_function="RMSE",
                random_state=RANDOM_STATE,
                verbose=False
            )
            cat.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
            cat_pred = cat.predict(X_test)
            cat_pred = self._clip_nonnegative(cat_pred)
            rmse_cat = self._rmse(y_test, cat_pred)
            metrics["rmse_catboost"] = rmse_cat
            print(f"   ✅ CatBoost RMSE: {rmse_cat:.3f}")
        else:
            print("[8/10] CatBoost not installed — skipping.")
            cat = None
            cat_pred = None

        # ---------------------------
        # Weighted Ensemble
        # ---------------------------
        print("[9/10] Ensembling predictions (inverse-RMSE weights)...")
        preds_dict = {
            "xgb": xgb_pred,
            "rf": rf_pred,
            "et": et_pred,
        }
        if cat_pred is not None:
            preds_dict["cat"] = cat_pred

        # Build weights from available models
        rmse_for_weights = {
            "xgb": metrics["rmse_xgb"],
            "rf": metrics["rmse_rf"],
            "et": metrics["rmse_extratrees"],
        }
        if cat_pred is not None:
            rmse_for_weights["cat"] = metrics["rmse_catboost"]

        weights = self._weights_from_rmses(rmse_for_weights)  # keys: xgb, rf, et, [cat]
        # Normalize just to be extra safe
        ws = sum(weights.values())
        if ws == 0:
            # fallback uniform
            k = len(weights)
            weights = {k_: 1.0 / k for k_ in weights}
        else:
            weights = {k_: v / ws for k_, v in weights.items()}

        # Weighted sum
        ensemble_pred = np.zeros_like(list(preds_dict.values())[0], dtype=float)
        for key, pred in preds_dict.items():
            w = weights.get(key, 0.0)
            ensemble_pred += w * pred
        ensemble_pred = self._clip_nonnegative(ensemble_pred)

        rmse_ensemble = self._rmse(y_test, ensemble_pred)
        metrics["rmse_ensemble"] = rmse_ensemble
        print(f"   ✅ Ensemble RMSE: {rmse_ensemble:.3f}")

        print("[10/10] Training complete.\n")
        self.models = TrainedModels(
            xgb=xgb,
            rf=rf,
            et=et,
            cat=cat,
            feature_names=NUMERIC_FEATURES.copy(),
            crop_name=crop_name,
            metrics=metrics,
            weights=weights
        )
        return self.models

    # ---------- Prediction ----------
    def predict(self, features: Dict[str, float]) -> Dict[str, float]:
        if self.models is None:
            raise RuntimeError("Models not trained. Call .train(crop_name) first.")

        # Build feature vector
        x = np.array([[features.get(k, np.nan) for k in self.models.feature_names]], dtype=float)
        # Fill any NaNs with 0 (you could also persist medians from training if needed)
        if np.isnan(x).any():
            for j in range(x.shape[1]):
                if np.isnan(x[0, j]):
                    x[0, j] = 0.0

        out: Dict[str, float] = {}

        # Individual predictions (clipped to nonnegative)
        if self.models.xgb is not None:
            out["xgboost"] = float(max(0.0, self.models.xgb.predict(x)[0]))
        if self.models.rf is not None:
            out["random_forest"] = float(max(0.0, self.models.rf.predict(x)[0]))
        if self.models.et is not None:
            out["extra_trees"] = float(max(0.0, self.models.et.predict(x)[0]))
        if self.models.cat is not None:
            out["catboost"] = float(max(0.0, self.models.cat.predict(x)[0]))

        # Weighted ensemble using stored weights (only for available models)
        denom = 0.0
        ens = 0.0
        for key, w in self.models.weights.items():
            if key == "xgb" and "xgboost" in out:
                ens += w * out["xgboost"]; denom += w
            elif key == "rf" and "random_forest" in out:
                ens += w * out["random_forest"]; denom += w
            elif key == "et" and "extra_trees" in out:
                ens += w * out["extra_trees"]; denom += w
            elif key == "cat" and "catboost" in out:
                ens += w * out["catboost"]; denom += w
        out["ensemble"] = float(max(0.0, ens / denom if denom > 0 else np.mean(list(out.values()))))

        return out

    # ---------- Text Tips ----------
    def recommend_text(self, features: Dict[str, float], predicted_yield: float) -> str:
        tips = []
        ph = features.get("ph", None)
        rainfall = features.get("rainfall", None)
        tmax = features.get("temp_max", None)
        hum = features.get("humidity", None)

        if ph is not None and ph < 5.5:
            tips.append("Soil pH is low—add agricultural lime to raise pH toward 6.0–7.0.")
        if rainfall is not None and rainfall < 100:
            tips.append("Rainfall is low—consider supplemental irrigation or mulching to retain moisture.")
        if tmax is not None and tmax > 35:
            tips.append("High daytime temps—use mulching/shade nets and irrigate during cooler hours.")
        if hum is not None and hum > 85:
            tips.append("High humidity—monitor for fungal disease; ensure airflow and consider prophylactic spray.")

        if not tips:
            tips.append("Conditions look broadly suitable for good yield.")
        return " ".join(tips)

    # ---------- Save / Load ----------
    def save(self, path: str = MODEL_OUT_PATH):
        if self.models is None:
            raise RuntimeError("Nothing to save—train the models first.")
        pkg = {
            "xgb": self.models.xgb,
            "rf": self.models.rf,
            "et": self.models.et,
            "cat": self.models.cat,
            "feature_names": self.models.feature_names,
            "crop_name": self.models.crop_name,
            "metrics": self.models.metrics,
            "weights": self.models.weights
        }
        joblib.dump(pkg, path)
        print(f"   💾 Saved ensemble to {path}")

    @staticmethod
    def load(path: str) -> "TrainedModels":
        pkg = joblib.load(path)
        return TrainedModels(
            xgb=pkg.get("xgb"),
            rf=pkg.get("rf"),
            et=pkg.get("et"),
            cat=pkg.get("cat"),
            feature_names=pkg.get("feature_names"),
            crop_name=pkg.get("crop_name"),
            metrics=pkg.get("metrics"),
            weights=pkg.get("weights"),
        )


# ========================
# Example run
# ========================
if __name__ == "__main__":
    crop = "rice"  # change as needed

    rec = CropRecommenderEnsemble(DATA_PATH)
    models = rec.train(crop_name=crop)

    print("== RMSE report ==")
    for k, v in models.metrics.items():
        print(f"{k}: {v:.3f}")

    sample = {
        "temp_max": 30,
        "temp_min": 20,
        "rainfall": 100,
        "humidity": 70,
        "ph": 6.0,
        "altitude": 200,
        "crop_year": 2023
    }

    print("\n[Prediction] Making per-model and ensemble predictions...")
    preds = rec.predict(sample)
    for name, val in preds.items():
        print(f"  {name:12s}: {val:.2f}")

    print("\n[Recommendation]")
    print(rec.recommend_text(sample, preds["ensemble"]))

    rec.save(MODEL_OUT_PATH)
