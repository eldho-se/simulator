from __future__ import annotations

import os
import calendar
from datetime import date, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from joblib import dump, load
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor

app = FastAPI(title="Overshoot Scenario Simulator")

# --------------------------- CORS -------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------- Config -----------------------------------------
MODEL_PATH = "rf_overshoot.joblib"
DATA_PATH = "training_matrix.csv"
EXCLUDE_COLS = {"Year", "Overshoot_Date", "DayOfYear"}
SPARSITY_THRESHOLD = 0.5   # drop features with >50% NA in training
CLIP_TO_HISTORY = True     # clip adjusted features to historical min/max
N_ESTIMATORS = 300
RANDOM_STATE = 42

# --------------------------- Pydantic models --------------------------------
class Scenario(BaseModel):
    # adjustments as fractional changes: 0.10 = +10%, -0.20 = -20%
    adjustments: Dict[str, float] = Field(
        ..., description="Map feature_name → percent change (e.g. {'co2_total': -0.1})"
    )
    # Baseline year to simulate (must exist in dataset)
    forecast_year: int = Field(..., ge=1900, le=2100)

class SimulationResult(BaseModel):
    year: int
    predicted_day_of_year: int
    predicted_date: date
    raw_prediction: float
    baseline_day_of_year: int
    baseline_raw_prediction: float
    delta_days: int

# --------------------------- Data & Model -----------------------------------

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Example derived feature: renewable share if ingredients exist
    cols = {"Wind_GWh", "Solar_GWh", "Coal_GWh", "Gas_GWh"}
    if cols.issubset(df.columns):
        denom = df[list(cols)].sum(axis=1).replace(0, np.nan)
        df["renewable_share"] = (df["Wind_GWh"] + df["Solar_GWh"]) / denom
    return df


def prepare_feature_metadata(df: pd.DataFrame):
    na_frac = df.isna().mean()
    # choose feature columns by excluding targets + dropping ultra-sparse
    candidate_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    feature_cols = [c for c in candidate_cols if na_frac.get(c, 0.0) <= SPARSITY_THRESHOLD]
    # stats for clipping & imputation
    mins = df[feature_cols].min(numeric_only=True)
    maxs = df[feature_cols].max(numeric_only=True)
    means = df[feature_cols].mean(numeric_only=True)
    return feature_cols, na_frac.to_dict(), mins, maxs, means


def train_model(df: pd.DataFrame, feature_cols: List[str]) -> RandomForestRegressor:
    dft = df.dropna(subset=["DayOfYear"]).copy()
    X = dft[feature_cols].copy()
    # simple mean imputation
    X = X.fillna(X.mean(numeric_only=True))
    y = dft["DayOfYear"].astype(float).values

    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X, y)
    dump(model, MODEL_PATH)
    return model


def load_data_and_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")

    df = pd.read_csv(DATA_PATH)
    if "Year" not in df.columns or "DayOfYear" not in df.columns:
        raise RuntimeError("training_matrix.csv must include 'Year' and 'DayOfYear'.")

    df = add_derived_features(df)

    # feature selection & stats
    feature_cols, na_frac_map, mins, maxs, means = prepare_feature_metadata(df)

    # try load existing model; retrain if feature count doesn't match
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = load(MODEL_PATH)
            if getattr(model, "n_features_in_", None) != len(feature_cols):
                model = train_model(df, feature_cols)
        except Exception:
            model = train_model(df, feature_cols)
    else:
        model = train_model(df, feature_cols)

    years_min, years_max = int(df["Year"].min()), int(df["Year"].max())

    return {
        "df": df,
        "model": model,
        "feature_cols": feature_cols,
        "na_frac_map": na_frac_map,
        "mins": mins,
        "maxs": maxs,
        "means": means,
        "years_min": years_min,
        "years_max": years_max,
    }


_store = load_data_and_model()
df_matrix: pd.DataFrame = _store["df"]
rf_model: RandomForestRegressor = _store["model"]
FEATURE_COLS: List[str] = _store["feature_cols"]
NA_FRAC_MAP: Dict[str, float] = _store["na_frac_map"]
FEATURE_MINS: pd.Series = _store["mins"]
FEATURE_MAXS: pd.Series = _store["maxs"]
FEATURE_MEANS: pd.Series = _store["means"]
YEARS_MIN: int = _store["years_min"]
YEARS_MAX: int = _store["years_max"]

# --------------------------- Utilities --------------------------------------

def _is_leap(y: int) -> bool:
    return calendar.isleap(y)


def _predict_day_of_year(vec: pd.Series, year: int) -> tuple[float, int, date]:
    raw = float(rf_model.predict([vec.values])[0])
    days_in_year = 366 if _is_leap(year) else 365
    doy = int(np.clip(round(raw), 1, days_in_year))
    sim_date = date(year, 1, 1) + timedelta(days=doy - 1)
    return raw, doy, sim_date


def _baseline_vector_for_year(year: int) -> pd.Series:
    row = df_matrix.loc[df_matrix["Year"] == year]
    if row.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No data for year {year}. Available range: {YEARS_MIN}–{YEARS_MAX}."
            ),
        )
    base = row[FEATURE_COLS].iloc[0]
    # impute missing using global column means
    base = base.fillna(FEATURE_MEANS)
    return base


def apply_adjustments(features: pd.Series, adjustments: Dict[str, float]) -> pd.Series:
    f = features.copy()
    for feat, pct in adjustments.items():
        if feat not in f.index:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Feature '{feat}' not found. Valid features: {', '.join(FEATURE_COLS)}"
                ),
            )
        f[feat] = f[feat] * (1 + pct)
    if CLIP_TO_HISTORY:
        f = f.clip(lower=FEATURE_MINS, upper=FEATURE_MAXS)
    return f

# --------------------------- Endpoints --------------------------------------

@app.post("/simulate", response_model=List[SimulationResult])
async def simulate(scenario: Scenario):
    """
    Scenario simulator:
    - Use the requested year's baseline row.
    - Apply percentage adjustments to that row's feature values.
    - Predict the overshoot DayOfYear for that same year.
    - Return raw predictions and delta vs baseline for transparency.
    """
    # Baseline
    baseline = _baseline_vector_for_year(scenario.forecast_year)
    base_raw, base_doy, _ = _predict_day_of_year(baseline, scenario.forecast_year)

    # Adjusted
    adjusted = apply_adjustments(baseline, scenario.adjustments)
    raw, doy, sim_date = _predict_day_of_year(adjusted, scenario.forecast_year)

    return [
        {
            "year": scenario.forecast_year,
            "predicted_day_of_year": doy,
            "predicted_date": sim_date,
            "raw_prediction": raw,
            "baseline_day_of_year": base_doy,
            "baseline_raw_prediction": base_raw,
            "delta_days": int(doy - base_doy),
        }
    ]


@app.post("/compare")
async def compare(scenario: Scenario):
    """
    Compare baseline vs adjusted for a given year.
    Returns raw and rounded values and the delta in days.
    """
    baseline = _baseline_vector_for_year(scenario.forecast_year)
    base_raw, base_doy, base_date = _predict_day_of_year(baseline, scenario.forecast_year)

    adjusted = apply_adjustments(baseline, scenario.adjustments)
    adj_raw, adj_doy, adj_date = _predict_day_of_year(adjusted, scenario.forecast_year)

    return {
        "year": scenario.forecast_year,
        "baseline": {
            "raw_prediction": base_raw,
            "day_of_year": base_doy,
            "date": base_date,
        },
        "adjusted": {
            "raw_prediction": adj_raw,
            "day_of_year": adj_doy,
            "date": adj_date,
        },
        "delta_days": int(adj_doy - base_doy),
    }


@app.get("/features")
def features():
    """List of valid feature names for adjustments (after sparsity filtering)."""
    return FEATURE_COLS


@app.get("/feature_stats")
def feature_stats():
    """Basic stats to help choose reasonable adjustments and debug stability."""
    stats = {}
    for c in FEATURE_COLS:
        stats[c] = {
            "na_frac": float(NA_FRAC_MAP.get(c, 0.0)),
            "min": float(FEATURE_MINS[c]) if pd.notna(FEATURE_MINS[c]) else None,
            "max": float(FEATURE_MAXS[c]) if pd.notna(FEATURE_MAXS[c]) else None,
            "mean": float(FEATURE_MEANS[c]) if pd.notna(FEATURE_MEANS[c]) else None,
        }
    return stats


@app.get("/health")
def health():
    return {
        "status": "ok",
        "years_min": YEARS_MIN,
        "years_max": YEARS_MAX,
        "n_features": len(FEATURE_COLS),
        "sparsity_threshold": SPARSITY_THRESHOLD,
        "clip_to_history": CLIP_TO_HISTORY,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)