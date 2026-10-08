"""Shared feature engineering for the wine-quality project.

Imported by BOTH the training notebook and the Streamlit app so that a wine is
transformed identically at training time and at prediction time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
]

ENGINEERED_FEATURES = [
    "is_red",
    "total_acidity",
    "free_so2_ratio",
    "bound_so2",
    "sugar_to_alcohol",
    "acidity_to_alcohol",
    "sulphates_x_alcohol",
]

MODEL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES

QUALITY_BANDS = {"low": "≤ 4", "medium": "5 – 6", "high": "≥ 7"}


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `data` with the engineered columns added.

    `data` must contain RAW_FEATURES plus either `wine_type` ('red'/'white')
    or an existing `is_red` column.
    """
    out = data.copy()
    out.columns = [c.strip().lower().replace(" ", "_") for c in out.columns]

    if "is_red" not in out.columns:
        out["is_red"] = (out["wine_type"].astype(str).str.lower() == "red").astype(int)

    eps = 1e-9
    out["total_acidity"] = out.fixed_acidity + out.volatile_acidity + out.citric_acid
    out["free_so2_ratio"] = out.free_sulfur_dioxide / (out.total_sulfur_dioxide + eps)
    out["bound_so2"] = (out.total_sulfur_dioxide - out.free_sulfur_dioxide).clip(lower=0)
    out["sugar_to_alcohol"] = out.residual_sugar / (out.alcohol + eps)
    out["acidity_to_alcohol"] = out.total_acidity / (out.alcohol + eps)
    out["sulphates_x_alcohol"] = out.sulphates * out.alcohol

    return out.replace([np.inf, -np.inf], np.nan)


def band_of(score: float) -> str:
    """Map a numeric quality score onto the low/medium/high band."""
    if score <= 4:
        return "low"
    if score <= 6:
        return "medium"
    return "high"
