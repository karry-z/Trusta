"""Feature preparation for the clinical elastic-net baseline."""

from __future__ import annotations

import pandas as pd

from clinical_baseline.constants import (
    CATEGORICAL_FEATURES,
    MAIN_NUMERIC_FEATURES,
    RAW_NUMERIC_FEATURES,
    REQUIRED_SOURCE_COLUMNS,
)


def validate_required_columns(df: pd.DataFrame, include_icu_hours: bool = False) -> None:
    """Raise a clear error when the source dataset is missing required columns."""
    required = list(REQUIRED_SOURCE_COLUMNS)
    if include_icu_hours:
        required.append("icu_hours")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")


def note_risk_median(train_df: pd.DataFrame) -> float:
    """Return the train-set median note risk score, falling back to 0.0 if all missing."""
    validate_required_columns(train_df)
    median = train_df["note_risk_score"].median()
    if pd.isna(median):
        return 0.0
    return float(median)


def prepare_model_frame(
    df: pd.DataFrame,
    note_median: float,
    include_icu_hours: bool = False,
) -> pd.DataFrame:
    """Create the exact model feature frame from source data."""
    validate_required_columns(df, include_icu_hours=include_icu_hours)

    out = pd.DataFrame(index=df.index)

    for col in CATEGORICAL_FEATURES:
        out[col] = df[col].astype("string").fillna("missing")

    for col in RAW_NUMERIC_FEATURES:
        out[col] = pd.to_numeric(df[col], errors="coerce")

    out["note_risk_score_missing"] = df["note_risk_score"].isna().astype(int)
    out["note_risk_score_imputed"] = (
        pd.to_numeric(df["note_risk_score"], errors="coerce")
        .fillna(note_median)
        .astype(float)
    )

    if include_icu_hours:
        out["icu_hours"] = pd.to_numeric(df["icu_hours"], errors="coerce")

    numeric_features = list(MAIN_NUMERIC_FEATURES)
    if include_icu_hours:
        numeric_features.append("icu_hours")

    ordered = CATEGORICAL_FEATURES + numeric_features
    return out[ordered]
