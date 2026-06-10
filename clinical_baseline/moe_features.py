"""Dynamic snapshot features for the data-availability MoE."""

from __future__ import annotations

import numpy as np
import pandas as pd

from clinical_baseline.constants import DYNAMIC_STAGES, TARGET
from clinical_baseline.note_nlp import NOTE_FEATURES, NoteNlpFeatureExtractor

STAGE_END_HOUR = {
    "t0": 0,
    "t6": 6,
    "t12": 12,
    "t24": 23,
}

MOE_CATEGORICAL_FEATURES = [
    "sex",
    "surgery_type",
    "admission_urgency",
]

MOE_STRUCT_NUMERIC_FEATURES = [
    "age",
    "asa_class",
    "op_duration_h",
    "blood_loss_imputed",
    "blood_loss_missing",
    "transfused",
    "has_diabetes",
    "has_hypertension",
    "preop_creatinine",
    "preop_wbc",
    "preop_lactate",
    "sofa_score",
    "icu_lactate",
    "icu_creatinine",
    "icu_wbc",
    "icu_bilirubin",
    "has_notes",
    "hr_mean",
    "hr_std",
    "rr_mean",
    "rr_std",
    "spo2_mean",
    "spo2_min",
    "sbp_mean",
    "temp_mean",
    "lactate_rolling_mean",
    "lactate_rolling_max",
    "lactate_slope",
]

DYNAMIC_NUMERIC_FEATURES = {
    "hr_mean",
    "hr_std",
    "rr_mean",
    "rr_std",
    "spo2_mean",
    "spo2_min",
    "sbp_mean",
    "temp_mean",
    "lactate_rolling_mean",
    "lactate_rolling_max",
    "lactate_slope",
}

E0_FEATURES = MOE_CATEGORICAL_FEATURES + MOE_STRUCT_NUMERIC_FEATURES
E2_FEATURES = E0_FEATURES
E1_BASIC_FEATURES = E0_FEATURES + ["note_risk_score"]
E1_AUX_FEATURES = E1_BASIC_FEATURES + NOTE_FEATURES

FORBIDDEN_DEPLOYABLE_COLUMNS = {
    "patient_id",
    TARGET,
    "major_complication_30d",
    "note_text",
    "icu_hours",
}


def validate_moe_source_columns(df: pd.DataFrame) -> None:
    required = set(MOE_CATEGORICAL_FEATURES)
    required.update(col for col in MOE_STRUCT_NUMERIC_FEATURES if col not in DYNAMIC_NUMERIC_FEATURES)
    required.update(
        {
            "patient_id",
            TARGET,
            "major_complication_30d",
            "note_text",
            "note_risk_score",
            "icu_hours",
        }
    )
    for vital in ("hr", "rr", "spo2", "sbp", "temp", "lactate"):
        required.update(f"{vital}_h{hour:02d}" for hour in range(24))
    missing = sorted(col for col in required if col not in df.columns)
    if missing:
        raise ValueError(f"Missing required MoE source columns: {missing}")


def build_stage_features(
    df: pd.DataFrame,
    stage: str,
    note_extractor: NoteNlpFeatureExtractor | None = None,
    include_note_features: bool = True,
) -> pd.DataFrame:
    """Build a same-schema patient snapshot for one simulated time point."""
    if stage not in STAGE_END_HOUR:
        raise ValueError(f"Unknown stage {stage}; expected one of {DYNAMIC_STAGES}")
    validate_moe_source_columns(df)

    out = pd.DataFrame(index=df.index)
    for col in MOE_CATEGORICAL_FEATURES:
        out[col] = df[col].astype("string").fillna("missing")

    for col in MOE_STRUCT_NUMERIC_FEATURES:
        if col not in DYNAMIC_NUMERIC_FEATURES:
            out[col] = pd.to_numeric(df[col], errors="coerce")

    rolling = rolling_vital_features(df, end_hour=STAGE_END_HOUR[stage])
    for col in rolling.columns:
        out[col] = rolling[col]

    out["note_risk_score"] = pd.to_numeric(df["note_risk_score"], errors="coerce").fillna(0.0)

    if include_note_features and note_extractor is not None:
        note_features = note_extractor.transform(df)
        for col in NOTE_FEATURES:
            out[col] = note_features[col]
    elif include_note_features:
        for col in NOTE_FEATURES:
            out[col] = 0.0

    return out


def build_stacked_stage_features(
    df: pd.DataFrame,
    note_extractor: NoteNlpFeatureExtractor | None = None,
    include_note_features: bool = True,
    stages: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Stack t0/t6/t12/t24 snapshots for one cohort."""
    stages = DYNAMIC_STAGES if stages is None else stages
    frames = []
    meta = []
    y = []
    for stage in stages:
        x_stage = build_stage_features(
            df,
            stage=stage,
            note_extractor=note_extractor,
            include_note_features=include_note_features,
        )
        frames.append(x_stage.reset_index(drop=True))
        meta.append(
            pd.DataFrame(
                {
                    "patient_id": df["patient_id"].astype(str).to_numpy(),
                    "stage": stage,
                    "has_notes": df["has_notes"].astype(int).to_numpy(),
                }
            )
        )
        y.append(df[TARGET].to_numpy())
    return (
        pd.concat(frames, ignore_index=True),
        np.concatenate(y),
        pd.concat(meta, ignore_index=True),
    )


def rolling_vital_features(df: pd.DataFrame, end_hour: int) -> pd.DataFrame:
    hours = list(range(end_hour + 1))
    out = pd.DataFrame(index=df.index)
    out["hr_mean"] = rolling_mean(df, "hr", hours)
    out["hr_std"] = rolling_std(df, "hr", hours)
    out["rr_mean"] = rolling_mean(df, "rr", hours)
    out["rr_std"] = rolling_std(df, "rr", hours)
    out["spo2_mean"] = rolling_mean(df, "spo2", hours)
    out["spo2_min"] = rolling_min(df, "spo2", hours)
    out["sbp_mean"] = rolling_mean(df, "sbp", hours)
    out["temp_mean"] = rolling_mean(df, "temp", hours)
    out["lactate_rolling_mean"] = rolling_mean(df, "lactate", hours)
    out["lactate_rolling_max"] = rolling_max(df, "lactate", hours)
    out["lactate_slope"] = rolling_slope(df, "lactate", hours)
    return out


def rolling_array(df: pd.DataFrame, vital: str, hours: list[int]) -> np.ndarray:
    cols = [f"{vital}_h{hour:02d}" for hour in hours]
    return df[cols].to_numpy(dtype=float)


def rolling_mean(df: pd.DataFrame, vital: str, hours: list[int]) -> np.ndarray:
    return np.nanmean(rolling_array(df, vital, hours), axis=1)


def rolling_std(df: pd.DataFrame, vital: str, hours: list[int]) -> np.ndarray:
    if len(hours) == 1:
        return np.zeros(len(df), dtype=float)
    return np.nanstd(rolling_array(df, vital, hours), axis=1)


def rolling_min(df: pd.DataFrame, vital: str, hours: list[int]) -> np.ndarray:
    return np.nanmin(rolling_array(df, vital, hours), axis=1)


def rolling_max(df: pd.DataFrame, vital: str, hours: list[int]) -> np.ndarray:
    return np.nanmax(rolling_array(df, vital, hours), axis=1)


def rolling_slope(df: pd.DataFrame, vital: str, hours: list[int]) -> np.ndarray:
    if len(hours) == 1:
        return np.zeros(len(df), dtype=float)
    y = rolling_array(df, vital, hours)
    x = np.asarray(hours, dtype=float)
    x_centered = x - x.mean()
    denom = np.sum(x_centered**2)
    y_centered = y - np.nanmean(y, axis=1, keepdims=True)
    return np.nansum(y_centered * x_centered, axis=1) / denom


def assert_no_forbidden_features(feature_names: list[str]) -> None:
    forbidden = sorted(set(feature_names).intersection(FORBIDDEN_DEPLOYABLE_COLUMNS))
    if forbidden:
        raise AssertionError(f"Forbidden deployable columns used as features: {forbidden}")
