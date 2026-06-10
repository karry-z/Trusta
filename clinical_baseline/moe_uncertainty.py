"""Uncertainty and dynamic belief helpers for the data-availability MoE."""

from __future__ import annotations

import numpy as np
import pandas as pd

from clinical_baseline.constants import DYNAMIC_STAGES, TARGET
from clinical_baseline.moe_features import build_stage_features, build_stacked_stage_features

STAGE_ORDER = {stage: i for i, stage in enumerate(DYNAMIC_STAGES)}

OUT_OF_RANGE_FEATURES = [
    "preop_lactate",
    "icu_lactate",
    "icu_creatinine",
    "icu_wbc",
    "sofa_score",
    "hr_mean",
    "rr_mean",
    "spo2_min",
    "sbp_mean",
    "temp_mean",
    "lactate_rolling_max",
]


def build_feature_bounds(train_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    frames = []
    for stage in DYNAMIC_STAGES:
        frames.append(build_stage_features(train_df, stage=stage, include_note_features=False))
    stacked = pd.concat(frames, ignore_index=True)
    bounds = {}
    for col in OUT_OF_RANGE_FEATURES:
        bounds[col] = {
            "p01": float(stacked[col].quantile(0.01)),
            "p99": float(stacked[col].quantile(0.99)),
        }
    return bounds


def validation_uncertainty_reference(models: list, validation_df: pd.DataFrame) -> dict[str, np.ndarray]:
    widths = []
    disagreements = []
    for stage in DYNAMIC_STAGES:
        pred = ensemble_stage_summary(models, validation_df, stage)
        widths.append(pred["risk_width"].to_numpy())
        disagreements.append(pred["expert_disagreement"].to_numpy())
    return {
        "risk_width": np.concatenate(widths),
        "expert_disagreement": np.concatenate(disagreements),
    }


def build_belief_trajectory(
    models: list,
    df: pd.DataFrame,
    threshold: float,
    uncertainty_reference: dict[str, np.ndarray],
    feature_bounds: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for stage in DYNAMIC_STAGES:
        pred = ensemble_stage_summary(models, df, stage)
        x_stage = build_stage_features(df, stage=stage, include_note_features=False)
        u_data = data_uncertainty(df, x_stage, feature_bounds)
        u_model = percentile_rank(pred["risk_width"].to_numpy(), uncertainty_reference["risk_width"])
        u_disagree = percentile_rank(
            pred["expert_disagreement"].to_numpy(),
            uncertainty_reference["expert_disagreement"],
        )
        risk_mean = pred["risk_mean"].to_numpy()
        u_entropy = 4 * risk_mean * (1 - risk_mean)
        u_total = 0.35 * u_model + 0.25 * u_disagree + 0.25 * u_data + 0.15 * u_entropy
        threshold_crossed = (pred["risk_low_5"].to_numpy() < threshold) & (
            threshold < pred["risk_high_95"].to_numpy()
        )
        high_uncertainty = (
            threshold_crossed
            | (u_model >= 0.80)
            | (u_disagree >= 0.80)
            | (u_data >= 0.50)
        )
        action = recommended_action(risk_mean, threshold, high_uncertainty)
        stage_rows = pred.copy()
        stage_rows["y_true"] = df[TARGET].astype(int).to_numpy()
        stage_rows["threshold"] = threshold
        stage_rows["u_model"] = u_model
        stage_rows["u_disagree"] = u_disagree
        stage_rows["u_data"] = u_data
        stage_rows["u_entropy"] = u_entropy
        stage_rows["u_total"] = u_total
        stage_rows["threshold_crossed"] = threshold_crossed
        stage_rows["high_uncertainty"] = high_uncertainty
        stage_rows["recommended_action"] = action
        stage_rows["alpha"] = np.clip(1 - 0.5 * u_total, 0.25, 0.85)
        rows.append(stage_rows)

    trajectory = pd.concat(rows, ignore_index=True)
    trajectory["stage_order"] = trajectory["stage"].map(STAGE_ORDER)
    trajectory = trajectory.sort_values(["patient_id", "stage_order"]).reset_index(drop=True)
    trajectory = add_belief_updates(trajectory)
    return trajectory.drop(columns=["stage_order"])


def ensemble_stage_summary(models: list, df: pd.DataFrame, stage: str) -> pd.DataFrame:
    per_model = [model.predict_stage(df, stage) for model in models]
    risk = np.vstack([pred["p_calibrated"].to_numpy() for pred in per_model])
    p_struct = np.vstack([pred["p_struct"].to_numpy() for pred in per_model])
    p_route = np.vstack([pred["p_route_expert"].to_numpy() for pred in per_model])
    first = per_model[0]
    risk_low = np.quantile(risk, 0.05, axis=0)
    risk_high = np.quantile(risk, 0.95, axis=0)
    p_struct_mean = np.nanmean(p_struct, axis=0)
    p_route_mean = np.nanmean(p_route, axis=0)
    return pd.DataFrame(
        {
            "patient_id": first["patient_id"],
            "stage": stage,
            "route": first["route"],
            "risk_mean": np.mean(risk, axis=0),
            "risk_low_5": risk_low,
            "risk_high_95": risk_high,
            "risk_width": risk_high - risk_low,
            "model_std": np.std(risk, axis=0),
            "p_struct": p_struct_mean,
            "p_route_expert": p_route_mean,
            "expert_disagreement": np.abs(p_route_mean - p_struct_mean),
            "lambda_route": first["lambda_route"],
        }
    )


def data_uncertainty(
    df: pd.DataFrame,
    x_stage: pd.DataFrame,
    feature_bounds: dict[str, dict[str, float]],
) -> np.ndarray:
    has_notes = df["has_notes"].astype(int).to_numpy()
    blood_loss_missing = df["blood_loss_missing"].astype(int).to_numpy()
    surgery = df["surgery_type"].astype(str).to_numpy()
    u = np.zeros(len(df), dtype=float)
    u += np.where(has_notes == 0, 0.45, 0.0)
    u += np.where(blood_loss_missing == 1, 0.30, 0.0)
    u += np.where((has_notes == 0) & np.isin(surgery, ["cardiac", "vascular"]), 0.15, 0.0)

    out_of_range = np.zeros(len(df), dtype=bool)
    for col, bounds in feature_bounds.items():
        values = pd.to_numeric(x_stage[col], errors="coerce").to_numpy()
        out_of_range |= (values < bounds["p01"]) | (values > bounds["p99"])
    u += np.where(out_of_range, 0.10, 0.0)
    return np.clip(u, 0.0, 1.0)


def percentile_rank(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    reference = reference[np.isfinite(reference)]
    if len(reference) == 0:
        return np.zeros(len(values), dtype=float)
    sorted_ref = np.sort(reference)
    ranks = np.searchsorted(sorted_ref, values, side="right") / len(sorted_ref)
    return np.clip(ranks, 0.0, 1.0)


def recommended_action(
    risk_mean: np.ndarray,
    threshold: float,
    high_uncertainty: np.ndarray,
) -> np.ndarray:
    action = np.full(len(risk_mean), "ROUTINE MONITORING", dtype=object)
    action[(risk_mean >= threshold) & ~high_uncertainty] = "ALERT"
    action[(risk_mean >= threshold) & high_uncertainty] = "ALERT + SENIOR REVIEW"
    action[(risk_mean < threshold) & high_uncertainty] = "MANUAL REVIEW / RECHECK DATA"
    return action


def add_belief_updates(trajectory: pd.DataFrame) -> pd.DataFrame:
    out = trajectory.copy()
    beliefs = []
    deltas = []
    for _, group in out.groupby("patient_id", sort=False):
        previous = None
        for row in group.itertuples(index=False):
            obs = float(row.risk_mean)
            if previous is None:
                belief = obs
                delta = 0.0
            else:
                belief = inv_logit((1 - row.alpha) * logit(previous) + row.alpha * logit(obs))
                delta = belief - previous
            beliefs.append(belief)
            deltas.append(delta)
            previous = belief
    out["belief_risk"] = beliefs
    out["belief_delta"] = deltas
    return out


def logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def inv_logit(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))
