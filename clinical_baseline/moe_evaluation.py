"""Metric table builders for the data-availability MoE."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix

from clinical_baseline.evaluation import classification_metrics
from clinical_baseline.moe_uncertainty import STAGE_ORDER

PRIMARY_SUBGROUP_COLUMNS = [
    "sex",
    "age_band",
    "surgery_type",
    "admission_urgency",
    "has_notes_route",
]

SECONDARY_SUBGROUP_COLUMNS = [
    "sofa_quartile",
    "asa_band",
    "blood_loss_missing_label",
    "transfused_label",
    "has_diabetes_label",
    "has_hypertension_label",
]

INTERSECTION_SUBGROUP_COLUMNS = [
    "has_notes_x_surgery_type",
    "has_notes_x_admission_urgency",
    "admission_urgency_x_surgery_type",
    "no_notes_cardiac_emergency",
]


def overall_stage_metrics(trajectory: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for stage, group in trajectory.groupby("stage", sort=False):
        row = metric_row(group, threshold)
        row["stage"] = stage
        rows.append(row)
    return order_stage_table(pd.DataFrame(rows))


def add_subgroup_labels(df: pd.DataFrame) -> pd.DataFrame:
    labels = pd.DataFrame({"patient_id": df["patient_id"].astype(str)})
    labels["sex"] = df["sex"].astype(str)
    labels["age_band"] = pd.cut(
        df["age"],
        bins=[-np.inf, 65, 75, np.inf],
        labels=["under_65", "65_to_74", "75_plus"],
        right=False,
    ).astype(str)
    labels["surgery_type"] = df["surgery_type"].astype(str)
    labels["admission_urgency"] = df["admission_urgency"].astype(str)
    labels["has_notes_route"] = np.where(df["has_notes"].astype(int) == 1, "with_notes", "no_notes")
    labels["sofa_quartile"] = pd.qcut(
        df["sofa_score"].rank(method="first"),
        q=4,
        labels=["q1_lowest", "q2", "q3", "q4_highest"],
    ).astype(str)
    labels["asa_band"] = pd.cut(
        df["asa_class"],
        bins=[0, 2, 3, np.inf],
        labels=["asa_1_2", "asa_3", "asa_4_plus"],
        right=True,
    ).astype(str)
    labels["blood_loss_missing_label"] = np.where(
        df["blood_loss_missing"].astype(int) == 1,
        "missing",
        "measured",
    )
    labels["transfused_label"] = np.where(df["transfused"].astype(int) == 1, "yes", "no")
    labels["has_diabetes_label"] = np.where(df["has_diabetes"].astype(int) == 1, "yes", "no")
    labels["has_hypertension_label"] = np.where(df["has_hypertension"].astype(int) == 1, "yes", "no")
    labels["has_notes_x_surgery_type"] = labels["has_notes_route"] + "+" + labels["surgery_type"]
    labels["has_notes_x_admission_urgency"] = labels["has_notes_route"] + "+" + labels["admission_urgency"]
    labels["admission_urgency_x_surgery_type"] = labels["admission_urgency"] + "+" + labels["surgery_type"]
    labels["no_notes_cardiac_emergency"] = np.where(
        (labels["has_notes_route"] == "no_notes")
        & (labels["surgery_type"] == "cardiac")
        & (labels["admission_urgency"] == "emergency"),
        "no_notes+cardiac+emergency",
        "other",
    )
    return labels


def attach_subgroup_labels(trajectory: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    labels = add_subgroup_labels(source_df)
    return trajectory.merge(labels, on="patient_id", how="left")


def subgroup_metric_table(
    trajectory_with_labels: pd.DataFrame,
    group_columns: list[str],
    threshold: float,
) -> pd.DataFrame:
    rows = []
    overall_by_stage = {
        stage: metric_row(group, threshold)
        for stage, group in trajectory_with_labels.groupby("stage", sort=False)
    }
    for stage, stage_df in trajectory_with_labels.groupby("stage", sort=False):
        for group_col in group_columns:
            for group_value, group in stage_df.groupby(group_col, dropna=False, sort=True):
                row = metric_row(group, threshold)
                row["stage"] = stage
                row["group_col"] = group_col
                row["group_value"] = str(group_value)
                row["sensitivity_gap_vs_overall"] = gap(row["sensitivity"], overall_by_stage[stage]["sensitivity"])
                row["auroc_gap_vs_overall"] = gap(row["auroc"], overall_by_stage[stage]["auroc"])
                row["brier_gap_vs_overall"] = gap(row["brier_score"], overall_by_stage[stage]["brier_score"])
                row["unstable_estimate"] = bool(row["n_total"] < 50 or row["n_positive"] < 10)
                rows.append(row)
    return order_stage_table(pd.DataFrame(rows))


def uncertainty_group_metrics(trajectory: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    df = trajectory.copy()
    df["u_total_quintile"] = (
        df.groupby("stage")["u_total"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=["q1_lowest", "q2", "q3", "q4", "q5_highest"]))
        .astype(str)
    )
    groups = ["u_total_quintile", "threshold_crossed", "recommended_action"]
    for stage, stage_df in df.groupby("stage", sort=False):
        for group_col in groups:
            for group_value, group in stage_df.groupby(group_col, dropna=False, sort=True):
                row = error_profile_row(group, threshold)
                row["stage"] = stage
                row["group_col"] = group_col
                row["group_value"] = str(group_value)
                rows.append(row)
    return order_stage_table(pd.DataFrame(rows))


def belief_stage_deltas(trajectory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient_id, group in trajectory.sort_values(["patient_id", "stage"]).groupby("patient_id"):
        by_stage = group.set_index("stage")
        rows.append(
            {
                "patient_id": patient_id,
                "y_true": int(group["y_true"].iloc[0]),
                "belief_t0": float(by_stage.loc["t0", "belief_risk"]),
                "belief_t6": float(by_stage.loc["t6", "belief_risk"]),
                "belief_t12": float(by_stage.loc["t12", "belief_risk"]),
                "belief_t24": float(by_stage.loc["t24", "belief_risk"]),
                "belief_delta_t6_minus_t0": float(by_stage.loc["t6", "belief_risk"] - by_stage.loc["t0", "belief_risk"]),
                "belief_delta_t12_minus_t6": float(by_stage.loc["t12", "belief_risk"] - by_stage.loc["t6", "belief_risk"]),
                "belief_delta_t24_minus_t12": float(by_stage.loc["t24", "belief_risk"] - by_stage.loc["t12", "belief_risk"]),
            }
        )
    return pd.DataFrame(rows)


def note_ablation_table(
    y_true: np.ndarray,
    basic_prob: np.ndarray,
    aux_prob: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": "E1_notes_basic", **classification_metrics(y_true, basic_prob, threshold)},
            {"model": "E1_notes_aux", **classification_metrics(y_true, aux_prob, threshold)},
        ]
    )


def metric_row(group: pd.DataFrame, threshold: float) -> dict:
    metrics = classification_metrics(
        group["y_true"].to_numpy(),
        group["belief_risk"].to_numpy(),
        threshold,
    )
    metrics["event_rate"] = float(group["y_true"].mean()) if len(group) else 0.0
    metrics["alert_count"] = int(group["recommended_action"].astype(str).str.startswith("ALERT").sum())
    metrics["manual_review_count"] = int(
        group["recommended_action"].eq("MANUAL REVIEW / RECHECK DATA").sum()
    )
    return metrics


def error_profile_row(group: pd.DataFrame, threshold: float) -> dict:
    y_true = group["y_true"].to_numpy().astype(int)
    y_pred = (group["belief_risk"].to_numpy() >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(group)),
        "n_positive": int(y_true.sum()),
        "event_rate": float(y_true.mean()) if len(group) else 0.0,
        "brier_score": float(brier_score_loss(y_true, group["belief_risk"].to_numpy())),
        "error_rate": float((y_pred != y_true).mean()) if len(group) else 0.0,
        "false_negative_rate": float(fn / (tp + fn)) if (tp + fn) else 0.0,
        "false_positive_rate": float(fp / (tn + fp)) if (tn + fp) else 0.0,
    }


def gap(value, overall):
    if value is None or overall is None or pd.isna(value) or pd.isna(overall):
        return None
    return float(value - overall)


def order_stage_table(df: pd.DataFrame) -> pd.DataFrame:
    if "stage" not in df.columns:
        return df
    out = df.copy()
    out["_stage_order"] = out["stage"].map(STAGE_ORDER)
    return out.sort_values([col for col in ["_stage_order", "group_col", "group_value"] if col in out.columns]).drop(
        columns=["_stage_order"]
    )
