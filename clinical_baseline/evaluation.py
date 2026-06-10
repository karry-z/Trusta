"""Evaluation helpers for the clinical elastic-net baseline."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from clinical_baseline.constants import SUBGROUP_COLUMNS


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    """Return discrimination, calibration, and threshold metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "auroc": safe_metric(roc_auc_score, y_true, y_prob),
        "auprc": safe_metric(average_precision_score, y_true, y_prob),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "ppv": float(precision_score(y_true, y_pred, zero_division=0)),
        "npv": float(tn / (tn + fn)) if (tn + fn) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n_total": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }


def safe_metric(func, y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    try:
        value = func(y_true, y_prob)
    except ValueError:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Bootstrap 95% CIs for AUROC, AUPRC, and sensitivity on test predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = {"auroc": [], "auprc": [], "sensitivity": []}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        p_b = y_prob[idx]
        if len(np.unique(y_b)) < 2:
            continue
        y_pred = (p_b >= threshold).astype(int)
        values["auroc"].append(float(roc_auc_score(y_b, p_b)))
        values["auprc"].append(float(average_precision_score(y_b, p_b)))
        values["sensitivity"].append(float(recall_score(y_b, y_pred, zero_division=0)))

    out = {}
    point = classification_metrics(y_true, y_prob, threshold)
    for metric, samples in values.items():
        arr = np.asarray(samples, dtype=float)
        if len(arr) == 0:
            out[metric] = {
                "point": point[metric],
                "ci_lower": None,
                "ci_upper": None,
                "n_bootstrap_valid": 0,
            }
            continue
        out[metric] = {
            "point": point[metric],
            "ci_lower": float(np.quantile(arr, 0.025)),
            "ci_upper": float(np.quantile(arr, 0.975)),
            "n_bootstrap_valid": int(len(arr)),
        }
    return out


def add_subgroup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add stable subgroup labels for final test-set evaluation."""
    out = df.copy()
    out["age_band"] = pd.cut(
        out["age"],
        bins=[-np.inf, 65, 75, np.inf],
        labels=["under_65", "65_to_74", "75_plus"],
        right=False,
    ).astype("string")
    out["sofa_quartile"] = pd.qcut(
        out["sofa_score"].rank(method="first"),
        q=4,
        labels=["q1_lowest", "q2", "q3", "q4_highest"],
    ).astype("string")
    return out


def subgroup_sensitivity_table(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Return per-subgroup sensitivity gaps vs overall test sensitivity."""
    df = add_subgroup_columns(df).reset_index(drop=True)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    overall_sensitivity = classification_metrics(y_true, y_prob, threshold)["sensitivity"]

    rows = []
    for col in SUBGROUP_COLUMNS:
        for value in sorted(df[col].dropna().astype(str).unique()):
            mask = (df[col].astype(str) == value).to_numpy()
            positives = int(y_true[mask].sum())
            if positives == 0:
                sensitivity = None
                gap = None
            else:
                sensitivity = float(((y_pred[mask] == 1) & (y_true[mask] == 1)).sum() / positives)
                gap = float(sensitivity - overall_sensitivity)
            rows.append(
                {
                    "group_col": col,
                    "group_value": value,
                    "n": int(mask.sum()),
                    "positives": positives,
                    "sensitivity": sensitivity,
                    "sensitivity_gap_vs_overall": gap,
                }
            )
    return pd.DataFrame(rows)


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict]:
    if thresholds is None:
        thresholds = np.round(np.arange(0.20, 0.81, 0.05), 2)
    rows = []
    for threshold in thresholds:
        m = classification_metrics(y_true, y_prob, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "ppv": m["ppv"],
                "n_alerts": int(m["tp"] + m["fp"]),
            }
        )
    return rows
