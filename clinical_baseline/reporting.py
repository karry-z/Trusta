"""Reference JSON report updates for the clinical baseline."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clinical_baseline.constants import MODEL_NAME, TARGET


def update_reference_reports(
    reference_dir: Path,
    metrics: dict,
    subgroup_df: pd.DataFrame,
    coefficients_df: pd.DataFrame,
    threshold_sweep_rows: list[dict],
    icu_hours_summary: dict,
) -> None:
    """Insert or update the clinical baseline in the two submission templates."""
    reference_dir.mkdir(parents=True, exist_ok=True)
    pathway_path = reference_dir / "omaib_pathway.json"
    safety_path = reference_dir / "model_safety_report.json"

    pathway = load_json(pathway_path)
    safety = load_json(safety_path)

    model_entry = build_pathway_model_entry(metrics, subgroup_df)
    safety_entry = build_safety_model_entry(
        metrics,
        subgroup_df,
        coefficients_df,
        threshold_sweep_rows,
        icu_hours_summary,
    )

    pathway.setdefault("schema_version", "omaib-clinical")
    pathway.setdefault("submission_type", "model_safety_report")
    pathway.setdefault("strand", "clinical")
    pathway.setdefault("hackathon", "MultimodalAI26")
    pathway["submitted"] = str(date.today())
    pathway.setdefault("team", {"name": "", "members": ""})
    pathway["models"] = upsert_model(pathway.get("models", []), model_entry)
    pathway.setdefault("overall_notes", "")

    safety.setdefault("schema_version", "omaib-clinical")
    safety.setdefault("report_type", "model_safety_report")
    safety.setdefault("strand", "clinical")
    safety.setdefault("hackathon", "MultimodalAI26")
    safety["evaluation_date"] = str(date.today())
    safety.setdefault("team", {"name": "", "members": ""})
    safety["models"] = upsert_model(safety.get("models", []), safety_entry)
    safety.setdefault("overall_notes", "")

    write_json(pathway_path, pathway)
    write_json(safety_path, safety)


def build_pathway_model_entry(metrics: dict, subgroup_df: pd.DataFrame) -> dict:
    calibrated = metrics["main"]["calibrated"]
    return {
        "name": MODEL_NAME,
        "verdict": "CONDITIONAL",
        "conditions": conditions_text(),
        "narrative": narrative_text(metrics),
        "metrics": compact_metrics(calibrated),
        "subgroup_gaps": subgroup_gap_dict(subgroup_df),
    }


def build_safety_model_entry(
    metrics: dict,
    subgroup_df: pd.DataFrame,
    coefficients_df: pd.DataFrame,
    threshold_sweep_rows: list[dict],
    icu_hours_summary: dict,
) -> dict:
    calibrated = metrics["main"]["calibrated"]
    top_pos = top_coefficients(coefficients_df, "positive")
    top_neg = top_coefficients(coefficients_df, "negative")
    worst = worst_subgroup(subgroup_df)
    note_gap = subgroup_lookup(subgroup_df, "has_notes")

    return {
        "name": MODEL_NAME,
        "label": "Clinical elastic-net logistic baseline",
        "verdict": "CONDITIONAL",
        "conditions": conditions_text(),
        "narrative": narrative_text(metrics),
        "metrics": compact_metrics(calibrated),
        "subgroup_gaps": subgroup_gap_dict(subgroup_df),
        "deployment_questions": {
            "q1_miss_rate": (
                f"At threshold {calibrated['threshold']:.4f}, the model missed "
                f"{calibrated['fn']} of {calibrated['n_positive']} deteriorations "
                f"on the held-out test set."
            ),
            "q2_alert_precision": (
                f"Alert precision (PPV) was {calibrated['ppv']:.3f}; "
                f"{calibrated['tp'] + calibrated['fp']} test patients were flagged."
            ),
            "q3_clearance_safety": (
                f"NPV was {calibrated['npv']:.3f}. Low-risk clearance should retain "
                "standard ICU safety-net monitoring because false negatives remain."
            ),
            "q4_discrimination": (
                f"Calibrated AUROC was {calibrated['auroc']:.3f} and AUPRC was "
                f"{calibrated['auprc']:.3f}. Largest observed subgroup sensitivity "
                f"gap was {worst['gap_text']} in {worst['label']}."
            ),
            "q5_calibration": (
                f"Calibrated Brier score was {calibrated['brier_score']:.3f}. "
                "Use calibrated probabilities for prioritisation; raw probabilities "
                "are retained only for comparison."
            ),
        },
        "failure_analysis": {
            "who_is_missed": (
                f"Misses concentrate most in {worst['label']} "
                f"(sensitivity {worst['sensitivity_text']}, n={worst['n']}, "
                f"positives={worst['positives']})."
            ),
            "false_alarm_profile": (
                f"The selected threshold produced {calibrated['fp']} false positives "
                f"and specificity {calibrated['specificity']:.3f}; alert burden should "
                "be reviewed before operational use."
            ),
            "feature_importance_interpretation": (
                f"Top positive coefficients: {', '.join(top_pos)}. "
                f"Top negative coefficients: {', '.join(top_neg)}."
            ),
            "model_disagreement": (
                "No second deployable model is used for disagreement analysis. The "
                f"icu_hours sensitivity check is marked non-deployable; AUROC "
                f"{icu_hours_summary.get('auroc_delta_text', 'delta unavailable')}."
            ),
        },
        "option_specific": {
            "type": "threshold_sensitivity_report",
            "title": "Clinical threshold sensitivity sweep",
            "content": {
                "thresholds_tested": [row["threshold"] for row in threshold_sweep_rows],
                "signal_at_each_threshold": threshold_sweep_rows,
                "recommended_threshold": calibrated["threshold"],
                "rationale": (
                    "Recommended threshold is the highest validation-set calibrated "
                    "risk threshold that achieved target sensitivity >= 0.80."
                ),
            },
        },
        "explainability": {
            "output_type": "standardized_log_odds_coefficients",
            "description": (
                "Linear coefficients from the elastic-net logistic model. Numeric "
                "features are median-imputed where needed and standardized; categorical "
                "terms are one-hot indicators relative to the dropped reference level."
            ),
        },
        "failure_catalogue": [
            {
                "mode": "Residual false negatives at clinical threshold",
                "example": (
                    f"{calibrated['fn']} deteriorating test patients were below the "
                    "selected risk threshold."
                ),
                "subgroup": worst["label"],
            },
            {
                "mode": "Alert burden from false positives",
                "example": (
                    f"{calibrated['fp']} non-deteriorating test patients were flagged "
                    f"at threshold {calibrated['threshold']:.4f}."
                ),
                "subgroup": "overall_test_set",
            },
            {
                "mode": "Notes availability sensitivity",
                "example": note_gap,
                "subgroup": "has_notes",
            },
        ],
    }


def compact_metrics(metrics: dict) -> dict:
    keys = [
        "auroc",
        "auprc",
        "brier_score",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "threshold",
        "n_total",
        "n_positive",
    ]
    return {key: round_value(metrics[key]) for key in keys}


def conditions_text() -> str:
    return (
        "Conditional baseline only: use as the minimum explainable comparator, not as "
        "standalone clinical automation. Keep icu_hours excluded from deployable use, "
        "validate calibration prospectively, and audit subgroup sensitivity before "
        "any escalation workflow."
    )


def narrative_text(metrics: dict) -> str:
    calibrated = metrics["main"]["calibrated"]
    raw = metrics["main"]["raw"]
    return (
        f"Clinical elastic-net logistic regression trained on structured, interpretable "
        f"features for {TARGET}. On held-out test data, calibrated AUROC was "
        f"{calibrated['auroc']:.3f}, AUPRC {calibrated['auprc']:.3f}, sensitivity "
        f"{calibrated['sensitivity']:.3f}, and Brier score {calibrated['brier_score']:.3f}. "
        f"Raw AUROC was {raw['auroc']:.3f}; calibrated probabilities are the reported "
        "risk output."
    )


def subgroup_gap_dict(subgroup_df: pd.DataFrame) -> dict:
    gaps = {}
    for row in subgroup_df.itertuples(index=False):
        key = f"{row.group_col}_{row.group_value}"
        gaps[key] = round_value(row.sensitivity_gap_vs_overall)
    return gaps


def worst_subgroup(subgroup_df: pd.DataFrame) -> dict:
    usable = subgroup_df.dropna(subset=["sensitivity_gap_vs_overall"]).copy()
    if usable.empty:
        return {
            "label": "no subgroup with positive cases",
            "gap_text": "unavailable",
            "sensitivity_text": "unavailable",
            "n": 0,
            "positives": 0,
        }
    row = usable.sort_values("sensitivity_gap_vs_overall").iloc[0]
    return {
        "label": f"{row['group_col']}={row['group_value']}",
        "gap_text": f"{row['sensitivity_gap_vs_overall']:.3f}",
        "sensitivity_text": f"{row['sensitivity']:.3f}",
        "n": int(row["n"]),
        "positives": int(row["positives"]),
    }


def subgroup_lookup(subgroup_df: pd.DataFrame, group_col: str) -> str:
    rows = subgroup_df[subgroup_df["group_col"] == group_col]
    if rows.empty:
        return "Notes availability subgroup metrics were unavailable."
    parts = []
    for row in rows.itertuples(index=False):
        sens = "unavailable" if pd.isna(row.sensitivity) else f"{row.sensitivity:.3f}"
        parts.append(f"{group_col}={row.group_value}: sensitivity {sens}, n={row.n}")
    return "; ".join(parts)


def top_coefficients(coefficients_df: pd.DataFrame, direction: str, n: int = 5) -> list[str]:
    rows = coefficients_df[coefficients_df["direction"] == direction]
    rows = rows.sort_values("coefficient", ascending=(direction == "negative")).head(n)
    return [f"{row.feature} ({row.coefficient:.3f})" for row in rows.itertuples(index=False)]


def upsert_model(models: list[dict], new_model: dict) -> list[dict]:
    out = []
    replaced = False
    for model in models:
        if model.get("name") == new_model["name"]:
            out.append(new_model)
            replaced = True
        else:
            out.append(model)
    if not replaced:
        out.append(new_model)
    return out


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(to_jsonable(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def round_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 4)
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if pd.isna(value):
        return None
    return value
