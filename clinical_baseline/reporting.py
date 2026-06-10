"""Reference JSON report updates for the clinical baseline."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clinical_baseline.constants import MODEL_NAME, NO_NOTES_MODEL_NAME, TARGET


PLACEHOLDER_MODEL_NAMES = {
    "Your model name A ",
    "Your model name B",
    "Your model name C",
    "model_a",
    "model_b",
    "model_c",
}


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
    pathway["models"] = upsert_model(
        prune_placeholder_models(pathway.get("models", [])), model_entry
    )
    pathway.setdefault("overall_notes", "")

    safety.setdefault("schema_version", "omaib-clinical")
    safety.setdefault("report_type", "model_safety_report")
    safety.setdefault("strand", "clinical")
    safety.setdefault("hackathon", "MultimodalAI26")
    safety["evaluation_date"] = str(date.today())
    safety.setdefault("team", {"name": "", "members": ""})
    safety["models"] = upsert_model(
        prune_placeholder_models(safety.get("models", [])), safety_entry
    )
    safety.setdefault("overall_notes", "")

    write_json(pathway_path, pathway)
    write_json(safety_path, safety)


def update_no_notes_reference_reports(
    reference_dir: Path,
    metrics: dict,
    subgroup_df: pd.DataFrame,
    coefficients_df: pd.DataFrame,
    threshold_sweep_rows: list[dict],
) -> None:
    """Insert or update the no-notes modality ablation in the submission JSON files."""
    reference_dir.mkdir(parents=True, exist_ok=True)
    pathway_path = reference_dir / "omaib_pathway.json"
    safety_path = reference_dir / "model_safety_report.json"

    pathway = load_json(pathway_path)
    safety = load_json(safety_path)

    model_entry = build_no_notes_pathway_model_entry(metrics, subgroup_df)
    safety_entry = build_no_notes_safety_model_entry(
        metrics,
        subgroup_df,
        coefficients_df,
        threshold_sweep_rows,
    )

    pathway.setdefault("schema_version", "omaib-clinical")
    pathway.setdefault("submission_type", "model_safety_report")
    pathway.setdefault("strand", "clinical")
    pathway.setdefault("hackathon", "MultimodalAI26")
    pathway["submitted"] = str(date.today())
    pathway.setdefault("team", {"name": "", "members": ""})
    pathway["models"] = upsert_model(
        prune_placeholder_models(pathway.get("models", [])), model_entry
    )
    if is_blank(pathway.get("overall_notes", "")):
        pathway["overall_notes"] = modality_comparison_notes()

    safety.setdefault("schema_version", "omaib-clinical")
    safety.setdefault("report_type", "model_safety_report")
    safety.setdefault("strand", "clinical")
    safety.setdefault("hackathon", "MultimodalAI26")
    safety["evaluation_date"] = str(date.today())
    safety.setdefault("team", {"name": "", "members": ""})
    safety["models"] = upsert_model(
        prune_placeholder_models(safety.get("models", [])), safety_entry
    )
    if is_blank(safety.get("overall_notes", "")):
        safety["overall_notes"] = modality_comparison_notes()

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
                "Use structured_no_notes_baseline as the notes-modality ablation and "
                "data_availability_moe as the dynamic multimodal comparator. The "
                "icu_hours sensitivity check remains non-deployable; AUROC "
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


def build_no_notes_pathway_model_entry(
    metrics: dict, subgroup_df: pd.DataFrame
) -> dict:
    calibrated = metrics["main"]["calibrated"]
    return {
        "name": NO_NOTES_MODEL_NAME,
        "verdict": "CONDITIONAL",
        "conditions": no_notes_conditions_text(),
        "narrative": no_notes_narrative_text(metrics),
        "metrics": compact_metrics(calibrated),
        "subgroup_gaps": subgroup_gap_dict(subgroup_df),
    }


def build_no_notes_safety_model_entry(
    metrics: dict,
    subgroup_df: pd.DataFrame,
    coefficients_df: pd.DataFrame,
    threshold_sweep_rows: list[dict],
) -> dict:
    calibrated = metrics["main"]["calibrated"]
    top_pos = top_coefficients(coefficients_df, "positive")
    top_neg = top_coefficients(coefficients_df, "negative")
    worst = worst_subgroup(subgroup_df)
    orthopaedic = subgroup_value_summary(subgroup_df, "surgery_type", "orthopaedic")
    low_sofa = subgroup_value_summary(subgroup_df, "sofa_quartile", "q1_lowest")
    elective = subgroup_value_summary(subgroup_df, "admission_urgency", "elective")

    return {
        "name": NO_NOTES_MODEL_NAME,
        "label": "Structured no-notes modality ablation",
        "verdict": "CONDITIONAL",
        "conditions": no_notes_conditions_text(),
        "narrative": no_notes_narrative_text(metrics),
        "metrics": compact_metrics(calibrated),
        "subgroup_gaps": subgroup_gap_dict(subgroup_df),
        "deployment_questions": {
            "q1_miss_rate": (
                f"At threshold {calibrated['threshold']:.4f}, the no-notes model missed "
                f"{calibrated['fn']} of {calibrated['n_positive']} deteriorations on the "
                "held-out test set."
            ),
            "q2_alert_precision": (
                f"Alert precision (PPV) was {calibrated['ppv']:.3f}; "
                f"{calibrated['tp'] + calibrated['fp']} test patients were flagged."
            ),
            "q3_clearance_safety": (
                f"NPV was {calibrated['npv']:.3f}. A low no-notes risk score should not "
                "clear a patient without standard bedside monitoring because the notes "
                "modality was intentionally unavailable."
            ),
            "q4_discrimination": (
                f"AUROC was {calibrated['auroc']:.3f} and AUPRC was "
                f"{calibrated['auprc']:.3f}. Largest observed subgroup sensitivity gap "
                f"was {worst['gap_text']} in {worst['label']}."
            ),
            "q5_calibration": (
                f"Calibrated Brier score was {calibrated['brier_score']:.3f}. "
                "Probabilities are calibrated on the validation split and should be "
                "rechecked prospectively if notes are absent in a new ward workflow."
            ),
        },
        "failure_analysis": {
            "who_is_missed": (
                f"Misses concentrate most in {worst['label']} "
                f"(sensitivity {worst['sensitivity_text']}, n={worst['n']}, "
                f"positives={worst['positives']})."
            ),
            "false_alarm_profile": (
                f"The threshold produced {calibrated['fp']} false positives and "
                f"specificity {calibrated['specificity']:.3f}; alert burden is similar "
                "to the note-aware structured baseline but without note context."
            ),
            "feature_importance_interpretation": (
                f"Top positive coefficients without notes: {', '.join(top_pos)}. "
                f"Top negative coefficients: {', '.join(top_neg)}."
            ),
            "model_disagreement": (
                "This model is the planned notes-modality ablation: compare it with "
                "clinical_elastic_net_baseline and data_availability_moe to isolate the "
                "incremental evidence carried by note-risk and NLP features."
            ),
        },
        "option_specific": {
            "type": "threshold_sensitivity_report",
            "title": "No-notes modality threshold sensitivity",
            "content": {
                "thresholds_tested": [row["threshold"] for row in threshold_sweep_rows],
                "signal_at_each_threshold": threshold_sweep_rows,
                "recommended_threshold": calibrated["threshold"],
                "rationale": (
                    "Recommended threshold is the highest validation-set calibrated "
                    "risk threshold that achieved target sensitivity >= 0.80 while "
                    "excluding note availability and note-risk features."
                ),
            },
        },
        "explainability": {
            "output_type": "standardized_log_odds_coefficients",
            "description": (
                "Linear coefficients from an elastic-net logistic model that excludes "
                "has_notes, note_risk_score_imputed, and note_risk_score_missing. "
                "The remaining structured, laboratory, and vital-sign features are "
                "median-imputed and standardized."
            ),
        },
        "failure_catalogue": [
            {
                "mode": "Orthopaedic sensitivity gap",
                "example": orthopaedic,
                "subgroup": "surgery_type=orthopaedic",
            },
            {
                "mode": "Low SOFA deterioration misses",
                "example": low_sofa,
                "subgroup": "sofa_quartile=q1_lowest",
            },
            {
                "mode": "Elective admission under-detection",
                "example": elective,
                "subgroup": "admission_urgency=elective",
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


def no_notes_conditions_text() -> str:
    return (
        "Conditional comparator only: use to quantify performance when clinical notes "
        "are unavailable, not as a preferred standalone deployment model. Any deployment "
        "must audit orthopaedic, low-SOFA, and elective-admission sensitivity."
    )


def no_notes_narrative_text(metrics: dict) -> str:
    calibrated = metrics["main"]["calibrated"]
    return (
        "Structured no-notes modality ablation trained the same calibrated elastic-net "
        "logistic architecture as the clinical baseline but excluded has_notes, "
        "note_risk_score_imputed, and note_risk_score_missing. On held-out test data, "
        f"AUROC was {calibrated['auroc']:.3f}, AUPRC {calibrated['auprc']:.3f}, "
        f"sensitivity {calibrated['sensitivity']:.3f}, and Brier score "
        f"{calibrated['brier_score']:.3f}. The result supports modality comparison "
        "against note-aware and MoE models rather than replacing them."
    )


def modality_comparison_notes() -> str:
    return (
        "Submission compares three ICU deterioration approaches: a structured no-notes "
        "ablation, a note-aware structured elastic-net baseline, and a data-availability "
        "MoE with dynamic vitals, notes, and uncertainty outputs."
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


def subgroup_value_summary(
    subgroup_df: pd.DataFrame, group_col: str, group_value: str
) -> str:
    rows = subgroup_df[
        (subgroup_df["group_col"] == group_col)
        & (subgroup_df["group_value"].astype(str) == group_value)
    ]
    if rows.empty:
        return f"{group_col}={group_value}: subgroup metrics unavailable."
    row = rows.iloc[0]
    sens = "unavailable" if pd.isna(row["sensitivity"]) else f"{row['sensitivity']:.3f}"
    gap = (
        "unavailable"
        if pd.isna(row["sensitivity_gap_vs_overall"])
        else f"{row['sensitivity_gap_vs_overall']:.3f}"
    )
    return (
        f"{group_col}={group_value}: sensitivity {sens}, gap vs overall {gap}, "
        f"n={int(row['n'])}, positives={int(row['positives'])}."
    )


def top_coefficients(
    coefficients_df: pd.DataFrame, direction: str, n: int = 5
) -> list[str]:
    rows = coefficients_df[coefficients_df["direction"] == direction]
    rows = rows.sort_values("coefficient", ascending=(direction == "negative")).head(n)
    return [
        f"{row.feature} ({row.coefficient:.3f})" for row in rows.itertuples(index=False)
    ]


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


def prune_placeholder_models(models: list[dict]) -> list[dict]:
    return [
        model for model in models if model.get("name") not in PLACEHOLDER_MODEL_NAMES
    ]


def is_blank(value: Any) -> bool:
    return not str(value).strip()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(to_jsonable(data), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
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
