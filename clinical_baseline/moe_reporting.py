"""Reference JSON updater for the data-availability MoE."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from clinical_baseline.constants import MOE_MODEL_NAME
from clinical_baseline.reporting import to_jsonable


def update_moe_reference_reports(
    reference_dir: Path,
    t24_metrics: dict,
    primary_subgroups: pd.DataFrame,
    route_metrics: pd.DataFrame,
    uncertainty_metrics: pd.DataFrame,
    model_summary: dict,
) -> None:
    reference_dir.mkdir(parents=True, exist_ok=True)
    pathway_path = reference_dir / "omaib_pathway.json"
    safety_path = reference_dir / "model_safety_report.json"

    pathway = load_json(pathway_path)
    safety = load_json(safety_path)
    subgroup_gaps = subgroup_gap_dict(primary_subgroups)
    worst = worst_subgroup(primary_subgroups)
    route_summary = route_metric_summary(route_metrics)
    uncertainty_summary = uncertainty_metric_summary(uncertainty_metrics)

    pathway.setdefault("schema_version", "omaib-clinical")
    pathway.setdefault("submission_type", "model_safety_report")
    pathway.setdefault("strand", "clinical")
    pathway.setdefault("hackathon", "MultimodalAI26")
    pathway["submitted"] = str(date.today())
    pathway.setdefault("team", {"name": "", "members": ""})
    pathway["models"] = upsert_model(
        pathway.get("models", []),
        {
            "name": MOE_MODEL_NAME,
            "verdict": "CONDITIONAL",
            "conditions": conditions_text(),
            "narrative": narrative_text(t24_metrics, model_summary),
            "metrics": compact_metrics(t24_metrics),
            "subgroup_gaps": subgroup_gaps,
        },
    )
    pathway.setdefault("overall_notes", "")

    safety.setdefault("schema_version", "omaib-clinical")
    safety.setdefault("report_type", "model_safety_report")
    safety.setdefault("strand", "clinical")
    safety.setdefault("hackathon", "MultimodalAI26")
    safety["evaluation_date"] = str(date.today())
    safety.setdefault("team", {"name": "", "members": ""})
    safety["models"] = upsert_model(
        safety.get("models", []),
        {
            "name": MOE_MODEL_NAME,
            "label": "Data-availability MoE with dynamic belief update",
            "verdict": "CONDITIONAL",
            "conditions": conditions_text(),
            "narrative": narrative_text(t24_metrics, model_summary),
            "metrics": compact_metrics(t24_metrics),
            "subgroup_gaps": subgroup_gaps,
            "deployment_questions": {
                "q1_miss_rate": (
                    f"At threshold {t24_metrics['threshold']:.4f}, the t24 belief model "
                    f"missed {t24_metrics['fn']} of {t24_metrics['n_positive']} held-out "
                    "deteriorations."
                ),
                "q2_alert_precision": (
                    f"PPV was {t24_metrics['ppv']:.3f}; alert count was "
                    f"{t24_metrics['alert_count']} at t24."
                ),
                "q3_clearance_safety": (
                    f"NPV was {t24_metrics['npv']:.3f}. Low-risk outputs with high "
                    "uncertainty are routed to manual review rather than safe all-clear."
                ),
                "q4_discrimination": (
                    f"t24 AUROC was {t24_metrics['auroc']:.3f} and AUPRC was "
                    f"{t24_metrics['auprc']:.3f}. Largest primary subgroup sensitivity "
                    f"gap was {worst['gap_text']} in {worst['label']}."
                ),
                "q5_calibration": (
                    f"t24 Brier score was {t24_metrics['brier_score']:.3f}. MoE outputs "
                    "use route-specific Platt calibration before belief updating."
                ),
            },
            "failure_analysis": {
                "who_is_missed": (
                    f"Most adverse primary subgroup gap: {worst['label']} with sensitivity "
                    f"{worst['sensitivity_text']}, n={worst['n']}, positives={worst['n_positive']}."
                ),
                "false_alarm_profile": (
                    f"At t24 the model produced {t24_metrics['fp']} false positives and "
                    f"specificity {t24_metrics['specificity']:.3f}."
                ),
                "feature_importance_interpretation": (
                    "Interpretation is route based: E0 is the structured fallback, E1 adds "
                    "note_risk_score plus auxiliary NLP indicators when notes exist, and E2 "
                    "corrects the no-notes MNAR subgroup if validation supports it."
                ),
                "model_disagreement": route_summary,
            },
            "option_specific": {
                "type": "threshold_sensitivity_report",
                "title": "Dynamic MoE threshold and uncertainty review",
                "content": {
                    "recommended_threshold": t24_metrics["threshold"],
                    "lambda_with": model_summary["lambda_with"],
                    "lambda_no": model_summary["lambda_no"],
                    "uncertainty_summary": uncertainty_summary,
                    "rationale": (
                        "Threshold was selected on validation t24 calibrated MoE risk to "
                        "achieve target sensitivity >= 0.80. High-uncertainty cases are "
                        "routed to senior/manual review."
                    ),
                },
            },
            "explainability": {
                "output_type": "route_experts_plus_uncertainty_components",
                "description": (
                    "The model reports route, E0/E1/E2 predictions, expert disagreement, "
                    "bootstrap risk interval, data uncertainty, entropy, belief updates, "
                    "and recommended action."
                ),
            },
            "failure_catalogue": [
                {
                    "mode": "Residual false negatives",
                    "example": (
                        f"{t24_metrics['fn']} deteriorating test patients remained below "
                        "the alert threshold at t24."
                    ),
                    "subgroup": worst["label"],
                },
                {
                    "mode": "Route-specific missingness risk",
                    "example": route_summary,
                    "subgroup": "has_notes_route",
                },
                {
                    "mode": "High uncertainty low-risk outputs",
                    "example": uncertainty_summary,
                    "subgroup": "uncertainty_groups",
                },
            ],
        },
    )
    safety.setdefault("overall_notes", "")

    write_json(pathway_path, pathway)
    write_json(safety_path, safety)


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


def subgroup_gap_dict(primary_subgroups: pd.DataFrame) -> dict:
    t24 = primary_subgroups[primary_subgroups["stage"] == "t24"]
    return {
        f"{row.group_col}_{row.group_value}": round_value(row.sensitivity_gap_vs_overall)
        for row in t24.itertuples(index=False)
    }


def worst_subgroup(primary_subgroups: pd.DataFrame) -> dict:
    t24 = primary_subgroups[
        (primary_subgroups["stage"] == "t24")
        & primary_subgroups["sensitivity_gap_vs_overall"].notna()
    ].copy()
    if t24.empty:
        return {
            "label": "unavailable",
            "gap_text": "unavailable",
            "sensitivity_text": "unavailable",
            "n": 0,
            "n_positive": 0,
        }
    row = t24.sort_values("sensitivity_gap_vs_overall").iloc[0]
    return {
        "label": f"{row['group_col']}={row['group_value']}",
        "gap_text": f"{row['sensitivity_gap_vs_overall']:.3f}",
        "sensitivity_text": f"{row['sensitivity']:.3f}",
        "n": int(row["n_total"]),
        "n_positive": int(row["n_positive"]),
    }


def route_metric_summary(route_metrics: pd.DataFrame) -> str:
    t24 = route_metrics[route_metrics["stage"] == "t24"]
    parts = []
    for row in t24.itertuples(index=False):
        parts.append(
            f"{row.group_value}: AUROC {fmt(row.auroc)}, Brier {fmt(row.brier_score)}, "
            f"sensitivity {fmt(row.sensitivity)}, n={row.n_total}"
        )
    return "; ".join(parts) if parts else "Route metrics unavailable."


def uncertainty_metric_summary(uncertainty_metrics: pd.DataFrame) -> str:
    t24 = uncertainty_metrics[
        (uncertainty_metrics["stage"] == "t24")
        & (uncertainty_metrics["group_col"] == "recommended_action")
    ]
    parts = []
    for row in t24.itertuples(index=False):
        parts.append(
            f"{row.group_value}: n={row.n}, event_rate {fmt(row.event_rate)}, "
            f"error_rate {fmt(row.error_rate)}"
        )
    return "; ".join(parts) if parts else "Uncertainty action metrics unavailable."


def conditions_text() -> str:
    return (
        "Conditional retrospective simulation only. Validate prospectively before use, "
        "treat t24 as full-information reference rather than clean prospective prediction, "
        "and audit no-notes, surgical, and high-uncertainty subgroups."
    )


def narrative_text(metrics: dict, model_summary: dict) -> str:
    return (
        "Data-availability MoE trained three shallow LightGBM experts: structured fallback, "
        "with-notes, and no-notes MNAR-context. At t24 on held-out test data, calibrated "
        f"belief AUROC was {metrics['auroc']:.3f}, AUPRC {metrics['auprc']:.3f}, "
        f"sensitivity {metrics['sensitivity']:.3f}, and Brier {metrics['brier_score']:.3f}. "
        f"Validation selected lambda_with={model_summary['lambda_with']:.1f} and "
        f"lambda_no={model_summary['lambda_no']:.1f}."
    )


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


def round_value(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 4)
    return value


def fmt(value) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"
