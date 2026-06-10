#!/usr/bin/env python3
"""Feature attribution analysis for the data-availability MoE."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import textwrap
from datetime import date
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_baseline.constants import TARGET  # noqa: E402
from clinical_baseline.moe_features import (  # noqa: E402
    E0_FEATURES,
    E2_FEATURES,
    MOE_CATEGORICAL_FEATURES,
    build_stage_features,
    build_stacked_stage_features,
    validate_moe_source_columns,
)
from clinical_baseline.moe_model import DataAvailabilityMoE  # noqa: E402


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
COLOR_FAMILIES = {
    "blue": {"base": "#A3BEFA", "dark": "#2E4780"},
    "gold": {"base": "#FFE15B", "dark": "#736422"},
    "orange": {"base": "#F0986E", "dark": "#804126"},
    "olive": {"base": "#A3D576", "dark": "#386411"},
    "pink": {"base": "#F390CA", "dark": "#8A3A6F"},
}
FEATURE_LABELS = {
    "sex": "Sex",
    "surgery_type": "Surgery type",
    "admission_urgency": "Admission urgency",
    "age": "Age",
    "asa_class": "ASA class",
    "op_duration_h": "Operation duration",
    "blood_loss_imputed": "Blood loss",
    "blood_loss_missing": "Blood loss missing",
    "transfused": "Transfused",
    "has_diabetes": "Diabetes",
    "has_hypertension": "Hypertension",
    "preop_creatinine": "Pre-op creatinine",
    "preop_wbc": "Pre-op WBC",
    "preop_lactate": "Pre-op lactate",
    "sofa_score": "SOFA score",
    "icu_lactate": "ICU lactate",
    "icu_creatinine": "ICU creatinine",
    "icu_wbc": "ICU WBC",
    "icu_bilirubin": "ICU bilirubin",
    "has_notes": "Clinical note present",
    "hr_mean": "Heart rate mean",
    "hr_std": "Heart rate variability",
    "rr_mean": "Respiratory rate mean",
    "rr_std": "Respiratory rate variability",
    "spo2_mean": "SpO2 mean",
    "spo2_min": "SpO2 minimum",
    "sbp_mean": "Systolic BP mean",
    "temp_mean": "Temperature mean",
    "lactate_rolling_mean": "Rolling lactate mean",
    "lactate_rolling_max": "Rolling lactate max",
    "lactate_slope": "Lactate slope",
    "note_risk_score": "Note risk score",
    "note_text_risk_score": "Note text risk score",
    "note_hemodynamic_instability": "Note: hemodynamic instability",
    "note_vasopressor_support": "Note: vasopressor support",
    "note_borderline_urine_output": "Note: borderline urine output",
    "note_lactate_elevated": "Note: lactate elevated",
    "note_blood_loss_not_measured": "Note: blood loss not measured",
    "note_transfusion_mentioned": "Note: transfusion mentioned",
    "note_close_monitoring": "Note: close monitoring",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoE feature attribution analysis.")
    parser.add_argument("--data", default=ROOT / "data/raw/icu_patients.csv", type=Path)
    parser.add_argument("--artifacts", default=ROOT / "artifacts/data_availability_moe", type=Path)
    parser.add_argument(
        "--out",
        default=ROOT / "artifacts/data_availability_moe/feature_attribution",
        type=Path,
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--target-sensitivity", default=0.80, type=float)
    parser.add_argument("--n-repeats", default=5, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = resolve_path(args.data)
    artifacts_dir = resolve_path(args.artifacts)
    out_dir = resolve_path(args.out)
    chart_dir = out_dir / "charts"
    table_dir = out_dir / "tables"
    chart_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    validate_moe_source_columns(df)
    train_df, validation_df, test_df = stratified_split(df, args.seed)

    model = DataAvailabilityMoE(
        target_sensitivity=args.target_sensitivity,
        seed=args.seed,
        use_aux_note_features=True,
    ).fit(train_df, validation_df, sample_label="feature_attribution")

    x_train_t24 = build_stage_features(
        train_df,
        stage="t24",
        note_extractor=model.note_extractor_,
        include_note_features=True,
    )
    x_test_t24 = build_stage_features(
        test_df,
        stage="t24",
        note_extractor=model.note_extractor_,
        include_note_features=True,
    )
    y_test = test_df[TARGET].astype(int).to_numpy()
    base_pred = predict_final_stage_from_features(model, test_df, x_test_t24)
    base_prob = base_pred["p_calibrated"]
    base_metrics = metric_dict(y_test, base_prob, model.threshold_)

    final_importance = permutation_importance_final(
        model=model,
        raw_df=test_df,
        x_stage=x_test_t24,
        y_true=y_test,
        features=E0_FEATURES,
        n_repeats=args.n_repeats,
        seed=args.seed,
    )
    with_notes_importance = permutation_importance_route_expert(
        model=model,
        route="with_notes",
        raw_df=test_df,
        x_stage=x_test_t24,
        y_true=y_test,
        n_repeats=args.n_repeats,
        seed=args.seed + 100,
    )
    no_notes_importance = permutation_importance_route_expert(
        model=model,
        route="no_notes",
        raw_df=test_df,
        x_stage=x_test_t24,
        y_true=y_test,
        n_repeats=args.n_repeats,
        seed=args.seed + 200,
    )

    baselines = feature_baselines(x_train_t24, E0_FEATURES)
    local_cases = select_local_cases(test_df, base_prob, model.threshold_)
    local_attr = local_counterfactual_attribution(
        model=model,
        raw_df=test_df,
        x_stage=x_test_t24,
        prob=base_prob,
        selected_cases=local_cases,
        baselines=baselines,
        features=E0_FEATURES,
    )
    partial_effects = partial_effect_table(
        model=model,
        raw_df=test_df,
        x_stage=x_test_t24,
        importance=final_importance,
        max_features=6,
    )

    final_importance.to_csv(table_dir / "final_risk_permutation_importance.csv", index=False)
    with_notes_importance.to_csv(table_dir / "with_notes_expert_permutation_importance.csv", index=False)
    no_notes_importance.to_csv(table_dir / "no_notes_expert_permutation_importance.csv", index=False)
    local_attr.to_csv(table_dir / "local_counterfactual_attributions.csv", index=False)
    partial_effects.to_csv(table_dir / "partial_effect_top_features.csv", index=False)

    original_metrics = read_json_if_exists(artifacts_dir / "metrics.json")
    original_weights = read_json_if_exists(artifacts_dir / "expert_weights.json")

    use_chart_theme()
    charts = {
        "global_importance": plot_global_importance(final_importance, chart_dir),
        "partial_effects": plot_partial_effects(partial_effects, chart_dir),
        "local_case": plot_local_case(local_attr, chart_dir),
        "route_importance": plot_route_importance(with_notes_importance, no_notes_importance, chart_dir),
    }

    summary = build_summary(
        args=args,
        base_metrics=base_metrics,
        model=model,
        final_importance=final_importance,
        local_attr=local_attr,
        partial_effects=partial_effects,
        original_metrics=original_metrics,
        original_weights=original_weights,
    )
    write_json(out_dir / "summary.json", summary)
    write_source_notes(out_dir / "source_notes.md", data_path, artifacts_dir, summary)
    write_report(out_dir / "report.html", charts, summary)

    print(f"Feature attribution report: {out_dir / 'report.html'}")
    print(f"Attribution tables: {table_dir}")
    print(f"Charts: {chart_dir}")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def stratified_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_validation_df, test_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df[TARGET],
        random_state=seed,
    )
    train_df, validation_df = train_test_split(
        train_validation_df,
        test_size=0.25,
        stratify=train_validation_df[TARGET],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), validation_df.reset_index(drop=True), test_df.reset_index(drop=True)


def predict_final_stage_from_features(
    model: DataAvailabilityMoE,
    raw_df: pd.DataFrame,
    x_stage: pd.DataFrame,
) -> dict[str, np.ndarray]:
    has_notes = raw_df["has_notes"].astype(int).to_numpy()
    notes_mask = has_notes == 1
    p0 = model.e0_.predict_proba(x_stage[E0_FEATURES])[:, 1]
    p1 = np.full(len(x_stage), np.nan)
    p2 = np.full(len(x_stage), np.nan)
    if notes_mask.any():
        p1[notes_mask] = model.e1_.predict_proba(x_stage.loc[notes_mask, model.e1_features_])[:, 1]
    if (~notes_mask).any():
        p2[~notes_mask] = model.e2_.predict_proba(x_stage.loc[~notes_mask, E2_FEATURES])[:, 1]
    p_raw = model.fuse_raw(p0, p1, p2, has_notes)
    p_calibrated = np.empty(len(x_stage), dtype=float)
    if notes_mask.any():
        p_calibrated[notes_mask] = model.calibrator_with_.predict(p_raw[notes_mask])
    if (~notes_mask).any():
        p_calibrated[~notes_mask] = model.calibrator_no_.predict(p_raw[~notes_mask])
    return {
        "p_struct": p0,
        "p_with_notes": p1,
        "p_no_notes": p2,
        "p_raw": p_raw,
        "p_calibrated": p_calibrated,
    }


def metric_dict(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = prob >= threshold
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return {
        "auroc": float(roc_auc_score(y_true, prob)),
        "auprc": float(average_precision_score(y_true, prob)),
        "brier_score": float(brier_score_loss(y_true, prob)),
        "threshold": float(threshold),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "ppv": safe_div(tp, tp + fp),
        "npv": safe_div(tn, tn + fn),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_total": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }


def safe_div(num: int, den: int) -> float:
    return float(num / den) if den else float("nan")


def permutation_importance_final(
    model: DataAvailabilityMoE,
    raw_df: pd.DataFrame,
    x_stage: pd.DataFrame,
    y_true: np.ndarray,
    features: list[str],
    n_repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_prob = predict_final_stage_from_features(model, raw_df, x_stage)["p_calibrated"]
    base_auroc = roc_auc_score(y_true, base_prob)
    base_brier = brier_score_loss(y_true, base_prob)
    rows = []
    for feature in features:
        auroc_drops = []
        brier_increases = []
        for _ in range(n_repeats):
            x_perm = x_stage.copy()
            x_perm[feature] = rng.permutation(x_perm[feature].to_numpy())
            prob = predict_final_stage_from_features(model, raw_df, x_perm)["p_calibrated"]
            auroc_drops.append(base_auroc - roc_auc_score(y_true, prob))
            brier_increases.append(brier_score_loss(y_true, prob) - base_brier)
        rows.append(
            {
                "feature": feature,
                "feature_label": feature_label(feature),
                "model_component": "final_t24_risk",
                "auroc_drop_mean": float(np.mean(auroc_drops)),
                "auroc_drop_std": float(np.std(auroc_drops)),
                "brier_increase_mean": float(np.mean(brier_increases)),
                "brier_increase_std": float(np.std(brier_increases)),
            }
        )
    out = pd.DataFrame(rows)
    out["importance_score"] = out["auroc_drop_mean"].clip(lower=0) + out["brier_increase_mean"].clip(lower=0)
    return out.sort_values(["importance_score", "auroc_drop_mean"], ascending=False).reset_index(drop=True)


def permutation_importance_route_expert(
    model: DataAvailabilityMoE,
    route: str,
    raw_df: pd.DataFrame,
    x_stage: pd.DataFrame,
    y_true: np.ndarray,
    n_repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if route == "with_notes":
        mask = raw_df["has_notes"].eq(1).to_numpy()
        estimator = model.e1_
        features = model.e1_features_
        component = "E1_with_notes_expert"
    elif route == "no_notes":
        mask = raw_df["has_notes"].eq(0).to_numpy()
        estimator = model.e2_
        features = E2_FEATURES
        component = "E2_no_notes_expert"
    else:
        raise ValueError(f"Unknown route: {route}")

    x_route = x_stage.loc[mask, features].reset_index(drop=True)
    y_route = y_true[mask]
    if len(np.unique(y_route)) < 2:
        return pd.DataFrame()

    base_prob = estimator.predict_proba(x_route)[:, 1]
    base_auroc = roc_auc_score(y_route, base_prob)
    base_brier = brier_score_loss(y_route, base_prob)
    rows = []
    for feature in features:
        auroc_drops = []
        brier_increases = []
        for _ in range(n_repeats):
            x_perm = x_route.copy()
            x_perm[feature] = rng.permutation(x_perm[feature].to_numpy())
            prob = estimator.predict_proba(x_perm)[:, 1]
            auroc_drops.append(base_auroc - roc_auc_score(y_route, prob))
            brier_increases.append(brier_score_loss(y_route, prob) - base_brier)
        rows.append(
            {
                "feature": feature,
                "feature_label": feature_label(feature),
                "model_component": component,
                "route": route,
                "n_route": int(len(y_route)),
                "auroc_drop_mean": float(np.mean(auroc_drops)),
                "auroc_drop_std": float(np.std(auroc_drops)),
                "brier_increase_mean": float(np.mean(brier_increases)),
                "brier_increase_std": float(np.std(brier_increases)),
            }
        )
    out = pd.DataFrame(rows)
    out["importance_score"] = out["auroc_drop_mean"].clip(lower=0) + out["brier_increase_mean"].clip(lower=0)
    return out.sort_values(["importance_score", "auroc_drop_mean"], ascending=False).reset_index(drop=True)


def feature_baselines(x_train: pd.DataFrame, features: list[str]) -> dict[str, object]:
    baselines: dict[str, object] = {}
    for feature in features:
        if feature in MOE_CATEGORICAL_FEATURES:
            mode = x_train[feature].mode(dropna=True)
            baselines[feature] = mode.iloc[0] if not mode.empty else "missing"
        else:
            baselines[feature] = float(pd.to_numeric(x_train[feature], errors="coerce").median())
    return baselines


def select_local_cases(raw_df: pd.DataFrame, prob: np.ndarray, threshold: float) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "row_index": np.arange(len(raw_df)),
            "patient_id": raw_df["patient_id"].astype(str).to_numpy(),
            "y_true": raw_df[TARGET].astype(int).to_numpy(),
            "risk": prob,
            "predicted_positive": prob >= threshold,
        }
    )
    cases = []
    tp = frame[(frame["y_true"].eq(1)) & frame["predicted_positive"]].sort_values("risk", ascending=False)
    fp = frame[(frame["y_true"].eq(0)) & frame["predicted_positive"]].sort_values("risk", ascending=False)
    fn = frame[(frame["y_true"].eq(1)) & ~frame["predicted_positive"]].sort_values("risk", ascending=False)
    for label, subset in [
        ("high_risk_true_positive", tp),
        ("high_risk_false_positive", fp),
        ("near_threshold_false_negative", fn),
    ]:
        if not subset.empty:
            row = subset.iloc[0].copy()
            row["case_label"] = label
            cases.append(row)
    return pd.DataFrame(cases)


def local_counterfactual_attribution(
    model: DataAvailabilityMoE,
    raw_df: pd.DataFrame,
    x_stage: pd.DataFrame,
    prob: np.ndarray,
    selected_cases: pd.DataFrame,
    baselines: dict[str, object],
    features: list[str],
) -> pd.DataFrame:
    rows = []
    for case in selected_cases.itertuples(index=False):
        idx = int(case.row_index)
        raw_one = raw_df.iloc[[idx]].reset_index(drop=True)
        x_one = x_stage.iloc[[idx]].reset_index(drop=True)
        original_risk = float(prob[idx])
        for feature in features:
            x_counterfactual = x_one.copy()
            x_counterfactual[feature] = baselines[feature]
            counterfactual_risk = float(
                predict_final_stage_from_features(model, raw_one, x_counterfactual)["p_calibrated"][0]
            )
            rows.append(
                {
                    "case_label": case.case_label,
                    "patient_id": case.patient_id,
                    "y_true": int(case.y_true),
                    "original_risk": original_risk,
                    "feature": feature,
                    "feature_label": feature_label(feature),
                    "observed_value": x_one[feature].iloc[0],
                    "baseline_value": baselines[feature],
                    "counterfactual_risk": counterfactual_risk,
                    "risk_contribution_vs_baseline": original_risk - counterfactual_risk,
                }
            )
    out = pd.DataFrame(rows)
    out["abs_contribution"] = out["risk_contribution_vs_baseline"].abs()
    return out.sort_values(["case_label", "abs_contribution"], ascending=[True, False]).reset_index(drop=True)


def partial_effect_table(
    model: DataAvailabilityMoE,
    raw_df: pd.DataFrame,
    x_stage: pd.DataFrame,
    importance: pd.DataFrame,
    max_features: int,
) -> pd.DataFrame:
    rows = []
    top_features = importance[~importance["feature"].isin(MOE_CATEGORICAL_FEATURES)].head(max_features)
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    for feature in top_features["feature"]:
        values = pd.to_numeric(x_stage[feature], errors="coerce")
        grid = np.nanquantile(values, quantiles)
        for quantile, value in zip(quantiles, grid):
            x_modified = x_stage.copy()
            x_modified[feature] = float(value)
            risk = predict_final_stage_from_features(model, raw_df, x_modified)["p_calibrated"]
            rows.append(
                {
                    "feature": feature,
                    "feature_label": feature_label(feature),
                    "quantile": float(quantile),
                    "quantile_label": f"p{int(quantile * 100):02d}",
                    "grid_value": float(value),
                    "mean_predicted_risk": float(np.mean(risk)),
                }
            )
    return pd.DataFrame(rows)


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )


def add_chart_header(fig: plt.Figure, ax: plt.Axes, title: str, subtitle: str) -> None:
    title = textwrap.fill(title, width=82, break_long_words=False)
    subtitle = textwrap.fill(subtitle, width=112, break_long_words=False)
    title_lines = title.count("\n") + 1
    subtitle_lines = subtitle.count("\n") + 1
    ax.set_title("")
    fig.subplots_adjust(
        top=max(0.58, 0.80 - 0.05 * (title_lines - 1) - 0.03 * (subtitle_lines - 1))
    )
    left = ax.get_position().x0
    fig.text(left, 0.975, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=TOKENS["ink"])
    fig.text(
        left,
        0.925 - 0.052 * (title_lines - 1),
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    sns.despine(ax=ax)


def save_chart(fig: plt.Figure, path: Path) -> str:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_global_importance(importance: pd.DataFrame, chart_dir: Path) -> str:
    plot_df = importance.head(14).sort_values("auroc_drop_mean")
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    sns.barplot(
        data=plot_df,
        x="auroc_drop_mean",
        y="feature_label",
        color=COLOR_FAMILIES["orange"]["base"],
        edgecolor=COLOR_FAMILIES["orange"]["dark"],
        linewidth=0.8,
        ax=ax,
    )
    ax.set_xlabel("AUROC drop after permutation")
    ax.set_ylabel("")
    add_chart_header(
        fig,
        ax,
        "Global attribution ranks features by how much permutation damages t24 risk ranking",
        "Attribution model retrained with the original split and MoE feature pipeline; larger AUROC drop means stronger global contribution.",
    )
    return save_chart(fig, chart_dir / "global_final_risk_permutation_importance.png")


def plot_partial_effects(partial: pd.DataFrame, chart_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    palette = [
        COLOR_FAMILIES["blue"]["base"],
        COLOR_FAMILIES["orange"]["base"],
        COLOR_FAMILIES["olive"]["base"],
        COLOR_FAMILIES["pink"]["base"],
        COLOR_FAMILIES["gold"]["base"],
        COLOR_FAMILIES["blue"]["dark"],
    ]
    sns.lineplot(
        data=partial,
        x="quantile",
        y="mean_predicted_risk",
        hue="feature_label",
        marker="o",
        palette=palette[: partial["feature_label"].nunique()],
        linewidth=1.2,
        ax=ax,
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Counterfactual feature quantile")
    ax.set_ylabel("Mean predicted risk")
    add_chart_header(
        fig,
        ax,
        "Partial effects show how feature quantiles shift average risk",
        "Each line sets one feature to selected quantiles while holding other t24 features fixed.",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, ncol=3, borderaxespad=0)
    fig.subplots_adjust(top=0.79, bottom=0.26)
    return save_chart(fig, chart_dir / "partial_effect_top_features.png")


def plot_local_case(local_attr: pd.DataFrame, chart_dir: Path) -> str:
    first_case = local_attr["case_label"].iloc[0]
    plot_df = local_attr[local_attr["case_label"].eq(first_case)].head(10).sort_values(
        "risk_contribution_vs_baseline"
    )
    colors = np.where(
        plot_df["risk_contribution_vs_baseline"].ge(0),
        COLOR_FAMILIES["orange"]["base"],
        COLOR_FAMILIES["blue"]["base"],
    )
    edges = np.where(
        plot_df["risk_contribution_vs_baseline"].ge(0),
        COLOR_FAMILIES["orange"]["dark"],
        COLOR_FAMILIES["blue"]["dark"],
    )
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    bars = ax.barh(plot_df["feature_label"], plot_df["risk_contribution_vs_baseline"], color=colors, edgecolor=edges)
    ax.axvline(0, color=TOKENS["ink"], linewidth=1.0)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Risk contribution versus train-set baseline")
    ax.set_ylabel("")
    for bar, value in zip(bars, plot_df["risk_contribution_vs_baseline"]):
        ax.text(
            value + (0.004 if value >= 0 else -0.004),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1%}",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=8,
            color=TOKENS["ink"],
        )
    add_chart_header(
        fig,
        ax,
        "Local attribution explains one prediction by resetting each feature to baseline",
        f"Case {plot_df['patient_id'].iloc[0]} ({first_case}); positive values pushed predicted risk above baseline.",
    )
    return save_chart(fig, chart_dir / "local_counterfactual_case.png")


def plot_route_importance(with_notes: pd.DataFrame, no_notes: pd.DataFrame, chart_dir: Path) -> str:
    keep = []
    for route_df, route_label in [(with_notes, "with_notes expert"), (no_notes, "no_notes expert")]:
        part = route_df.head(8).copy()
        part["component_label"] = route_label
        keep.append(part)
    plot_df = pd.concat(keep, ignore_index=True)
    plot_df["display_feature"] = plot_df["component_label"] + " | " + plot_df["feature_label"]
    plot_df = plot_df.sort_values("auroc_drop_mean")
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    sns.barplot(
        data=plot_df,
        x="auroc_drop_mean",
        y="display_feature",
        hue="component_label",
        palette={
            "with_notes expert": COLOR_FAMILIES["gold"]["base"],
            "no_notes expert": COLOR_FAMILIES["pink"]["base"],
        },
        dodge=False,
        edgecolor=TOKENS["ink"],
        linewidth=0.7,
        ax=ax,
    )
    ax.set_xlabel("AUROC drop after permutation")
    ax.set_ylabel("")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2, borderaxespad=0)
    add_chart_header(
        fig,
        ax,
        "Route-specific experts use different strongest features even when final lambda is zero",
        "Standalone E1/E2 expert attribution; useful for diagnostics and disagreement review, not direct final-risk weighting.",
    )
    return save_chart(fig, chart_dir / "route_expert_permutation_importance.png")


def build_summary(
    args: argparse.Namespace,
    base_metrics: dict,
    model: DataAvailabilityMoE,
    final_importance: pd.DataFrame,
    local_attr: pd.DataFrame,
    partial_effects: pd.DataFrame,
    original_metrics: dict,
    original_weights: dict,
) -> dict:
    top_feature = final_importance.iloc[0].to_dict()
    local_top = local_attr.iloc[0].to_dict()
    original_t24 = (original_metrics or {}).get("t24_metrics", {})
    return {
        "generated_on": str(date.today()),
        "seed": args.seed,
        "method": "permutation_importance_and_counterfactual_baseline_attribution",
        "attribution_model": {
            "reason": (
                "The saved LightGBM model.joblib cannot be safely unpickled in the current environment, "
                "so attribution is computed from a same-split, same-feature-pipeline MoE refit."
            ),
            "expert_model": model.model_summary()["expert_model"],
            "lambda_with": float(model.lambda_with_),
            "lambda_no": float(model.lambda_no_),
            "threshold": float(model.threshold_),
            "metrics": base_metrics,
        },
        "original_artifact_reference": {
            "threshold": original_t24.get("threshold"),
            "auroc": original_t24.get("auroc"),
            "auprc": original_t24.get("auprc"),
            "brier_score": original_t24.get("brier_score"),
            "sensitivity": original_t24.get("sensitivity"),
            "lambda_with": (original_weights or {}).get("lambda_with"),
            "lambda_no": (original_weights or {}).get("lambda_no"),
        },
        "top_global_feature": top_feature,
        "top_local_feature": local_top,
        "partial_effect_features": partial_effects["feature"].drop_duplicates().tolist(),
    }


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


def write_source_notes(path: Path, data_path: Path, artifacts_dir: Path, summary: dict) -> None:
    text = f"""# MoE Feature Attribution Source Notes

Generated: {summary["generated_on"]}

## Sources

- Raw data: `{data_path}`
- Existing MoE artifacts: `{artifacts_dir}`
- Existing metrics reference: `{artifacts_dir / "metrics.json"}`
- Existing expert weights reference: `{artifacts_dir / "expert_weights.json"}`

## Method

- Recreated the original stratified 60/20/20 train/validation/test split with seed {summary["seed"]}.
- Refit `DataAvailabilityMoE` with the same feature engineering and routing logic in the current environment.
- Computed global permutation importance against t24 calibrated risk.
- Computed local counterfactual attribution by replacing one feature at a time with the train-set t24 baseline value.
- Computed route-specific standalone permutation importance for E1 and E2.

## Caveat

The attribution model is not the serialized historical LightGBM object. The saved `model.joblib`
was created under an older scikit-learn/LightGBM environment and cannot be safely unpickled on
this machine without restoring that environment. Existing artifact metrics are preserved as a
reference, while feature attribution is computed from the compatible refit.
"""
    path.write_text(text, encoding="utf-8")


def write_report(path: Path, charts: dict[str, str], summary: dict) -> None:
    rel = {name: Path(value).relative_to(path.parent) for name, value in charts.items()}
    attr = summary["attribution_model"]
    orig = summary["original_artifact_reference"]
    top = summary["top_global_feature"]
    local = summary["top_local_feature"]
    report = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Data-Availability MoE Feature Attribution</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 40px 20px 72px; }}
    header, section {{ margin-bottom: 34px; }}
    h1, h2, h3 {{ line-height: 1.18; margin: 0 0 12px; letter-spacing: 0; }}
    h1 {{ font-size: 32px; }}
    h2 {{ font-size: 22px; border-top: 1px solid #dbe3ef; padding-top: 24px; }}
    h3 {{ font-size: 17px; margin-top: 22px; }}
    p, li {{ line-height: 1.68; font-size: 15px; }}
    ul, ol {{ padding-left: 22px; }}
    .summary {{ background: #ffffff; border: 1px solid #dbe3ef; border-radius: 8px; padding: 18px 20px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }}
    .metric {{ background: #ffffff; border: 1px solid #dbe3ef; border-radius: 8px; padding: 14px; }}
    .metric strong {{ display: block; font-size: 24px; line-height: 1.1; color: #2E4780; }}
    .metric span {{ color: #475569; font-size: 13px; }}
    figure {{ margin: 20px 0 26px; background: #ffffff; border: 1px solid #dbe3ef; border-radius: 8px; padding: 14px; }}
    figure img {{ display: block; max-width: 100%; height: auto; margin: 0 auto; }}
    figcaption {{ color: #475569; font-size: 13px; margin-top: 8px; line-height: 1.5; }}
    table {{ border-collapse: collapse; width: 100%; background: #ffffff; border: 1px solid #dbe3ef; margin: 16px 0 20px; }}
    th, td {{ border-bottom: 1px solid #dbe3ef; padding: 9px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #edf2f7; }}
    code, pre {{ background: #edf2f7; border-radius: 4px; }}
    code {{ padding: 2px 5px; }}
    pre {{ padding: 14px; overflow-x: auto; }}
    .caveat {{ color: #475569; }}
    @media (max-width: 760px) {{
      main {{ padding: 26px 14px 56px; }}
      .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <main data-report-audience="technical">
    <header data-contract-section="title">
      <h1>Data-Availability MoE Feature Attribution</h1>
      <p class="caveat">生成日期：{esc(summary["generated_on"])}。目标：解释 t24 MoE 风险预测到底被哪些指标推动，以及指标变动方向如何改变预测风险。</p>
    </header>

    <section data-contract-section="technical-summary">
      <h2>技术结论</h2>
      <div class="summary">
        <ul>
          <li><strong>可以做 feature attribution。</strong>本报告实现了 global permutation attribution、partial-effect direction 和 local counterfactual attribution。</li>
          <li><strong>当前最终风险优先解释 E0。</strong>原始 artifacts 的 validation 权重是 <code>lambda_with={fmt(orig.get("lambda_with"))}</code>、<code>lambda_no={fmt(orig.get("lambda_no"))}</code>；当前 attribution refit 得到 <code>lambda_with={attr["lambda_with"]:.1f}</code>、<code>lambda_no={attr["lambda_no"]:.1f}</code>。因此 E1/E2 更适合作为 route-specific diagnostics，而不是声称它们主导最终风险。</li>
          <li><strong>当前 attribution refit 的 t24 表现：</strong>AUROC={attr["metrics"]["auroc"]:.3f}，AUPRC={attr["metrics"]["auprc"]:.3f}，Brier={attr["metrics"]["brier_score"]:.3f}，threshold={attr["threshold"]:.4f}。</li>
          <li><strong>最大 global attribution 特征：</strong>{esc(top["feature_label"])}，permutation 后 AUROC 平均下降 {top["auroc_drop_mean"]:.3f}，Brier 平均增加 {top["brier_increase_mean"]:.3f}。</li>
        </ul>
      </div>
      <div class="metric-grid">
        <div class="metric"><strong>{attr["metrics"]["auroc"]:.3f}</strong><span>Attribution refit AUROC</span></div>
        <div class="metric"><strong>{fmt(orig.get("auroc"))}</strong><span>Original artifact AUROC</span></div>
        <div class="metric"><strong>{attr["expert_model"]}</strong><span>Attribution model expert</span></div>
        <div class="metric"><strong>{attr["threshold"]:.4f}</strong><span>Attribution threshold</span></div>
      </div>
    </section>

    <section data-contract-section="key-findings">
      <h2>Feature Attribution 结果</h2>

      <h3>Global attribution：打乱重要特征会明显损害风险排序</h3>
      <p>Permutation importance 的读法是：只打乱一个特征，其他特征保持不变，然后观察 AUROC/Brier 变差多少。它回答的是“模型全局上依赖哪些指标做 t24 风险排序”。这里最重要的特征是 <strong>{esc(top["feature_label"])}</strong>。</p>
      <figure>
        <img src="{esc(str(rel["global_importance"]))}" alt="Global final-risk permutation importance">
        <figcaption>图 1：final t24 risk 的 global permutation attribution。越靠上说明该指标被打乱后模型越难排序。</figcaption>
      </figure>

      <h3>Feature direction：指标值变化会改变平均预测风险</h3>
      <p>Partial-effect 曲线把一个指标设置到不同分位点，同时保持其他 t24 特征不变。曲线向上表示该指标升高会把平均预测风险推高；曲线向下表示该指标升高会压低平均预测风险。这个图比单纯排序更接近你说的“根据指标变化然后预测”。</p>
      <figure>
        <img src="{esc(str(rel["partial_effects"]))}" alt="Partial effects for top attribution features">
        <figcaption>图 2：top attribution features 的 counterfactual partial effect。注意这是平均效应，不代表每个个体都单调。</figcaption>
      </figure>

      <h3>Local attribution：单个病人的风险由哪些指标推高或压低</h3>
      <p>Local counterfactual attribution 的读法是：对一个病人，把某个指标替换成 train-set baseline，再看风险下降或上升多少。正值表示该病人的实际指标值相对 baseline 把风险推高；负值表示压低风险。示例病例 <strong>{esc(local["patient_id"])}</strong> 的最大局部贡献来自 <strong>{esc(local["feature_label"])}</strong>，贡献 {float(local["risk_contribution_vs_baseline"]):+.1%}。</p>
      <figure>
        <img src="{esc(str(rel["local_case"]))}" alt="Local counterfactual attribution case">
        <figcaption>图 3：一个病例的 local counterfactual attribution。支持表里还有其他示例病例。</figcaption>
      </figure>

      <h3>Route-specific attribution：E1/E2 可解释专家差异，但不是最终风险主权重</h3>
      <p>因为最终融合权重为 0，E1/E2 的 attribution 不能直接解释 final risk；但它们仍能解释 route-specific expert 为什么和 E0 意见不同。这个信息适合用于 high-disagreement review、notes ablation 和 MNAR 分析。</p>
      <figure>
        <img src="{esc(str(rel["route_importance"]))}" alt="Route-specific expert permutation importance">
        <figcaption>图 4：E1 with-notes expert 和 E2 no-notes expert 的 standalone permutation attribution。</figcaption>
      </figure>
    </section>

    <section data-contract-section="scope-data-and-metric-definitions">
      <h2>范围、数据和定义</h2>
      <p>分析使用 <code>data/raw/icu_patients.csv</code>，按原脚本的 stratified 60/20/20 split 和 seed={summary["seed"]} 重建 train/validation/test。归因对象是 t24 calibrated risk，不是完整跨阶段 <code>belief_risk</code>。原因是 feature attribution 要解释“某个 t24 指标改变如何影响当前风险输出”；belief update 还混入 t0/t6/t12 历史状态和 uncertainty smoothing。</p>
      <table>
        <thead><tr><th>Attribution output</th><th>Meaning</th></tr></thead>
        <tbody>
          <tr><td>Permutation importance</td><td>全局重要性：打乱该指标后 AUROC 下降和 Brier 上升。</td></tr>
          <tr><td>Partial effect</td><td>方向性：把该指标设为不同分位点时，平均预测风险如何变化。</td></tr>
          <tr><td>Local counterfactual</td><td>单病人解释：把该病人的指标替换为 baseline 后，风险变化多少。</td></tr>
        </tbody>
      </table>
    </section>

    <section data-contract-section="methodology">
      <h2>方法</h2>
      <p>最终风险的 feature attribution 使用下面的定义：</p>
      <pre>global_importance(feature) =
  baseline_AUROC - AUROC(model(x with feature permuted))

local_contribution(patient, feature) =
  risk(patient observed features) - risk(patient with feature set to train baseline)</pre>
      <p>这个方法不是 SHAP，但属于清晰可复现的 feature attribution。它直接回答“模型是否依赖这个指标”和“该指标相对 baseline 把某个病人的风险推高还是压低”。如果需要严格 SHAP，需要恢复原 LightGBM joblib 的运行环境或重新保存一个当前环境可加载的 LightGBM/TreeExplainer 版本。</p>
    </section>

    <section data-contract-section="limitations-uncertainty-and-robustness-checks">
      <h2>限制和 caveat</h2>
      <ul>
        <li><strong>不是旧 joblib 的严格树归因：</strong>{esc(attr["reason"])}</li>
        <li><strong>final risk 与 route expert 分开解释：</strong>final risk 当前主要解释 E0；E1/E2 attribution 只能作为 route-specific diagnostics。</li>
        <li><strong>Permutation 是相关性敏感的：</strong>如果两个指标高度相关，打乱其中一个可能低估或高估真实依赖。</li>
        <li><strong>Counterfactual baseline 不是临床干预：</strong>把 lactate 设为 median 只是解释模型，不表示真实治疗能产生同样风险变化。</li>
      </ul>
    </section>

    <section data-contract-section="recommended-next-steps">
      <h2>建议下一步</h2>
      <ol>
        <li>答辩时用图 1 说明“模型主要看哪些指标”，用图 2 说明“指标变动方向”，用图 3 说明“单个病人怎么解释”。</li>
        <li>如果要提交更强版本，恢复原 LightGBM 环境后补 SHAP beeswarm 和 3 个 patient-level waterfall。</li>
        <li>把 local counterfactual 表接到 demo UI：输入 patient_id，显示 top positive/negative contributors。</li>
      </ol>
    </section>

    <section data-contract-section="further-questions">
      <h2>仍需回答的问题</h2>
      <ul>
        <li>临床评委是否要求严格 SHAP，还是 permutation/counterfactual attribution 已足够？</li>
        <li>是否要把 attribution 从 t24 扩展到 t0/t6/t12，解释早期预警阶段？</li>
        <li>是否要为 false negatives 单独输出 attribution，用于 failure catalogue？</li>
      </ul>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(report, encoding="utf-8")


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
