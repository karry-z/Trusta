#!/usr/bin/env python3
"""Train and evaluate the data-availability MoE with dynamic belief updates."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_baseline.constants import DYNAMIC_STAGES, MOE_MODEL_NAME, TARGET  # noqa: E402
from clinical_baseline.evaluation import bootstrap_confidence_intervals  # noqa: E402
from clinical_baseline.moe_evaluation import (  # noqa: E402
    INTERSECTION_SUBGROUP_COLUMNS,
    PRIMARY_SUBGROUP_COLUMNS,
    SECONDARY_SUBGROUP_COLUMNS,
    attach_subgroup_labels,
    belief_stage_deltas,
    note_ablation_table,
    overall_stage_metrics,
    subgroup_metric_table,
    uncertainty_group_metrics,
)
from clinical_baseline.moe_features import (  # noqa: E402
    E0_FEATURES,
    E1_AUX_FEATURES,
    E1_BASIC_FEATURES,
    E2_FEATURES,
    FORBIDDEN_DEPLOYABLE_COLUMNS,
    build_stage_features,
    build_stacked_stage_features,
    validate_moe_source_columns,
)
from clinical_baseline.moe_model import (  # noqa: E402
    DataAvailabilityMoE,
    bootstrap_train_df,
    make_expert_pipeline,
)
from clinical_baseline.moe_reporting import update_moe_reference_reports  # noqa: E402
from clinical_baseline.moe_uncertainty import (  # noqa: E402
    build_belief_trajectory,
    build_feature_bounds,
    validation_uncertainty_reference,
)
from clinical_baseline.reporting import to_jsonable  # noqa: E402

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the data-availability MoE.")
    parser.add_argument("--data", default="data/raw/icu_patients.csv", type=Path)
    parser.add_argument("--out", default="artifacts/data_availability_moe", type=Path)
    parser.add_argument("--bootstrap", default=20, type=int)
    parser.add_argument("--target-sensitivity", default=0.80, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--no-update-reference",
        action="store_true",
        help="Write artifacts only; do not update reference/*.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = resolve_path(args.data)
    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    validate_moe_source_columns(df)
    train_df, validation_df, test_df = stratified_split(df, args.seed)

    main_model = DataAvailabilityMoE(
        target_sensitivity=args.target_sensitivity,
        seed=args.seed,
        use_aux_note_features=True,
    ).fit(train_df, validation_df, sample_label="main")

    bootstrap_models = []
    for b in range(args.bootstrap):
        boot_train = bootstrap_train_df(train_df, seed=args.seed + 10_000 + b)
        model = DataAvailabilityMoE(
            target_sensitivity=args.target_sensitivity,
            seed=args.seed + 1_000 + b,
            use_aux_note_features=True,
        ).fit(boot_train, validation_df, sample_label=f"bootstrap_{b + 1:02d}")
        bootstrap_models.append(model)

    ensemble_models = bootstrap_models if bootstrap_models else [main_model]
    threshold = main_model.threshold_
    feature_bounds = build_feature_bounds(train_df)
    uncertainty_reference = validation_uncertainty_reference(ensemble_models, validation_df)

    validation_trajectory = build_belief_trajectory(
        ensemble_models,
        validation_df,
        threshold=threshold,
        uncertainty_reference=uncertainty_reference,
        feature_bounds=feature_bounds,
    )
    test_trajectory = build_belief_trajectory(
        ensemble_models,
        test_df,
        threshold=threshold,
        uncertainty_reference=uncertainty_reference,
        feature_bounds=feature_bounds,
    )

    test_with_labels = attach_subgroup_labels(test_trajectory, test_df)
    validation_with_labels = attach_subgroup_labels(validation_trajectory, validation_df)

    overall_metrics = overall_stage_metrics(test_trajectory, threshold)
    primary_subgroups = subgroup_metric_table(test_with_labels, PRIMARY_SUBGROUP_COLUMNS, threshold)
    secondary_subgroups = subgroup_metric_table(test_with_labels, SECONDARY_SUBGROUP_COLUMNS, threshold)
    intersection_subgroups = subgroup_metric_table(test_with_labels, INTERSECTION_SUBGROUP_COLUMNS, threshold)
    uncertainty_groups = uncertainty_group_metrics(test_trajectory, threshold)
    uncertainty_validation = uncertainty_group_metrics(validation_trajectory, threshold)
    route_metrics = subgroup_metric_table(test_trajectory, ["route"], threshold)
    expert_disagreement = test_trajectory[
        [
            "patient_id",
            "stage",
            "route",
            "p_struct",
            "p_route_expert",
            "expert_disagreement",
            "u_disagree",
        ]
    ].copy()
    belief_deltas = belief_stage_deltas(test_trajectory)
    note_ablation = build_note_ablation(main_model, train_df, test_df, threshold, args.seed)
    moe_predictions_t24 = test_trajectory[test_trajectory["stage"] == "t24"].reset_index(drop=True)

    t24_metrics = overall_metrics[overall_metrics["stage"] == "t24"].iloc[0].to_dict()
    bootstrap_ci = bootstrap_confidence_intervals(
        moe_predictions_t24["y_true"].to_numpy(),
        moe_predictions_t24["belief_risk"].to_numpy(),
        threshold=threshold,
        n_bootstrap=1000,
        seed=args.seed,
    )
    metrics_json = build_metrics_json(
        args=args,
        data_path=data_path,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        main_model=main_model,
        overall_metrics=overall_metrics,
        t24_metrics=t24_metrics,
    )

    run_contract_assertions(
        main_model=main_model,
        test_trajectory=test_trajectory,
        primary_subgroups=primary_subgroups,
        secondary_subgroups=secondary_subgroups,
        intersection_subgroups=intersection_subgroups,
    )

    joblib.dump(
        {
            "main_model": main_model,
            "bootstrap_models": bootstrap_models,
            "threshold": threshold,
            "feature_bounds": feature_bounds,
            "uncertainty_reference": uncertainty_reference,
        },
        out_dir / "model.joblib",
    )
    write_json(out_dir / "metrics.json", metrics_json)
    write_json(out_dir / "bootstrap_ci.json", bootstrap_ci)
    write_json(
        out_dir / "expert_weights.json",
        {
            "threshold": threshold,
            "lambda_with": main_model.lambda_with_,
            "lambda_no": main_model.lambda_no_,
            "lambda_selection": main_model.lambda_selection_,
            "bootstrap_models": args.bootstrap,
        },
    )
    overall_metrics.to_csv(out_dir / "overall_stage_metrics.csv", index=False)
    primary_subgroups.to_csv(out_dir / "primary_subgroup_metrics.csv", index=False)
    secondary_subgroups.to_csv(out_dir / "secondary_subgroup_metrics.csv", index=False)
    intersection_subgroups.to_csv(out_dir / "intersection_subgroup_metrics.csv", index=False)
    uncertainty_groups.to_csv(out_dir / "uncertainty_group_metrics.csv", index=False)
    route_metrics.to_csv(out_dir / "route_metrics.csv", index=False)
    expert_disagreement.to_csv(out_dir / "expert_disagreement.csv", index=False)
    uncertainty_validation.to_csv(out_dir / "uncertainty_validation.csv", index=False)
    moe_predictions_t24.to_csv(out_dir / "moe_predictions_t24.csv", index=False)
    test_trajectory.to_csv(out_dir / "belief_trajectory.csv", index=False)
    belief_deltas.to_csv(out_dir / "belief_stage_deltas.csv", index=False)
    note_ablation.to_csv(out_dir / "note_ablation.csv", index=False)

    if not args.no_update_reference:
        update_moe_reference_reports(
            reference_dir=ROOT / "reference",
            t24_metrics=t24_metrics,
            primary_subgroups=primary_subgroups,
            route_metrics=route_metrics,
            uncertainty_metrics=uncertainty_groups,
            model_summary=main_model.model_summary(),
        )

    print(f"Trained {MOE_MODEL_NAME}")
    print(f"Artifacts: {out_dir}")
    print(
        "t24 belief metrics: "
        f"AUROC={t24_metrics['auroc']:.3f} "
        f"AUPRC={t24_metrics['auprc']:.3f} "
        f"Sensitivity={t24_metrics['sensitivity']:.3f} "
        f"Brier={t24_metrics['brier_score']:.3f} "
        f"Threshold={threshold:.4f}"
    )
    print(f"lambda_with={main_model.lambda_with_:.1f} lambda_no={main_model.lambda_no_:.1f}")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def stratified_split(
    df: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_note_ablation(
    main_model: DataAvailabilityMoE,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float,
    seed: int,
) -> pd.DataFrame:
    x_train, y_train, meta_train = build_stacked_stage_features(
        train_df,
        note_extractor=main_model.note_extractor_,
        include_note_features=True,
    )
    notes_train = meta_train["has_notes"].eq(1).to_numpy()
    basic = make_expert_pipeline(E1_BASIC_FEATURES, seed + 50_000)
    basic.fit(x_train.loc[notes_train, E1_BASIC_FEATURES], y_train[notes_train])

    notes_test = test_df["has_notes"].eq(1).to_numpy()
    x_test_t24 = build_stage_features(
        test_df,
        stage="t24",
        note_extractor=main_model.note_extractor_,
        include_note_features=True,
    )
    y_notes = test_df.loc[notes_test, TARGET].to_numpy()
    basic_prob = basic.predict_proba(x_test_t24.loc[notes_test, E1_BASIC_FEATURES])[:, 1]
    aux_prob = main_model.e1_.predict_proba(x_test_t24.loc[notes_test, E1_AUX_FEATURES])[:, 1]
    return note_ablation_table(y_notes, basic_prob, aux_prob, threshold)


def build_metrics_json(
    args: argparse.Namespace,
    data_path: Path,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    main_model: DataAvailabilityMoE,
    overall_metrics: pd.DataFrame,
    t24_metrics: dict,
) -> dict:
    return {
        "model": main_model.model_summary(),
        "data": {
            "source": str(data_path),
            "split": "stratified 60/20/20 train/validation/test",
            "target": TARGET,
            "seed": args.seed,
            "n_total": int(len(train_df) + len(validation_df) + len(test_df)),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
            "test_rows": int(len(test_df)),
            "train_positive_rate": float(train_df[TARGET].mean()),
            "validation_positive_rate": float(validation_df[TARGET].mean()),
            "test_positive_rate": float(test_df[TARGET].mean()),
        },
        "dynamic_simulation": {
            "stages": DYNAMIC_STAGES,
            "t24_caveat": (
                "t24 is a full-information retrospective reference and may include "
                "information inside the prediction window."
            ),
            "belief_update": "logit smoothing with alpha=clip(1 - 0.5*u_total, 0.25, 0.85)",
        },
        "bootstrap_ensemble": {
            "B": args.bootstrap,
            "meaning": "B bootstrap MoE replicas; each replica trains E0/E1/E2.",
        },
        "overall_stage_metrics": overall_metrics.to_dict(orient="records"),
        "t24_metrics": t24_metrics,
    }


def run_contract_assertions(
    main_model: DataAvailabilityMoE,
    test_trajectory: pd.DataFrame,
    primary_subgroups: pd.DataFrame,
    secondary_subgroups: pd.DataFrame,
    intersection_subgroups: pd.DataFrame,
) -> None:
    for label, features in {
        "E0": E0_FEATURES,
        "E1": main_model.e1_features_,
        "E2": E2_FEATURES,
    }.items():
        forbidden = sorted(set(features).intersection(FORBIDDEN_DEPLOYABLE_COLUMNS))
        if forbidden:
            raise AssertionError(f"{label} uses forbidden deployable columns: {forbidden}")
    if "note_text" in main_model.e1_features_:
        raise AssertionError("E1 uses raw note_text instead of derived NLP features.")
    if "icu_hours" in set(E0_FEATURES + main_model.e1_features_ + E2_FEATURES):
        raise AssertionError("icu_hours entered a deployable expert.")

    bounded_cols = [
        "risk_mean",
        "risk_low_5",
        "risk_high_95",
        "u_model",
        "u_disagree",
        "u_data",
        "u_entropy",
        "u_total",
        "belief_risk",
    ]
    for col in bounded_cols:
        values = test_trajectory[col].to_numpy()
        if np.any(values < 0) or np.any(values > 1):
            raise AssertionError(f"{col} contains values outside [0, 1].")

    stage_counts = test_trajectory.groupby("patient_id")["stage"].nunique()
    if not (stage_counts == len(DYNAMIC_STAGES)).all():
        raise AssertionError("Not every test patient has all dynamic stages.")

    if set(primary_subgroups["group_col"].unique()) != set(PRIMARY_SUBGROUP_COLUMNS):
        raise AssertionError("Primary subgroup table is missing configured groups.")
    if set(secondary_subgroups["group_col"].unique()) != set(SECONDARY_SUBGROUP_COLUMNS):
        raise AssertionError("Secondary subgroup table is missing configured groups.")
    if set(intersection_subgroups["group_col"].unique()) != set(INTERSECTION_SUBGROUP_COLUMNS):
        raise AssertionError("Intersection subgroup table is missing configured groups.")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
