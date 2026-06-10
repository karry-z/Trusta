#!/usr/bin/env python3
"""Train and evaluate the clinical elastic-net baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_baseline.constants import (  # noqa: E402
    FORBIDDEN_MAIN_COLUMNS,
    MODEL_NAME,
    SUBGROUP_COLUMNS,
    TARGET,
)
from clinical_baseline.evaluation import (  # noqa: E402
    bootstrap_confidence_intervals,
    classification_metrics,
    subgroup_sensitivity_table,
    threshold_sweep,
)
from clinical_baseline.features import validate_required_columns  # noqa: E402
from clinical_baseline.model import (  # noqa: E402
    ClinicalBaselineRiskModel,
    choose_threshold_for_sensitivity,
)
from clinical_baseline.reporting import to_jsonable, update_reference_reports  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the structured clinical elastic-net baseline.",
    )
    parser.add_argument("--data", default="data/raw/icu_patients.csv", type=Path)
    parser.add_argument("--out", default="artifacts/clinical_baseline", type=Path)
    parser.add_argument("--target-sensitivity", default=0.80, type=float)
    parser.add_argument("--bootstrap", default=1000, type=int)
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
    validate_required_columns(df)
    validate_required_columns(df, include_icu_hours=True)

    train_df, validation_df, test_df = stratified_split(df, args.seed)
    y_train = train_df[TARGET].to_numpy()
    y_validation = validation_df[TARGET].to_numpy()
    y_test = test_df[TARGET].to_numpy()

    main_model = ClinicalBaselineRiskModel(
        include_icu_hours=False,
        target_sensitivity=args.target_sensitivity,
        seed=args.seed,
    ).fit(train_df, y_train, validation_df, y_validation)

    raw_validation_prob = main_model.predict_raw_proba(validation_df)
    raw_threshold = choose_threshold_for_sensitivity(
        y_validation,
        raw_validation_prob,
        target_sensitivity=args.target_sensitivity,
    )
    raw_test_prob = main_model.predict_raw_proba(test_df)
    calibrated_test_prob = main_model.predict_calibrated_proba(test_df)
    calibrated_validation_prob = main_model.predict_calibrated_proba(validation_df)

    metrics = {
        "model": main_model.model_summary(),
        "data": {
            "source": str(data_path),
            "split": "stratified 60/20/20 train/validation/test",
            "seed": args.seed,
            "n_total": int(len(df)),
            "target": TARGET,
            "target_rate": float(df[TARGET].mean()),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
            "test_rows": int(len(test_df)),
            "train_positive_rate": float(train_df[TARGET].mean()),
            "validation_positive_rate": float(validation_df[TARGET].mean()),
            "test_positive_rate": float(test_df[TARGET].mean()),
        },
        "threshold_policy": {
            "target_sensitivity": args.target_sensitivity,
            "selection_set": "validation",
            "selected_on": "calibrated_probability",
            "rule": "highest threshold with validation sensitivity >= target",
        },
        "validation": {
            "calibrated": classification_metrics(
                y_validation,
                calibrated_validation_prob,
                main_model.threshold_,
            ),
            "raw": classification_metrics(y_validation, raw_validation_prob, raw_threshold),
        },
        "main": {
            "raw": classification_metrics(y_test, raw_test_prob, raw_threshold),
            "calibrated": classification_metrics(
                y_test,
                calibrated_test_prob,
                main_model.threshold_,
            ),
        },
    }

    bootstrap_ci = bootstrap_confidence_intervals(
        y_test,
        calibrated_test_prob,
        main_model.threshold_,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    coefficients = main_model.coefficient_frame()
    subgroup_df = subgroup_sensitivity_table(
        test_df,
        y_test,
        calibrated_test_prob,
        main_model.threshold_,
    )
    predictions_df = test_predictions_frame(
        test_df,
        y_test,
        raw_test_prob,
        calibrated_test_prob,
        main_model.threshold_,
    )
    sweep_rows = threshold_sweep(y_test, calibrated_test_prob)

    icu_hours_summary = run_icu_hours_sensitivity(
        train_df,
        validation_df,
        test_df,
        y_train,
        y_validation,
        y_test,
        args.target_sensitivity,
        args.seed,
        metrics["main"]["calibrated"],
    )

    run_contract_assertions(
        main_model=main_model,
        icu_model_summary=icu_hours_summary["model"],
        raw_prob=raw_test_prob,
        calibrated_prob=calibrated_test_prob,
        test_df=test_df,
        subgroup_df=subgroup_df,
    )

    joblib.dump(main_model, out_dir / "model.joblib")
    write_json(out_dir / "split_ids.json", split_payload(train_df, validation_df, test_df, args.seed))
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "bootstrap_ci.json", bootstrap_ci)
    coefficients.to_csv(out_dir / "coefficients.csv", index=False)
    subgroup_df.to_csv(out_dir / "subgroup_sensitivity.csv", index=False)
    predictions_df.to_csv(out_dir / "test_predictions.csv", index=False)
    write_json(out_dir / "icu_hours_sensitivity.json", icu_hours_summary)

    if not args.no_update_reference:
        update_reference_reports(
            reference_dir=ROOT / "reference",
            metrics=metrics,
            subgroup_df=subgroup_df,
            coefficients_df=coefficients,
            threshold_sweep_rows=sweep_rows,
            icu_hours_summary=icu_hours_summary,
        )

    calibrated = metrics["main"]["calibrated"]
    print(f"Trained {MODEL_NAME}")
    print(f"Artifacts: {out_dir}")
    print(
        "Calibrated test metrics: "
        f"AUROC={calibrated['auroc']:.3f} "
        f"AUPRC={calibrated['auprc']:.3f} "
        f"Sensitivity={calibrated['sensitivity']:.3f} "
        f"Brier={calibrated['brier_score']:.3f} "
        f"Threshold={calibrated['threshold']:.4f}"
    )


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


def run_icu_hours_sensitivity(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
    target_sensitivity: float,
    seed: int,
    main_calibrated_metrics: dict,
) -> dict:
    model = ClinicalBaselineRiskModel(
        include_icu_hours=True,
        target_sensitivity=target_sensitivity,
        seed=seed,
    ).fit(train_df, y_train, validation_df, y_validation)
    prob = model.predict_calibrated_proba(test_df)
    metrics = classification_metrics(y_test, prob, model.threshold_)
    deltas = {
        key: (
            None
            if metrics.get(key) is None or main_calibrated_metrics.get(key) is None
            else float(metrics[key] - main_calibrated_metrics[key])
        )
        for key in ("auroc", "auprc", "brier_score", "sensitivity")
    }
    auroc_delta = deltas["auroc"]
    auroc_delta_text = "delta unavailable" if auroc_delta is None else f"delta {auroc_delta:+.3f}"
    return {
        "warning": (
            "icu_hours is described as total hours in ICU before step-down or event. "
            "This is likely unavailable at prediction time and is treated as a "
            "non-deployable leakage-risk feature."
        ),
        "model": model.model_summary(),
        "metrics": metrics,
        "delta_vs_main_calibrated": deltas,
        "auroc_delta_text": auroc_delta_text,
    }


def test_predictions_frame(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    raw_prob: np.ndarray,
    calibrated_prob: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": test_df["patient_id"].astype(str),
            "y_true": y_test.astype(int),
            "risk_probability": raw_prob,
            "calibrated_risk_probability": calibrated_prob,
            "threshold": threshold,
            "predicted_high_risk": (calibrated_prob >= threshold).astype(int),
        }
    )


def run_contract_assertions(
    main_model: ClinicalBaselineRiskModel,
    icu_model_summary: dict,
    raw_prob: np.ndarray,
    calibrated_prob: np.ndarray,
    test_df: pd.DataFrame,
    subgroup_df: pd.DataFrame,
) -> None:
    main_source = set(main_model.source_features_)
    forbidden_used = main_source.intersection(FORBIDDEN_MAIN_COLUMNS)
    if forbidden_used:
        raise AssertionError(f"Forbidden columns entered main model: {sorted(forbidden_used)}")
    if "icu_hours" in main_source:
        raise AssertionError("icu_hours entered the main deployable model.")
    if "icu_hours" not in set(icu_model_summary["source_features"]):
        raise AssertionError("icu_hours sensitivity model did not include icu_hours.")

    for label, prob in {"raw": raw_prob, "calibrated": calibrated_prob}.items():
        if len(prob) != len(test_df):
            raise AssertionError(f"{label} probability length does not match test rows.")
        if np.any(prob < 0) or np.any(prob > 1):
            raise AssertionError(f"{label} probabilities outside [0, 1].")

    observed_subgroups = set(subgroup_df["group_col"].unique())
    expected_subgroups = set(SUBGROUP_COLUMNS)
    if observed_subgroups != expected_subgroups:
        raise AssertionError(
            f"Subgroup table mismatch: expected {sorted(expected_subgroups)}, "
            f"got {sorted(observed_subgroups)}"
        )


def split_payload(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
) -> dict:
    return {
        "seed": seed,
        "target": TARGET,
        "split": "stratified 60/20/20 train/validation/test",
        "train_patient_ids": train_df["patient_id"].astype(str).tolist(),
        "validation_patient_ids": validation_df["patient_id"].astype(str).tolist(),
        "test_patient_ids": test_df["patient_id"].astype(str).tolist(),
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
