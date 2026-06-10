#!/usr/bin/env python3
"""Train and evaluate the structured no-notes modality baseline."""

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
    NOTE_MODALITY_FEATURES,
    NO_NOTES_MODEL_NAME,
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
from clinical_baseline.reporting import (  # noqa: E402
    to_jsonable,
    update_no_notes_reference_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the structured elastic-net baseline without notes features.",
    )
    parser.add_argument("--data", default="data/raw/icu_patients.csv", type=Path)
    parser.add_argument(
        "--out", default="artifacts/structured_no_notes_baseline", type=Path
    )
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

    train_df, validation_df, test_df = stratified_split(df, args.seed)
    y_train = train_df[TARGET].to_numpy()
    y_validation = validation_df[TARGET].to_numpy()
    y_test = test_df[TARGET].to_numpy()

    model = ClinicalBaselineRiskModel(
        include_note_features=False,
        target_sensitivity=args.target_sensitivity,
        seed=args.seed,
    ).fit(train_df, y_train, validation_df, y_validation)

    raw_validation_prob = model.predict_raw_proba(validation_df)
    raw_threshold = choose_threshold_for_sensitivity(
        y_validation,
        raw_validation_prob,
        target_sensitivity=args.target_sensitivity,
    )
    raw_test_prob = model.predict_raw_proba(test_df)
    calibrated_validation_prob = model.predict_calibrated_proba(validation_df)
    calibrated_test_prob = model.predict_calibrated_proba(test_df)

    metrics = {
        "model": model.model_summary(),
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
        "excluded_modality_features": list(NOTE_MODALITY_FEATURES),
        "validation": {
            "calibrated": classification_metrics(
                y_validation,
                calibrated_validation_prob,
                model.threshold_,
            ),
            "raw": classification_metrics(
                y_validation, raw_validation_prob, raw_threshold
            ),
        },
        "main": {
            "raw": classification_metrics(y_test, raw_test_prob, raw_threshold),
            "calibrated": classification_metrics(
                y_test, calibrated_test_prob, model.threshold_
            ),
        },
    }

    bootstrap_ci = bootstrap_confidence_intervals(
        y_test,
        calibrated_test_prob,
        model.threshold_,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    coefficients = model.coefficient_frame()
    subgroup_df = subgroup_sensitivity_table(
        test_df,
        y_test,
        calibrated_test_prob,
        model.threshold_,
    )
    predictions_df = test_predictions_frame(
        test_df,
        y_test,
        raw_test_prob,
        calibrated_test_prob,
        model.threshold_,
    )
    sweep_rows = threshold_sweep(y_test, calibrated_test_prob)

    run_contract_assertions(
        model=model,
        raw_prob=raw_test_prob,
        calibrated_prob=calibrated_test_prob,
        test_df=test_df,
        subgroup_df=subgroup_df,
    )

    joblib.dump(model, out_dir / "model.joblib")
    write_json(
        out_dir / "split_ids.json",
        split_payload(train_df, validation_df, test_df, args.seed),
    )
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "bootstrap_ci.json", bootstrap_ci)
    coefficients.to_csv(out_dir / "coefficients.csv", index=False)
    subgroup_df.to_csv(out_dir / "subgroup_sensitivity.csv", index=False)
    predictions_df.to_csv(out_dir / "test_predictions.csv", index=False)

    if not args.no_update_reference:
        update_no_notes_reference_reports(
            reference_dir=ROOT / "reference",
            metrics=metrics,
            subgroup_df=subgroup_df,
            coefficients_df=coefficients,
            threshold_sweep_rows=sweep_rows,
        )

    calibrated = metrics["main"]["calibrated"]
    print(f"Trained {NO_NOTES_MODEL_NAME}")
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
    model: ClinicalBaselineRiskModel,
    raw_prob: np.ndarray,
    calibrated_prob: np.ndarray,
    test_df: pd.DataFrame,
    subgroup_df: pd.DataFrame,
) -> None:
    source_features = set(model.source_features_)
    forbidden_used = source_features.intersection(FORBIDDEN_MAIN_COLUMNS)
    if forbidden_used:
        raise AssertionError(
            f"Forbidden columns entered no-notes model: {sorted(forbidden_used)}"
        )
    note_features_used = source_features.intersection(NOTE_MODALITY_FEATURES)
    if note_features_used:
        raise AssertionError(
            f"Notes modality features entered no-notes model: {sorted(note_features_used)}"
        )
    if model.name != NO_NOTES_MODEL_NAME:
        raise AssertionError(f"Unexpected model name: {model.name}")

    for label, prob in {"raw": raw_prob, "calibrated": calibrated_prob}.items():
        if len(prob) != len(test_df):
            raise AssertionError(
                f"{label} probability length does not match test rows."
            )
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
        json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
