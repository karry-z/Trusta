"""Data-availability mixture-of-experts model with route calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from clinical_baseline.constants import DYNAMIC_STAGES, MOE_MODEL_NAME, TARGET
from clinical_baseline.moe_features import (
    E0_FEATURES,
    E1_AUX_FEATURES,
    E1_BASIC_FEATURES,
    E2_FEATURES,
    MOE_CATEGORICAL_FEATURES,
    build_stage_features,
    build_stacked_stage_features,
)
from clinical_baseline.note_nlp import NoteNlpFeatureExtractor

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - fallback for environments without LightGBM.
    LGBMClassifier = None


@dataclass
class LambdaSelection:
    value: float
    brier: float
    sensitivity_at_050: float
    met_sensitivity_constraint: bool
    candidates: list[dict]


class DataAvailabilityMoE:
    """Hard-routed MoE with a fallback structured expert."""

    def __init__(
        self,
        target_sensitivity: float = 0.80,
        seed: int = 42,
        use_aux_note_features: bool = True,
    ) -> None:
        self.target_sensitivity = target_sensitivity
        self.seed = seed
        self.use_aux_note_features = use_aux_note_features
        self.name = MOE_MODEL_NAME

    def fit(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        sample_label: str = "main",
    ) -> "DataAvailabilityMoE":
        self.sample_label_ = sample_label
        self.note_extractor_ = NoteNlpFeatureExtractor(seed=self.seed)
        train_has_notes = train_df["has_notes"].eq(1).to_numpy()
        self.note_extractor_.fit(
            train_df.loc[train_has_notes],
            train_df.loc[train_has_notes, TARGET].to_numpy(),
        )

        x_train, y_train, meta_train = build_stacked_stage_features(
            train_df,
            note_extractor=self.note_extractor_,
            include_note_features=True,
        )
        x_val, y_val, meta_val = build_stacked_stage_features(
            validation_df,
            note_extractor=self.note_extractor_,
            include_note_features=True,
        )

        self.e0_ = make_expert_pipeline(E0_FEATURES, self.seed)
        self.e0_.fit(x_train[E0_FEATURES], y_train)

        notes_train = meta_train["has_notes"].eq(1).to_numpy()
        self.e1_features_ = E1_AUX_FEATURES if self.use_aux_note_features else E1_BASIC_FEATURES
        self.e1_ = make_expert_pipeline(self.e1_features_, self.seed + 1)
        self.e1_.fit(x_train.loc[notes_train, self.e1_features_], y_train[notes_train])

        no_notes_train = meta_train["has_notes"].eq(0).to_numpy()
        self.e2_ = make_expert_pipeline(E2_FEATURES, self.seed + 2)
        self.e2_.fit(x_train.loc[no_notes_train, E2_FEATURES], y_train[no_notes_train])

        p0_val = self.e0_.predict_proba(x_val[E0_FEATURES])[:, 1]
        p1_val = np.full(len(x_val), np.nan)
        p2_val = np.full(len(x_val), np.nan)
        notes_val = meta_val["has_notes"].eq(1).to_numpy()
        no_notes_val = ~notes_val
        p1_val[notes_val] = self.e1_.predict_proba(x_val.loc[notes_val, self.e1_features_])[:, 1]
        p2_val[no_notes_val] = self.e2_.predict_proba(x_val.loc[no_notes_val, E2_FEATURES])[:, 1]

        with_selection = choose_lambda(
            y_true=y_val[notes_val],
            fallback_prob=p0_val[notes_val],
            expert_prob=p1_val[notes_val],
            target_sensitivity=self.target_sensitivity,
        )
        no_selection = choose_lambda(
            y_true=y_val[no_notes_val],
            fallback_prob=p0_val[no_notes_val],
            expert_prob=p2_val[no_notes_val],
            target_sensitivity=self.target_sensitivity,
        )
        self.lambda_with_ = with_selection.value
        self.lambda_no_ = no_selection.value
        self.lambda_selection_ = {
            "with_notes": with_selection.__dict__,
            "no_notes": no_selection.__dict__,
        }

        raw_val = self.fuse_raw(p0_val, p1_val, p2_val, meta_val["has_notes"].to_numpy())
        self.calibrator_with_ = fit_platt(raw_val[notes_val], y_val[notes_val])
        self.calibrator_no_ = fit_platt(raw_val[no_notes_val], y_val[no_notes_val])

        calibrated_val = self.apply_route_calibration(raw_val, meta_val["has_notes"].to_numpy())
        t24_mask = meta_val["stage"].eq("t24").to_numpy()
        self.threshold_ = choose_threshold_for_sensitivity(
            y_val[t24_mask],
            calibrated_val[t24_mask],
            target_sensitivity=self.target_sensitivity,
        )
        self.validation_stage_metrics_basis_ = {
            "lambda_tuning": "stacked train/validation snapshots",
            "threshold_selection": "validation t24 calibrated MoE risk",
        }
        return self

    def predict_stage(self, df: pd.DataFrame, stage: str) -> pd.DataFrame:
        x = build_stage_features(
            df,
            stage=stage,
            note_extractor=self.note_extractor_,
            include_note_features=True,
        )
        has_notes = df["has_notes"].astype(int).to_numpy()
        p0 = self.e0_.predict_proba(x[E0_FEATURES])[:, 1]
        p1 = np.full(len(x), np.nan)
        p2 = np.full(len(x), np.nan)

        notes_mask = has_notes == 1
        no_notes_mask = ~notes_mask
        if notes_mask.any():
            p1[notes_mask] = self.e1_.predict_proba(x.loc[notes_mask, self.e1_features_])[:, 1]
        if no_notes_mask.any():
            p2[no_notes_mask] = self.e2_.predict_proba(x.loc[no_notes_mask, E2_FEATURES])[:, 1]

        p_raw = self.fuse_raw(p0, p1, p2, has_notes)
        p_calibrated = self.apply_route_calibration(p_raw, has_notes)
        route_expert = np.where(has_notes == 1, p1, p2)
        return pd.DataFrame(
            {
                "patient_id": df["patient_id"].astype(str).to_numpy(),
                "stage": stage,
                "route": np.where(has_notes == 1, "with_notes", "no_notes"),
                "p_struct": p0,
                "p_route_expert": route_expert,
                "p_raw": p_raw,
                "p_calibrated": p_calibrated,
                "expert_disagreement": np.abs(route_expert - p0),
                "lambda_route": np.where(has_notes == 1, self.lambda_with_, self.lambda_no_),
            }
        )

    def predict_all_stages(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [self.predict_stage(df, stage=stage) for stage in DYNAMIC_STAGES],
            ignore_index=True,
        )

    def fuse_raw(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        has_notes: np.ndarray,
    ) -> np.ndarray:
        has_notes = np.asarray(has_notes).astype(int)
        out = np.empty(len(p0), dtype=float)
        notes_mask = has_notes == 1
        out[notes_mask] = (1 - self.lambda_with_) * p0[notes_mask] + self.lambda_with_ * p1[notes_mask]
        out[~notes_mask] = (1 - self.lambda_no_) * p0[~notes_mask] + self.lambda_no_ * p2[~notes_mask]
        return out

    def apply_route_calibration(self, p_raw: np.ndarray, has_notes: np.ndarray) -> np.ndarray:
        has_notes = np.asarray(has_notes).astype(int)
        out = np.empty(len(p_raw), dtype=float)
        notes_mask = has_notes == 1
        out[notes_mask] = self.calibrator_with_.predict(p_raw[notes_mask])
        out[~notes_mask] = self.calibrator_no_.predict(p_raw[~notes_mask])
        return out

    def model_summary(self) -> dict:
        return {
            "name": self.name,
            "target": TARGET,
            "sample_label": self.sample_label_,
            "target_sensitivity": self.target_sensitivity,
            "threshold": self.threshold_,
            "lambda_with": self.lambda_with_,
            "lambda_no": self.lambda_no_,
            "expert_model": "LightGBMClassifier" if LGBMClassifier is not None else "HistGradientBoostingClassifier",
            "e0_features": E0_FEATURES,
            "e1_features": self.e1_features_,
            "e2_features": E2_FEATURES,
            "forbidden_columns_excluded": [
                "patient_id",
                TARGET,
                "major_complication_30d",
                "note_text",
                "icu_hours",
            ],
        }


class PlattCalibrator:
    def __init__(self, model: LogisticRegression | None = None, constant: float | None = None) -> None:
        self.model = model
        self.constant = constant

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        p_raw = np.asarray(p_raw).reshape(-1, 1)
        if self.constant is not None:
            return np.full(p_raw.shape[0], self.constant, dtype=float)
        return self.model.predict_proba(p_raw)[:, 1]


def make_expert_pipeline(feature_names: list[str], seed: int) -> Pipeline:
    categorical = [col for col in MOE_CATEGORICAL_FEATURES if col in feature_names]
    numeric = [col for col in feature_names if col not in categorical]
    classifier = make_classifier(seed)
    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "numeric",
                            Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                            numeric,
                        ),
                        (
                            "categorical",
                            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                            categorical,
                        ),
                    ],
                    remainder="drop",
                ),
            ),
            ("classifier", classifier),
        ]
    )


def make_classifier(seed: int):
    if LGBMClassifier is not None:
        return LGBMClassifier(
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=12,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
    return HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_leaf_nodes=12,
        min_samples_leaf=50,
        max_iter=250,
        random_state=seed,
    )


def choose_lambda(
    y_true: np.ndarray,
    fallback_prob: np.ndarray,
    expert_prob: np.ndarray,
    target_sensitivity: float,
) -> LambdaSelection:
    candidates = []
    y_true = np.asarray(y_true).astype(int)
    for value in np.round(np.arange(0.0, 1.01, 0.1), 1):
        p = (1 - value) * fallback_prob + value * expert_prob
        y_pred = (p >= 0.5).astype(int)
        sensitivity = float(recall_score(y_true, y_pred, zero_division=0))
        candidates.append(
            {
                "lambda": float(value),
                "brier": float(brier_score_loss(y_true, p)),
                "sensitivity_at_050": sensitivity,
            }
        )
    feasible = [row for row in candidates if row["sensitivity_at_050"] >= target_sensitivity]
    if feasible:
        best = min(feasible, key=lambda row: (row["brier"], -row["sensitivity_at_050"]))
        met = True
    else:
        best = min(candidates, key=lambda row: (-row["sensitivity_at_050"], row["brier"]))
        met = False
    return LambdaSelection(
        value=float(best["lambda"]),
        brier=float(best["brier"]),
        sensitivity_at_050=float(best["sensitivity_at_050"]),
        met_sensitivity_constraint=met,
        candidates=candidates,
    )


def fit_platt(p_raw: np.ndarray, y_true: np.ndarray) -> PlattCalibrator:
    p_raw = np.asarray(p_raw, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return PlattCalibrator(constant=float(np.mean(y_true)))
    model = LogisticRegression(solver="lbfgs")
    model.fit(p_raw.reshape(-1, 1), y_true)
    return PlattCalibrator(model=model)


def choose_threshold_for_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_sensitivity: float,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    if (y_true == 1).sum() == 0:
        raise ValueError("Cannot choose threshold without positive labels.")
    for threshold in np.sort(np.unique(y_prob))[::-1]:
        y_pred = (y_prob >= threshold).astype(int)
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        if sensitivity >= target_sensitivity:
            return float(threshold)
    return float(np.min(y_prob))


def bootstrap_train_df(train_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(train_df), size=len(train_df))
    return train_df.iloc[idx].reset_index(drop=True)
