"""Elastic-net logistic baseline with validation-set calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from clinical_baseline.constants import (
    CATEGORICAL_FEATURES,
    MAIN_NUMERIC_FEATURES,
    MODEL_NAME,
    TARGET,
)
from clinical_baseline.features import note_risk_median, prepare_model_frame


class ClinicalBaselineRiskModel:
    """Structured clinical elastic-net logistic model.

    The main model deliberately excludes identifiers, outcomes, free text, and
    ICU length of stay. Set include_icu_hours=True only for leakage sensitivity
    analysis.
    """

    def __init__(
        self,
        include_icu_hours: bool = False,
        target_sensitivity: float = 0.80,
        seed: int = 42,
    ) -> None:
        self.include_icu_hours = include_icu_hours
        self.target_sensitivity = target_sensitivity
        self.seed = seed
        self.name = MODEL_NAME if not include_icu_hours else f"{MODEL_NAME}_with_icu_hours"

    def fit(
        self,
        train_df: pd.DataFrame,
        y_train: np.ndarray,
        validation_df: pd.DataFrame,
        y_validation: np.ndarray,
    ) -> "ClinicalBaselineRiskModel":
        self.note_risk_median_ = note_risk_median(train_df)
        self.numeric_features_ = list(MAIN_NUMERIC_FEATURES)
        if self.include_icu_hours:
            self.numeric_features_.append("icu_hours")
        self.source_features_ = list(CATEGORICAL_FEATURES) + self.numeric_features_

        x_train = prepare_model_frame(
            train_df,
            note_median=self.note_risk_median_,
            include_icu_hours=self.include_icu_hours,
        )
        x_validation = prepare_model_frame(
            validation_df,
            note_median=self.note_risk_median_,
            include_icu_hours=self.include_icu_hours,
        )

        self.pipeline_ = Pipeline(
            steps=[
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            (
                                "numeric",
                                Pipeline(
                                    steps=[
                                        ("imputer", SimpleImputer(strategy="median")),
                                        ("scaler", StandardScaler()),
                                    ]
                                ),
                                self.numeric_features_,
                            ),
                            (
                                "categorical",
                                OneHotEncoder(
                                    drop="first",
                                    handle_unknown="ignore",
                                    sparse_output=False,
                                ),
                                CATEGORICAL_FEATURES,
                            ),
                        ],
                        remainder="drop",
                        verbose_feature_names_out=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegressionCV(
                        Cs=np.logspace(-3, 2, 12),
                        cv=StratifiedKFold(
                            n_splits=5,
                            shuffle=True,
                            random_state=self.seed,
                        ),
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratios=[0.1, 0.5, 0.9],
                        scoring="roc_auc",
                        max_iter=5000,
                        n_jobs=-1,
                        random_state=self.seed,
                        refit=True,
                    ),
                ),
            ]
        )
        self.pipeline_.fit(x_train, y_train)

        validation_scores = self.decision_function(validation_df)
        self.calibrator_ = LogisticRegression(solver="lbfgs")
        self.calibrator_.fit(validation_scores.reshape(-1, 1), y_validation)

        validation_calibrated = self.predict_calibrated_proba(validation_df)
        self.threshold_ = choose_threshold_for_sensitivity(
            y_validation,
            validation_calibrated,
            target_sensitivity=self.target_sensitivity,
        )
        self.validation_sensitivity_at_threshold_ = sensitivity_at_threshold(
            y_validation,
            validation_calibrated,
            self.threshold_,
        )
        return self

    def _feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        return prepare_model_frame(
            df,
            note_median=self.note_risk_median_,
            include_icu_hours=self.include_icu_hours,
        )

    def decision_function(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline_.decision_function(self._feature_frame(df)))

    def predict_raw_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline_.predict_proba(self._feature_frame(df))[:, 1])

    def predict_calibrated_proba(self, df: pd.DataFrame) -> np.ndarray:
        scores = self.decision_function(df).reshape(-1, 1)
        return np.asarray(self.calibrator_.predict_proba(scores)[:, 1])

    def predict(self, df: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        threshold = self.threshold_ if threshold is None else threshold
        return (self.predict_calibrated_proba(df) >= threshold).astype(int)

    def transformed_feature_names(self) -> list[str]:
        names = self.pipeline_.named_steps["preprocess"].get_feature_names_out()
        return [clean_feature_name(str(name)) for name in names]

    def coefficient_frame(self) -> pd.DataFrame:
        classifier = self.pipeline_.named_steps["classifier"]
        coefs = classifier.coef_[0]
        names = self.transformed_feature_names()
        out = pd.DataFrame(
            {
                "feature": names,
                "coefficient": coefs,
                "abs_coefficient": np.abs(coefs),
                "direction": np.where(coefs >= 0, "positive", "negative"),
            }
        )
        return out.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    def model_summary(self) -> dict:
        classifier = self.pipeline_.named_steps["classifier"]
        return {
            "name": self.name,
            "target": TARGET,
            "include_icu_hours": self.include_icu_hours,
            "note_risk_score_train_median": self.note_risk_median_,
            "target_sensitivity": self.target_sensitivity,
            "selected_threshold": self.threshold_,
            "validation_sensitivity_at_threshold": self.validation_sensitivity_at_threshold_,
            "selected_C": float(classifier.C_[0]),
            "selected_l1_ratio": float(classifier.l1_ratio_[0]),
            "source_features": self.source_features_,
            "transformed_features": self.transformed_feature_names(),
        }


def clean_feature_name(name: str) -> str:
    for prefix in ("numeric__", "categorical__"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def sensitivity_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    positives = y_true == 1
    if positives.sum() == 0:
        return float("nan")
    return float(((y_pred == 1) & positives).sum() / positives.sum())


def choose_threshold_for_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_sensitivity: float,
) -> float:
    """Choose the highest validation threshold that reaches target sensitivity."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    if (y_true == 1).sum() == 0:
        raise ValueError("Cannot choose sensitivity threshold without positive labels.")

    for threshold in np.sort(np.unique(y_prob))[::-1]:
        if sensitivity_at_threshold(y_true, y_prob, float(threshold)) >= target_sensitivity:
            return float(threshold)
    return float(np.min(y_prob))
