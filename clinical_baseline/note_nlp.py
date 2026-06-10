"""Auxiliary NLP note features for the data-availability MoE."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

NOTE_INDICATOR_FEATURES = [
    "note_hemodynamic_instability",
    "note_vasopressor_support",
    "note_borderline_urine_output",
    "note_lactate_elevated",
    "note_blood_loss_not_measured",
    "note_transfusion_mentioned",
    "note_close_monitoring",
]

NOTE_FEATURES = ["note_text_risk_score"] + NOTE_INDICATOR_FEATURES


class NoteNlpFeatureExtractor:
    """TF-IDF note risk score plus small rule-based clinical indicators."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.model_ = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=500,
                        ngram_range=(1, 2),
                        min_df=3,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        solver="liblinear",
                        max_iter=1000,
                        random_state=seed,
                    ),
                ),
            ]
        )

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "NoteNlpFeatureExtractor":
        notes = clean_notes(df["note_text"])
        has_text = notes.str.len() > 0
        y_fit = np.asarray(y)[has_text.to_numpy()]
        if len(np.unique(y_fit)) < 2:
            self.constant_risk_ = float(np.mean(y_fit)) if len(y_fit) else 0.0
            self.is_constant_ = True
            return self
        self.model_.fit(notes[has_text], y_fit)
        self.is_constant_ = False
        self.constant_risk_ = None
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        notes = clean_notes(df.get("note_text", pd.Series("", index=df.index)))
        out = pd.DataFrame(index=df.index)
        if getattr(self, "is_constant_", False):
            out["note_text_risk_score"] = float(self.constant_risk_)
        else:
            nonempty = notes.str.len() > 0
            risk = np.zeros(len(df), dtype=float)
            if nonempty.any():
                risk[nonempty.to_numpy()] = self.model_.predict_proba(notes[nonempty])[:, 1]
            out["note_text_risk_score"] = risk

        indicators = notes.apply(extract_note_indicators)
        for feature in NOTE_INDICATOR_FEATURES:
            out[feature] = indicators.apply(lambda values: values[feature]).astype(int)
        return out[NOTE_FEATURES]


def clean_notes(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower()


def extract_note_indicators(text: str) -> dict[str, int]:
    text = normalize_text(text)
    return {
        "note_hemodynamic_instability": int(has_hemodynamic_instability(text)),
        "note_vasopressor_support": int(has_vasopressor_support(text)),
        "note_borderline_urine_output": int(has_borderline_urine_output(text)),
        "note_lactate_elevated": int(has_lactate_elevated(text)),
        "note_blood_loss_not_measured": int(has_blood_loss_not_measured(text)),
        "note_transfusion_mentioned": int(has_transfusion_administered(text)),
        "note_close_monitoring": int(has_close_monitoring(text)),
    }


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def has_hemodynamic_instability(text: str) -> bool:
    negative = [
        "haemodynamically stable",
        "hemodynamically stable",
        "cardiovascularly stable",
        "cardiovascular stability achieved",
    ]
    if any(term in text for term in negative):
        return False
    positive = [
        "haemodynamically compromised",
        "hemodynamically compromised",
        "haemodynamic support",
        "hemodynamic support",
        "cardiovascular instability",
        "cardiovascularly unstable",
    ]
    return any(term in text for term in positive)


def has_vasopressor_support(text: str) -> bool:
    negative = [
        "no ongoing pressor requirement",
        "no pressor requirement",
        "no ongoing vasopressor requirement",
        "no vasopressor requirement",
        "no vasopressors",
        "without vasopressor",
    ]
    if any(term in text for term in negative):
        return False
    positive = [
        "vasopressor support",
        "ongoing vasopressor",
        "pressor requirement",
        "vasopressors",
        "vasopressor requirement",
    ]
    return any(term in text for term in positive)


def has_borderline_urine_output(text: str) -> bool:
    positive = [
        "urine output borderline",
        "borderline urine",
        "poor urine output",
        "oliguria",
    ]
    return any(term in text for term in positive)


def has_lactate_elevated(text: str) -> bool:
    if "lactate" not in text:
        return False
    if "elevated" in text or "trending" in text:
        return True
    match = re.search(r"lactate\s+([0-9]+(?:\.[0-9]+)?)", text)
    return bool(match and float(match.group(1)) >= 2.5)


def has_blood_loss_not_measured(text: str) -> bool:
    positive = [
        "blood loss not formally measured",
        "blood loss not measured",
        "not formally measured intraoperatively",
    ]
    return any(term in text for term in positive)


def has_transfusion_administered(text: str) -> bool:
    negative = [
        "no intraoperative transfusion required",
        "no transfusion required",
        "without transfusion",
    ]
    if any(term in text for term in negative):
        return False
    positive = [
        "transfusion administered",
        "red-cell transfusion administered",
        "received transfusion",
    ]
    return any(term in text for term in positive)


def has_close_monitoring(text: str) -> bool:
    positive = [
        "close monitoring warranted",
        "icu team reviewing twice daily",
        "continued icu-level care",
        "continue standard monitoring",
    ]
    return any(term in text for term in positive)
