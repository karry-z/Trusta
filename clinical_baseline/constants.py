"""Feature contract for the clinical elastic-net baseline."""

MODEL_NAME = "clinical_elastic_net_baseline"
MOE_MODEL_NAME = "data_availability_moe"
TARGET = "deteriorated_24h"
SECONDARY_TARGET = "major_complication_30d"

ID_COLUMNS = ["patient_id"]
TEXT_COLUMNS = ["note_text"]
NON_DEPLOYABLE_COLUMNS = ["icu_hours"]
FORBIDDEN_MAIN_COLUMNS = set(ID_COLUMNS + [TARGET, SECONDARY_TARGET] + TEXT_COLUMNS + NON_DEPLOYABLE_COLUMNS)

CATEGORICAL_FEATURES = [
    "sex",
    "surgery_type",
    "admission_urgency",
]

RAW_NUMERIC_FEATURES = [
    "age",
    "asa_class",
    "op_duration_h",
    "blood_loss_imputed",
    "blood_loss_missing",
    "transfused",
    "has_diabetes",
    "has_hypertension",
    "preop_creatinine",
    "preop_wbc",
    "preop_lactate",
    "sofa_score",
    "hr_mean",
    "hr_std",
    "rr_mean",
    "rr_std",
    "spo2_mean",
    "spo2_min",
    "sbp_mean",
    "temp_mean",
    "icu_lactate",
    "icu_creatinine",
    "icu_wbc",
    "icu_bilirubin",
    "has_notes",
]

DERIVED_NUMERIC_FEATURES = [
    "note_risk_score_imputed",
    "note_risk_score_missing",
]

MAIN_NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES
MAIN_FEATURES = CATEGORICAL_FEATURES + MAIN_NUMERIC_FEATURES

REQUIRED_SOURCE_COLUMNS = CATEGORICAL_FEATURES + RAW_NUMERIC_FEATURES + ["note_risk_score", TARGET, "patient_id"]

SUBGROUP_COLUMNS = [
    "age_band",
    "sex",
    "admission_urgency",
    "surgery_type",
    "has_notes",
    "sofa_quartile",
]

DYNAMIC_STAGES = ["t0", "t6", "t12", "t24"]
