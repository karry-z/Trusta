# MoE Feature Attribution Source Notes

Generated: 2026-06-10

## Sources

- Raw data: `/Users/pu22650/work/mmai26-hackathon/data/raw/icu_patients.csv`
- Existing MoE artifacts: `/Users/pu22650/work/mmai26-hackathon/artifacts/data_availability_moe`
- Existing metrics reference: `/Users/pu22650/work/mmai26-hackathon/artifacts/data_availability_moe/metrics.json`
- Existing expert weights reference: `/Users/pu22650/work/mmai26-hackathon/artifacts/data_availability_moe/expert_weights.json`

## Method

- Recreated the original stratified 60/20/20 train/validation/test split with seed 42.
- Refit `DataAvailabilityMoE` with the same feature engineering and routing logic in the current environment.
- Computed global permutation importance against t24 calibrated risk.
- Computed local counterfactual attribution by replacing one feature at a time with the train-set t24 baseline value.
- Computed route-specific standalone permutation importance for E1 and E2.

## Caveat

The attribution model is not the serialized historical LightGBM object. The saved `model.joblib`
was created under an older scikit-learn/LightGBM environment and cannot be safely unpickled on
this machine without restoring that environment. Existing artifact metrics are preserved as a
reference, while feature attribution is computed from the compatible refit.
