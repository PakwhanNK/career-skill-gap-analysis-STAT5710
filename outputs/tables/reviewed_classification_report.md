# Reviewed-Only Classification Summary

## Model
- Model type: custom target-encoded logistic regression implemented in pure NumPy.
- Features: target-encoded school, major, degree, university country, user country; plus prestige, number of connections, entry-job year, and seniority.
- Train/test split: stratified 80/20.

## Results
- Reviewed rows modeled: 25,188
- Target rows: 11,021
- Non-target rows: 14,167
- Test AUC: 0.809
- Test accuracy: 0.750
- Test precision: 0.712
- Test recall: 0.720

## Caveats
- This is a reviewed-only cohort, so the model is trained on a manually labeled employer subset rather than the full employer universe.
- The bundled runtime did not include `scikit-learn`, so this is a custom logistic baseline rather than the earlier random forest / boosting pipeline.
- Raw skill coverage for the reviewed cohort is sparse in the available local skill file: 553 of 25,188 users (2.2%).
- Because of that sparse overlap, the current model intentionally excludes skill features and should be interpreted as a profile-and-education baseline.
