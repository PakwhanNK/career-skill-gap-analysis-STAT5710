# Ready Analysis Classification Report

## Scope
- Uses the cleaned `data/processed/ready_analysis` universe where education, positions, profiles, and skills overlap.
- Restricts modeling to first-job employers that have a reviewed `manual_decision` label.
- Keeps only `seniority == 2` roles to focus on entry-level full-time roles and reduce internship/research-assistant contamination.

## Model
- Custom NumPy logistic regression.
- Features: target-encoded school/major/degree/country, profile numeric fields, number of skills, and top skill indicators selected by train-only log odds.
- Seniority is excluded as a predictor because it defines the analytic scope.

## Test Results
- Rows modeled: 15,999
- Target rows: 9,641
- Non-target rows: 6,358
- Test AUC: 0.755
- Test accuracy: 0.742
- Test precision: 0.718
- Test recall: 0.939

## Primary Non-Skill Regression
- This version excludes skill indicators and `n_skills` to avoid self-reporting/profile-completeness bias.
- Non-skill test AUC: 0.692
- Non-skill test accuracy: 0.678
- Non-skill test precision: 0.699
- Non-skill test recall: 0.819

## Alternative Models
- Decision tree and random forest are fit with local NumPy/Pandas implementations because `sklearn` is not installed in this runtime.
- True XGBoost was attempted but skipped because `xgboost` is not installed in this runtime.
- See `ready_classification_model_comparison.csv` for model metrics and `ready_*_feature_importance.csv` files for feature importances.

## Logistic Inference
- See `ready_logistic_log_odds_p_values.csv` for standardized log-odds coefficients, odds ratios, standard errors, z-statistics, and approximate Wald p-values.

## Caveats
- The employer labels are manually reviewed, so results apply to the reviewed employer subset.
- Skill indicators are based on self-reported/profile-derived skills and should be interpreted as noisy signals.
- The model is a transparent baseline rather than a tuned production classifier.
