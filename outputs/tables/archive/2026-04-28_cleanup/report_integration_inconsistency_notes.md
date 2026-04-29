# Report Integration Notes

## What the README/Final Report Should Now Have

The final report should be organized around the cleaned modeling pipeline:

1. Research question and target definition.
2. Data integration and cleaned overlap universe.
3. Manual employer review process and reviewed-employer caveat.
4. Seniority sensitivity check and rationale for using `seniority == 2`.
5. EDA: target/non-target counts, target schools, target majors, role composition, and 2024-vs-2025 observed landscape.
6. Primary non-skill logistic regression.
7. Exploratory skill analysis and skill-augmented models.
8. Alternative classifiers and robustness checks.
9. Limitations and interpretation caveats.

## Major Inconsistencies in the Older LaTeX Report

### 1. Outcome Definition

Old report:
- Defines outcome as "popular company" using `data.xlsx`.
- Any employer not in the Excel file is treated as unpopular.

Current analysis:
- Defines outcome using manually reviewed target labels in `outputs/tables/employer_target_review.csv`.
- Target means selective tech, finance, consulting, and quant-oriented employers.
- Unreviewed employers are excluded from supervised modeling.

Decision needed:
- Replace "popular company" everywhere with "reviewed target employer".
- Or preserve old wording and treat the new analysis as an appendix/extension.

Recommended choice:
- Replace old target definition. The reviewed target label is more defensible than assuming every non-Excel employer is non-target.

### 2. Sample Definition and Counts

Old report:
- Final modeled dataset: 15,140 users.
- Positive class: 1,948 users, 12.9%.
- T20 bachelor-focused framing.
- Excludes internships/seasonal roles using title filters.

Current analysis:
- Final main modeled dataset: 15,999 users.
- Target: 9,641 users.
- Non-target: 6,358 users.
- Target rate: 60.3%.
- Uses reviewed employers and `seniority == 2`.
- Seniority 1 is excluded because it is dominated by internship-like/research/teaching/support roles.

Decision needed:
- Should the final report keep the original T20 bachelor wording, or should it describe the scope as the cleaned `Revelio_EDU_18-22` overlap universe?

Recommended choice:
- Use the cleaned overlap universe wording unless we can prove every user is still from the original T20 universe.

### 3. Main Model

Old report:
- Main model is random forest.
- Test AUC = 0.839.
- Accuracy = 0.862.
- Random forest selected as best model.

Current analysis:
- Main interpretive model should be non-skill logistic regression.
- Non-skill logistic test AUC = 0.692.
- Non-skill accuracy = 0.678.
- Skill-augmented logistic AUC = 0.755.
- Decision tree AUC = 0.701.
- Random forest AUC = 0.732.
- XGBoost was not run because `xgboost` is not installed.

Decision needed:
- Should the report prioritize interpretability/regression or predictive performance?

Recommended choice:
- Use non-skill logistic regression as the primary model and skill-augmented/tree models as exploratory robustness checks. This matches our discussion about self-reported skills.

### 4. Skills as Main Predictor

Old report:
- Claims skill text is by far the most important predictor.
- Uses skills in PCA, clustering, logistic regression, random forest, and feature importance.

Current analysis:
- Skills are analyzed descriptively and in exploratory models.
- We decided raw skill indicators should not be the primary regression features because they are self-reported, sparse, and potentially post-outcome.
- Skills add predictive signal but carry interpretation risk.

Decision needed:
- Keep skill analysis as a central project claim, or downgrade it to exploratory evidence?

Recommended choice:
- Downgrade skills to exploratory. The main claim should be that academic pathway and profile/network features predict target placement, while skills provide supplemental descriptive signal.

### 5. PCA and Clustering

Old report:
- Includes PCA on important skill matrix.
- Includes k-means clustering on PCA components.
- Reports cluster profiles and popular-company rates.

Current analysis:
- No PCA or k-means clustering was run in the cleaned `ready_analysis` pipeline.
- Older PCA/cluster figures exist only in archived outputs from the earlier pipeline.

Decision needed:
- Do we want to re-run PCA/k-means on the cleaned seniority-2 ready-analysis dataset, or remove PCA/clustering from the report?

Options:
- Option A: Remove PCA/clustering entirely from the final report.
- Option B: Keep a short "older exploratory analysis" appendix, clearly marked as pre-cleaning.
- Option C: Implement a new cleaned PCA/SVD + k-means workflow using current seniority-2 data.

Recommended choice:
- Option A for speed and consistency, or Option C if the report needs an unsupervised-learning component.

### 6. Figures

Old report references PNG files:
- `eda_popular_rate_by_entry_job_year.png`
- `eda_top_employers_excel_label.png`
- `skill_cloud_excel_popular.png`
- `skill_cloud_nonpopular.png`
- `pca_important_skills_popular_vs_nonpopular.png`
- `clusters_in_important_skill_pca_space.png`
- `classification_roc_curves.png`
- `classification_confusion_matrix.png`
- `classification_feature_importance.png`
- `skill_regression_coefficients.png`

Current report-ready figures are SVG files:
- `ready_eda_target_vs_nontarget_counts.svg`
- `ready_eda_top20_target_schools.svg`
- `ready_eda_target_major_pie.svg`
- `ready_classification_model_comparison_auc.svg`
- `ready_classification_confusion_matrix.svg`
- `ready_non_skill_logistic_feature_importance.svg`
- `ready_skill_word_cloud_target.svg`
- `ready_skill_word_cloud_non_target.svg`
- `ready_seniority_sensitivity_auc.svg`
- `ready_seniority_sensitivity_target_rate.svg`
- `ready_yoy_target_share_by_year.svg`
- `ready_yoy_top_roles_percentages_2024_2025.svg`

Decision needed:
- Should we convert the SVGs to PDFs/PNGs for LaTeX compatibility, or configure LaTeX to include SVGs?

Recommended choice:
- Convert selected SVGs to PDF or PNG for LaTeX. Most LaTeX workflows handle PDF/PNG more reliably than raw SVG.

### 7. Year-over-Year Section

Old report:
- Mentions entry-job year and 2024/2025 word clouds.

Current analysis:
- Includes explicit 2024-vs-2025 observed composition comparison.
- 2024 rows: 4,064, target share 61.3%.
- 2025 rows: 1,013, target share 58.9%.
- 2025 is smaller and likely right-censored.
- Top roles are reported as percentages of each year's observed cohort.

Decision needed:
- Add a standalone "Observed 2024 vs 2025 Landscape" section?

Recommended choice:
- Yes, add it, but frame it as observed composition rather than a full labor-market trend.

## Recommended Integration Strategy

Recommended path:

1. Replace the Executive Summary with the current cleaned-analysis summary.
2. Rewrite Data Description around the ready-analysis overlap universe and manual employer review labels.
3. Replace EDA figures with the `ready_*` figures.
4. Remove or rewrite PCA/clustering unless we re-run it under the cleaned pipeline.
5. Make non-skill logistic regression the primary model.
6. Keep skill word clouds/log-odds as exploratory.
7. Add seniority sensitivity and 2024-vs-2025 observed-composition sections.
8. Update limitations to emphasize reviewed-employer scope, seniority-2 scope, skill self-reporting, and 2025 right-censoring.

## Open Decisions for the Team

1. Should the report keep the old "T20 graduates" framing, or switch fully to the cleaned overlap universe framing?
2. Should "popular company" be replaced everywhere with "reviewed target employer"?
3. Should PCA/clustering be removed, kept as old exploratory appendix, or re-run on the cleaned data?
4. Should skills be central to the main model, or kept as exploratory/descriptive?
5. Should the final report emphasize regression interpretation or classification prediction?
6. Should we convert the selected SVG figures to PDF/PNG for LaTeX?
