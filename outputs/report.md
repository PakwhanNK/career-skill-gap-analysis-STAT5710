# Popular vs Unpopular Company Skill Analysis

## What Was Classified
The classifier predicts:

`popular_company = 1` if a student's first qualifying non-intern entry job is at a company whose `rcid` or `ultimate_parent_rcid` appears in `data/data.xlsx`.

`popular_company = 0` for every other company.

The Excel file is the source of truth for popular companies. Any first-job company not found in that file is classified as unpopular.

## Features Fed Into Prediction
The core classifier uses pre-outcome / student-side features:

- Education: `school`, `major`, `university_country`
- Profile context: `user_country`, `prestige`, `numconnections`
- Timing: `entry_job_year`
- Skills: phrase-level mapped skills from `user_skills.csv`, represented as TF-IDF/SVD components
- Cluster feature: `skill_cluster`, learned from important skills

Demographic variables are not used in prediction or regression outputs.

## PCA Method
PCA was done on skills in a deliberately interpretable way:

1. Each user was represented by a phrase-level skill set from `user_skills.csv`.
2. We calculated skill importance using smoothed log-odds: skills overrepresented among Excel-popular outcomes versus non-popular outcomes received high absolute scores.
3. We selected the top 250 important skills.
4. We built a user-by-skill binary matrix for those skills.
5. We standardized the matrix and ran PCA.
6. The first two PCs are plotted in `figures/pca_important_skills_popular_vs_nonpopular.png`.

This means the PCA plot is not using every noisy skill. It focuses on skills that actually help separate popular versus non-popular first-job outcomes.

## Clustering Method
KMeans clustering was run on the first 10 PCA components from the important-skill matrix. Each cluster is summarized with:

- number of users
- Excel-popular company rate
- average number of skills
- most common skills
- most common majors
- most common schools

See `tables/cluster_profiles.csv` and `figures/clusters_in_important_skill_pca_space.png`.

## Classification Results
- Users modeled: 15,140
- Popular-company RCIDs from Excel: 106
- Positive users: 1,948
- Positive rate: 12.9%
- Best core model: `random_forest`
- Test AUC: 0.839
- Test accuracy: 0.862

## Skill Regression Output
I also fit a skill-only logistic regression using the important-skill binary matrix. Its coefficients show which skills are positively or negatively associated with popular-company placement.

- Skill regression AUC: 0.757
- Skill regression accuracy: 0.587
- Coefficients table: `tables/skill_logistic_regression_coefficients.csv`
- Coefficient figure: `figures/skill_regression_coefficients.png`

## Main Figures
- `figures/eda_popular_rate_by_entry_job_year.png`
- `figures/eda_top_employers_excel_label.png`
- `figures/skill_cloud_excel_popular.png`
- `figures/skill_cloud_nonpopular.png`
- `figures/skill_cloud_excel_popular_entry_2024.png`
- `figures/skill_cloud_excel_popular_entry_2025.png`
- `figures/pca_important_skills_popular_vs_nonpopular.png`
- `figures/clusters_in_important_skill_pca_space.png`
- `figures/cluster_popular_rates.png`
- `figures/classification_roc_curves.png`
- `figures/classification_feature_importance.png`
- `figures/skill_regression_coefficients.png`
