# Career Skill Gap Analysis

This repository contains a STAT 5710 analysis of early-career placement into selective technology, finance, consulting, and quant-oriented employers using Revelio Labs data. The project studies which educational and profile characteristics are associated with entering a reviewed target employer, and how observed early-career role composition differs between 2024 and 2025.

## Research Question

The main outcome is binary:

> Did a user enter a reviewed target employer in their first observed early-career role?

Target employers are manually reviewed selective firms in tech, finance, consulting, and quant-adjacent markets. The analysis avoids interpreting the target label as "good" versus "bad" employment; it is a specific labor-market category.

## Analysis Scope

The cleaned modeling cohort uses users with overlapping education, position, profile, and skill records. The primary analysis is restricted to:

- First observed job records with manually reviewed employer labels.
- Revelio `seniority == 2`, used as the closest available proxy for early-career full-time roles.
- Reviewed target and reviewed non-target employers.

The seniority restriction is intentional. Sensitivity checks showed that `seniority == 1` is dominated by internship-like, research-assistant, teaching-assistant, and support roles, while `seniority == 2` better matches the project's focus on early-career full-time placement.

## Repository Structure

```text
scripts/
  Reproducible data preparation, employer review, EDA, and modeling scripts.

outputs/tables/
  Aggregate result tables, model reports, sensitivity checks, and the main summary report.

outputs/figures/
  Report-ready SVG visualizations for EDA, classification, seniority sensitivity, and 2024-vs-2025 comparisons.

data/processed/ready_analysis/
  Non-row-level summary files for the cleaned analysis universe.

archive/
  Older exploratory outputs moved out of the active analysis path.
```

Raw and row-level processed data are intentionally not committed. The `.gitignore` excludes raw CSVs, processed CSVs, user ID files, and row-level modeling datasets to avoid pushing large or potentially sensitive data.

## Reproducible Pipeline

The main analysis script is:

```powershell
python scripts/run_ready_analysis_classification.py
```

This script builds the seniority-2 analysis outputs, including:

- EDA tables and figures.
- Primary non-skill logistic regression.
- Exploratory skill-augmented classification.
- Decision-tree and random-forest comparison models.
- Seniority sensitivity checks.
- 2024-vs-2025 observed role landscape comparisons.

Supporting scripts in `scripts/` reconstruct the first-job scope, merge manual employer reviews, build overlap-ready files, and generate earlier reviewed-only diagnostic outputs.

## Main Outputs

The most useful report-facing files are:

- `outputs/tables/ready_analysis_results_summary_report.md`
- `outputs/tables/ready_classification_report.md`
- `outputs/tables/ready_non_skill_logistic_metrics.csv`
- `outputs/tables/ready_non_skill_logistic_log_odds_p_values_sorted_by_significance.csv`
- `outputs/tables/ready_seniority_sensitivity_report.md`
- `outputs/tables/ready_yoy_2024_2025_report.md`

Key figures include:

- `outputs/figures/ready_eda_target_vs_nontarget_counts.svg`
- `outputs/figures/ready_eda_top20_target_schools.svg`
- `outputs/figures/ready_eda_target_major_pie.svg`
- `outputs/figures/ready_classification_model_comparison_auc.svg`
- `outputs/figures/ready_seniority_sensitivity_auc.svg`
- `outputs/figures/ready_yoy_top_roles_percentages_2024_2025.svg`

## Headline Results

The final seniority-2 modeling cohort contains 15,999 reviewed users:

- Target: 9,641 users
- Non-target: 6,358 users
- Target rate: 60.3%

The primary non-skill logistic regression excludes self-reported skill indicators and `n_skills` to reduce profile-completeness and post-outcome bias.

Primary non-skill model performance:

| Metric | Test Value |
|---|---:|
| AUC | 0.692 |
| Accuracy | 0.678 |
| Precision | 0.699 |
| Recall | 0.819 |

The most stable predictors are major, number of connections, school, user country, and entry-job year. Skill-augmented models perform better predictively, but skills are treated as exploratory because they are self-reported/profile-derived and may be updated after employment.

## 2024 vs 2025 Comparison

The year comparison should be interpreted as an observed composition comparison, not a complete labor-market time series. The 2025 cohort is smaller and likely right-censored because many recent graduates may not yet have complete first-job observations.

Observed seniority-2 reviewed rows:

| Year | Users | Target Share |
|---|---:|---:|
| 2024 | 4,064 | 61.3% |
| 2025 | 1,013 | 58.9% |

For role comparisons, the report emphasizes within-year percentages while still reporting counts for coverage transparency.

## Limitations

- Employer labels are manually reviewed, so results apply to the reviewed-employer subset.
- Unreviewed employers are excluded from supervised modeling and may differ systematically from reviewed employers.
- The `seniority == 2` filter improves role-scope clarity but narrows generalizability.
- Major and school are target-encoded in the regression, so they should be interpreted as predictive category signals rather than causal effects.
- Skill features are noisy, self-reported/profile-derived, and used only as exploratory signals.
- The 2025 landscape comparison is partially observed and should not be overinterpreted as a completed annual trend.

## Recommended Citation of Results

Use `outputs/tables/ready_analysis_results_summary_report.md` as the primary written summary of the modeling pipeline, EDA findings, regression results, sensitivity checks, and caveats.
