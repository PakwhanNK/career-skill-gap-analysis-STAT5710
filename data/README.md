# Data Directory

This folder contains the local data inputs used by the reproducible pipeline.

## `raw/`

Raw Revelio/WRDS exports used by the project:

- `Revelio_EDU_18-22.csv`
- `User_positions_grouped.csv`
- `user_profiles.csv`
- `user_skills.csv`

These files are intentionally ignored by Git because they are large row-level data files.

## `processed/ready_analysis/`

Processed overlap-ready analysis files created from the raw files:

- `ready_revelio_edu_18_22.csv`
- `ready_user_positions_grouped.csv`
- `ready_user_profiles.csv`
- `ready_user_skills.csv`
- `ready_user_skill_agg.csv`
- `ready_first_job.csv`
- `ready_analysis_summary.md`
- `ready_analysis_summary.json`

The processed CSVs are also ignored by Git, while summary files are tracked so the project structure and row counts remain visible on GitHub.
