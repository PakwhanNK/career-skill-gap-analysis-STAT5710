# Processed Data Manifest

Expected local processed files:

| File | Approx. size | Description |
|---|---:|---|
| `core_user_ids.txt` | 2 MB | User IDs present across the core education, position, profile, and skill sources. |
| `ready_revelio_edu_18_22.csv` | 19 MB | Cleaned education records for the analysis universe. |
| `ready_user_positions_grouped.csv` | 61 MB | Position records restricted to overlapping users. |
| `ready_user_profiles.csv` | 34 MB | Profile records restricted to overlapping users. |
| `ready_user_skills.csv` | 397 MB | Skill rows restricted to overlapping users. |
| `ready_user_skill_agg.csv` | 126 MB | User-level aggregated skill text and skill counts. |
| `ready_first_job.csv` | 7 MB | Reconstructed first observed job per user. |
| `ready_analysis_summary.md` | small | Tracked summary of row counts and overlap construction. |
| `ready_analysis_summary.json` | small | Machine-readable version of the overlap summary. |

The large processed CSVs are intentionally not tracked in Git because they are row-level data. The summary and manifest files are tracked so the GitHub repository still documents the processed-data structure.
