# Raw Data Manifest

Expected local raw files:

| File | Approx. size | Description |
|---|---:|---|
| `Revelio_EDU_18-22.csv` | 692 MB | Revelio education records filtered to the T20 / 2018-2022 education universe. |
| `User_positions_grouped.csv` | 61 MB | Grouped user position records used to reconstruct first observed jobs. |
| `user_profiles.csv` | 43 MB | User profile fields including `prestige`, `numconnections`, and `user_country`. |
| `user_skills.csv` | 449 MB | User-skill rows used to aggregate skill text and skill indicators. |

These files are intentionally not tracked in Git because they are large row-level data files.
