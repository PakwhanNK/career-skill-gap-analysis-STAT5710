# Reviewed-Only Visualization Caveats

## Scope
- The charts use only the reviewed cohort: 20,846 rows.
- This includes 10,066 target rows and 10,780 reviewed non-target rows.
- It excludes 36,962 unreviewed rows.

## Caveats
- The reviewed-only cohort is selection-biased toward employers we explicitly labeled.
- The excluded unreviewed rows are mostly less common employers in the long tail.
- After dropping missing-employer rows, only 1 remaining unreviewed employers have at least 34 users; the largest non-missing unreviewed employer has 34 users.
- Because of that, the reviewed-only charts emphasize common, high-visibility employers and underrepresent rarer career destinations.
- The first-job cohort is reconstructed from `User_positions_grouped.csv`, not the original hidden parquet artifact from the prior pipeline.
- The first-job definition here uses the earliest non-intern position on or after the latest education end date in `Revelio_EDU_18-22.csv` when available.
- Major and school fields still contain some missingness, so composition charts should be interpreted as approximate rather than definitive.
- Employer labels reflect the current manual review taxonomy, which is a research decision rather than an objective ground truth.

## Output Files
- Figures: `reviewed_target_vs_nontarget_counts.svg`, `reviewed_target_top_universities.svg`, `reviewed_target_major_pie.svg`
- Source tables: `reviewed_target_top_universities.csv`, `reviewed_target_major_pie_source.csv`
