# Target Scope Summary

## Important Caveat
The original first-job user-level artifact used in the earlier report is not present in this checkout.
Because of that, the user-level school/major summary below comes from `data/processed/t20_all_positions.csv`, which is a much narrower local cohort with `has_position = True` rows.
The employer review counts still provide a useful aggregate view of which reviewed employers are target versus non-target.

## Row and Column Counts
- Available local user-level rows: 9,412
- Available local user-level columns: 13
- User-level rows with positions: 1,999
- User-level columns after adding local cleaned major/school fields in-memory: 15
- Reviewed employers total: 124
- Reviewed target employers: 91
- Reviewed non-target employers: 33

## Reviewed Top-Employer Slice
- Visible top-first-job rows across reviewed employers: 3,234
- Target rows in that visible slice: 940
- Non-target or excluded rows in that visible slice: 2,294

## Dominant Majors In Local Target-Like Cohort
- Engineering: 1,022 (51.1%)
- Missing: 596 (29.8%)
- Business: 254 (12.7%)
- Economics: 23 (1.2%)
- Marketing: 21 (1.1%)
- Statistics: 17 (0.9%)
- Education: 13 (0.7%)
- Law: 12 (0.6%)
- Mathematics: 11 (0.6%)
- Architecture: 6 (0.3%)

## Dominant Schools In Local Target-Like Cohort
- Stanford University: 327 (16.4%)
- University of California Berkeley: 281 (14.1%)
- Cornell University: 235 (11.8%)
- Massachusetts Institute of Technology: 156 (7.8%)
- Columbia University in the City of New York: 102 (5.1%)
- Harvard University: 96 (4.8%)
- Northwestern University: 70 (3.5%)
- University of Pennsylvania: 64 (3.2%)
- Duke University: 50 (2.5%)
- Harvard Business School: 46 (2.3%)
