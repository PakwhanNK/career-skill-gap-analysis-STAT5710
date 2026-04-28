# Reconstructed First-Job Scope

Assumptions:
- Use `User_positions_grouped.csv` as the position-history base.
- Exclude roles whose `role_k17000_v3` contains internship-like terms.
- Use the latest education record per user from `Revelio_EDU_18-22.csv`.
- Keep the earliest non-intern position with `startdate >= latest education enddate` when education end date is present.
- If education end date is missing, keep the earliest non-intern position available.

## Counts
- Reconstructed cohort rows: 57,808
- Reconstructed cohort columns: 23
- Target rows (`manual_decision = 1`): 11,021
- Reviewed non-target/excluded rows (`manual_decision = 0`): 14,167
- Unreviewed rows: 32,620
- Rows left if we exclude unreviewed rows: 25,188
- Columns left if we exclude unreviewed rows: 23

## Dominant Majors In Target
- Engineering: 5,608 (50.9%)
- Missing: 2,203 (20.0%)
- Economics: 1,655 (15.0%)
- Business: 796 (7.2%)
- Mathematics: 234 (2.1%)
- Finance: 157 (1.4%)
- Statistics: 152 (1.4%)
- Biology: 50 (0.5%)
- Physics: 39 (0.4%)
- Marketing: 27 (0.2%)

## Dominant Schools In Target
- University of California Berkeley: 1,779 (16.1%)
- University of Michigan: 1,219 (11.1%)
- Cornell University: 1,057 (9.6%)
- Columbia University in the City of New York: 620 (5.6%)
- University of Notre Dame: 521 (4.7%)
- Duke University: 511 (4.6%)
- University of Pennsylvania: 485 (4.4%)
- Northwestern University: 446 (4.0%)
- Vanderbilt University: 431 (3.9%)
- University of Chicago: 419 (3.8%)
