# Seniority Sensitivity Check

## Why
The original Revelio seniority scale is 1-7, but this analysis only keeps early-career levels 1 and 2. Because seniority was the strongest feature in the baseline model, this check tests whether it is acting as a shortcut.

## Scenarios
- Original: keep seniority levels 1 and 2, and include `seniority_num` as a predictor.
- A: keep seniority levels 1 and 2, but remove `seniority_num` from the model.
- B: restrict to seniority level 1 only.
- C: restrict to seniority level 2 only.

## Results
- Levels 1-2, with seniority predictor: rows=25,042, target_rate=43.9%, AUC=0.829, accuracy=0.768, precision=0.708, recall=0.801
- Levels 1-2, no seniority predictor: rows=25,042, target_rate=43.9%, AUC=0.764, accuracy=0.692, precision=0.636, recall=0.698
- Seniority 1 only: rows=9,043, target_rate=15.0%, AUC=0.787, accuracy=0.863, precision=0.725, recall=0.137
- Seniority 2 only: rows=15,999, target_rate=60.3%, AUC=0.755, accuracy=0.742, precision=0.718, recall=0.939

## Interpretation
The target rate differs sharply between seniority 1 and seniority 2, so the two junior seniority buckets are not interchangeable. Treat the data as early-career first observed roles rather than pure entry-level jobs, and prefer the no-seniority model or separate seniority-specific checks for substantive interpretation.
