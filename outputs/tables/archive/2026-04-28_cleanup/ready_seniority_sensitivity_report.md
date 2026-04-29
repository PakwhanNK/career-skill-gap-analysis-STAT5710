# Seniority Sensitivity Check

## Why
The original Revelio seniority scale is 1-7, but this analysis only keeps early-career levels 1 and 2. Because seniority was the strongest feature in the baseline model, this check tests whether it is acting as a shortcut.

## Scenarios
- Original: keep seniority levels 1 and 2, and include `seniority_num` as a predictor.
- A: keep seniority levels 1 and 2, but remove `seniority_num` from the model.
- B: restrict to seniority level 1 only.
- C: restrict to seniority level 2 only.

## Results
- Levels 1-2, with seniority predictor: rows=25,042, target_rate=43.9%, AUC=0.829, accuracy=0.768, precision=0.710, recall=0.797
- Levels 1-2, no seniority predictor: rows=25,042, target_rate=43.9%, AUC=0.764, accuracy=0.690, precision=0.635, recall=0.693
- Seniority 1 only: rows=9,043, target_rate=15.0%, AUC=0.787, accuracy=0.864, precision=0.778, recall=0.129
- Seniority 2 only: rows=15,999, target_rate=60.3%, AUC=0.756, accuracy=0.741, precision=0.718, recall=0.938

## Interpretation
The target rate differs sharply between seniority 1 and seniority 2, so the two junior seniority buckets are not interchangeable. Treat the data as early-career first observed roles rather than pure entry-level jobs, and prefer the no-seniority model or separate seniority-specific checks for substantive interpretation.
