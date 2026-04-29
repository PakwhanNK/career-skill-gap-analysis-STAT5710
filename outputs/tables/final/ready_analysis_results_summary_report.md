# Results Summary: Modeling Pipeline and EDA

## Research Framing

This analysis studies early-career placement into selective target employers, defined as well-known technology, finance, consulting, and quant-oriented firms. The main outcome is binary: whether a user's first observed early-career role is at a reviewed target employer or at a reviewed non-target employer.

The report should avoid framing the outcome as "good job" versus "bad job." A more precise framing is: target-company entry among reviewed employers in tech, finance, consulting, and quant-adjacent labor markets.

## Data Scope and Cleaning Pipeline

The cleaned analysis uses the `data/processed/ready_analysis` universe, which keeps users with overlap across education, position, profile, and skill files. The modeling dataset is further restricted to first-job employer rows with manually reviewed target labels.

The most important cleaning decision was seniority filtering. Revelio seniority originally ranges from 1 to 7. Earlier checks showed that seniority level 1 and 2 are not interchangeable. Level 1 was dominated by roles such as research support specialist, teaching assistant, graduate researcher, and lab/clinical support. Level 2 was more aligned with the target project scope, including software engineering, investment analyst, analyst, corporate legal, and quantitative analyst roles.

The main analysis therefore keeps only `seniority == 2`. Seniority itself is excluded as a model feature because it defines the analytic scope.

Final main modeling cohort:

| Group | Users |
|---|---:|
| Target | 9,641 |
| Non-target | 6,358 |
| Total | 15,999 |

The target rate in the main seniority-2 cohort is 60.3%, so the sample is not severely imbalanced after filtering. A simple always-target classifier would achieve about 60.3% accuracy, which is a useful baseline when interpreting model performance.

## EDA Findings

The target group is heavily concentrated in a small set of universities and fields of study. The largest target-producing schools are University of California Berkeley, University of Michigan, Cornell University, Columbia University, Duke University, University of Pennsylvania, University of Notre Dame, Northwestern University, and Vanderbilt University.

Target-major composition is also concentrated:

| Major | Target Users | Share of Target |
|---|---:|---:|
| Engineering | 5,062 | 52.5% |
| Missing | 1,808 | 18.8% |
| Economics | 1,459 | 15.1% |
| Business | 663 | 6.9% |
| Mathematics | 195 | 2.0% |
| Finance | 149 | 1.5% |
| Statistics | 134 | 1.4% |

This supports the idea that target-company entry is strongly structured by academic pathway. Engineering dominates overall target placement, while economics and business also contribute meaningfully.

## Main Regression Strategy

Two logistic-regression versions were estimated:

1. Primary non-skill regression: excludes raw skill indicators and excludes `n_skills`.
2. Exploratory skill-augmented regression: includes top skill indicators selected using train-only log odds plus `n_skills`.

The primary model is the cleaner specification for the report because LinkedIn-style skills are self-reported, sparse, and may be updated after employment. The skill model is useful as an exploratory comparison, but it should not be the main interpretive model.

Primary non-skill logistic regression performance:

| Metric | Test Value |
|---|---:|
| AUC | 0.692 |
| Accuracy | 0.678 |
| Precision | 0.699 |
| Recall | 0.819 |

Exploratory skill-augmented logistic regression performance:

| Metric | Test Value |
|---|---:|
| AUC | 0.755 |
| Accuracy | 0.742 |
| Precision | 0.718 |
| Recall | 0.939 |

Skill indicators improve predictive performance by about 0.063 AUC, but this lift should be interpreted cautiously because skills may capture profile curation, self-reporting behavior, or post-outcome updates.

## Primary Regression Results

The most stable predictors in the non-skill regression are major, number of LinkedIn connections, school, user country, and entry year.

| Feature | Log Odds | Odds Ratio | p-value |
|---|---:|---:|---:|
| Major target encoding | 0.693 | 2.000 | < 1e-200 |
| Number of connections | 0.307 | 1.359 | < 1e-50 |
| School target encoding | 0.251 | 1.286 | < 1e-20 |
| User country target encoding | 0.113 | 1.120 | < 1e-7 |
| Entry job year | -0.099 | 0.906 | < 1e-6 |
| Prestige | 0.013 | 1.013 | 0.511 |

Substantively, major and school carry the strongest interpretable signal. Number of connections is also predictive, but it should be interpreted carefully because it may capture LinkedIn activity, professional network size, or profile completeness rather than ability alone. Prestige is not significant in this specification.

Because major and school are target-encoded features, the regression coefficients should be interpreted as association with high-target-rate categories, not as direct causal effects of attending a specific school or choosing a specific major.

## Alternative Classifiers

Several classifiers were compared on the seniority-2 cohort:

| Model | AUC | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|
| Logistic regression with skills | 0.755 | 0.742 | 0.718 | 0.939 |
| Decision tree | 0.701 | 0.693 | 0.669 | 0.971 |
| Random forest | 0.732 | 0.693 | 0.666 | 0.986 |

The logistic model remains the best-performing model among the models fit in this environment. Decision tree and random forest were implemented locally because `sklearn` is not installed in the runtime. XGBoost was skipped because the `xgboost` package is not installed.

For the final report, the non-skill logistic regression should be treated as the main interpretable model. The skill-augmented logistic model and tree-based models are useful for robustness and exploratory comparison.

## Seniority Sensitivity Check

The seniority analysis strongly supports filtering to level 2 for the final analytic scope.

| Scenario | Rows | Target Rate | AUC | Accuracy |
|---|---:|---:|---:|---:|
| Levels 1-2, with seniority predictor | 25,042 | 43.9% | 0.829 | 0.768 |
| Levels 1-2, no seniority predictor | 25,042 | 43.9% | 0.764 | 0.692 |
| Seniority 1 only | 9,043 | 15.0% | 0.787 | 0.863 |
| Seniority 2 only | 15,999 | 60.3% | 0.755 | 0.742 |

The large target-rate difference between seniority 1 and seniority 2 means these rows represent different labor-market segments. Keeping both and including seniority as a predictor gives strong performance, but it risks letting the model use seniority as a shortcut. The final report should describe the main sample as seniority-2 early-career roles, not all entry-level or all first jobs.

## Skill Analysis

Skills were analyzed in three ways:

1. Frequency tables for target and non-target users.
2. Skill log-odds comparing target versus non-target users.
3. Exploratory skill-augmented classification models.

The skill analysis shows that biomedical, clinical, nursing, and research-oriented skills tend to be more common among non-target outcomes, while target outcomes are more aligned with software, data, finance, and business-oriented profiles. However, individual skill coefficients are risky to over-interpret because many skill indicators are sparse.

Main risks of using skills as regression features:

- Skills are self-reported or profile-derived rather than direct ability measures.
- Skill coverage and profile maintenance may differ across career tracks.
- Skills may be updated after the first job, creating possible post-outcome contamination.
- Rare skills can produce large but unstable coefficients.
- Skill indicators can proxy for major, industry, or career track rather than independent skill effects.

Recommended report framing: use skills descriptively and exploratorily, not as the main causal or inferential regression specification.

## 2024 vs 2025 Landscape Shift

The year-over-year analysis compares 2024 and 2025 within the same seniority-2 reviewed cohort. This is not a full time-series analysis because only two years are compared.

| Year | Users | Target Users | Target Share |
|---|---:|---:|---:|
| 2024 | 4,064 | 2,492 | 61.3% |
| 2025 | 1,013 | 597 | 58.9% |

The target share is slightly lower in 2025, but the 2025 sample is much smaller, so the result should be interpreted as descriptive rather than definitive.

Top role shares show a possible compositional shift:

| Role | 2024 Share | 2025 Share |
|---|---:|---:|
| Software Engineering | 23.5% | 16.5% |
| Investment Analyst | 8.8% | 6.5% |
| Corporate Transactions Lawyer | 4.6% | 4.1% |
| Software Development | 3.7% | 3.3% |
| General Analyst | 3.8% | 1.8% |
| Corporate Legal | 0.5% | 3.0% |
| Legal Research | 0.3% | 2.1% |
| Quantitative Analyst | 0.5% | 1.0% |

Software engineering remains the largest role category, but its share is lower in 2025. Corporate legal and legal research have higher shares in 2025. These shifts may reflect real labor-market changes, data timing, or partial observation of 2025 outcomes.

Target-major mix also shifts from 2024 to 2025. Engineering falls from 55.0% of target rows in 2024 to 42.5% in 2025, while missing major, economics, and business take larger shares. This should be discussed alongside the smaller 2025 sample size.

## Recommended Final Report Structure

The final report should use the following structure:

1. Research question and outcome definition.
2. Data integration and cleaning pipeline.
3. Manual employer-labeling process and reviewed-employer caveat.
4. Seniority sensitivity check and justification for seniority-2 scope.
5. EDA: target counts, schools, majors, roles, and year-over-year shifts.
6. Primary non-skill logistic regression results.
7. Exploratory skill analysis and skill-augmented models.
8. Robustness checks and limitations.

## Key Caveats

The analysis is strongest when framed as predictive and descriptive, not causal. The reviewed-employer subset excludes unreviewed less-common employers, which may bias the sample toward recognizable firms. The seniority-2 filter improves role-scope clarity but narrows generalizability. Major and school effects are target-encoded and should be interpreted as category-level predictive signals. Skill results are exploratory because skill fields are self-reported/profile-derived and may not represent pre-employment ability.

## Main Takeaway

Among reviewed seniority-2 first observed roles, target-company entry is strongly associated with academic pathway, especially major and school, and moderately associated with profile/network features such as number of connections. Skill data adds predictive signal but introduces interpretability risks, so the cleanest primary model excludes skills and treats them as exploratory descriptors. The 2024-2025 comparison suggests a modest decrease in target share and a possible shift away from software-engineering-heavy outcomes, but the smaller 2025 sample means this should be presented as descriptive evidence rather than a definitive trend.
