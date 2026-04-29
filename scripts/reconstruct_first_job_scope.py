from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path(r"C:\Users\knich\Projects\career-skill-gap-analysis-STAT5710")
POSITIONS = BASE / "data" / "raw" / "User_positions_grouped.csv"
EDUCATION = BASE / "data" / "raw" / "Revelio_EDU_18-22.csv"
REVIEW = BASE / "outputs" / "tables" / "employer_review" / "reviewed_companies.csv"
OUT_DIR = BASE / "outputs" / "tables" / "archive" / "reconstruction"


def clean_text(series: pd.Series, missing: str = "Missing") -> pd.Series:
    return series.fillna(missing).astype(str).str.strip().replace({"": missing, "nan": missing})


def normalize_decision(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().replace({"nan": "", "1.0": "1", "0.0": "0"})
    return s


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    positions = pd.read_csv(
        POSITIONS,
        usecols=[
            "user_id",
            "seniority",
            "startdate",
            "enddate",
            "role_k17000_v3",
            "rcid",
            "ultimate_parent_rcid",
            "company",
            "ultimate_parent_company_name",
        ],
    )
    education = pd.read_csv(
        EDUCATION,
        usecols=["user_id", "enddate", "university_name", "degree", "field", "university_country"],
    )
    review = pd.read_csv(REVIEW)

    education["enddate"] = pd.to_datetime(education["enddate"], errors="coerce")
    latest_education = education.sort_values(["user_id", "enddate"]).groupby("user_id").tail(1).copy()
    latest_education = latest_education.rename(columns={"enddate": "edu_enddate"})

    positions["startdate"] = pd.to_datetime(positions["startdate"], errors="coerce")
    positions["role_k17000_v3"] = positions["role_k17000_v3"].fillna("")
    non_intern = ~positions["role_k17000_v3"].str.contains(
        r"intern|internship|co-op|coop|trainee|apprentice|fellow", case=False, regex=True
    )
    positions = positions.loc[non_intern].copy()

    cohort = positions.merge(
        latest_education[["user_id", "edu_enddate", "university_name", "degree", "field", "university_country"]],
        on="user_id",
        how="left",
    )
    cohort = cohort[(cohort["edu_enddate"].isna()) | (cohort["startdate"] >= cohort["edu_enddate"])].copy()
    cohort = cohort.sort_values(["user_id", "startdate"]).groupby("user_id").head(1).copy()

    cohort["employer_name"] = clean_text(cohort["ultimate_parent_company_name"].where(
        clean_text(cohort["ultimate_parent_company_name"]).ne("Missing"),
        cohort["company"],
    ))
    review["company_norm"] = review["company_name"].astype(str).str.strip().str.lower()
    review["manual_decision"] = review["manual_decision"].astype(str).str.strip()
    cohort["company_norm"] = cohort["employer_name"].astype(str).str.strip().str.lower()
    cohort = cohort.merge(
        review[["company_norm", "manual_decision", "suggested_bucket"]],
        on="company_norm",
        how="left",
    )
    cohort["manual_decision"] = normalize_decision(cohort["manual_decision"])

    cohort["target_flag"] = cohort["manual_decision"].eq("1")
    cohort["reviewed_non_target_flag"] = cohort["manual_decision"].eq("0")
    cohort["unreviewed_flag"] = cohort["manual_decision"].isna() | cohort["manual_decision"].eq("")

    cohort["major"] = clean_text(cohort["field"])
    cohort["school"] = clean_text(cohort["university_name"])

    target = cohort[cohort["target_flag"]].copy()
    reviewed_scope = cohort[~cohort["unreviewed_flag"]].copy()

    target_majors = target["major"].value_counts().rename_axis("major").reset_index(name="users")
    target_majors["share"] = target_majors["users"] / len(target) if len(target) else 0
    target_schools = target["school"].value_counts().rename_axis("school").reset_index(name="users")
    target_schools["share"] = target_schools["users"] / len(target) if len(target) else 0

    target_majors.to_csv(OUT_DIR / "reconstructed_target_top_majors.csv", index=False)
    target_schools.to_csv(OUT_DIR / "reconstructed_target_top_schools.csv", index=False)
    cohort.to_csv(OUT_DIR / "reconstructed_first_job_scope_sample.csv", index=False)

    summary = {
        "reconstructed_rows": int(len(cohort)),
        "reconstructed_cols": int(cohort.shape[1]),
        "target_rows": int(cohort["target_flag"].sum()),
        "reviewed_non_target_rows": int(cohort["reviewed_non_target_flag"].sum()),
        "unreviewed_rows": int(cohort["unreviewed_flag"].sum()),
        "reviewed_scope_rows": int(len(reviewed_scope)),
        "reviewed_scope_cols": int(reviewed_scope.shape[1]),
        "target_share_all_rows": float(cohort["target_flag"].mean()),
        "target_share_reviewed_scope": float(target["target_flag"].sum() / len(reviewed_scope)) if len(reviewed_scope) else None,
    }
    (OUT_DIR / "reconstructed_first_job_scope_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Reconstructed First-Job Scope",
        "",
        "Assumptions:",
        "- Use `User_positions_grouped.csv` as the position-history base.",
        "- Exclude roles whose `role_k17000_v3` contains internship-like terms.",
        "- Use the latest education record per user from `Revelio_EDU_18-22.csv`.",
        "- Keep the earliest non-intern position with `startdate >= latest education enddate` when education end date is present.",
        "- If education end date is missing, keep the earliest non-intern position available.",
        "",
        "## Counts",
        f"- Reconstructed cohort rows: {summary['reconstructed_rows']:,}",
        f"- Reconstructed cohort columns: {summary['reconstructed_cols']:,}",
        f"- Target rows (`manual_decision = 1`): {summary['target_rows']:,}",
        f"- Reviewed non-target/excluded rows (`manual_decision = 0`): {summary['reviewed_non_target_rows']:,}",
        f"- Unreviewed rows: {summary['unreviewed_rows']:,}",
        f"- Rows left if we exclude unreviewed rows: {summary['reviewed_scope_rows']:,}",
        f"- Columns left if we exclude unreviewed rows: {summary['reviewed_scope_cols']:,}",
        "",
        "## Dominant Majors In Target",
    ]
    for _, row in target_majors.head(10).iterrows():
        lines.append(f"- {row['major']}: {int(row['users']):,} ({row['share']:.1%})")
    lines.extend(["", "## Dominant Schools In Target"])
    for _, row in target_schools.head(10).iterrows():
        lines.append(f"- {row['school']}: {int(row['users']):,} ({row['share']:.1%})")

    (OUT_DIR / "reconstructed_first_job_scope_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote reconstructed first-job scope outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
