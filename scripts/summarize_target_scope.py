from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path(r"C:\Users\knich\Projects\career-skill-gap-analysis-STAT5710")
POSITIONS = BASE / "data" / "processed" / "t20_all_positions.csv"
REVIEW = BASE / "outputs" / "tables" / "employer_review" / "reviewed_companies.csv"
OUT_DIR = BASE / "outputs" / "tables" / "archive" / "target_scope"
OUT_MD = OUT_DIR / "target_scope_summary.md"
OUT_JSON = OUT_DIR / "target_scope_summary.json"
OUT_MAJORS = OUT_DIR / "target_scope_top_majors.csv"
OUT_SCHOOLS = OUT_DIR / "target_scope_top_schools.csv"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    positions = pd.read_csv(POSITIONS)
    review = pd.read_csv(REVIEW)

    review["manual_decision"] = review["manual_decision"].astype(str).str.strip()
    review["top_first_job_count"] = pd.to_numeric(review["top_first_job_count"], errors="coerce").fillna(0).astype(int)

    with_position = positions[positions["has_position"] == True].copy()
    with_position["major"] = with_position["field"].fillna("Missing").astype(str).str.strip().replace({"": "Missing"})
    with_position["school"] = with_position["university_name"].fillna("Missing").astype(str).str.strip().replace({"": "Missing"})

    top_majors = with_position["major"].value_counts().rename_axis("major").reset_index(name="users")
    top_majors["share"] = top_majors["users"] / len(with_position)
    top_schools = with_position["school"].value_counts().rename_axis("school").reset_index(name="users")
    top_schools["share"] = top_schools["users"] / len(with_position)

    top_majors.to_csv(OUT_MAJORS, index=False)
    top_schools.to_csv(OUT_SCHOOLS, index=False)

    summary = {
        "available_user_level_rows": int(len(positions)),
        "available_user_level_cols": int(positions.shape[1]),
        "user_level_rows_with_positions": int(len(with_position)),
        "user_level_cols_with_positions": int(with_position.shape[1]),
        "review_employers_total": int(len(review)),
        "review_target_employers": int((review["manual_decision"] == "1").sum()),
        "review_non_target_employers": int((review["manual_decision"] == "0").sum()),
        "top_first_job_rows_visible_in_review_slice": int(review["top_first_job_count"].sum()),
        "top_first_job_target_rows_visible_in_review_slice": int(
            review.loc[review["manual_decision"] == "1", "top_first_job_count"].sum()
        ),
        "top_first_job_non_target_rows_visible_in_review_slice": int(
            review.loc[review["manual_decision"] == "0", "top_first_job_count"].sum()
        ),
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Target Scope Summary",
        "",
        "## Important Caveat",
        "The original first-job user-level artifact used in the earlier report is not present in this checkout.",
        "Because of that, the user-level school/major summary below comes from `data/processed/t20_all_positions.csv`, which is a much narrower local cohort with `has_position = True` rows.",
        "The employer review counts still provide a useful aggregate view of which reviewed employers are target versus non-target.",
        "",
        "## Row and Column Counts",
        f"- Available local user-level rows: {summary['available_user_level_rows']:,}",
        f"- Available local user-level columns: {summary['available_user_level_cols']:,}",
        f"- User-level rows with positions: {summary['user_level_rows_with_positions']:,}",
        f"- User-level columns after adding local cleaned major/school fields in-memory: {summary['user_level_cols_with_positions']:,}",
        f"- Reviewed employers total: {summary['review_employers_total']:,}",
        f"- Reviewed target employers: {summary['review_target_employers']:,}",
        f"- Reviewed non-target employers: {summary['review_non_target_employers']:,}",
        "",
        "## Reviewed Top-Employer Slice",
        f"- Visible top-first-job rows across reviewed employers: {summary['top_first_job_rows_visible_in_review_slice']:,}",
        f"- Target rows in that visible slice: {summary['top_first_job_target_rows_visible_in_review_slice']:,}",
        f"- Non-target or excluded rows in that visible slice: {summary['top_first_job_non_target_rows_visible_in_review_slice']:,}",
        "",
        "## Dominant Majors In Local Target-Like Cohort",
    ]

    for _, row in top_majors.head(10).iterrows():
        lines.append(f"- {row['major']}: {int(row['users']):,} ({row['share']:.1%})")

    lines.extend(
        [
            "",
            "## Dominant Schools In Local Target-Like Cohort",
        ]
    )
    for _, row in top_schools.head(10).iterrows():
        lines.append(f"- {row['school']}: {int(row['users']):,} ({row['share']:.1%})")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MAJORS}")
    print(f"Wrote {OUT_SCHOOLS}")


if __name__ == "__main__":
    main()
