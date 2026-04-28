from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path(r"C:\Users\knich\Projects\career-skill-gap-analysis-STAT5710")
MAIN_REVIEW = BASE / "outputs" / "tables" / "employer_target_review.csv"
PRIORITY_REVIEW = BASE / "outputs" / "tables" / "priority_unreviewed_employers.csv"


def main() -> None:
    main_df = pd.read_csv(MAIN_REVIEW)
    priority_df = pd.read_csv(PRIORITY_REVIEW)

    main_df["company_norm"] = main_df["company_name"].astype(str).str.strip().str.lower()
    main_df["manual_decision"] = main_df["manual_decision"].astype("string").fillna("").str.strip()
    priority_df["company_norm"] = priority_df["employer_name"].astype(str).str.strip().str.lower()
    priority_df["manual_decision"] = priority_df["manual_decision"].astype(str).str.strip()

    update_map = (
        priority_df.loc[priority_df["manual_decision"].ne(""), ["company_norm", "manual_decision"]]
        .drop_duplicates("company_norm", keep="last")
        .set_index("company_norm")["manual_decision"]
    )

    matched = main_df["company_norm"].isin(update_map.index)
    main_df.loc[matched, "manual_decision"] = main_df.loc[matched, "company_norm"].map(update_map)

    missing_from_main = priority_df.loc[~priority_df["company_norm"].isin(main_df["company_norm"]), :].copy()
    if not missing_from_main.empty:
        append_df = pd.DataFrame(
            {
                "company_name": missing_from_main["employer_name"],
                "rcid": "",
                "top_first_job_count": 0,
                "latest_profile_count_proxy": 0,
                "in_excel_popular_list": 0,
                "excel_industry": "",
                "hq_state": "",
                "suggested_bucket": missing_from_main["suggested_bucket"],
                "suggested_action": "review",
                "confidence": missing_from_main["confidence"],
                "manual_decision": missing_from_main["manual_decision"],
                "notes": missing_from_main["notes"],
                "company_norm": missing_from_main["company_norm"],
            }
        )
        main_df = pd.concat([main_df, append_df], ignore_index=True)

    main_df = main_df.drop(columns=["company_norm"])
    main_df.to_csv(MAIN_REVIEW, index=False)

    print(f"Updated {MAIN_REVIEW}")
    print(f"Matched existing employers: {int(matched.sum())}")
    print(f"Appended new employers: {len(missing_from_main)}")


if __name__ == "__main__":
    main()
