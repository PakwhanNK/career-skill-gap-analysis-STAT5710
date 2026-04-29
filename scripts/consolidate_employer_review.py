from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


BASE = Path(r"C:\Users\knich\Projects\career-skill-gap-analysis-STAT5710")
ROOT_TABLES = BASE / "outputs" / "tables"
REVIEW_DIR = ROOT_TABLES / "employer_review"
OUT_FILE = REVIEW_DIR / "reviewed_companies.csv"
SUMMARY_FILE = REVIEW_DIR / "reviewed_companies_summary.json"

INPUT_FILES = [
    REVIEW_DIR / "archive_batches" / "employer_target_review.csv",
    REVIEW_DIR / "archive_batches" / "priority_unreviewed_employers.csv",
    REVIEW_DIR / "archive_batches" / "priority_unreviewed_employers_prefilled.csv",
    REVIEW_DIR / "archive_batches" / "priority_unreviewed_employers_batch2.csv",
]


def normalize_company(value: str) -> str:
    value = str(value or "").strip().lower()
    value = re.sub(r"^the\s+", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def clean_decision(value: object) -> str:
    value = str(value).strip()
    if value in {"1", "1.0"}:
        return "1"
    if value in {"0", "0.0"}:
        return "0"
    return ""


def load_one(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "company_name" not in df.columns and "employer_name" in df.columns:
        df = df.rename(columns={"employer_name": "company_name"})
    if "top_first_job_count" not in df.columns and "users" in df.columns:
        df = df.rename(columns={"users": "top_first_job_count"})
    for col in [
        "company_name",
        "rcid",
        "top_first_job_count",
        "latest_profile_count_proxy",
        "in_excel_popular_list",
        "excel_industry",
        "hq_state",
        "suggested_bucket",
        "suggested_action",
        "confidence",
        "manual_decision",
        "notes",
        "top_major",
        "top_school",
    ]:
        if col not in df.columns:
            df[col] = ""
    df["source_table"] = path.name
    df["company_norm"] = df["company_name"].map(normalize_company)
    df["manual_decision"] = df["manual_decision"].map(clean_decision)
    df["top_first_job_count"] = pd.to_numeric(df["top_first_job_count"], errors="coerce").fillna(0).astype(int)
    return df


def first_nonempty(values: pd.Series) -> str:
    for value in values:
        if pd.notna(value) and str(value).strip() not in {"", "nan"}:
            return str(value).strip()
    return ""


def combine_group(group: pd.DataFrame) -> dict[str, object]:
    # Later files should win for manual decisions, but keep the largest count.
    reviewed = group[group["manual_decision"].isin(["0", "1"])]
    decision = reviewed.iloc[-1]["manual_decision"] if len(reviewed) else ""
    return {
        "company_name": first_nonempty(group["company_name"]),
        "company_norm": group.iloc[0]["company_norm"],
        "rcid": first_nonempty(group["rcid"]),
        "top_first_job_count": int(group["top_first_job_count"].max()),
        "latest_profile_count_proxy": first_nonempty(group["latest_profile_count_proxy"]),
        "in_excel_popular_list": first_nonempty(group["in_excel_popular_list"]),
        "excel_industry": first_nonempty(group["excel_industry"]),
        "hq_state": first_nonempty(group["hq_state"]),
        "suggested_bucket": first_nonempty(group["suggested_bucket"]),
        "suggested_action": first_nonempty(group["suggested_action"]),
        "confidence": first_nonempty(group["confidence"]),
        "manual_decision": decision,
        "notes": " | ".join(sorted({str(x).strip() for x in group["notes"] if pd.notna(x) and str(x).strip() not in {"", "nan"}})),
        "top_major": first_nonempty(group["top_major"]),
        "top_school": first_nonempty(group["top_school"]),
        "source_tables": "; ".join(sorted(set(group["source_table"]))),
    }


def main() -> None:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pieces = [load_one(path) for path in INPUT_FILES]
    pieces = [piece for piece in pieces if len(piece)]
    if not pieces:
        raise FileNotFoundError("No employer review files found to consolidate.")
    all_rows = pd.concat(pieces, ignore_index=True)
    all_rows = all_rows[all_rows["company_norm"].ne("")]
    consolidated = pd.DataFrame([combine_group(group) for _, group in all_rows.groupby("company_norm", sort=False)])
    consolidated = consolidated.sort_values(["manual_decision", "top_first_job_count"], ascending=[False, False])
    consolidated.to_csv(OUT_FILE, index=False)
    summary = {
        "rows": int(len(consolidated)),
        "reviewed_target_employers": int((consolidated["manual_decision"] == "1").sum()),
        "reviewed_non_target_employers": int((consolidated["manual_decision"] == "0").sum()),
        "unreviewed_employers": int((consolidated["manual_decision"] == "").sum()),
        "source_files": [str(path.relative_to(BASE)) for path in INPUT_FILES if path.exists()],
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_FILE}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
