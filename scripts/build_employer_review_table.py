from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(r"C:\Users\knich\Projects\career-skill-gap-analysis-STAT5710")
RAW_POSITIONS = BASE / "data" / "raw" / "Revelio.csv"
PROCESSED_POSITIONS = BASE / "data" / "processed" / "t20_all_positions.csv"
POPULAR_COMPANIES = BASE / "outputs" / "tables" / "popular_company_list_from_data_xlsx.csv"
TOP_FIRST_JOB = BASE / "outputs" / "tables" / "eda_top_first_job_employers_with_excel_label.csv"
OUT_DIR = BASE / "outputs" / "tables"
OUT_FILE = OUT_DIR / "employer_target_review.csv"


TARGET_EXACT = {
    "accenture plc": "consulting",
    "adobe, inc.": "tech",
    "alphabet, inc.": "tech",
    "amazon.com, inc.": "tech",
    "american express co.": "finance",
    "apollo global management, inc.": "finance",
    "apple, inc.": "tech",
    "blackrock, inc.": "finance",
    "capital one financial corp.": "finance",
    "citadel llc": "quant",
    "citadel securities llc": "quant",
    "deloitte llp": "consulting",
    "goldman sachs group, inc.": "finance",
    "hudson river trading llc": "quant",
    "imc trading b.v.": "quant",
    "jane street group, llc": "quant",
    "jpmorgan chase & co.": "finance",
    "kpmg llp": "consulting",
    "meta platforms, inc.": "tech",
    "mckinsey & company, inc.": "consulting",
    "microsoft corp.": "tech",
    "morgan stanley": "finance",
    "optiver holding b.v.": "quant",
    "palantir technologies inc.": "tech",
    "pricewaterhousecoopers llp": "consulting",
    "stripe, inc.": "tech",
    "the boston consulting group, inc.": "consulting",
    "two sigma investments, lp": "quant",
}

TECH_PATTERNS = [
    r"\bsoftware\b",
    r"\btechnolog",
    r"\bcloud\b",
    r"\bdata\b",
    r"\bsemiconductor",
    r"\binternet\b",
    r"\bplatform",
    r"\bsystems\b",
]
FINANCE_PATTERNS = [
    r"\bbank\b",
    r"\bcapital\b",
    r"\bfinancial\b",
    r"\basset\b",
    r"\binvest",
    r"\bsecurities\b",
    r"\bpayments?\b",
    r"\bcredit\b",
]
CONSULTING_PATTERNS = [
    r"\bconsult",
    r"\badvis",
    r"\bstrategy\b",
]
QUANT_PATTERNS = [
    r"\btrading\b",
    r"\bquant",
    r"\bmarket mak",
    r"\bhedge fund\b",
]
EXCLUDE_PATTERNS = [
    r"\buniversity\b",
    r"\bcollege\b",
    r"\bschool\b",
    r"\binstitute\b",
    r"\bhospital\b",
    r"\bmedical center\b",
    r"\bhealth\b",
    r"\bgovernment\b",
    r"\bcity of\b",
    r"\bstate of\b",
    r"\bcounty\b",
    r"\bfoundation\b",
    r"\bnonprofit\b",
]


def clean_name(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"^the\s+", "", value)
    return value


def suggest_bucket(name: str) -> tuple[str, str, str]:
    normalized = clean_name(name)
    if not normalized:
        return "unknown", "exclude", "empty company name"
    if normalized in TARGET_EXACT:
        return "high", TARGET_EXACT[normalized], "exact target list match"
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, normalized):
            return "high", "exclude", f"exclude pattern: {pattern}"
    for pattern in QUANT_PATTERNS:
        if re.search(pattern, normalized):
            return "medium", "quant", f"quant keyword: {pattern}"
    for pattern in CONSULTING_PATTERNS:
        if re.search(pattern, normalized):
            return "medium", "consulting", f"consulting keyword: {pattern}"
    for pattern in FINANCE_PATTERNS:
        if re.search(pattern, normalized):
            return "medium", "finance", f"finance keyword: {pattern}"
    for pattern in TECH_PATTERNS:
        if re.search(pattern, normalized):
            return "medium", "tech", f"tech keyword: {pattern}"
    return "low", "other_or_review", "no exact match or sector keyword"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_rows = read_csv(RAW_POSITIONS)
    processed_rows = read_csv(PROCESSED_POSITIONS)
    popular_rows = read_csv(POPULAR_COMPANIES)
    first_job_rows = read_csv(TOP_FIRST_JOB)

    company_name_by_rcid: dict[str, str] = {}
    for row in raw_rows:
        rcid = clean_name(row.get("(ultimate_parent_rcid) Ultimate parent RCID", ""))
        name = (row.get("(ultimate_parent_company_name) Ultimate parent company name", "") or "").strip()
        if rcid and name and rcid not in company_name_by_rcid:
            company_name_by_rcid[rcid] = name
        rcid_direct = clean_name(row.get("(rcid) Revelio Company ID", ""))
        company_direct = (row.get("(company) Company name", "") or "").strip()
        if rcid_direct and company_direct and rcid_direct not in company_name_by_rcid:
            company_name_by_rcid[rcid_direct] = company_direct

    latest_counts: Counter[tuple[str, str]] = Counter()
    for row in processed_rows:
        rcid = clean_name(row.get("latest_rcid", ""))
        if not rcid:
            continue
        company = company_name_by_rcid.get(rcid, "")
        if not company:
            continue
        latest_counts[(rcid, company)] += 1

    first_job_counts: dict[str, int] = {}
    first_job_excel_label: dict[str, str] = {}
    for row in first_job_rows:
        company = (row.get("ultimate_parent_company_name", "") or "").strip()
        if not company:
            continue
        first_job_counts[company] = int(float(row.get("first_jobs", "0") or 0))
        first_job_excel_label[company] = row.get("popular_company_excel", "")

    popular_lookup: dict[str, dict[str, str]] = {}
    for row in popular_rows:
        company = (row.get("company", "") or "").strip()
        rcid = clean_name(row.get("rcid_int", "") or row.get("rcid", ""))
        if company:
            popular_lookup[company] = {
                "popular_excel": "1",
                "popular_rcid": rcid,
                "industry": (row.get("industry", "") or "").strip(),
                "hq_state": (row.get("hq_state", "") or "").strip(),
            }

    combined: defaultdict[tuple[str, str], dict[str, str | int]] = defaultdict(
        lambda: {
            "latest_profile_count": 0,
            "top_first_job_count": 0,
            "in_excel_popular_list": "0",
            "excel_industry": "",
            "hq_state": "",
        }
    )

    for (rcid, company), count in latest_counts.items():
        item = combined[(rcid, company)]
        item["latest_profile_count"] = count

    for company, count in first_job_counts.items():
        matched_key = None
        for (rcid, existing_company) in combined.keys():
            if existing_company == company:
                matched_key = (rcid, existing_company)
                break
        if matched_key is None:
            matched_key = ("", company)
        item = combined[matched_key]
        item["top_first_job_count"] = count

    for company, meta in popular_lookup.items():
        matched_key = None
        for (rcid, existing_company) in combined.keys():
            if existing_company == company or (meta["popular_rcid"] and rcid == meta["popular_rcid"]):
                matched_key = (rcid, existing_company)
                break
        if matched_key is None:
            matched_key = (meta["popular_rcid"], company)
        item = combined[matched_key]
        item["in_excel_popular_list"] = "1"
        item["excel_industry"] = meta["industry"]
        item["hq_state"] = meta["hq_state"]

    rows = []
    for (rcid, company), meta in combined.items():
        company_name = company.strip()
        if not company_name:
            continue
        confidence, suggested_bucket, reason = suggest_bucket(company_name)
        top_first_job_count = int(meta["top_first_job_count"])
        latest_profile_count = int(meta["latest_profile_count"])
        source_priority = top_first_job_count * 1000 + latest_profile_count
        rows.append(
            {
                "company_name": company_name,
                "rcid": rcid,
                "top_first_job_count": top_first_job_count,
                "latest_profile_count_proxy": latest_profile_count,
                "in_excel_popular_list": meta["in_excel_popular_list"],
                "excel_industry": meta["excel_industry"],
                "hq_state": meta["hq_state"],
                "suggested_bucket": suggested_bucket,
                "suggested_action": "review" if confidence in {"medium", "low"} else "accept_if_sensible",
                "confidence": confidence,
                "manual_decision": "",
                "notes": reason,
                "_sort_key": source_priority,
            }
        )

    rows.sort(key=lambda r: (r["_sort_key"], r["in_excel_popular_list"] == "1"), reverse=True)

    with OUT_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row.pop("_sort_key", None)
            writer.writerow(row)

    print(f"Wrote {len(rows):,} employer rows to {OUT_FILE}")
    print("\nTop review rows:")
    for row in rows[:25]:
        print(
            f"fj={row['top_first_job_count']:>4} | lp={row['latest_profile_count_proxy']:>4} | "
            f"{row['company_name'][:55]:<55} | {row['suggested_bucket']:<15} | "
            f"excel={row['in_excel_popular_list']} | {row['confidence']}"
        )


if __name__ == "__main__":
    main()
