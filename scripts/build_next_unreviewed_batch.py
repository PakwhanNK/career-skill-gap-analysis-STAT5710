from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


BASE = Path(r"C:\Users\knich\Projects\career-skill-gap-analysis-STAT5710")
COHORT = BASE / "outputs" / "tables" / "reconstructed_first_job_scope_sample.csv"
OUT = BASE / "outputs" / "tables" / "priority_unreviewed_employers_batch2.csv"

TECH_PATTERNS = [
    r"\bsoftware\b",
    r"\btechnolog",
    r"\bcloud\b",
    r"\bdata\b",
    r"\bsemiconductor",
    r"\binternet\b",
    r"\bplatform",
    r"\bsystems?\b",
    r"\bai\b",
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
    r"\bpartners\b",
]
CONSULTING_PATTERNS = [
    r"\bconsult",
    r"\badvis",
    r"\bstrategy\b",
    r"cornerstone research",
    r"alphasights",
]
QUANT_PATTERNS = [
    r"\btrading\b",
    r"\bquant",
    r"\bmarket mak",
    r"\bhedge fund\b",
    r"two sigma",
    r"citadel",
    r"jane street",
    r"optiver",
    r"imc",
]
EXCLUDE_PATTERNS = [
    r"\buniversity\b",
    r"\bcollege\b",
    r"\bschool\b",
    r"\binstitute\b",
    r"\bhospital\b",
    r"\bmedical\b",
    r"\bhealth\b",
    r"\bstate of\b",
    r"\bgovernment\b",
    r"\bfoundation\b",
]

EXACT = {
    "merck & co., inc.": "other_or_review",
    "university of cambridge": "exclude",
    "stealth startup": "other_or_review",
    "texas instruments incorporated": "tech",
    "mizuho financial group, inc.": "finance",
    "johnson & johnson": "other_or_review",
    "huron consulting group, inc.": "consulting",
    "the toronto-dominion bank": "finance",
    "moelis & co.": "finance",
    "asml holding nv": "tech",
    "government of singapore": "exclude",
    "perella weinberg partners": "finance",
    "united airlines holdings, inc.": "other_or_review",
    "davis polk & wardwell llp": "other_or_review",
    "scale ai, inc.": "tech",
    "aon plc": "consulting",
    "boston university": "exclude",
    "mcmaster-carr supply co.": "other_or_review",
    "oppenheimer holdings, inc.": "finance",
    "zurich insurance group ag": "other_or_review",
    "freshfields llp": "other_or_review",
    "morgan, lewis & bockius llp": "other_or_review",
    "d. e. shaw & co., l.p.": "quant",
    "de shaw & co. lp": "quant",
    "goldman sachs asset management lp": "finance",
    "morgan, lewis & bockius llp.": "other_or_review",
    "white & case llp": "other_or_review",
    "skadden, arps, slate, meagher & flom llp": "other_or_review",
}


def clean(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"^the\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def suggest(name: str) -> tuple[str, str, str]:
    n = clean(name)
    if n in EXACT:
        return EXACT[n], "high", "exact mapping"
    for p in EXCLUDE_PATTERNS:
        if re.search(p, n):
            return "exclude", "high", f"exclude pattern: {p}"
    for p in QUANT_PATTERNS:
        if re.search(p, n):
            return "quant", "medium", f"quant keyword: {p}"
    for p in CONSULTING_PATTERNS:
        if re.search(p, n):
            return "consulting", "medium", f"consulting keyword: {p}"
    for p in FINANCE_PATTERNS:
        if re.search(p, n):
            return "finance", "medium", f"finance keyword: {p}"
    for p in TECH_PATTERNS:
        if re.search(p, n):
            return "tech", "medium", f"tech keyword: {p}"
    return "other_or_review", "low", "no rule match"


def main() -> None:
    df = pd.read_csv(COHORT)
    manual_raw = df["manual_decision"]
    manual = manual_raw.astype(str).str.strip().replace({"nan": "", "1.0": "1", "0.0": "0"})
    df["employer_name"] = df["employer_name"].fillna("Missing").astype(str).str.strip()
    df["major"] = df["major"].fillna("Missing").astype(str).str.strip()
    df["school"] = df["school"].fillna("Missing").astype(str).str.strip()

    unreviewed = df.loc[manual_raw.isna() | manual.eq("")].copy()
    unreviewed = unreviewed.loc[unreviewed["employer_name"].str.lower() != "missing"].copy()

    summary = (
        unreviewed.groupby("employer_name")
        .agg(
            users=("user_id", "nunique"),
            top_major=("major", lambda s: pd.Series(s).value_counts().index[0]),
            top_school=("school", lambda s: pd.Series(s).value_counts().index[0]),
        )
        .reset_index()
        .sort_values(["users", "employer_name"], ascending=[False, True])
        .head(200)
        .copy()
    )
    total_non_missing_unreviewed = int(unreviewed["user_id"].nunique())
    summary["cum_share_of_nonmissing_unreviewed"] = summary["users"].cumsum() / max(1, summary["users"].sum())
    suggestion_df = pd.DataFrame(
        summary["employer_name"].map(suggest).tolist(),
        columns=["suggested_bucket", "confidence", "notes"],
        index=summary.index,
    )
    summary[["suggested_bucket", "confidence", "notes"]] = suggestion_df
    summary["manual_decision"] = summary["suggested_bucket"].apply(
        lambda x: "1" if x in {"tech", "finance", "consulting", "quant"} else "0"
    )
    summary.to_csv(OUT, index=False)

    covered_rows = int(summary["users"].sum())
    print(f"Wrote {OUT}")
    print(f"Non-missing unreviewed rows available: {total_non_missing_unreviewed}")
    print(f"Rows covered by this 200-employer batch: {covered_rows}")
    print(f"Coverage of non-missing unreviewed rows: {covered_rows / max(1, total_non_missing_unreviewed):.1%}")
    print(summary[["employer_name", "users", "suggested_bucket", "confidence", "manual_decision"]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
