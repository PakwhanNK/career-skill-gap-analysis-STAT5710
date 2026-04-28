from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


BASE = Path(r"C:\Users\knich\Projects\career-skill-gap-analysis-STAT5710")
OUT_DIR = BASE / "data" / "processed" / "ready_analysis"

USER_IDS = BASE / "notebooks" / "user_ids.txt"
EDU = BASE / "data" / "raw" / "Revelio_EDU_18-22.csv"
SKILLS = BASE / "data" / "raw" / "user_skills.csv"
POSITIONS = BASE / "data" / "raw" / "User_positions_grouped.csv"
PROFILES = BASE / "data" / "raw" / "user_profiles.csv"


def parse_user_id(value: str) -> int | None:
    try:
        return int(float(value))
    except Exception:
        return None


def parse_date(value: str):
    s = (value or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def load_user_ids_txt(path: Path) -> set[int]:
    ids: set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            uid = parse_user_id(line.strip())
            if uid is not None:
                ids.add(uid)
    return ids


def load_csv_ids(path: Path, user_col: str) -> set[int]:
    ids: set[int] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = parse_user_id(row.get(user_col, ""))
            if uid is not None:
                ids.add(uid)
    return ids


def stream_filter_csv(src: Path, dst: Path, user_col: str, keep_ids: set[int]) -> int:
    kept = 0
    with src.open("r", encoding="utf-8-sig", newline="") as fin, dst.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            uid = parse_user_id(row.get(user_col, ""))
            if uid is None or uid not in keep_ids:
                continue
            writer.writerow(row)
            kept += 1
    return kept


def build_skill_agg(src: Path, dst: Path, keep_ids: set[int]) -> int:
    skills_by_user: dict[int, set[str]] = defaultdict(set)
    with src.open("r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            uid = parse_user_id(row.get("user_id", ""))
            if uid is None or uid not in keep_ids:
                continue
            skill = (row.get("skill_k35000") or row.get("skill_translated") or row.get("skill_raw") or "").strip()
            if skill:
                skills_by_user[uid].add(skill)
    with dst.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["user_id", "n_skills", "skill_text"])
        writer.writeheader()
        for uid in sorted(skills_by_user):
            vals = sorted(skills_by_user[uid])
            writer.writerow({"user_id": uid, "n_skills": len(vals), "skill_text": " ; ".join(vals)})
    return len(skills_by_user)


def build_first_job(src_positions: Path, src_edu: Path, dst: Path, keep_ids: set[int]) -> int:
    latest_edu_end: dict[int, datetime.date | None] = {}
    with src_edu.open("r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            uid = parse_user_id(row.get("user_id", ""))
            if uid is None or uid not in keep_ids:
                continue
            end = parse_date(row.get("enddate", ""))
            prev = latest_edu_end.get(uid)
            if prev is None or (end is not None and end > prev):
                latest_edu_end[uid] = end

    best_rows: dict[int, dict[str, str]] = {}
    best_dates: dict[int, datetime.date] = {}
    with src_positions.open("r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            uid = parse_user_id(row.get("user_id", ""))
            if uid is None or uid not in keep_ids:
                continue
            role = (row.get("role_k17000_v3") or "").lower()
            if any(tok in role for tok in ["intern", "internship", "co-op", "coop", "trainee", "apprentice", "fellow"]):
                continue
            start = parse_date(row.get("startdate", ""))
            if start is None:
                continue
            edu_end = latest_edu_end.get(uid)
            if edu_end is not None and start < edu_end:
                continue
            if uid not in best_dates or start < best_dates[uid]:
                best_dates[uid] = start
                best_rows[uid] = row

    fieldnames = [
        "user_id",
        "job_start_date",
        "job_end_date",
        "role_k17000_v3",
        "seniority",
        "rcid",
        "ultimate_parent_rcid",
        "company",
        "ultimate_parent_company_name",
    ]
    with dst.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for uid in sorted(best_rows):
            row = best_rows[uid]
            writer.writerow(
                {
                    "user_id": uid,
                    "job_start_date": row.get("startdate", ""),
                    "job_end_date": row.get("enddate", ""),
                    "role_k17000_v3": row.get("role_k17000_v3", ""),
                    "seniority": row.get("seniority", ""),
                    "rcid": row.get("rcid", ""),
                    "ultimate_parent_rcid": row.get("ultimate_parent_rcid", ""),
                    "company": row.get("company", ""),
                    "ultimate_parent_company_name": row.get("ultimate_parent_company_name", ""),
                }
            )
    return len(best_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ids = load_user_ids_txt(USER_IDS)
    edu_ids = load_csv_ids(EDU, "user_id")
    skill_ids = load_csv_ids(SKILLS, "user_id")
    pos_ids = load_csv_ids(POSITIONS, "user_id")
    profile_ids = load_csv_ids(PROFILES, "user_id")

    core_ids = ids & edu_ids & skill_ids & pos_ids & profile_ids

    (OUT_DIR / "core_user_ids.txt").write_text("\n".join(str(x) for x in sorted(core_ids)) + "\n", encoding="utf-8")

    counts = {
        "ready_revelio_edu_18_22_rows": stream_filter_csv(EDU, OUT_DIR / "ready_revelio_edu_18_22.csv", "user_id", core_ids),
        "ready_user_skills_rows": stream_filter_csv(SKILLS, OUT_DIR / "ready_user_skills.csv", "user_id", core_ids),
        "ready_user_positions_grouped_rows": stream_filter_csv(POSITIONS, OUT_DIR / "ready_user_positions_grouped.csv", "user_id", core_ids),
        "ready_user_profiles_rows": stream_filter_csv(PROFILES, OUT_DIR / "ready_user_profiles.csv", "user_id", core_ids),
    }
    counts["ready_user_skill_agg_users"] = build_skill_agg(SKILLS, OUT_DIR / "ready_user_skill_agg.csv", core_ids)
    counts["ready_first_job_users"] = build_first_job(POSITIONS, EDU, OUT_DIR / "ready_first_job.csv", core_ids)

    summary = {
        "core_overlap_users": len(core_ids),
        "input_user_sets": {
            "user_ids": len(ids),
            "edu18_22": len(edu_ids),
            "user_skills": len(skill_ids),
            "positions": len(pos_ids),
            "profiles": len(profile_ids),
        },
        "output_counts": counts,
        "output_dir": str(OUT_DIR),
    }
    (OUT_DIR / "ready_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Ready Analysis Files",
        "",
        f"- Core overlapping users across `user_ids`, `Revelio_EDU_18-22`, `user_skills`, `User_positions_grouped`, and `user_profiles`: {len(core_ids):,}",
        "",
        "## Files",
        f"- `core_user_ids.txt`: {len(core_ids):,} users",
        f"- `ready_revelio_edu_18_22.csv`: {counts['ready_revelio_edu_18_22_rows']:,} rows",
        f"- `ready_user_skills.csv`: {counts['ready_user_skills_rows']:,} rows",
        f"- `ready_user_positions_grouped.csv`: {counts['ready_user_positions_grouped_rows']:,} rows",
        f"- `ready_user_profiles.csv`: {counts['ready_user_profiles_rows']:,} rows",
        f"- `ready_user_skill_agg.csv`: {counts['ready_user_skill_agg_users']:,} users",
        f"- `ready_first_job.csv`: {counts['ready_first_job_users']:,} users",
        "",
        "These files are filtered to the overlapping core user universe and are intended as analysis-ready inputs.",
    ]
    (OUT_DIR / "ready_analysis_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
