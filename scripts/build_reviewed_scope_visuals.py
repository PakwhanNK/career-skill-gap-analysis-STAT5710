from __future__ import annotations

import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd


BASE = Path(r"C:\Users\knich\Projects\career-skill-gap-analysis-STAT5710")
COHORT = BASE / "outputs" / "tables" / "reconstructed_first_job_scope_sample.csv"
OUT_TABLES = BASE / "outputs" / "tables"
OUT_FIG = BASE / "outputs" / "figures"

COLORS = {
    "target": "#0f766e",
    "non_target": "#c05621",
    "blue": "#2b6cb0",
    "gold": "#d69e2e",
    "purple": "#6b46c1",
    "slate": "#475569",
    "green": "#2f855a",
    "rose": "#be185d",
    "amber": "#b45309",
    "gray": "#94a3b8",
    "bg": "#faf7f2",
    "ink": "#1f2937",
    "grid": "#e5ded2",
}


def clean_text(series: pd.Series, missing: str = "Missing") -> pd.Series:
    return series.fillna(missing).astype(str).str.strip().replace({"": missing, "nan": missing})


def normalize_decision(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace({"nan": "", "1.0": "1", "0.0": "0"})


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def render_vertical_bar_chart(title: str, categories: list[str], values: list[int], colors: list[str], out: Path) -> None:
    width, height = 860, 560
    left, right, top, bottom = 80, 40, 80, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_val = max(values) if values else 1
    bar_w = plot_w / max(1, len(values)) * 0.55
    gap = plot_w / max(1, len(values))

    parts = [
        svg_header(width, height),
        f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>',
        f'<text x="{width/2}" y="38" text-anchor="middle" font-size="24" font-weight="700" fill="{COLORS["ink"]}">{escape(title)}</text>',
    ]

    for i in range(6):
        y = top + plot_h * i / 5
        val = max_val * (1 - i / 5)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="{COLORS["ink"]}">{int(val):,}</text>')

    for idx, (cat, val, color) in enumerate(zip(categories, values, colors)):
        x = left + gap * idx + (gap - bar_w) / 2
        h = 0 if max_val == 0 else plot_h * val / max_val
        y = top + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="6" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-size="14" fill="{COLORS["ink"]}">{val:,}</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{height-40}" text-anchor="middle" font-size="14" fill="{COLORS["ink"]}">{escape(cat)}</text>')

    parts.append("</svg>")
    write_text(out, "\n".join(parts))


def render_horizontal_bar_chart(title: str, labels: list[str], values: list[int], color: str, out: Path) -> None:
    width = 1100
    row_h = 34
    top = 90
    left = 350
    right = 70
    bottom = 40
    height = top + bottom + row_h * len(labels)
    plot_w = width - left - right
    max_val = max(values) if values else 1

    parts = [
        svg_header(width, height),
        f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>',
        f'<text x="{width/2}" y="38" text-anchor="middle" font-size="24" font-weight="700" fill="{COLORS["ink"]}">{escape(title)}</text>',
    ]

    for idx, (label, val) in enumerate(zip(labels, values)):
        y = top + idx * row_h
        bar_w = 0 if max_val == 0 else plot_w * val / max_val
        parts.append(f'<text x="{left-12}" y="{y+20}" text-anchor="end" font-size="13" fill="{COLORS["ink"]}">{escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y+6}" width="{bar_w:.1f}" height="20" rx="5" fill="{color}"/>')
        parts.append(f'<text x="{left + bar_w + 8:.1f}" y="{y+21}" font-size="13" fill="{COLORS["ink"]}">{val:,}</text>')

    parts.append("</svg>")
    write_text(out, "\n".join(parts))


def polar_to_cartesian(cx: float, cy: float, r: float, angle_deg: float) -> tuple[float, float]:
    angle_rad = math.radians(angle_deg - 90)
    return cx + r * math.cos(angle_rad), cy + r * math.sin(angle_rad)


def describe_arc(cx: float, cy: float, r: float, start_angle: float, end_angle: float) -> str:
    start = polar_to_cartesian(cx, cy, r, end_angle)
    end = polar_to_cartesian(cx, cy, r, start_angle)
    large_arc = 1 if end_angle - start_angle > 180 else 0
    return f"M {cx} {cy} L {start[0]:.2f} {start[1]:.2f} A {r} {r} 0 {large_arc} 0 {end[0]:.2f} {end[1]:.2f} Z"


def render_pie_chart(title: str, labels: list[str], values: list[int], out: Path) -> None:
    width, height = 1050, 620
    cx, cy, r = 280, 330, 180
    total = sum(values) or 1
    palette = [
        COLORS["target"],
        COLORS["blue"],
        COLORS["gold"],
        COLORS["purple"],
        COLORS["rose"],
        COLORS["green"],
        COLORS["amber"],
        COLORS["slate"],
        COLORS["gray"],
    ]

    parts = [
        svg_header(width, height),
        f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>',
        f'<text x="{width/2}" y="38" text-anchor="middle" font-size="24" font-weight="700" fill="{COLORS["ink"]}">{escape(title)}</text>',
    ]

    angle = 0.0
    legend_x = 560
    legend_y = 120
    for idx, (label, val) in enumerate(zip(labels, values)):
        sweep = val / total * 360
        color = palette[idx % len(palette)]
        parts.append(f'<path d="{describe_arc(cx, cy, r, angle, angle + sweep)}" fill="{color}" stroke="{COLORS["bg"]}" stroke-width="2"/>')
        mid = angle + sweep / 2
        tx, ty = polar_to_cartesian(cx, cy, r * 0.68, mid)
        if sweep >= 12:
            parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" font-size="13" fill="white">{val / total * 100:.1f}%</text>')
        ly = legend_y + idx * 36
        parts.append(f'<rect x="{legend_x}" y="{ly-12}" width="18" height="18" rx="3" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x + 28}" y="{ly+2}" font-size="14" fill="{COLORS["ink"]}">{escape(label)} ({val:,})</text>'
        )
        angle += sweep

    parts.append("</svg>")
    write_text(out, "\n".join(parts))


def main() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(COHORT)
    df["manual_decision"] = normalize_decision(df["manual_decision"])
    df["major"] = clean_text(df["major"])
    df["school"] = clean_text(df["school"])
    df["employer_name"] = clean_text(df["employer_name"])

    unreviewed_mask = df["manual_decision"].isin(["", "nan"]) | df["manual_decision"].isna()
    reviewed = df.loc[~unreviewed_mask].copy()
    target = reviewed.loc[reviewed["manual_decision"] == "1"].copy()

    unreviewed = df.loc[unreviewed_mask].copy()
    non_missing_unreviewed = unreviewed.loc[unreviewed["employer_name"].str.lower() != "missing"].copy()
    unreviewed_counts = non_missing_unreviewed["employer_name"].value_counts()

    count_df = pd.DataFrame(
        {
            "group": ["Target", "Non-target"],
            "users": [
                int((reviewed["manual_decision"] == "1").sum()),
                int((reviewed["manual_decision"] == "0").sum()),
            ],
        }
    )
    render_vertical_bar_chart(
        "Reviewed Cohort: Target vs Non-Target",
        count_df["group"].tolist(),
        count_df["users"].tolist(),
        [COLORS["target"], COLORS["non_target"]],
        OUT_FIG / "reviewed_target_vs_nontarget_counts.svg",
    )

    school_df = target["school"].value_counts().head(12).rename_axis("school").reset_index(name="users")
    school_df.to_csv(OUT_TABLES / "reviewed_target_top_universities.csv", index=False)
    render_horizontal_bar_chart(
        "Top Universities Among Target Entrants",
        school_df["school"].tolist(),
        school_df["users"].tolist(),
        COLORS["blue"],
        OUT_FIG / "reviewed_target_top_universities.svg",
    )

    major_counts = target["major"].value_counts()
    top_majors = major_counts.head(8)
    other_count = int(major_counts.iloc[8:].sum())
    pie_df = top_majors.rename_axis("major").reset_index(name="users")
    if other_count > 0:
        pie_df.loc[len(pie_df)] = {"major": "Other", "users": other_count}
    pie_df["share"] = pie_df["users"] / pie_df["users"].sum()
    pie_df.to_csv(OUT_TABLES / "reviewed_target_major_pie_source.csv", index=False)
    render_pie_chart(
        "Major Mix Among Target Entrants",
        pie_df["major"].tolist(),
        pie_df["users"].tolist(),
        OUT_FIG / "reviewed_target_major_pie.svg",
    )

    summary = {
        "reviewed_rows": int(len(reviewed)),
        "target_rows": int((reviewed["manual_decision"] == "1").sum()),
        "non_target_rows": int((reviewed["manual_decision"] == "0").sum()),
        "excluded_unreviewed_rows": int(len(unreviewed)),
        "distinct_non_missing_unreviewed_employers": int(non_missing_unreviewed["employer_name"].nunique()),
        "remaining_unreviewed_employers_ge_34_users": int((unreviewed_counts >= 34).sum()),
        "largest_non_missing_unreviewed_employer_count": int(unreviewed_counts.max()) if len(unreviewed_counts) else 0,
    }
    write_text(OUT_TABLES / "reviewed_scope_visual_summary.json", json.dumps(summary, indent=2))

    caveat_lines = [
        "# Reviewed-Only Visualization Caveats",
        "",
        "## Scope",
        f"- The charts use only the reviewed cohort: {summary['reviewed_rows']:,} rows.",
        f"- This includes {summary['target_rows']:,} target rows and {summary['non_target_rows']:,} reviewed non-target rows.",
        f"- It excludes {summary['excluded_unreviewed_rows']:,} unreviewed rows.",
        "",
        "## Caveats",
        "- The reviewed-only cohort is selection-biased toward employers we explicitly labeled.",
        "- The excluded unreviewed rows are mostly less common employers in the long tail.",
        f"- After dropping missing-employer rows, only {summary['remaining_unreviewed_employers_ge_34_users']} remaining unreviewed employers have at least 34 users; the largest non-missing unreviewed employer has {summary['largest_non_missing_unreviewed_employer_count']} users.",
        "- Because of that, the reviewed-only charts emphasize common, high-visibility employers and underrepresent rarer career destinations.",
        "- The first-job cohort is reconstructed from `User_positions_grouped.csv`, not the original hidden parquet artifact from the prior pipeline.",
        "- The first-job definition here uses the earliest non-intern position on or after the latest education end date in `Revelio_EDU_18-22.csv` when available.",
        "- Major and school fields still contain some missingness, so composition charts should be interpreted as approximate rather than definitive.",
        "- Employer labels reflect the current manual review taxonomy, which is a research decision rather than an objective ground truth.",
        "",
        "## Output Files",
        "- Figures: `reviewed_target_vs_nontarget_counts.svg`, `reviewed_target_top_universities.svg`, `reviewed_target_major_pie.svg`",
        "- Source tables: `reviewed_target_top_universities.csv`, `reviewed_target_major_pie_source.csv`",
    ]
    write_text(OUT_TABLES / "reviewed_scope_visual_caveats.md", "\n".join(caveat_lines) + "\n")
    print("Wrote reviewed-scope SVG figures and caveat notes.")


if __name__ == "__main__":
    main()
