from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties


BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
OUT = BASE / "outputs"
FIG = OUT / "figures"
TABLES = OUT / "tables"

FIG.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 4710

COLORS = {
    "bg": "#fbfaf7",
    "ink": "#253044",
    "muted": "#6b7280",
    "popular": "#159a8c",
    "nonpopular": "#e76f51",
    "blue": "#457b9d",
    "gold": "#e9c46a",
    "purple": "#7768d8",
}

sns.set_theme(style="whitegrid", context="talk")


def clean_skill(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = " ".join(s.split())
    bad = {"", "nan", "none", "unknown", ".", "null"}
    return "" if s in bad else s


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()


def skill_card_cloud(freq, path, title, subtitle="", max_words=70, color=COLORS["popular"]):
    """Organic word-cloud-style layout with collision checks and no overlapping labels."""
    freq = freq.dropna()
    freq = freq[freq > 0].head(max_words)
    if freq.empty:
        return

    vals = freq.values.astype(float)
    vals = vals / vals.max()
    palette = [
        color,
        COLORS["blue"],
        COLORS["purple"],
        COLORS["gold"],
        COLORS["ink"],
        COLORS["nonpopular"],
    ]
    rng = np.random.default_rng(RANDOM_STATE)

    fig, ax = plt.subplots(figsize=(14.5, 8.8))
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle(title, fontsize=22, weight="bold", color=COLORS["ink"], y=0.98)
    if subtitle:
        ax.set_title(subtitle, fontsize=12, color=COLORS["muted"], pad=8)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []

    for i, ((skill, _count), scale) in enumerate(zip(freq.items(), vals)):
        txt = str(skill)
        if len(txt) > 30:
            txt = txt[:28] + "..."
        fs = 9 + 22 * scale
        placed = False

        for attempt in range(450):
            if attempt < 180:
                x = np.clip(rng.normal(0.5, 0.19), 0.06, 0.94)
                y = np.clip(rng.normal(0.50, 0.22), 0.08, 0.88)
            else:
                x = rng.uniform(0.06, 0.94)
                y = rng.uniform(0.08, 0.88)

            text = ax.text(
                x,
                y,
                txt,
                ha="center",
                va="center",
                fontsize=fs,
                fontproperties=FontProperties(weight="bold" if scale > 0.68 else "normal"),
                color=palette[i % len(palette)],
                alpha=0.94,
                transform=ax.transAxes,
            )
            fig.canvas.draw()
            bbox = text.get_window_extent(renderer=renderer).expanded(1.10, 1.20)

            if bbox.x0 < 0 or bbox.y0 < 0 or bbox.x1 > fig.bbox.width or bbox.y1 > fig.bbox.height:
                text.remove()
                continue
            if all(not bbox.overlaps(old) for old in boxes):
                boxes.append(bbox)
                placed = True
                break
            text.remove()

        if not placed and fs > 11:
            fs = 10

    savefig(path)


def build_skill_labels():
    features_path = TABLES / "features_excel_popular_sample.csv"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Expected {features_path}. Run the main analysis first, or update this script to point to your feature file."
        )

    features = pd.read_csv(features_path, usecols=["user_id", "popular_company", "entry_job_year"])
    feature_ids = set(features["user_id"])

    skill_parts = []
    skill_cols = ["user_id", "skill_raw", "skill_translated", "skill_k35000"]
    for chunk in pd.read_csv(RAW / "user_skills.csv", usecols=skill_cols, chunksize=500_000):
        chunk = chunk[chunk["user_id"].isin(feature_ids)].copy()
        if chunk.empty:
            continue
        skill = chunk["skill_k35000"].fillna(chunk["skill_translated"]).fillna(chunk["skill_raw"])
        chunk["skill"] = skill.map(clean_skill)
        chunk = chunk[chunk["skill"].ne("")]
        skill_parts.append(chunk[["user_id", "skill"]])

    if not skill_parts:
        return pd.DataFrame(columns=["user_id", "skill", "popular_company", "entry_job_year"])

    skill_df = pd.concat(skill_parts, ignore_index=True).drop_duplicates()
    return skill_df.merge(features, on="user_id", how="inner")


def main():
    skill_labeled = build_skill_labels()

    pop_freq = skill_labeled[skill_labeled["popular_company"] == 1]["skill"].value_counts()
    non_freq = skill_labeled[skill_labeled["popular_company"] == 0]["skill"].value_counts()

    pop_freq.head(200).rename_axis("skill").reset_index(name="users").to_csv(
        TABLES / "skills_popular_frequency.csv", index=False
    )
    non_freq.head(200).rename_axis("skill").reset_index(name="users").to_csv(
        TABLES / "skills_nonpopular_frequency.csv", index=False
    )

    skill_card_cloud(
        pop_freq,
        FIG / "skill_cloud_excel_popular.png",
        "Skills Among Popular First Jobs",
        "Companies in data.xlsx are popular; all others are unpopular",
        color=COLORS["popular"],
    )
    skill_card_cloud(
        non_freq,
        FIG / "skill_cloud_nonpopular.png",
        "Skills Among Non-Popular First Jobs",
        "Phrase-level mapped skills; no overlapping labels",
        color=COLORS["nonpopular"],
    )

    for year in [2024, 2025]:
        ydf = skill_labeled[skill_labeled["entry_job_year"] == year]
        skill_card_cloud(
            ydf[ydf["popular_company"] == 1]["skill"].value_counts(),
            FIG / f"skill_cloud_excel_popular_entry_{year}.png",
            f"Popular-Company Skills, Entry Job {year}",
            "Split by entry job year",
            color=COLORS["popular"],
        )
        skill_card_cloud(
            ydf[ydf["popular_company"] == 0]["skill"].value_counts(),
            FIG / f"skill_cloud_nonpopular_entry_{year}.png",
            f"Non-Popular Skills, Entry Job {year}",
            "Split by entry job year",
            color=COLORS["nonpopular"],
        )


if __name__ == "__main__":
    main()
