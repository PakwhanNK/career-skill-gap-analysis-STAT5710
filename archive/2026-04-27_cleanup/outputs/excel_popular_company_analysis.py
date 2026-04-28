from pathlib import Path
import json
import pickle
import textwrap
import warnings
import zipfile
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler


BASE = Path("/Users/chekaylimeyer/Desktop/Current Courses/STAT 4710 Data Mining/Final Project")
RAW = BASE / "data"
PRIOR = BASE / "top_company_prediction" / "data"
OUT = BASE / "outputs"
FIG = OUT / "figures"
TABLES = OUT / "tables"
MODELS = OUT / "models"
for d in [OUT, FIG, TABLES, MODELS]:
    d.mkdir(exist_ok=True)

for old_file in [
    TABLES / "fairness_audit_by_demographics.csv",
    TABLES / "eda_popular_rate_by_ethnicity_predicted.csv",
    TABLES / "eda_popular_rate_by_sex_predicted.csv",
    FIG / "eda_popular_rate_by_ethnicity_predicted.png",
    FIG / "eda_popular_rate_by_sex_predicted.png",
]:
    old_file.unlink(missing_ok=True)

RANDOM_STATE = 4710
MIN_SKILL_USERS = 10
N_IMPORTANT_SKILLS = 250

COLORS = {
    "bg": "#fbfaf7",
    "ink": "#253044",
    "muted": "#6b7280",
    "popular": "#159a8c",
    "nonpopular": "#e76f51",
    "blue": "#457b9d",
    "gold": "#e9c46a",
    "purple": "#7768d8",
    "gray": "#e5e7eb",
}

sns.set_theme(style="whitegrid", context="talk")


def read_first_xlsx_sheet(path: Path) -> pd.DataFrame:
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("main:si", ns):
                shared.append("".join(t.text or "" for t in si.findall(".//main:t", ns)))
        sheet = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet"))[0]
        root = ET.fromstring(z.read(sheet))
        rows = []
        for row in root.findall(".//main:sheetData/main:row", ns):
            vals = []
            for cell in row.findall("main:c", ns):
                v = cell.find("main:v", ns)
                val = "" if v is None else v.text
                if cell.attrib.get("t") == "s" and val != "":
                    val = shared[int(val)]
                vals.append(val)
            rows.append(vals)
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    return pd.DataFrame(rows[1:], columns=rows[0])


def clean_skill(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = " ".join(s.split())
    bad = {"", "nan", "none", "unknown", ".", "null"}
    return "" if s in bad else s


def style_ax(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(COLORS["bg"])
    ax.grid(True, axis="y", color="#e7e5df", linewidth=0.9)
    ax.grid(False, axis="x")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#d6d3cc")
    ax.spines["bottom"].set_color("#d6d3cc")
    if title:
        ax.set_title(title, fontsize=16, weight="bold", color=COLORS["ink"], pad=14)
    if xlabel:
        ax.set_xlabel(xlabel, color=COLORS["ink"])
    if ylabel:
        ax.set_ylabel(ylabel, color=COLORS["ink"])


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()


def wrap_label(label, width=20):
    return "\n".join(textwrap.wrap(str(label), width=width, break_long_words=False)) or str(label)


def valid_label(x):
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s not in {"", "missing", "nan", "none", "unknown", ".", "null", "other"}


def skill_card_cloud(freq, path, title, subtitle="", max_words=70, color=COLORS["popular"]):
    """Organic word-cloud-style layout with collision checks, no cards, no overlap."""
    freq = freq.dropna()
    freq = freq[freq > 0].head(max_words)
    if freq.empty:
        return
    vals = freq.values.astype(float)
    vals = vals / vals.max()
    palette = [color, COLORS["blue"], COLORS["purple"], COLORS["gold"], COLORS["ink"], COLORS["nonpopular"]]
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

    for i, ((skill, count), scale) in enumerate(zip(freq.items(), vals)):
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


def safe_auc(y, p):
    return roc_auc_score(y, p) if pd.Series(y).nunique() == 2 else np.nan


def normalize_cat(s, min_count=20):
    x = s.fillna("Missing").astype(str).str.strip().replace("", "Missing")
    vc = x.value_counts()
    return x.where(x.isin(vc[vc >= min_count].index), "Other")


# Load reusable cohort/first-job artifacts from the previous clean extraction.
features_base = pd.read_parquet(PRIOR / "features.parquet")
first_job = pd.read_parquet(PRIOR / "first_job.parquet")

# Excel-defined popular companies.
popular = read_first_xlsx_sheet(RAW / "data.xlsx").rename(
    columns={
        "(rcid) Revelio Labs Company ID": "rcid",
        "(company) Company name": "company",
        "(rics_k50) Revelio Industry Classification K=50": "industry",
        "(hq_state) Headquarters state": "hq_state",
    }
)
popular["rcid_int"] = pd.to_numeric(popular["rcid"], errors="coerce").astype("Int64")
popular_rcids = set(popular["rcid_int"].dropna().astype(int))
popular.to_csv(TABLES / "excel_popular_company_list.csv", index=False)

fj = first_job.copy()
fj["entry_job_year"] = pd.to_datetime(fj["job_start_date"], errors="coerce").dt.year
fj["rcid_int"] = pd.to_numeric(fj["rcid"], errors="coerce").astype("Int64")
fj["ultimate_parent_rcid_int"] = pd.to_numeric(fj["ultimate_parent_rcid"], errors="coerce").astype("Int64")
fj["popular_company_excel"] = (
    fj["rcid_int"].astype("float").isin(popular_rcids) | fj["ultimate_parent_rcid_int"].astype("float").isin(popular_rcids)
).astype(int)

label_cols = ["user_id", "entry_job_year", "company", "ultimate_parent_company_name", "rcid", "ultimate_parent_rcid", "popular_company_excel"]
features = features_base.drop(columns=["top_company"], errors="ignore").merge(fj[label_cols], on="user_id", how="inner")

# Phrase-level skill aggregation.
feature_ids = set(features["user_id"])
skill_parts = []
for chunk in pd.read_csv(RAW / "user_skills.csv", usecols=["user_id", "skill_raw", "skill_translated", "skill_k35000"], chunksize=500_000):
    chunk = chunk[chunk["user_id"].isin(feature_ids)].copy()
    if chunk.empty:
        continue
    skill = chunk["skill_k35000"].fillna(chunk["skill_translated"]).fillna(chunk["skill_raw"])
    chunk["skill"] = skill.map(clean_skill)
    chunk = chunk[chunk["skill"].ne("")]
    skill_parts.append(chunk[["user_id", "skill"]])

skill_df = pd.concat(skill_parts, ignore_index=True).drop_duplicates() if skill_parts else pd.DataFrame(columns=["user_id", "skill"])
skill_agg = skill_df.groupby("user_id").agg(
    skill_text=("skill", lambda s: " ; ".join(sorted(set(s)))),
    n_skills=("skill", "nunique"),
    skill_list=("skill", lambda s: sorted(set(s))),
).reset_index()
features = features.drop(columns=["skill_text", "n_skills"], errors="ignore").merge(skill_agg, on="user_id", how="inner")
features["popular_company"] = features["popular_company_excel"].astype(int)

for col in ["school", "major", "university_country", "user_country"]:
    if col in features.columns:
        features[col] = normalize_cat(features[col], min_count=20)

features.to_parquet(OUT / "features_excel_popular.parquet", index=False)
features.drop(columns=["skill_list"], errors="ignore").to_csv(TABLES / "features_excel_popular_sample.csv", index=False)

skill_labeled = skill_df.merge(features[["user_id", "popular_company", "entry_job_year"]], on="user_id", how="inner")

# Skill importance by smoothed log-odds.
skill_stats = skill_labeled.groupby("skill").agg(
    users=("user_id", "nunique"),
    popular_users=("popular_company", "sum"),
).reset_index()
total_pop = features["popular_company"].sum()
total_non = len(features) - total_pop
skill_stats["nonpopular_users"] = skill_stats["users"] - skill_stats["popular_users"]
skill_stats = skill_stats[skill_stats["users"] >= MIN_SKILL_USERS].copy()
skill_stats["popular_rate"] = skill_stats["popular_users"] / skill_stats["users"]
skill_stats["log_odds_popular"] = np.log((skill_stats["popular_users"] + 0.5) / (total_pop + 1)) - np.log(
    (skill_stats["nonpopular_users"] + 0.5) / (total_non + 1)
)
skill_stats["abs_log_odds"] = skill_stats["log_odds_popular"].abs()
skill_stats.sort_values("abs_log_odds", ascending=False).to_csv(TABLES / "skill_importance_log_odds.csv", index=False)
important_skills = skill_stats.sort_values("abs_log_odds", ascending=False).head(N_IMPORTANT_SKILLS)["skill"].tolist()

# EDA tables and graphs.
summary_by_year = features.groupby("entry_job_year").agg(
    users=("user_id", "nunique"),
    popular_rate=("popular_company", "mean"),
).reset_index()
summary_by_year.to_csv(TABLES / "eda_popular_rate_by_entry_job_year.csv", index=False)

fig, ax = plt.subplots(figsize=(9.5, 5.8))
sns.lineplot(data=summary_by_year, x="entry_job_year", y="popular_rate", marker="o", linewidth=2.7, color=COLORS["popular"], ax=ax)
for _, r in summary_by_year.iterrows():
    ax.text(r["entry_job_year"], r["popular_rate"] + 0.01, f"n={int(r['users'])}", ha="center", fontsize=9, color=COLORS["muted"])
style_ax(ax, "Popular Company Rate by Entry Job Year", "Entry job year", "Share at popular company")
savefig(FIG / "eda_popular_rate_by_entry_job_year.png")

for col, title in [("school", "School"), ("major", "Career Field / Major")]:
    tbl = features.groupby(col).agg(users=("user_id", "nunique"), popular_rate=("popular_company", "mean")).reset_index()
    tbl = tbl[tbl[col].map(valid_label)]
    tbl = tbl[tbl["users"] >= 30].sort_values("popular_rate", ascending=False).head(18)
    tbl.to_csv(TABLES / f"eda_popular_rate_by_{col}.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, max(5.5, len(tbl) * 0.38)))
    plot = tbl.sort_values("popular_rate")
    sns.barplot(data=plot, x="popular_rate", y=col, color=COLORS["blue"] if col in ["school", "major"] else COLORS["purple"], ax=ax)
    style_ax(ax, f"Popular Company Rate by {title}", "Popular-company rate", "")
    savefig(FIG / f"eda_popular_rate_by_{col}.png")

top_employers = (
    fj.groupby(["ultimate_parent_company_name", "popular_company_excel"])
    .size()
    .reset_index(name="first_jobs")
    .sort_values("first_jobs", ascending=False)
    .head(25)
)
top_employers.to_csv(TABLES / "eda_top_first_job_employers_with_excel_label.csv", index=False)
fig, ax = plt.subplots(figsize=(11, 8))
plot = top_employers.sort_values("first_jobs")
bar_colors = [COLORS["popular"] if x == 1 else COLORS["nonpopular"] for x in plot["popular_company_excel"]]
ax.barh(plot["ultimate_parent_company_name"].map(lambda x: wrap_label(x, 34)), plot["first_jobs"], color=bar_colors)
style_ax(ax, "Top First-Job Employers, Colored by Popular Label", "First-job count", "")
savefig(FIG / "eda_top_employers_excel_label.png")

# Skill card clouds by label and 2024/2025 entry-job year.
pop_freq = skill_labeled[skill_labeled["popular_company"] == 1]["skill"].value_counts()
non_freq = skill_labeled[skill_labeled["popular_company"] == 0]["skill"].value_counts()
pop_freq.head(200).rename_axis("skill").reset_index(name="users").to_csv(TABLES / "skills_popular_frequency.csv", index=False)
non_freq.head(200).rename_axis("skill").reset_index(name="users").to_csv(TABLES / "skills_nonpopular_frequency.csv", index=False)
skill_card_cloud(pop_freq, FIG / "skill_cloud_excel_popular.png", "Skills Among Popular First Jobs", "Companies in data.xlsx are popular; all others are unpopular", color=COLORS["popular"])
skill_card_cloud(non_freq, FIG / "skill_cloud_nonpopular.png", "Skills Among Non-Popular First Jobs", "Phrase-level mapped skills; no overlapping labels", color=COLORS["nonpopular"])
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

# PCA by important skills.
skill_sets = features["skill_list"].apply(lambda xs: [s for s in xs if s in important_skills])
mlb = MultiLabelBinarizer(classes=important_skills)
X_skill = mlb.fit_transform(skill_sets)
scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X_skill)
pca = PCA(n_components=20, random_state=RANDOM_STATE)
pcs = pca.fit_transform(X_scaled)
features["pca_skill_1"] = pcs[:, 0]
features["pca_skill_2"] = pcs[:, 1]

loadings = pd.DataFrame(
    {
        "skill": important_skills,
        "pc1_loading": pca.components_[0],
        "pc2_loading": pca.components_[1],
        "pc1_abs": np.abs(pca.components_[0]),
        "pc2_abs": np.abs(pca.components_[1]),
    }
).sort_values("pc1_abs", ascending=False)
loadings.to_csv(TABLES / "pca_important_skill_loadings.csv", index=False)

fig, ax = plt.subplots(figsize=(10.5, 7.2))
sample = features.sample(n=min(6500, len(features)), random_state=RANDOM_STATE)
palette = {0: COLORS["nonpopular"], 1: COLORS["popular"]}
for val, label in [(0, "Non-popular"), (1, "Popular")]:
    g = sample[sample["popular_company"] == val]
    ax.scatter(g["pca_skill_1"], g["pca_skill_2"], s=17, alpha=0.42, c=palette[val], label=label, edgecolors="none")
centers = features.groupby("popular_company")[["pca_skill_1", "pca_skill_2"]].mean()
for val, label in [(0, "NON-POPULAR CENTER"), (1, "POPULAR CENTER")]:
    ax.scatter(centers.loc[val, "pca_skill_1"], centers.loc[val, "pca_skill_2"], s=220, c=palette[val], edgecolor="white", linewidth=2.2)
    dx = 0.4 if val == 1 else 0.25
    dy = -0.6 if val == 1 else 0.35
    ax.text(centers.loc[val, "pca_skill_1"] + dx, centers.loc[val, "pca_skill_2"] + dy, label, weight="bold", color=COLORS["ink"], fontsize=10)
style_ax(
    ax,
    "PCA of Important Skills: Popular vs Non-Popular",
    f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
    f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
)
ax.legend(frameon=False, loc="best")
savefig(FIG / "pca_important_skills_popular_vs_nonpopular.png")

# Clustering on important-skill PCA components.
k = 6
clusterer = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=30)
features["skill_cluster"] = clusterer.fit_predict(pcs[:, :10])
cluster_rows = []
for cluster, g in features.groupby("skill_cluster"):
    users = set(g["user_id"])
    subskills = skill_labeled[skill_labeled["user_id"].isin(users)]
    top_skills = ", ".join(subskills["skill"].value_counts().head(8).index.tolist())
    top_majors = ", ".join([x for x in g["major"].value_counts().index.astype(str).tolist() if valid_label(x)][:4])
    top_schools = ", ".join([x for x in g["school"].value_counts().index.astype(str).tolist() if valid_label(x)][:4])
    cluster_rows.append(
        {
            "skill_cluster": cluster,
            "users": len(g),
            "popular_rate": g["popular_company"].mean(),
            "avg_n_skills": g["n_skills"].mean(),
            "top_skills": top_skills,
            "top_majors": top_majors,
            "top_schools": top_schools,
        }
    )
cluster_summary = pd.DataFrame(cluster_rows).sort_values("popular_rate", ascending=False)
cluster_summary.to_csv(TABLES / "cluster_profiles.csv", index=False)

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.ravel()
rate_min = cluster_summary["popular_rate"].min()
rate_max = cluster_summary["popular_rate"].max()
for ax, (_, row) in zip(axes, cluster_summary.iterrows()):
    ax.set_facecolor("white")
    ax.axis("off")
    intensity = 0 if rate_max == rate_min else (row["popular_rate"] - rate_min) / (rate_max - rate_min)
    header_color = COLORS["popular"] if intensity > 0.5 else COLORS["blue"]
    ax.text(0.02, 0.92, f"Cluster {int(row['skill_cluster'])}", fontsize=17, weight="bold", color=header_color, transform=ax.transAxes)
    ax.text(0.02, 0.80, f"Users: {int(row['users']):,}   Popular rate: {row['popular_rate']:.1%}   Avg skills: {row['avg_n_skills']:.1f}", fontsize=10.5, color=COLORS["ink"], transform=ax.transAxes)
    ax.text(0.02, 0.64, "Top skills", fontsize=11.5, weight="bold", color=COLORS["ink"], transform=ax.transAxes)
    ax.text(0.02, 0.44, wrap_label(row["top_skills"], 72), fontsize=9.2, color=COLORS["muted"], transform=ax.transAxes)
    ax.text(0.02, 0.25, "Top majors", fontsize=11.5, weight="bold", color=COLORS["ink"], transform=ax.transAxes)
    ax.text(0.02, 0.12, wrap_label(row["top_majors"], 72), fontsize=9.2, color=COLORS["muted"], transform=ax.transAxes)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor="#e2e8f0", linewidth=1.2))
fig.suptitle("Skill Cluster Profiles", fontsize=20, weight="bold", color=COLORS["ink"], y=0.995)
savefig(FIG / "cluster_profile_cards.png")

fig, ax = plt.subplots(figsize=(10.5, 7))
plot = features.sample(n=min(7000, len(features)), random_state=RANDOM_STATE)
sns.scatterplot(
    data=plot,
    x="pca_skill_1",
    y="pca_skill_2",
    hue="skill_cluster",
    style="popular_company",
    palette="tab10",
    s=34,
    alpha=0.65,
    ax=ax,
)
style_ax(ax, "Skill Clusters in PCA Space", "Important-skill PC1", "Important-skill PC2")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, title="Cluster / popular")
plt.setp(ax.get_legend().get_texts(), fontsize=9)
ax.get_legend().get_title().set_fontsize(10)
savefig(FIG / "clusters_in_important_skill_pca_space.png")

fig, ax = plt.subplots(figsize=(10, 5.8))
cs = cluster_summary.sort_values("popular_rate")
sns.barplot(data=cs, x="popular_rate", y=cs["skill_cluster"].astype(str), color=COLORS["popular"], ax=ax)
style_ax(ax, "Popular Company Rate by Skill Cluster", "Popular-company rate", "Skill cluster")
savefig(FIG / "cluster_popular_rates.png")

# Classification: predict popular first-job company using pre-job features only.
core_cat = ["school", "major", "university_country", "user_country", "skill_cluster"]
core_num = ["entry_job_year", "n_skills", "prestige", "numconnections"]
for col in core_cat:
    if col in features.columns:
        features[col] = normalize_cat(features[col], min_count=20)

model_cols = sorted(set(core_cat + core_num + ["skill_text"]))
model_df = features[["user_id", "popular_company"] + model_cols].copy()
y = model_df["popular_company"].astype(int)
train_idx, test_idx = train_test_split(model_df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
X_train, X_test = model_df.loc[train_idx, model_cols], model_df.loc[test_idx, model_cols]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]


def make_preprocessor():
    cats = core_cat
    nums = core_num
    return ColumnTransformer(
        [
            (
                "skills",
                Pipeline(
                    [
                        ("tfidf", TfidfVectorizer(max_features=1200, min_df=5, ngram_range=(1, 2), token_pattern=r"(?u)\b[\w+#.-]+\b")),
                        ("svd", TruncatedSVD(n_components=60, random_state=RANDOM_STATE)),
                        ("scale", StandardScaler()),
                    ]
                ),
                "skill_text",
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cats),
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), nums),
        ]
    )


models = {
    "logistic_regression": LogisticRegression(max_iter=5000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=450, min_samples_leaf=6, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
    "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.045, l2_regularization=0.1, random_state=RANDOM_STATE),
}

results = []
fitted = {}
variant = "student_skills_only"
for name, estimator in models.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("model", clone(estimator))])
    pipe.fit(X_train, y_train)
    p = pipe.predict_proba(X_test)[:, 1]
    q = (p >= 0.5).astype(int)
    results.append(
        {
            "variant": variant,
            "model": name,
            "test_auc": safe_auc(y_test, p),
            "test_accuracy": accuracy_score(y_test, q),
            "precision": precision_score(y_test, q, zero_division=0),
            "recall": recall_score(y_test, q, zero_division=0),
            "n_train": len(y_train),
            "n_test": len(y_test),
        }
        )
    fitted[(variant, name)] = pipe

metrics = pd.DataFrame(results).sort_values(["test_auc", "test_accuracy"], ascending=False)
metrics.to_csv(TABLES / "classification_metrics.csv", index=False)
best_row = metrics.iloc[0]
best = fitted[(best_row["variant"], best_row["model"])]
p = best.predict_proba(X_test)[:, 1]
q = (p >= 0.5).astype(int)
pd.DataFrame(classification_report(y_test, q, output_dict=True, zero_division=0)).T.to_csv(TABLES / "classification_report_best_core.csv")
pd.DataFrame(confusion_matrix(y_test, q), index=["actual_nonpopular", "actual_popular"], columns=["pred_nonpopular", "pred_popular"]).to_csv(
    TABLES / "confusion_matrix_best_core.csv"
)

# Skill-only logistic regression: interpretable regression output for skills.
skill_reg = LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")
skill_reg.fit(X_skill[train_idx], y_train)
skill_reg_proba = skill_reg.predict_proba(X_skill[test_idx])[:, 1]
skill_reg_pred = (skill_reg_proba >= 0.5).astype(int)
skill_reg_metrics = pd.DataFrame(
    [
        {
            "model": "skill_only_logistic_regression",
            "test_auc": safe_auc(y_test, skill_reg_proba),
            "test_accuracy": accuracy_score(y_test, skill_reg_pred),
            "precision": precision_score(y_test, skill_reg_pred, zero_division=0),
            "recall": recall_score(y_test, skill_reg_pred, zero_division=0),
            "n_train": len(y_train),
            "n_test": len(y_test),
        }
    ]
)
skill_reg_metrics.to_csv(TABLES / "skill_regression_metrics.csv", index=False)
coef = pd.DataFrame(
    {
        "skill": important_skills,
        "coefficient": skill_reg.coef_[0],
    }
)
coef["odds_ratio"] = np.exp(coef["coefficient"])
coef["direction"] = np.where(coef["coefficient"] >= 0, "popular", "unpopular")
coef.sort_values("coefficient", ascending=False).to_csv(TABLES / "skill_logistic_regression_coefficients.csv", index=False)

coef_plot = pd.concat([coef.nlargest(8, "coefficient"), coef.nsmallest(8, "coefficient")]).sort_values("coefficient")
fig, ax = plt.subplots(figsize=(11, 8.5))
coef_colors = [COLORS["nonpopular"] if c < 0 else COLORS["popular"] for c in coef_plot["coefficient"]]
ax.barh(coef_plot["skill"].map(lambda x: wrap_label(x, 24)), coef_plot["coefficient"], color=coef_colors)
ax.axvline(0, color=COLORS["ink"], linewidth=1)
style_ax(ax, "Skill Logistic Regression Coefficients", "Coefficient: positive = popular-company association", "")
ax.tick_params(axis="y", labelsize=10)
ax.tick_params(axis="x", labelsize=11)
savefig(FIG / "skill_regression_coefficients.png")

fig, ax = plt.subplots(figsize=(8, 6))
for key, pipe in fitted.items():
    RocCurveDisplay.from_predictions(y_test, pipe.predict_proba(X_test)[:, 1], name=key[1], ax=ax)
style_ax(ax, "Classifier ROC: Popular Company Outcome", "False positive rate", "True positive rate")
savefig(FIG / "classification_roc_curves.png")

fig, ax = plt.subplots(figsize=(8.5, 5.5))
mm = metrics.sort_values("test_auc")
sns.barplot(data=mm, x="test_auc", y="model", color=COLORS["blue"], ax=ax)
ax.set_xlim(0.45, min(1, max(0.75, mm["test_auc"].max() + 0.05)))
style_ax(ax, "Classification Model Comparison", "Test AUC", "")
savefig(FIG / "classification_model_auc.png")

cm = confusion_matrix(y_test, q)
fig, ax = plt.subplots(figsize=(5.8, 5.2))
sns.heatmap(cm, annot=True, fmt=",", cmap=sns.light_palette(COLORS["popular"], as_cmap=True), cbar=False, ax=ax)
ax.set_xticklabels(["Pred non-popular", "Pred popular"], rotation=15, ha="right")
ax.set_yticklabels(["Actual non-popular", "Actual popular"], rotation=0)
style_ax(ax, "Best Core Classifier Confusion Matrix", "", "")
savefig(FIG / "classification_confusion_matrix.png")

perm = permutation_importance(best, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1)
perm_df = pd.DataFrame({"feature_block": model_cols, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values(
    "importance_mean", ascending=False
)
perm_df.to_csv(TABLES / "classification_feature_importance.csv", index=False)
core_blocks = set(core_cat + core_num + ["skill_text"])
fig, ax = plt.subplots(figsize=(9.5, 6))
pp = perm_df[perm_df["feature_block"].isin(core_blocks)].head(12).sort_values("importance_mean")
pp = pp[pp["importance_mean"] > 0]
sns.barplot(data=pp, x="importance_mean", y="feature_block", color=COLORS["blue"], ax=ax)
style_ax(ax, "Feature Importance: Popular Company Classifier", "Permutation AUC decrease", "")
savefig(FIG / "classification_feature_importance.png")

with open(MODELS / "best_excel_popular_classifier.pkl", "wb") as f:
    pickle.dump(best, f)

summary = {
    "target": "popular_company = first qualifying non-intern entry job at a company whose rcid or ultimate_parent_rcid appears in data.xlsx; all other companies are unpopular",
    "rows_modeled": int(len(features)),
    "excel_popular_rcids": int(len(popular_rcids)),
    "positive_users": int(features["popular_company"].sum()),
    "positive_rate": float(features["popular_company"].mean()),
    "important_skills_for_pca": int(len(important_skills)),
    "pca_variance_pc1": float(pca.explained_variance_ratio_[0]),
    "pca_variance_pc2": float(pca.explained_variance_ratio_[1]),
    "clusters": int(k),
    "best_core_model": str(best_row["model"]),
    "best_core_auc": float(best_row["test_auc"]),
    "best_core_accuracy": float(best_row["test_accuracy"]),
    "core_features": core_cat + core_num + ["skill_text"],
    "demographic_features_used": [],
    "skill_regression_auc": float(skill_reg_metrics.iloc[0]["test_auc"]),
    "skill_regression_accuracy": float(skill_reg_metrics.iloc[0]["test_accuracy"]),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))

report = f"""# Popular vs Unpopular Company Skill Analysis

## What Was Classified
The classifier predicts:

`popular_company = 1` if a student's first qualifying non-intern entry job is at a company whose `rcid` or `ultimate_parent_rcid` appears in `data/data.xlsx`.

`popular_company = 0` for every other company.

The Excel file is the source of truth for popular companies. Any first-job company not found in that file is classified as unpopular.

## Features Fed Into Prediction
The core classifier uses pre-outcome / student-side features:

- Education: `school`, `major`, `university_country`
- Profile context: `user_country`, `prestige`, `numconnections`
- Timing: `entry_job_year`
- Skills: phrase-level mapped skills from `user_skills.csv`, represented as TF-IDF/SVD components
- Cluster feature: `skill_cluster`, learned from important skills

Demographic variables are not used in prediction or regression outputs.

## PCA Method
PCA was done on skills in a deliberately interpretable way:

1. Each user was represented by a phrase-level skill set from `user_skills.csv`.
2. We calculated skill importance using smoothed log-odds: skills overrepresented among Excel-popular outcomes versus non-popular outcomes received high absolute scores.
3. We selected the top {len(important_skills)} important skills.
4. We built a user-by-skill binary matrix for those skills.
5. We standardized the matrix and ran PCA.
6. The first two PCs are plotted in `figures/pca_important_skills_popular_vs_nonpopular.png`.

This means the PCA plot is not using every noisy skill. It focuses on skills that actually help separate popular versus non-popular first-job outcomes.

## Clustering Method
KMeans clustering was run on the first 10 PCA components from the important-skill matrix. Each cluster is summarized with:

- number of users
- Excel-popular company rate
- average number of skills
- most common skills
- most common majors
- most common schools

See `tables/cluster_profiles.csv` and `figures/clusters_in_important_skill_pca_space.png`.

## Classification Results
- Users modeled: {summary['rows_modeled']:,}
- Popular-company RCIDs from Excel: {summary['excel_popular_rcids']:,}
- Positive users: {summary['positive_users']:,}
- Positive rate: {summary['positive_rate']:.1%}
- Best core model: `{summary['best_core_model']}`
- Test AUC: {summary['best_core_auc']:.3f}
- Test accuracy: {summary['best_core_accuracy']:.3f}

## Skill Regression Output
I also fit a skill-only logistic regression using the important-skill binary matrix. Its coefficients show which skills are positively or negatively associated with popular-company placement.

- Skill regression AUC: {summary['skill_regression_auc']:.3f}
- Skill regression accuracy: {summary['skill_regression_accuracy']:.3f}
- Coefficients table: `tables/skill_logistic_regression_coefficients.csv`
- Coefficient figure: `figures/skill_regression_coefficients.png`

## Main Figures
- `figures/eda_popular_rate_by_entry_job_year.png`
- `figures/eda_top_employers_excel_label.png`
- `figures/skill_cloud_excel_popular.png`
- `figures/skill_cloud_nonpopular.png`
- `figures/skill_cloud_excel_popular_entry_2024.png`
- `figures/skill_cloud_excel_popular_entry_2025.png`
- `figures/pca_important_skills_popular_vs_nonpopular.png`
- `figures/clusters_in_important_skill_pca_space.png`
- `figures/cluster_popular_rates.png`
- `figures/classification_roc_curves.png`
- `figures/classification_feature_importance.png`
- `figures/skill_regression_coefficients.png`
"""
(OUT / "report.md").write_text(report)

print(json.dumps(summary, indent=2))
print("Wrote outputs to", OUT)
