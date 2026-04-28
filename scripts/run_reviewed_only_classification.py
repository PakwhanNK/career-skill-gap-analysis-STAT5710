from __future__ import annotations

import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


BASE = Path(r"C:\Users\knich\Projects\career-skill-gap-analysis-STAT5710")
COHORT = BASE / "outputs" / "tables" / "reconstructed_first_job_scope_sample.csv"
PROFILES = BASE / "data" / "raw" / "user_profiles.csv"
RAW_SKILLS = BASE / "data" / "raw" / "Revelio User Skill.csv"
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

RANDOM_SEED = 5710


def clean_text(series: pd.Series, missing: str = "Missing") -> pd.Series:
    return series.fillna(missing).astype(str).str.strip().replace({"": missing, "nan": missing})


def normalize_decision(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace({"nan": "", "1.0": "1", "0.0": "0"})


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 0.08,
    l2: float = 1e-3,
    epochs: int = 2000,
) -> tuple[np.ndarray, float]:
    n, d = x.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    for _ in range(epochs):
        p = sigmoid(x @ w + b)
        err = p - y
        grad_w = (x.T @ err) / n + l2 * w
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def roc_auc_score_manual(y: np.ndarray, scores: np.ndarray) -> float:
    y = y.astype(int)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores), dtype=float) + 1.0
    pos_ranks_sum = ranks[pos].sum()
    auc = (pos_ranks_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": roc_auc_score_manual(y_true, prob),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def stratified_split(y: np.ndarray, test_frac: float = 0.2, seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    n_pos_test = max(1, int(round(len(pos_idx) * test_frac)))
    n_neg_test = max(1, int(round(len(neg_idx) * test_frac)))
    test_idx = np.concatenate([pos_idx[:n_pos_test], neg_idx[:n_neg_test]])
    train_idx = np.concatenate([pos_idx[n_pos_test:], neg_idx[n_neg_test:]])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def make_target_encoding(train_series: pd.Series, y_train: pd.Series, min_count: int = 25, alpha: float = 25.0) -> tuple[dict[str, float], float]:
    tmp = pd.DataFrame({"cat": train_series.astype(str), "y": y_train.astype(float)})
    grouped = tmp.groupby("cat").agg(n=("y", "size"), s=("y", "sum")).reset_index()
    global_rate = float(y_train.mean())
    mapping: dict[str, float] = {}
    for _, row in grouped.iterrows():
        cat = row["cat"]
        n = int(row["n"])
        if n < min_count:
            continue
        s = float(row["s"])
        mapping[cat] = (s + alpha * global_rate) / (n + alpha)
    return mapping, global_rate


def apply_target_encoding(series: pd.Series, mapping: dict[str, float], default: float) -> np.ndarray:
    return series.astype(str).map(mapping).fillna(default).to_numpy(dtype=float)


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (train - mean) / std, (test - mean) / std, mean, std


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def render_vertical_bar_chart(title: str, categories: list[str], values: list[float], colors: list[str], out: Path, y_fmt: str = "int") -> None:
    width, height = 900, 560
    left, right, top, bottom = 80, 40, 80, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_val = max(values) if values else 1.0
    bar_w = plot_w / max(1, len(values)) * 0.55
    gap = plot_w / max(1, len(values))
    parts = [svg_header(width, height), f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>']
    parts.append(f'<text x="{width/2}" y="38" text-anchor="middle" font-size="24" font-weight="700" fill="{COLORS["ink"]}">{escape(title)}</text>')

    for i in range(6):
        y = top + plot_h * i / 5
        val = max_val * (1 - i / 5)
        tick = f"{int(round(val)):,}" if y_fmt == "int" else f"{val:.2f}"
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="{COLORS["ink"]}">{tick}</text>')

    for idx, (cat, val, color) in enumerate(zip(categories, values, colors)):
        x = left + gap * idx + (gap - bar_w) / 2
        h = 0 if max_val == 0 else plot_h * val / max_val
        y = top + plot_h - h
        label = f"{int(round(val)):,}" if y_fmt == "int" else f"{val:.3f}"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="6" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-size="14" fill="{COLORS["ink"]}">{label}</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{height-40}" text-anchor="middle" font-size="14" fill="{COLORS["ink"]}">{escape(cat)}</text>')

    parts.append("</svg>")
    write_text(out, "\n".join(parts))


def render_horizontal_bar_chart(title: str, labels: list[str], values: list[float], color: str, out: Path, value_fmt: str = "int") -> None:
    width = 1200
    row_h = 30
    top = 90
    left = 420
    right = 80
    bottom = 40
    height = top + bottom + row_h * len(labels)
    plot_w = width - left - right
    max_val = max(values) if values else 1.0
    parts = [svg_header(width, height), f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>']
    parts.append(f'<text x="{width/2}" y="38" text-anchor="middle" font-size="24" font-weight="700" fill="{COLORS["ink"]}">{escape(title)}</text>')
    for idx, (label, val) in enumerate(zip(labels, values)):
        y = top + idx * row_h
        bar_w = 0 if max_val == 0 else plot_w * val / max_val
        val_label = f"{int(round(val)):,}" if value_fmt == "int" else f"{val:.1%}"
        parts.append(f'<text x="{left-12}" y="{y+19}" text-anchor="end" font-size="13" fill="{COLORS["ink"]}">{escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y+5}" width="{bar_w:.1f}" height="18" rx="5" fill="{color}"/>')
        parts.append(f'<text x="{left + bar_w + 8:.1f}" y="{y+19}" font-size="13" fill="{COLORS["ink"]}">{val_label}</text>')
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
    width, height = 1100, 650
    cx, cy, r = 290, 350, 190
    total = sum(values) or 1
    palette = [COLORS["target"], COLORS["blue"], COLORS["gold"], COLORS["purple"], COLORS["rose"], COLORS["green"], COLORS["amber"], COLORS["slate"], COLORS["gray"]]
    parts = [svg_header(width, height), f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>']
    parts.append(f'<text x="{width/2}" y="38" text-anchor="middle" font-size="24" font-weight="700" fill="{COLORS["ink"]}">{escape(title)}</text>')
    angle = 0.0
    legend_x = 600
    legend_y = 120
    for idx, (label, val) in enumerate(zip(labels, values)):
        sweep = val / total * 360
        color = palette[idx % len(palette)]
        parts.append(f'<path d="{describe_arc(cx, cy, r, angle, angle + sweep)}" fill="{color}" stroke="{COLORS["bg"]}" stroke-width="2"/>')
        mid = angle + sweep / 2
        tx, ty = polar_to_cartesian(cx, cy, r * 0.66, mid)
        if sweep >= 10:
            parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" font-size="13" fill="white">{val / total * 100:.1f}%</text>')
        ly = legend_y + idx * 38
        parts.append(f'<rect x="{legend_x}" y="{ly-12}" width="18" height="18" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 28}" y="{ly+2}" font-size="14" fill="{COLORS["ink"]}">{escape(label)} ({val:,})</text>')
        angle += sweep
    parts.append("</svg>")
    write_text(out, "\n".join(parts))


def main() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(COHORT)
    cohort["manual_decision"] = normalize_decision(cohort["manual_decision"])
    reviewed = cohort[cohort["manual_decision"].isin(["0", "1"])].copy()
    reviewed["target"] = reviewed["manual_decision"].eq("1").astype(int)
    reviewed["major"] = clean_text(reviewed["major"])
    reviewed["school"] = clean_text(reviewed["school"])
    reviewed["degree"] = clean_text(reviewed["degree"])
    reviewed["university_country"] = clean_text(reviewed["university_country"])

    profiles = pd.read_csv(PROFILES, usecols=["user_id", "prestige", "numconnections", "user_country"])
    profiles["user_country"] = clean_text(profiles["user_country"])
    reviewed = reviewed.merge(profiles, on="user_id", how="left")
    reviewed["user_country"] = clean_text(reviewed["user_country"])
    reviewed["prestige"] = pd.to_numeric(reviewed["prestige"], errors="coerce")
    reviewed["numconnections"] = pd.to_numeric(reviewed["numconnections"], errors="coerce")
    reviewed["entry_job_year"] = pd.to_datetime(reviewed["startdate"], errors="coerce").dt.year
    reviewed["seniority_num"] = pd.to_numeric(reviewed["seniority"], errors="coerce")

    skill_rows = 0
    skill_users = set()
    with RAW_SKILLS.open("r", encoding="utf-8-sig", newline="") as f:
        import csv
        reader = csv.DictReader(f)
        review_ids = set(reviewed["user_id"].astype(int).tolist())
        for row in reader:
            try:
                uid = int(float(row["user_id"]))
            except Exception:
                continue
            if uid in review_ids:
                skill_rows += 1
                skill_users.add(uid)

    y = reviewed["target"].to_numpy(dtype=float)
    train_idx, test_idx = stratified_split(y, test_frac=0.2)
    train = reviewed.iloc[train_idx].copy()
    test = reviewed.iloc[test_idx].copy()

    # Train-only target encodings for categorical features.
    cat_cols = ["school", "major", "degree", "university_country", "user_country"]
    encoded_train_cols = []
    encoded_test_cols = []
    feature_names: list[str] = []
    for col in cat_cols:
        mapping, default = make_target_encoding(train[col], train["target"], min_count=30, alpha=30.0)
        encoded_train_cols.append(apply_target_encoding(train[col], mapping, default))
        encoded_test_cols.append(apply_target_encoding(test[col], mapping, default))
        feature_names.append(f"{col}_te")

    num_cols = ["prestige", "numconnections", "entry_job_year", "seniority_num"]
    train_num = train[num_cols].copy()
    test_num = test[num_cols].copy()
    train_num["prestige"] = train_num["prestige"].fillna(train_num["prestige"].median())
    test_num["prestige"] = test_num["prestige"].fillna(train_num["prestige"].median())
    train_num["numconnections"] = train_num["numconnections"].fillna(train_num["numconnections"].median())
    test_num["numconnections"] = test_num["numconnections"].fillna(train_num["numconnections"].median())
    train_num["entry_job_year"] = train_num["entry_job_year"].fillna(train_num["entry_job_year"].median())
    test_num["entry_job_year"] = test_num["entry_job_year"].fillna(train_num["entry_job_year"].median())
    train_num["seniority_num"] = train_num["seniority_num"].fillna(train_num["seniority_num"].median())
    test_num["seniority_num"] = test_num["seniority_num"].fillna(train_num["seniority_num"].median())

    x_train = np.column_stack(encoded_train_cols + [train_num[c].to_numpy(dtype=float) for c in num_cols])
    x_test = np.column_stack(encoded_test_cols + [test_num[c].to_numpy(dtype=float) for c in num_cols])
    feature_names.extend(num_cols)
    x_train, x_test, mean, std = standardize(x_train, x_test)

    w, b = fit_logistic_regression(x_train, train["target"].to_numpy(dtype=float))
    train_prob = sigmoid(x_train @ w + b)
    test_prob = sigmoid(x_test @ w + b)

    train_metrics = classification_metrics(train["target"].to_numpy(dtype=int), train_prob)
    test_metrics = classification_metrics(test["target"].to_numpy(dtype=int), test_prob)

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": w,
            "abs_coefficient": np.abs(w),
        }
    ).sort_values("abs_coefficient", ascending=False)
    coef_df.to_csv(OUT_TABLES / "reviewed_classification_feature_coefficients.csv", index=False)

    metrics_df = pd.DataFrame(
        [
            {"split": "train", **train_metrics},
            {"split": "test", **test_metrics},
        ]
    )
    metrics_df.to_csv(OUT_TABLES / "reviewed_classification_metrics.csv", index=False)

    confusion_df = pd.DataFrame(
        [[test_metrics["tn"], test_metrics["fp"]], [test_metrics["fn"], test_metrics["tp"]]],
        index=["actual_non_target", "actual_target"],
        columns=["pred_non_target", "pred_target"],
    )
    confusion_df.to_csv(OUT_TABLES / "reviewed_classification_confusion_matrix.csv")

    summary = {
        "reviewed_rows": int(len(reviewed)),
        "target_rows": int(reviewed["target"].sum()),
        "non_target_rows": int(len(reviewed) - reviewed["target"].sum()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_auc": float(test_metrics["auc"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "skill_rows_overlap_raw_skill_file": int(skill_rows),
        "skill_user_overlap_raw_skill_file": int(len(skill_users)),
        "skill_user_overlap_share": float(len(skill_users) / len(reviewed)),
        "model_type": "custom_target_encoded_logistic_regression",
    }
    write_text(OUT_TABLES / "reviewed_classification_summary.json", json.dumps(summary, indent=2))

    report_lines = [
        "# Reviewed-Only Classification Summary",
        "",
        "## Model",
        "- Model type: custom target-encoded logistic regression implemented in pure NumPy.",
        "- Features: target-encoded school, major, degree, university country, user country; plus prestige, number of connections, entry-job year, and seniority.",
        "- Train/test split: stratified 80/20.",
        "",
        "## Results",
        f"- Reviewed rows modeled: {summary['reviewed_rows']:,}",
        f"- Target rows: {summary['target_rows']:,}",
        f"- Non-target rows: {summary['non_target_rows']:,}",
        f"- Test AUC: {summary['test_auc']:.3f}",
        f"- Test accuracy: {summary['test_accuracy']:.3f}",
        f"- Test precision: {summary['test_precision']:.3f}",
        f"- Test recall: {summary['test_recall']:.3f}",
        "",
        "## Caveats",
        "- This is a reviewed-only cohort, so the model is trained on a manually labeled employer subset rather than the full employer universe.",
        "- The bundled runtime did not include `scikit-learn`, so this is a custom logistic baseline rather than the earlier random forest / boosting pipeline.",
        f"- Raw skill coverage for the reviewed cohort is sparse in the available local skill file: {summary['skill_user_overlap_raw_skill_file']:,} of {summary['reviewed_rows']:,} users ({summary['skill_user_overlap_share']:.1%}).",
        "- Because of that sparse overlap, the current model intentionally excludes skill features and should be interpreted as a profile-and-education baseline.",
    ]
    write_text(OUT_TABLES / "reviewed_classification_report.md", "\n".join(report_lines) + "\n")

    # EDA source tables
    top_target_schools = reviewed.loc[reviewed["target"] == 1, "school"].value_counts().head(20).rename_axis("school").reset_index(name="users")
    top_target_schools.to_csv(OUT_TABLES / "eda_reviewed_top20_target_schools.csv", index=False)

    target_major_counts = reviewed.loc[reviewed["target"] == 1, "major"].value_counts()
    pie_major = target_major_counts.head(8).rename_axis("major").reset_index(name="users")
    other_major_count = int(target_major_counts.iloc[8:].sum())
    if other_major_count > 0:
        pie_major.loc[len(pie_major)] = {"major": "Other", "users": other_major_count}
    pie_major["share"] = pie_major["users"] / pie_major["users"].sum()
    pie_major.to_csv(OUT_TABLES / "eda_reviewed_target_major_pie.csv", index=False)

    counts_df = pd.DataFrame({"group": ["Target", "Non-target"], "users": [int(reviewed["target"].sum()), int((1 - reviewed["target"]).sum())]})
    counts_df.to_csv(OUT_TABLES / "eda_reviewed_target_vs_nontarget_counts.csv", index=False)

    major_rate_df = (
        reviewed.groupby("major")
        .agg(users=("user_id", "size"), target_rate=("target", "mean"))
        .reset_index()
        .sort_values("users", ascending=False)
    )
    major_rate_df = major_rate_df[major_rate_df["users"] >= 75].head(12)
    major_rate_df.to_csv(OUT_TABLES / "eda_reviewed_target_rate_by_major.csv", index=False)

    # Visuals
    render_vertical_bar_chart(
        "Reviewed Cohort: Target vs Non-Target",
        counts_df["group"].tolist(),
        counts_df["users"].tolist(),
        [COLORS["target"], COLORS["non_target"]],
        OUT_FIG / "reviewed_eda_target_vs_nontarget_counts.svg",
        y_fmt="int",
    )
    render_horizontal_bar_chart(
        "Top 20 Target Universities",
        top_target_schools["school"].tolist(),
        top_target_schools["users"].tolist(),
        COLORS["blue"],
        OUT_FIG / "reviewed_eda_top20_target_schools.svg",
        value_fmt="int",
    )
    render_pie_chart(
        "Top Target Majors",
        pie_major["major"].tolist(),
        pie_major["users"].astype(int).tolist(),
        OUT_FIG / "reviewed_eda_target_major_pie.svg",
    )
    render_horizontal_bar_chart(
        "Target Rate by Major (Reviewed Cohort)",
        major_rate_df["major"].tolist(),
        major_rate_df["target_rate"].tolist(),
        COLORS["gold"],
        OUT_FIG / "reviewed_eda_target_rate_by_major.svg",
        value_fmt="pct",
    )
    render_vertical_bar_chart(
        "Largest Feature Coefficients",
        coef_df.head(8)["feature"].tolist(),
        coef_df.head(8)["abs_coefficient"].tolist(),
        [COLORS["purple"]] * min(8, len(coef_df)),
        OUT_FIG / "reviewed_classification_top_feature_coefficients.svg",
        y_fmt="float",
    )
    print("Wrote reviewed-only classification outputs and EDA visuals.")


if __name__ == "__main__":
    main()
