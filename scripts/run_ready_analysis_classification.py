from __future__ import annotations

import json
import math
import re
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


BASE = Path(r"C:\Users\knich\Projects\career-skill-gap-analysis-STAT5710")
READY = BASE / "data" / "processed" / "ready_analysis"
OUT_TABLES = BASE / "outputs" / "tables"
OUT_FIG = BASE / "outputs" / "figures"

FIRST_JOB = READY / "ready_first_job.csv"
EDU = READY / "ready_revelio_edu_18_22.csv"
PROFILES = READY / "ready_user_profiles.csv"
SKILL_AGG = READY / "ready_user_skill_agg.csv"
EMPLOYER_REVIEW = OUT_TABLES / "employer_target_review.csv"

RANDOM_SEED = 5710
TOP_SKILLS = 90
TREE_MAX_DEPTH = 5
FOREST_TREES = 35

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


def clean_series(series: pd.Series, missing: str = "Missing") -> pd.Series:
    return series.fillna(missing).astype(str).str.strip().replace({"": missing, "nan": missing})


def normalize_company(value: str) -> str:
    value = str(value or "").strip().lower()
    value = re.sub(r"^the\s+", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_decision(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace({"nan": "", "1.0": "1", "0.0": "0"})


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_logistic(x: np.ndarray, y: np.ndarray, lr: float = 0.05, l2: float = 0.003, epochs: int = 1800) -> tuple[np.ndarray, float]:
    n, d = x.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        p = sigmoid(x @ w + b)
        err = p - y
        w -= lr * ((x.T @ err) / n + l2 * w)
        b -= lr * float(err.mean())
    return w, b


def normal_two_sided_p(z: float) -> float:
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def stratified_split(y: np.ndarray, test_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    pos_test = max(1, int(round(len(pos) * test_frac)))
    neg_test = max(1, int(round(len(neg) * test_frac)))
    test = np.concatenate([pos[:pos_test], neg[:neg_test]])
    train = np.concatenate([pos[pos_test:], neg[neg_test:]])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def auc_manual(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(int)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(score), dtype=float) + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / len(y)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc_manual(y, prob), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def target_encoding(train_col: pd.Series, y: pd.Series, min_count: int = 30, alpha: float = 30) -> tuple[dict[str, float], float]:
    global_rate = float(y.mean())
    tmp = pd.DataFrame({"cat": train_col.astype(str), "y": y.astype(float)})
    rows = tmp.groupby("cat").agg(n=("y", "size"), s=("y", "sum")).reset_index()
    mapping = {}
    for _, row in rows.iterrows():
        if int(row["n"]) < min_count:
            continue
        mapping[str(row["cat"])] = (float(row["s"]) + alpha * global_rate) / (int(row["n"]) + alpha)
    return mapping, global_rate


def apply_encoding(series: pd.Series, mapping: dict[str, float], default: float) -> np.ndarray:
    return series.astype(str).map(mapping).fillna(default).to_numpy(dtype=float)


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (train - mean) / std, (test - mean) / std


def split_skills(text: str) -> set[str]:
    if not isinstance(text, str) or not text.strip():
        return set()
    return {x.strip() for x in text.split(";") if x.strip() and x.strip().lower() not in {"unknown", "nan", "none"}}


def select_train_skills(train_skill_sets: list[set[str]], y_train: np.ndarray) -> list[str]:
    counts: dict[str, list[int]] = {}
    total_pos = int(y_train.sum())
    total_neg = int(len(y_train) - total_pos)
    for skills, y in zip(train_skill_sets, y_train):
        for skill in skills:
            if skill not in counts:
                counts[skill] = [0, 0]
            counts[skill][int(y)] += 1
    rows = []
    for skill, (neg, pos) in counts.items():
        users = neg + pos
        if users < 25:
            continue
        log_odds = math.log((pos + 0.5) / (total_pos + 1)) - math.log((neg + 0.5) / (total_neg + 1))
        rows.append((skill, users, pos, neg, log_odds, abs(log_odds)))
    rows.sort(key=lambda x: x[-1], reverse=True)
    pd.DataFrame(rows, columns=["skill", "users", "target_users", "non_target_users", "log_odds_target", "abs_log_odds"]).to_csv(
        OUT_TABLES / "ready_skill_log_odds.csv", index=False
    )
    return [row[0] for row in rows[:TOP_SKILLS]]


def make_skill_matrix(skill_sets: list[set[str]], skills: list[str]) -> np.ndarray:
    index = {skill: i for i, skill in enumerate(skills)}
    x = np.zeros((len(skill_sets), len(skills)), dtype=float)
    for r, row_skills in enumerate(skill_sets):
        for skill in row_skills:
            c = index.get(skill)
            if c is not None:
                x[r, c] = 1.0
    return x


def prepare_design(df: pd.DataFrame, include_seniority: bool = False, include_skills: bool = True, include_n_skills: bool = True) -> dict[str, object]:
    y = df["target"].to_numpy(dtype=int)
    train_idx, test_idx = stratified_split(y)
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()
    y_train = train["target"].to_numpy(dtype=float)
    y_test = test["target"].to_numpy(dtype=int)

    cat_cols = ["school", "major", "degree", "university_country", "user_country"]
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    feature_names: list[str] = []
    for col in cat_cols:
        mapping, default = target_encoding(train[col], train["target"])
        train_parts.append(apply_encoding(train[col], mapping, default))
        test_parts.append(apply_encoding(test[col], mapping, default))
        feature_names.append(f"{col}_te")

    selected_skills: list[str] = []
    train_skill_matrix = np.zeros((len(train), 0), dtype=float)
    test_skill_matrix = np.zeros((len(test), 0), dtype=float)
    if include_skills:
        train_skill_sets = train["skill_set"].tolist()
        test_skill_sets = test["skill_set"].tolist()
        selected_skills = select_train_skills(train_skill_sets, y_train.astype(int))
        train_skill_matrix = make_skill_matrix(train_skill_sets, selected_skills)
        test_skill_matrix = make_skill_matrix(test_skill_sets, selected_skills)

    num_cols = ["prestige", "numconnections", "entry_job_year"]
    if include_n_skills:
        num_cols.append("n_skills")
    if include_seniority:
        num_cols.insert(3, "seniority_num")
    train_num = train[num_cols].copy()
    test_num = test[num_cols].copy()
    for col in num_cols:
        median = train_num[col].median()
        train_num[col] = train_num[col].fillna(median)
        test_num[col] = test_num[col].fillna(median)

    x_train_raw = np.column_stack(train_parts + [train_num[col].to_numpy(dtype=float) for col in num_cols] + [train_skill_matrix])
    x_test_raw = np.column_stack(test_parts + [test_num[col].to_numpy(dtype=float) for col in num_cols] + [test_skill_matrix])
    feature_names.extend(num_cols)
    feature_names.extend([f"skill::{s}" for s in selected_skills])
    x_train, x_test = standardize(x_train_raw, x_test_raw)
    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "selected_skills": selected_skills,
        "train": train,
        "test": test,
    }


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def render_horizontal(title: str, labels: list[str], values: list[float], color: str, out: Path, fmt: str = "int") -> None:
    width = 1250
    row_h = 30
    top = 90
    left = 430
    right = 90
    bottom = 40
    height = top + bottom + row_h * len(labels)
    max_val = max(values) if values else 1
    plot_w = width - left - right
    parts = [svg_header(width, height), f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>']
    parts.append(f'<text x="{width/2}" y="38" text-anchor="middle" font-size="24" font-weight="700" fill="{COLORS["ink"]}">{escape(title)}</text>')
    for i, (label, value) in enumerate(zip(labels, values)):
        y = top + i * row_h
        w = 0 if max_val == 0 else plot_w * value / max_val
        value_label = f"{int(round(value)):,}" if fmt == "int" else f"{value:.1%}" if fmt == "pct" else f"{value:.3f}"
        parts.append(f'<text x="{left-12}" y="{y+19}" text-anchor="end" font-size="13" fill="{COLORS["ink"]}">{escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y+5}" width="{w:.1f}" height="18" rx="5" fill="{color}"/>')
        parts.append(f'<text x="{left+w+8:.1f}" y="{y+19}" font-size="13" fill="{COLORS["ink"]}">{value_label}</text>')
    parts.append("</svg>")
    write_text(out, "\n".join(parts))


def render_vertical(title: str, labels: list[str], values: list[float], colors: list[str], out: Path, max_y: float | None = None, fmt: str = "int") -> None:
    width, height = 900, 560
    left, right, top, bottom = 80, 40, 80, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_val = max_y if max_y is not None else max(values) if values else 1
    gap = plot_w / len(labels)
    bar_w = gap * 0.55
    parts = [svg_header(width, height), f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>']
    parts.append(f'<text x="{width/2}" y="38" text-anchor="middle" font-size="24" font-weight="700" fill="{COLORS["ink"]}">{escape(title)}</text>')
    for i in range(6):
        y = top + plot_h * i / 5
        tick = max_val * (1 - i / 5)
        tick_label = f"{int(round(tick)):,}" if fmt == "int" else f"{tick:.1f}"
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="{COLORS["ink"]}">{tick_label}</text>')
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = left + gap * i + (gap - bar_w) / 2
        h = 0 if max_val == 0 else plot_h * value / max_val
        y = top + plot_h - h
        value_label = f"{int(round(value)):,}" if fmt == "int" else f"{value:.3f}"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="6" fill="{color}"/>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-size="14" fill="{COLORS["ink"]}">{value_label}</text>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{height-40}" text-anchor="middle" font-size="14" fill="{COLORS["ink"]}">{escape(label)}</text>')
    parts.append("</svg>")
    write_text(out, "\n".join(parts))


def render_grouped_bars(title: str, labels: list[str], first_values: list[float], second_values: list[float], first_label: str, second_label: str, out: Path, fmt: str = "int", max_y: float | None = None) -> None:
    width, height = 1150, 620
    left, right, top, bottom = 85, 50, 90, 130
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_values = first_values + second_values
    max_val = max_y if max_y is not None else max(all_values) if all_values else 1
    group_w = plot_w / max(1, len(labels))
    bar_w = group_w * 0.28
    parts = [svg_header(width, height), f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>']
    parts.append(f'<text x="{width/2}" y="38" text-anchor="middle" font-size="24" font-weight="700" fill="{COLORS["ink"]}">{escape(title)}</text>')
    parts.append(f'<rect x="{width-250}" y="58" width="16" height="16" fill="{COLORS["blue"]}"/><text x="{width-226}" y="71" font-size="14" fill="{COLORS["ink"]}">{escape(first_label)}</text>')
    parts.append(f'<rect x="{width-140}" y="58" width="16" height="16" fill="{COLORS["gold"]}"/><text x="{width-116}" y="71" font-size="14" fill="{COLORS["ink"]}">{escape(second_label)}</text>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="{COLORS["grid"]}" stroke-width="2"/>')
    for i, label in enumerate(labels):
        x_center = left + group_w * (i + 0.5)
        for value, dx, color in [(first_values[i], -bar_w * 0.6, COLORS["blue"]), (second_values[i], bar_w * 0.6, COLORS["gold"])]:
            h = 0 if max_val == 0 else plot_h * value / max_val
            x = x_center + dx - bar_w / 2
            y = top + plot_h - h
            value_label = f"{int(round(value)):,}" if fmt == "int" else f"{value:.1%}" if fmt == "pct" else f"{value:.3f}"
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="5" fill="{color}"/>')
            parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-5:.1f}" text-anchor="middle" font-size="11" fill="{COLORS["ink"]}">{value_label}</text>')
        parts.append(f'<text x="{x_center}" y="{top+plot_h+28}" text-anchor="middle" font-size="12" fill="{COLORS["ink"]}" transform="rotate(-28 {x_center} {top+plot_h+28})">{escape(str(label)[:24])}</text>')
    parts.append("</svg>")
    write_text(out, "\n".join(parts))


def render_confusion(cm: pd.DataFrame, out: Path) -> None:
    width, height = 760, 620
    cell = 150
    left, top = 210, 150
    vals = cm.to_numpy()
    max_val = vals.max()
    parts = [svg_header(width, height), f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>']
    parts.append(f'<text x="{width/2}" y="44" text-anchor="middle" font-size="24" font-weight="700" fill="{COLORS["ink"]}">Ready Analysis Confusion Matrix</text>')
    for i, label in enumerate(["Pred Non-Target", "Pred Target"]):
        parts.append(f'<text x="{left+i*cell+cell/2}" y="{top-24}" text-anchor="middle" font-size="15" fill="{COLORS["ink"]}">{escape(label)}</text>')
    for i, label in enumerate(["Actual Non-Target", "Actual Target"]):
        parts.append(f'<text x="{left-18}" y="{top+i*cell+cell/2+6}" text-anchor="end" font-size="15" fill="{COLORS["ink"]}">{escape(label)}</text>')
    for r in range(2):
        for c in range(2):
            x = left + c * cell
            y = top + r * cell
            val = int(vals[r, c])
            opacity = 0.35 + 0.55 * val / max_val
            fill = "#c6f6d5" if r == c else "#fbd38d"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" fill-opacity="{opacity:.2f}" stroke="{COLORS["grid"]}" stroke-width="2"/>')
            parts.append(f'<text x="{x+cell/2}" y="{y+cell/2+8}" text-anchor="middle" font-size="30" font-weight="700" fill="{COLORS["ink"]}">{val:,}</text>')
    parts.append("</svg>")
    write_text(out, "\n".join(parts))


def polar(cx: float, cy: float, r: float, deg: float) -> tuple[float, float]:
    rad = math.radians(deg - 90)
    return cx + r * math.cos(rad), cy + r * math.sin(rad)


def arc_path(cx: float, cy: float, r: float, start: float, end: float) -> str:
    a = polar(cx, cy, r, end)
    b = polar(cx, cy, r, start)
    large = 1 if end - start > 180 else 0
    return f"M {cx} {cy} L {a[0]:.2f} {a[1]:.2f} A {r} {r} 0 {large} 0 {b[0]:.2f} {b[1]:.2f} Z"


def render_pie(title: str, labels: list[str], values: list[int], out: Path) -> None:
    width, height = 1100, 650
    cx, cy, r = 290, 350, 190
    total = sum(values) or 1
    palette = [COLORS["target"], COLORS["blue"], COLORS["gold"], COLORS["purple"], COLORS["rose"], COLORS["green"], COLORS["amber"], COLORS["slate"], COLORS["gray"]]
    parts = [svg_header(width, height), f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>']
    parts.append(f'<text x="{width/2}" y="38" text-anchor="middle" font-size="24" font-weight="700" fill="{COLORS["ink"]}">{escape(title)}</text>')
    angle = 0.0
    for i, (label, value) in enumerate(zip(labels, values)):
        sweep = value / total * 360
        color = palette[i % len(palette)]
        parts.append(f'<path d="{arc_path(cx, cy, r, angle, angle+sweep)}" fill="{color}" stroke="{COLORS["bg"]}" stroke-width="2"/>')
        if sweep >= 10:
            x, y = polar(cx, cy, r * 0.65, angle + sweep / 2)
            parts.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" font-size="13" fill="white">{value/total:.1%}</text>')
        ly = 120 + i * 38
        parts.append(f'<rect x="600" y="{ly-12}" width="18" height="18" rx="3" fill="{color}"/>')
        parts.append(f'<text x="628" y="{ly+2}" font-size="14" fill="{COLORS["ink"]}">{escape(label)} ({value:,})</text>')
        angle += sweep
    parts.append("</svg>")
    write_text(out, "\n".join(parts))


def render_word_cloud(title: str, freq: pd.Series, out: Path, color: str) -> None:
    width, height = 1200, 760
    words = freq.head(55)
    max_count = float(words.max()) if len(words) else 1.0
    parts = [svg_header(width, height), f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>']
    parts.append(f'<text x="{width/2}" y="44" text-anchor="middle" font-size="26" font-weight="700" fill="{COLORS["ink"]}">{escape(title)}</text>')
    cols = 5
    cell_w = width / cols
    y = 105
    for i, (word, count) in enumerate(words.items()):
        col = i % cols
        row = i // cols
        x = 40 + col * cell_w
        y = 105 + row * 54
        size = 13 + 25 * (float(count) / max_count) ** 0.65
        opacity = 0.62 + 0.35 * (float(count) / max_count)
        label = str(word)[:32]
        parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size:.1f}" fill="{color}" fill-opacity="{opacity:.2f}" font-weight="700">{escape(label)}</text>')
    parts.append("</svg>")
    write_text(out, "\n".join(parts))


def fit_model(df: pd.DataFrame, include_seniority: bool = True) -> tuple[dict[str, float], dict[str, float], pd.DataFrame, pd.DataFrame, list[str]]:
    y = df["target"].to_numpy(dtype=int)
    train_idx, test_idx = stratified_split(y)
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()
    y_train = train["target"].to_numpy(dtype=float)
    y_test = test["target"].to_numpy(dtype=int)

    cat_cols = ["school", "major", "degree", "university_country", "user_country"]
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    feature_names: list[str] = []
    for col in cat_cols:
        mapping, default = target_encoding(train[col], train["target"])
        train_parts.append(apply_encoding(train[col], mapping, default))
        test_parts.append(apply_encoding(test[col], mapping, default))
        feature_names.append(f"{col}_te")

    train_skill_sets = train["skill_set"].tolist()
    test_skill_sets = test["skill_set"].tolist()
    selected_skills = select_train_skills(train_skill_sets, y_train.astype(int))
    train_skill_matrix = make_skill_matrix(train_skill_sets, selected_skills)
    test_skill_matrix = make_skill_matrix(test_skill_sets, selected_skills)

    num_cols = ["prestige", "numconnections", "entry_job_year", "n_skills"]
    if include_seniority:
        num_cols.insert(3, "seniority_num")
    train_num = train[num_cols].copy()
    test_num = test[num_cols].copy()
    for col in num_cols:
        median = train_num[col].median()
        train_num[col] = train_num[col].fillna(median)
        test_num[col] = test_num[col].fillna(median)

    x_train = np.column_stack(train_parts + [train_num[col].to_numpy(dtype=float) for col in num_cols] + [train_skill_matrix])
    x_test = np.column_stack(test_parts + [test_num[col].to_numpy(dtype=float) for col in num_cols] + [test_skill_matrix])
    feature_names.extend(num_cols)
    feature_names.extend([f"skill::{s}" for s in selected_skills])

    x_train, x_test = standardize(x_train, x_test)
    w, b = fit_logistic(x_train, y_train)
    train_prob = sigmoid(x_train @ w + b)
    test_prob = sigmoid(x_test @ w + b)
    train_metrics = metrics(y_train.astype(int), train_prob)
    test_metrics = metrics(y_test, test_prob)

    cm = pd.DataFrame([[test_metrics["tn"], test_metrics["fp"]], [test_metrics["fn"], test_metrics["tp"]]], index=["actual_non_target", "actual_target"], columns=["pred_non_target", "pred_target"])
    coef = pd.DataFrame({"feature": feature_names, "coefficient": w, "abs_coefficient": np.abs(w)}).sort_values("abs_coefficient", ascending=False)
    return train_metrics, test_metrics, cm, coef, selected_skills


def logistic_inference_table(x: np.ndarray, y: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    # Fit an unpenalized logistic model for approximate Wald inference. The main
    # predictive model remains the regularized version above.
    n, d = x.shape
    x_aug = np.column_stack([np.ones(n), x])
    beta = np.zeros(d + 1)
    ridge = 1e-5
    for _ in range(80):
        p = sigmoid(x_aug @ beta)
        weight = np.clip(p * (1 - p), 1e-6, None)
        grad = x_aug.T @ (y - p)
        hess = (x_aug.T * weight) @ x_aug + ridge * np.eye(d + 1)
        step = np.linalg.solve(hess, grad)
        beta += step
        if float(np.max(np.abs(step))) < 1e-6:
            break
    p = sigmoid(x_aug @ beta)
    weight = np.clip(p * (1 - p), 1e-6, None)
    fisher = (x_aug.T * weight) @ x_aug + ridge * np.eye(d + 1)
    cov = np.linalg.pinv(fisher)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    z = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    names = ["Intercept"] + feature_names
    out = pd.DataFrame({
        "feature": names,
        "log_odds_coefficient": beta,
        "odds_ratio": np.exp(np.clip(beta, -20, 20)),
        "std_error": se,
        "z_value": z,
        "p_value": [normal_two_sided_p(float(v)) for v in z],
        "abs_log_odds": np.abs(beta),
    })
    return out.sort_values("abs_log_odds", ascending=False)


def build_logistic_inference(df: pd.DataFrame) -> pd.DataFrame:
    design = prepare_design(df, include_seniority=False, include_skills=True, include_n_skills=True)
    return logistic_inference_table(
        design["x_train"],
        design["y_train"].astype(float),
        design["feature_names"],
    )


def build_non_skill_regression_outputs(df: pd.DataFrame) -> dict[str, float]:
    design = prepare_design(df, include_seniority=False, include_skills=False, include_n_skills=False)
    x_train = design["x_train"]
    x_test = design["x_test"]
    y_train = design["y_train"].astype(float)
    y_test = design["y_test"].astype(int)
    feature_names = design["feature_names"]

    w, b = fit_logistic(x_train, y_train)
    train_prob = sigmoid(x_train @ w + b)
    test_prob = sigmoid(x_test @ w + b)
    train_metrics = metrics(y_train.astype(int), train_prob)
    test_metrics = metrics(y_test, test_prob)
    coef = pd.DataFrame({"feature": feature_names, "coefficient": w, "abs_coefficient": np.abs(w)}).sort_values("abs_coefficient", ascending=False)
    inference = logistic_inference_table(x_train, y_train, feature_names)

    pd.DataFrame([{"split": "train", **train_metrics}, {"split": "test", **test_metrics}]).to_csv(OUT_TABLES / "ready_non_skill_logistic_metrics.csv", index=False)
    coef.to_csv(OUT_TABLES / "ready_non_skill_logistic_coefficients.csv", index=False)
    inference.to_csv(OUT_TABLES / "ready_non_skill_logistic_log_odds_p_values.csv", index=False)
    inference.sort_values("p_value").to_csv(OUT_TABLES / "ready_non_skill_logistic_log_odds_p_values_sorted_by_significance.csv", index=False)
    render_horizontal(
        "Non-Skill Logistic Coefficients",
        coef.head(12)["feature"].tolist(),
        coef.head(12)["abs_coefficient"].tolist(),
        COLORS["blue"],
        OUT_FIG / "ready_non_skill_logistic_feature_importance.svg",
        fmt="float",
    )
    return {f"non_skill_test_{k}": v for k, v in test_metrics.items() if k in {"accuracy", "precision", "recall", "f1", "auc"}}


def gini(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    p = float(y.mean())
    return 2 * p * (1 - p)


def best_split(x: np.ndarray, y: np.ndarray, features: np.ndarray, min_leaf: int) -> tuple[int | None, float | None, float]:
    parent = gini(y)
    best_feature = None
    best_threshold = None
    best_gain = 0.0
    for feature in features:
        col = x[:, feature]
        qs = np.unique(np.quantile(col, [0.2, 0.4, 0.6, 0.8]))
        for threshold in qs:
            left = col <= threshold
            n_left = int(left.sum())
            n_right = len(y) - n_left
            if n_left < min_leaf or n_right < min_leaf:
                continue
            gain = parent - (n_left / len(y)) * gini(y[left]) - (n_right / len(y)) * gini(y[~left])
            if gain > best_gain:
                best_feature = int(feature)
                best_threshold = float(threshold)
                best_gain = float(gain)
    return best_feature, best_threshold, best_gain


def fit_tree(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    max_depth: int = TREE_MAX_DEPTH,
    min_leaf: int = 80,
    feature_subsample: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, object]:
    rng = rng or np.random.default_rng(RANDOM_SEED)
    importances = np.zeros(x.shape[1])

    def grow(rows: np.ndarray, depth: int) -> dict[str, object]:
        row_y = y[rows]
        prob = float(row_y.mean()) if len(row_y) else 0.0
        node: dict[str, object] = {"prob": prob, "n": int(len(rows))}
        if depth >= max_depth or len(rows) < 2 * min_leaf or gini(row_y) < 1e-8:
            return node
        if feature_subsample is None:
            features = np.arange(x.shape[1])
        else:
            size = min(feature_subsample, x.shape[1])
            features = rng.choice(x.shape[1], size=size, replace=False)
        feature, threshold, gain = best_split(x[rows], row_y, features, min_leaf)
        if feature is None or threshold is None or gain <= 0:
            return node
        left_mask = x[rows, feature] <= threshold
        left_rows = rows[left_mask]
        right_rows = rows[~left_mask]
        importances[feature] += gain * len(rows)
        node.update({
            "feature": feature,
            "threshold": threshold,
            "left": grow(left_rows, depth + 1),
            "right": grow(right_rows, depth + 1),
        })
        return node

    tree = grow(np.arange(len(y)), 0)
    total = importances.sum()
    if total > 0:
        importances = importances / total
    return {"tree": tree, "importances": importances, "feature_names": feature_names}


def predict_tree(model: dict[str, object], x: np.ndarray) -> np.ndarray:
    tree = model["tree"]

    def one(row: np.ndarray, node: dict[str, object]) -> float:
        if "feature" not in node:
            return float(node["prob"])
        child = node["left"] if row[int(node["feature"])] <= float(node["threshold"]) else node["right"]
        return one(row, child)

    return np.array([one(row, tree) for row in x], dtype=float)


def fit_random_forest(x: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict[str, object]:
    rng = np.random.default_rng(RANDOM_SEED)
    trees = []
    importances = np.zeros(x.shape[1])
    feature_subsample = max(4, int(math.sqrt(x.shape[1])))
    for _ in range(FOREST_TREES):
        rows = rng.integers(0, len(y), len(y))
        model = fit_tree(x[rows], y[rows], feature_names, max_depth=TREE_MAX_DEPTH, min_leaf=70, feature_subsample=feature_subsample, rng=rng)
        trees.append(model)
        importances += model["importances"]
    importances /= max(1, len(trees))
    total = importances.sum()
    if total > 0:
        importances /= total
    return {"trees": trees, "importances": importances, "feature_names": feature_names}


def predict_random_forest(model: dict[str, object], x: np.ndarray) -> np.ndarray:
    preds = [predict_tree(tree, x) for tree in model["trees"]]
    return np.mean(np.vstack(preds), axis=0)


def write_importance(name: str, importances: np.ndarray, feature_names: list[str], out_csv: Path, out_svg: Path) -> None:
    imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    imp.to_csv(out_csv, index=False)
    top = imp.head(15)
    render_horizontal(name, top["feature"].tolist(), top["importance"].tolist(), COLORS["green"], out_svg, fmt="float")


def build_model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    design = prepare_design(df, include_seniority=False)
    x_train = design["x_train"]
    x_test = design["x_test"]
    y_train = design["y_train"].astype(int)
    y_test = design["y_test"].astype(int)
    feature_names = design["feature_names"]

    rows = []
    w, b = fit_logistic(x_train, y_train.astype(float))
    logistic_prob = sigmoid(x_test @ w + b)
    rows.append({"model": "Logistic regression", "status": "fit", **{f"test_{k}": v for k, v in metrics(y_test, logistic_prob).items() if k in {"accuracy", "precision", "recall", "f1", "auc"}}})

    tree = fit_tree(x_train, y_train, feature_names)
    tree_prob = predict_tree(tree, x_test)
    rows.append({"model": "Decision tree", "status": "fit", **{f"test_{k}": v for k, v in metrics(y_test, tree_prob).items() if k in {"accuracy", "precision", "recall", "f1", "auc"}}})
    write_importance("Decision Tree Feature Importance", tree["importances"], feature_names, OUT_TABLES / "ready_decision_tree_feature_importance.csv", OUT_FIG / "ready_decision_tree_feature_importance.svg")

    forest = fit_random_forest(x_train, y_train, feature_names)
    forest_prob = predict_random_forest(forest, x_test)
    rows.append({"model": "Random forest", "status": "fit", **{f"test_{k}": v for k, v in metrics(y_test, forest_prob).items() if k in {"accuracy", "precision", "recall", "f1", "auc"}}})
    write_importance("Random Forest Feature Importance", forest["importances"], feature_names, OUT_TABLES / "ready_random_forest_feature_importance.csv", OUT_FIG / "ready_random_forest_feature_importance.svg")

    rows.append({
        "model": "XGBoost",
        "status": "skipped: xgboost package is not installed in this runtime",
        "test_accuracy": np.nan,
        "test_precision": np.nan,
        "test_recall": np.nan,
        "test_f1": np.nan,
        "test_auc": np.nan,
    })
    return pd.DataFrame(rows)


def build_model_and_outputs(df: pd.DataFrame) -> None:
    train_metrics, test_metrics, cm, coef, selected_skills = fit_model(df, include_seniority=False)
    comparison = build_model_comparison(df)
    inference = build_logistic_inference(df)
    non_skill_metrics = build_non_skill_regression_outputs(df)

    pd.DataFrame([{"split": "train", **train_metrics}, {"split": "test", **test_metrics}]).to_csv(OUT_TABLES / "ready_classification_metrics.csv", index=False)
    cm.to_csv(OUT_TABLES / "ready_classification_confusion_matrix.csv")
    coef.to_csv(OUT_TABLES / "ready_classification_feature_coefficients.csv", index=False)
    comparison.to_csv(OUT_TABLES / "ready_classification_model_comparison.csv", index=False)
    inference.to_csv(OUT_TABLES / "ready_logistic_log_odds_p_values.csv", index=False)
    inference.sort_values("p_value").to_csv(OUT_TABLES / "ready_logistic_log_odds_p_values_sorted_by_significance.csv", index=False)

    summary = {
        "rows": int(len(df)),
        "target_rows": int(df["target"].sum()),
        "non_target_rows": int(len(df) - df["target"].sum()),
        "train_rows": int(round(len(df) * 0.8)),
        "test_rows": int(len(df) - round(len(df) * 0.8)),
        "test_auc": float(test_metrics["auc"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "non_skill_test_auc": float(non_skill_metrics["non_skill_test_auc"]),
        "non_skill_test_accuracy": float(non_skill_metrics["non_skill_test_accuracy"]),
        "non_skill_test_precision": float(non_skill_metrics["non_skill_test_precision"]),
        "non_skill_test_recall": float(non_skill_metrics["non_skill_test_recall"]),
        "top_skill_features": TOP_SKILLS,
        "model_type": "custom_target_encoded_logistic_regression_with_skill_indicators_no_seniority",
        "scope": "reviewed employers, seniority level 2 only",
    }
    write_text(OUT_TABLES / "ready_classification_summary.json", json.dumps(summary, indent=2))

    report = [
        "# Ready Analysis Classification Report",
        "",
        "## Scope",
        "- Uses the cleaned `data/processed/ready_analysis` universe where education, positions, profiles, and skills overlap.",
        "- Restricts modeling to first-job employers that have a reviewed `manual_decision` label.",
        "- Keeps only `seniority == 2` roles to focus on entry-level full-time roles and reduce internship/research-assistant contamination.",
        "",
        "## Model",
        "- Custom NumPy logistic regression.",
        "- Features: target-encoded school/major/degree/country, profile numeric fields, number of skills, and top skill indicators selected by train-only log odds.",
        "- Seniority is excluded as a predictor because it defines the analytic scope.",
        "",
        "## Test Results",
        f"- Rows modeled: {summary['rows']:,}",
        f"- Target rows: {summary['target_rows']:,}",
        f"- Non-target rows: {summary['non_target_rows']:,}",
        f"- Test AUC: {summary['test_auc']:.3f}",
        f"- Test accuracy: {summary['test_accuracy']:.3f}",
        f"- Test precision: {summary['test_precision']:.3f}",
        f"- Test recall: {summary['test_recall']:.3f}",
        "",
        "## Primary Non-Skill Regression",
        "- This version excludes skill indicators and `n_skills` to avoid self-reporting/profile-completeness bias.",
        f"- Non-skill test AUC: {summary['non_skill_test_auc']:.3f}",
        f"- Non-skill test accuracy: {summary['non_skill_test_accuracy']:.3f}",
        f"- Non-skill test precision: {summary['non_skill_test_precision']:.3f}",
        f"- Non-skill test recall: {summary['non_skill_test_recall']:.3f}",
        "",
        "## Alternative Models",
        "- Decision tree and random forest are fit with local NumPy/Pandas implementations because `sklearn` is not installed in this runtime.",
        "- True XGBoost was attempted but skipped because `xgboost` is not installed in this runtime.",
        "- See `ready_classification_model_comparison.csv` for model metrics and `ready_*_feature_importance.csv` files for feature importances.",
        "",
        "## Logistic Inference",
        "- See `ready_logistic_log_odds_p_values.csv` for standardized log-odds coefficients, odds ratios, standard errors, z-statistics, and approximate Wald p-values.",
        "",
        "## Caveats",
        "- The employer labels are manually reviewed, so results apply to the reviewed employer subset.",
        "- Skill indicators are based on self-reported/profile-derived skills and should be interpreted as noisy signals.",
        "- The model is a transparent baseline rather than a tuned production classifier.",
    ]
    write_text(OUT_TABLES / "ready_classification_report.md", "\n".join(report) + "\n")

    # Model visuals.
    render_confusion(cm, OUT_FIG / "ready_classification_confusion_matrix.svg")
    top_coef = coef.head(12)
    colors = [COLORS["target"] if x >= 0 else COLORS["non_target"] for x in top_coef["coefficient"]]
    render_horizontal("Ready Analysis Feature Importance", top_coef["feature"].tolist(), top_coef["abs_coefficient"].tolist(), COLORS["purple"], OUT_FIG / "ready_classification_feature_importance.svg", fmt="float")
    render_vertical("Ready Analysis Test Metrics", ["AUC", "Accuracy", "Precision", "Recall", "F1"], [test_metrics[k] for k in ["auc", "accuracy", "precision", "recall", "f1"]], [COLORS["target"], COLORS["blue"], COLORS["gold"], COLORS["purple"], COLORS["slate"]], OUT_FIG / "ready_classification_test_metrics.svg", max_y=1.0, fmt="float")
    fitted_comparison = comparison.dropna(subset=["test_auc"])
    render_vertical(
        "Seniority 2 Model Comparison: AUC",
        fitted_comparison["model"].tolist(),
        fitted_comparison["test_auc"].tolist(),
        [COLORS["purple"], COLORS["blue"], COLORS["target"]],
        OUT_FIG / "ready_classification_model_comparison_auc.svg",
        max_y=1.0,
        fmt="float",
    )


def build_seniority_sensitivity_outputs(df: pd.DataFrame) -> None:
    scenarios = [
        ("Original_with_seniority", "Levels 1-2, with seniority predictor", df.copy(), True),
        ("A_no_seniority_feature", "Levels 1-2, no seniority predictor", df.copy(), False),
        ("B_seniority_1_only", "Seniority 1 only", df[df["seniority_num"] == 1].copy(), False),
        ("C_seniority_2_only", "Seniority 2 only", df[df["seniority_num"] == 2].copy(), False),
    ]
    rows = []
    for scenario, label, subset, include_seniority in scenarios:
        train_metrics, test_metrics, cm, coef, selected_skills = fit_model(subset, include_seniority=include_seniority)
        target_rows = int(subset["target"].sum())
        non_target_rows = int(len(subset) - target_rows)
        rows.append({
            "scenario": scenario,
            "label": label,
            "rows": int(len(subset)),
            "target_rows": target_rows,
            "non_target_rows": non_target_rows,
            "target_rate": target_rows / len(subset) if len(subset) else 0,
            "test_auc": test_metrics["auc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
        })
        cm.to_csv(OUT_TABLES / f"ready_seniority_sensitivity_{scenario}_confusion_matrix.csv")
        coef.to_csv(OUT_TABLES / f"ready_seniority_sensitivity_{scenario}_feature_coefficients.csv", index=False)

    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(OUT_TABLES / "ready_seniority_sensitivity_metrics.csv", index=False)
    render_vertical(
        "Seniority Sensitivity: Test AUC",
        sensitivity["label"].tolist(),
        sensitivity["test_auc"].tolist(),
        [COLORS["purple"], COLORS["blue"], COLORS["target"], COLORS["gold"]],
        OUT_FIG / "ready_seniority_sensitivity_auc.svg",
        max_y=1.0,
        fmt="float",
    )
    render_vertical(
        "Seniority Sensitivity: Target Rate",
        sensitivity["label"].tolist(),
        sensitivity["target_rate"].tolist(),
        [COLORS["purple"], COLORS["blue"], COLORS["target"], COLORS["gold"]],
        OUT_FIG / "ready_seniority_sensitivity_target_rate.svg",
        max_y=1.0,
        fmt="pct",
    )
    report = [
        "# Seniority Sensitivity Check",
        "",
        "## Why",
        "The original Revelio seniority scale is 1-7, but this analysis only keeps early-career levels 1 and 2. Because seniority was the strongest feature in the baseline model, this check tests whether it is acting as a shortcut.",
        "",
        "## Scenarios",
        "- Original: keep seniority levels 1 and 2, and include `seniority_num` as a predictor.",
        "- A: keep seniority levels 1 and 2, but remove `seniority_num` from the model.",
        "- B: restrict to seniority level 1 only.",
        "- C: restrict to seniority level 2 only.",
        "",
        "## Results",
    ]
    for row in rows:
        report.append(f"- {row['label']}: rows={row['rows']:,}, target_rate={row['target_rate']:.1%}, AUC={row['test_auc']:.3f}, accuracy={row['test_accuracy']:.3f}, precision={row['test_precision']:.3f}, recall={row['test_recall']:.3f}")
    report.extend([
        "",
        "## Interpretation",
        "The target rate differs sharply between seniority 1 and seniority 2, so the two junior seniority buckets are not interchangeable. Treat the data as early-career first observed roles rather than pure entry-level jobs, and prefer the no-seniority model or separate seniority-specific checks for substantive interpretation.",
    ])
    write_text(OUT_TABLES / "ready_seniority_sensitivity_report.md", "\n".join(report) + "\n")


def build_eda_and_clouds(df: pd.DataFrame) -> None:
    counts = pd.DataFrame({"group": ["Target", "Non-target"], "users": [int(df["target"].sum()), int((1 - df["target"]).sum())]})
    counts.to_csv(OUT_TABLES / "ready_eda_target_vs_nontarget_counts.csv", index=False)
    render_vertical("Ready Analysis: Target vs Non-Target", counts["group"].tolist(), counts["users"].tolist(), [COLORS["target"], COLORS["non_target"]], OUT_FIG / "ready_eda_target_vs_nontarget_counts.svg")

    top_schools = df[df["target"] == 1]["school"].value_counts().head(20).rename_axis("school").reset_index(name="users")
    top_schools.to_csv(OUT_TABLES / "ready_eda_top20_target_schools.csv", index=False)
    render_horizontal("Top 20 Target Schools", top_schools["school"].tolist(), top_schools["users"].tolist(), COLORS["blue"], OUT_FIG / "ready_eda_top20_target_schools.svg")

    major_counts = df[df["target"] == 1]["major"].value_counts()
    major_pie = major_counts.head(8).rename_axis("major").reset_index(name="users")
    other = int(major_counts.iloc[8:].sum())
    if other:
        major_pie.loc[len(major_pie)] = {"major": "Other", "users": other}
    major_pie["share"] = major_pie["users"] / major_pie["users"].sum()
    major_pie.to_csv(OUT_TABLES / "ready_eda_target_major_pie.csv", index=False)
    render_pie("Target Major Mix", major_pie["major"].tolist(), major_pie["users"].astype(int).tolist(), OUT_FIG / "ready_eda_target_major_pie.svg")

    major_rate = df.groupby("major").agg(users=("user_id", "size"), target_rate=("target", "mean")).reset_index()
    major_rate = major_rate[major_rate["users"] >= 75].sort_values("target_rate", ascending=False).head(15)
    major_rate.to_csv(OUT_TABLES / "ready_eda_target_rate_by_major.csv", index=False)
    render_horizontal("Target Rate by Major", major_rate["major"].tolist(), major_rate["target_rate"].tolist(), COLORS["gold"], OUT_FIG / "ready_eda_target_rate_by_major.svg", fmt="pct")

    target_freq: dict[str, int] = {}
    non_freq: dict[str, int] = {}
    for skills, target in zip(df["skill_set"], df["target"]):
        dest = target_freq if target == 1 else non_freq
        for skill in skills:
            dest[skill] = dest.get(skill, 0) + 1
    target_series = pd.Series(target_freq).sort_values(ascending=False)
    non_series = pd.Series(non_freq).sort_values(ascending=False)
    target_series.rename_axis("skill").reset_index(name="users").head(200).to_csv(OUT_TABLES / "ready_target_skill_frequency.csv", index=False)
    non_series.rename_axis("skill").reset_index(name="users").head(200).to_csv(OUT_TABLES / "ready_non_target_skill_frequency.csv", index=False)
    render_word_cloud("Target Skill Word Cloud", target_series, OUT_FIG / "ready_skill_word_cloud_target.svg", COLORS["target"])
    render_word_cloud("Non-Target Skill Word Cloud", non_series, OUT_FIG / "ready_skill_word_cloud_non_target.svg", COLORS["non_target"])


def build_yoy_outputs(df: pd.DataFrame) -> None:
    yoy = df[df["entry_job_year"].isin([2024, 2025])].copy()
    if yoy.empty:
        return

    yearly = yoy.groupby("entry_job_year").agg(
        users=("user_id", "size"),
        target_rows=("target", "sum"),
        target_rate=("target", "mean"),
    ).reset_index()
    yearly["non_target_rows"] = yearly["users"] - yearly["target_rows"]
    yearly.to_csv(OUT_TABLES / "ready_yoy_2024_2025_target_summary.csv", index=False)
    render_vertical(
        "Target Share by Entry Year",
        yearly["entry_job_year"].astype(int).astype(str).tolist(),
        yearly["target_rate"].tolist(),
        [COLORS["blue"], COLORS["gold"]][: len(yearly)],
        OUT_FIG / "ready_yoy_target_share_by_year.svg",
        max_y=1.0,
        fmt="pct",
    )
    render_grouped_bars(
        "Target vs Non-Target Counts by Year",
        yearly["entry_job_year"].astype(int).astype(str).tolist(),
        yearly["target_rows"].astype(float).tolist(),
        yearly["non_target_rows"].astype(float).tolist(),
        "Target",
        "Non-target",
        OUT_FIG / "ready_yoy_target_non_target_counts.svg",
    )

    def pivot_top(field: str, filename: str, top_n: int = 15) -> pd.DataFrame:
        top = yoy[field].value_counts().head(top_n).index.tolist()
        out = yoy[yoy[field].isin(top)].pivot_table(index=field, columns="entry_job_year", values="user_id", aggfunc="size", fill_value=0)
        for year in [2024, 2025]:
            if year not in out.columns:
                out[year] = 0
        out = out[[2024, 2025]].reset_index()
        out["total"] = out[2024] + out[2025]
        out = out.sort_values("total", ascending=False)
        out.to_csv(OUT_TABLES / filename, index=False)
        return out

    top_roles = pivot_top("role_k17000_v3", "ready_yoy_2024_2025_top_roles.csv", top_n=15)
    render_grouped_bars(
        "Top Roles: 2024 vs 2025",
        top_roles["role_k17000_v3"].tolist(),
        top_roles[2024].astype(float).tolist(),
        top_roles[2025].astype(float).tolist(),
        "2024",
        "2025",
        OUT_FIG / "ready_yoy_top_roles_2024_2025.svg",
    )
    year_totals = yoy.groupby("entry_job_year")["user_id"].size()
    top_roles_pct = top_roles[["role_k17000_v3", 2024, 2025]].copy()
    top_roles_pct["share_2024"] = top_roles_pct[2024] / max(1, int(year_totals.get(2024, 0)))
    top_roles_pct["share_2025"] = top_roles_pct[2025] / max(1, int(year_totals.get(2025, 0)))
    top_roles_pct.to_csv(OUT_TABLES / "ready_yoy_2024_2025_top_roles_percentages.csv", index=False)
    render_grouped_bars(
        "Top Role Categories as % of Each Year's Observed Cohort",
        top_roles_pct["role_k17000_v3"].tolist(),
        top_roles_pct["share_2024"].astype(float).tolist(),
        top_roles_pct["share_2025"].astype(float).tolist(),
        "2024",
        "2025",
        OUT_FIG / "ready_yoy_top_roles_percentages_2024_2025.svg",
        fmt="pct",
    )

    top_employers = pivot_top("employer_name", "ready_yoy_2024_2025_top_employers.csv", top_n=15)
    render_grouped_bars(
        "Top Employers: 2024 vs 2025",
        top_employers["employer_name"].tolist(),
        top_employers[2024].astype(float).tolist(),
        top_employers[2025].astype(float).tolist(),
        "2024",
        "2025",
        OUT_FIG / "ready_yoy_top_employers_2024_2025.svg",
    )

    target_yoy = yoy[yoy["target"] == 1].copy()
    major = target_yoy.pivot_table(index="major", columns="entry_job_year", values="user_id", aggfunc="size", fill_value=0)
    for year in [2024, 2025]:
        if year not in major.columns:
            major[year] = 0
    major = major[[2024, 2025]]
    major["total"] = major[2024] + major[2025]
    major = major.sort_values("total", ascending=False).head(10).reset_index()
    totals = target_yoy.groupby("entry_job_year")["user_id"].size()
    major["share_2024"] = major[2024] / max(1, int(totals.get(2024, 0)))
    major["share_2025"] = major[2025] / max(1, int(totals.get(2025, 0)))
    major.to_csv(OUT_TABLES / "ready_yoy_2024_2025_target_major_mix.csv", index=False)
    render_grouped_bars(
        "Target Major Mix: 2024 vs 2025",
        major["major"].tolist(),
        major["share_2024"].astype(float).tolist(),
        major["share_2025"].astype(float).tolist(),
        "2024",
        "2025",
        OUT_FIG / "ready_yoy_target_major_mix_2024_2025.svg",
        fmt="pct",
    )

    major_rate = yoy.groupby(["entry_job_year", "major"]).agg(users=("user_id", "size"), target_rate=("target", "mean")).reset_index()
    major_totals = yoy["major"].value_counts()
    keep_majors = major_totals[major_totals >= 100].head(10).index.tolist()
    major_rate = major_rate[major_rate["major"].isin(keep_majors)]
    rate_wide = major_rate.pivot(index="major", columns="entry_job_year", values="target_rate").fillna(0)
    for year in [2024, 2025]:
        if year not in rate_wide.columns:
            rate_wide[year] = 0
    rate_wide = rate_wide[[2024, 2025]].reset_index()
    rate_wide["avg_rate"] = (rate_wide[2024] + rate_wide[2025]) / 2
    rate_wide = rate_wide.sort_values("avg_rate", ascending=False)
    rate_wide.to_csv(OUT_TABLES / "ready_yoy_2024_2025_target_rate_by_major.csv", index=False)
    render_grouped_bars(
        "Target Rate by Major: 2024 vs 2025",
        rate_wide["major"].tolist(),
        rate_wide[2024].astype(float).tolist(),
        rate_wide[2025].astype(float).tolist(),
        "2024",
        "2025",
        OUT_FIG / "ready_yoy_target_rate_by_major_2024_2025.svg",
        fmt="pct",
        max_y=1.0,
    )

    report = [
        "# 2024 vs 2025 Landscape Shift",
        "",
        "## Scope",
        "- Uses the main analysis cohort: reviewed employers, seniority level 2 only, and overlapping education/profile/skills files.",
        "- This is a year-over-year comparison, not a full time-series analysis.",
        "",
        "## Target Share",
    ]
    for _, row in yearly.iterrows():
        report.append(f"- {int(row['entry_job_year'])}: {int(row['users']):,} users, {int(row['target_rows']):,} target, target share {row['target_rate']:.1%}")
    report.extend([
        "",
        "## Outputs",
        "- `ready_yoy_2024_2025_target_summary.csv`",
        "- `ready_yoy_2024_2025_top_roles.csv`",
        "- `ready_yoy_2024_2025_top_roles_percentages.csv`",
        "- `ready_yoy_2024_2025_top_employers.csv`",
        "- `ready_yoy_2024_2025_target_major_mix.csv`",
        "- `ready_yoy_2024_2025_target_rate_by_major.csv`",
        "",
        "Note: top-role percentages are each role's share of that year's full observed cohort. The displayed roles do not sum to 100% because only the top role categories are shown.",
    ])
    write_text(OUT_TABLES / "ready_yoy_2024_2025_report.md", "\n".join(report) + "\n")


def main() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    first_job = pd.read_csv(FIRST_JOB)
    first_job["employer_name"] = clean_series(first_job["ultimate_parent_company_name"].where(clean_series(first_job["ultimate_parent_company_name"]).ne("Missing"), first_job["company"]))
    first_job["company_norm"] = first_job["employer_name"].map(normalize_company)

    review = pd.read_csv(EMPLOYER_REVIEW)
    review["company_norm"] = review["company_name"].map(normalize_company)
    review["manual_decision"] = normalize_decision(review["manual_decision"])
    review = review[review["manual_decision"].isin(["0", "1"])].drop_duplicates("company_norm", keep="last")

    df = first_job.merge(review[["company_norm", "manual_decision", "suggested_bucket"]], on="company_norm", how="inner")
    df["target"] = df["manual_decision"].eq("1").astype(int)
    df["entry_job_year"] = pd.to_datetime(df["job_start_date"], errors="coerce").dt.year
    df["seniority_num"] = pd.to_numeric(df["seniority"], errors="coerce")

    edu = pd.read_csv(EDU, usecols=["user_id", "enddate", "university_name", "degree", "field", "university_country"])
    edu["enddate"] = pd.to_datetime(edu["enddate"], errors="coerce")
    latest_edu = edu.sort_values(["user_id", "enddate"]).groupby("user_id").tail(1)
    latest_edu = latest_edu.rename(columns={"university_name": "school", "field": "major"})

    profiles = pd.read_csv(PROFILES, usecols=["user_id", "prestige", "numconnections", "user_country"])
    skills = pd.read_csv(SKILL_AGG)

    df = df.merge(latest_edu[["user_id", "school", "major", "degree", "university_country"]], on="user_id", how="left")
    df = df.merge(profiles, on="user_id", how="left")
    df = df.merge(skills, on="user_id", how="left")
    for col in ["school", "major", "degree", "university_country", "user_country"]:
        df[col] = clean_series(df[col])
    df["prestige"] = pd.to_numeric(df["prestige"], errors="coerce")
    df["numconnections"] = pd.to_numeric(df["numconnections"], errors="coerce")
    df["n_skills"] = pd.to_numeric(df["n_skills"], errors="coerce").fillna(0)
    df["skill_text"] = df["skill_text"].fillna("")
    df["skill_set"] = df["skill_text"].map(split_skills)

    df.drop(columns=["skill_set"]).to_csv(OUT_TABLES / "ready_modeling_dataset_all_reviewed_seniority_1_2.csv", index=False)
    analysis_df = df[df["seniority_num"] == 2].copy()
    analysis_df.drop(columns=["skill_set"]).to_csv(OUT_TABLES / "ready_modeling_dataset.csv", index=False)
    build_eda_and_clouds(analysis_df)
    build_yoy_outputs(analysis_df)
    build_model_and_outputs(analysis_df)
    build_seniority_sensitivity_outputs(df)
    print("Wrote ready-analysis classification, EDA charts, skill word clouds, and seniority sensitivity checks.")


if __name__ == "__main__":
    main()
