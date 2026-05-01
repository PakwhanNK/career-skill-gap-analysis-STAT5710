from __future__ import annotations

import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


BASE = Path(r"C:\Users\knich\Projects\career-skill-gap-analysis-STAT5710")
MODEL_DATA = BASE / "outputs" / "tables" / "modeling_data" / "ready_modeling_dataset.csv"
OUT_TABLES = BASE / "outputs" / "tables" / "final"
OUT_FIG = BASE / "outputs" / "figures"

COLORS = {
    "positive": "#0f766e",
    "negative": "#c05621",
    "bg": "#faf7f2",
    "ink": "#1f2937",
    "grid": "#e5ded2",
}


def normalize_school_display(value: str) -> str:
    value = str(value or "").strip()
    if value == "Wharton School of Business at University of Pennsylvania":
        return "University of Pennsylvania"
    if value == "Duke University School of Nursing":
        return "Duke University"
    return value


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def normal_two_sided_p(z: float) -> float:
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def fit_logistic_inference(x: np.ndarray, y: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Small IRLS logistic model with ridge-stabilized covariance."""
    n, d = x.shape
    x_aug = np.column_stack([np.ones(n), x])
    beta = np.zeros(d + 1)
    ridge = 1e-5
    for _ in range(100):
        p = sigmoid(x_aug @ beta)
        weight = np.clip(p * (1 - p), 1e-6, None)
        grad = x_aug.T @ (y - p)
        hess = (x_aug.T * weight) @ x_aug + ridge * np.eye(d + 1)
        step = np.linalg.solve(hess, grad)
        beta += step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    p = sigmoid(x_aug @ beta)
    weight = np.clip(p * (1 - p), 1e-6, None)
    fisher = (x_aug.T * weight) @ x_aug + ridge * np.eye(d + 1)
    cov = np.linalg.pinv(fisher)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    z = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    return pd.DataFrame({
        "feature": ["Intercept"] + feature_names,
        "log_odds_coefficient": beta,
        "odds_ratio": np.exp(np.clip(beta, -20, 20)),
        "std_error": se,
        "z_value": z,
        "p_value": [normal_two_sided_p(float(v)) for v in z],
    })


def controls(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    out = []
    names = []
    for col in ["prestige", "numconnections", "entry_job_year"]:
        values = pd.to_numeric(df[col], errors="coerce")
        values = values.fillna(values.median())
        std = float(values.std(ddof=0))
        if std < 1e-9:
            std = 1.0
        out.append(((values - values.mean()) / std).to_numpy(dtype=float))
        names.append(f"{col}_z")
    return np.column_stack(out), names


def one_vs_rest(df: pd.DataFrame, col: str, min_users: int) -> pd.DataFrame:
    y = df["target"].to_numpy(dtype=float)
    base_controls, control_names = controls(df)
    rows = []
    for category, count in df[col].value_counts().sort_index().items():
        if int(count) < min_users:
            continue
        indicator = df[col].eq(category).astype(float).to_numpy()
        x = np.column_stack([indicator, base_controls])
        table = fit_logistic_inference(x, y, [f"{col}::{category}", *control_names])
        row = table.iloc[1].to_dict()
        in_group = df[col].eq(category)
        rows.append({
            col: category,
            "users": int(in_group.sum()),
            "target_users": int(df.loc[in_group, "target"].sum()),
            "target_rate": float(df.loc[in_group, "target"].mean()),
            "overall_target_rate": float(df["target"].mean()),
            "comparison": f"{category} vs all other {col} values",
            **row,
            "direction": "increases odds vs rest" if row["log_odds_coefficient"] > 0 else "decreases odds vs rest",
        })
    return pd.DataFrame(rows).sort_values("log_odds_coefficient", ascending=False)


def render_signed_bar(path: Path, title: str, df: pd.DataFrame, label_col: str) -> None:
    plot = pd.concat([df.head(10), df.tail(10)]).drop_duplicates(label_col)
    plot = plot.sort_values("log_odds_coefficient")
    width = 1250
    row_h = 30
    left = 500
    right = 120
    top = 90
    bottom = 45
    height = top + bottom + row_h * len(plot)
    plot_w = width - left - right
    max_abs = max(float(plot["log_odds_coefficient"].abs().max()), 0.1)
    zero_x = left + plot_w / 2
    scale = (plot_w / 2) / max_abs
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>',
        f'<text x="{width/2}" y="38" text-anchor="middle" font-size="24" font-weight="700" fill="{COLORS["ink"]}">{escape(title)}</text>',
        f'<line x1="{zero_x:.1f}" y1="{top-10}" x2="{zero_x:.1f}" y2="{height-bottom+8}" stroke="{COLORS["ink"]}" stroke-width="1"/>',
    ]
    for i, row in enumerate(plot.itertuples(index=False)):
        label = str(getattr(row, label_col))[:42]
        value = float(row.log_odds_coefficient)
        y = top + i * row_h
        if value >= 0:
            x = zero_x
            w = value * scale
            color = COLORS["positive"]
        else:
            x = zero_x + value * scale
            w = -value * scale
            color = COLORS["negative"]
        parts.append(f'<text x="{left-125}" y="{y+19}" text-anchor="end" font-size="13" fill="{COLORS["ink"]}">{escape(label)}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{y+5}" width="{w:.1f}" height="18" rx="5" fill="{color}"/>')
        text_x = x + w + 8 if value >= 0 else x - 8
        anchor = "start" if value >= 0 else "end"
        parts.append(f'<text x="{text_x:.1f}" y="{y+19}" text-anchor="{anchor}" font-size="13" fill="{COLORS["ink"]}">{value:+.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def top_bottom(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    return pd.concat([df.head(n), df.tail(n)]).drop_duplicates("feature")


def main() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MODEL_DATA)
    df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int)
    for col in ["school", "major", "user_country"]:
        df[col] = df[col].fillna("Missing").astype(str).str.strip().replace({"": "Missing", "nan": "Missing"})
    df["school"] = df["school"].map(normalize_school_display)

    school = one_vs_rest(df, "school", min_users=30)
    major = one_vs_rest(df, "major", min_users=50)
    country = one_vs_rest(df, "user_country", min_users=30)
    country_us = country[country["user_country"].eq("United States")].copy()

    school.to_csv(OUT_TABLES / "ready_non_skill_one_vs_rest_school_coefficients.csv", index=False)
    major.to_csv(OUT_TABLES / "ready_non_skill_one_vs_rest_major_coefficients.csv", index=False)
    country.to_csv(OUT_TABLES / "ready_non_skill_one_vs_rest_user_country_coefficients.csv", index=False)
    country_us.to_csv(OUT_TABLES / "ready_non_skill_user_country_united_states_coefficient.csv", index=False)

    render_signed_bar(OUT_FIG / "ready_non_skill_school_one_vs_rest_coefficients.svg", "School Coefficients: One-vs-Rest Non-Skill Logistic Models", school, "school")
    render_signed_bar(OUT_FIG / "ready_non_skill_school_top3_bottom3_coefficients.svg", "School Coefficients: Top 3 and Bottom 3", top_bottom(school, 3), "school")
    render_signed_bar(OUT_FIG / "ready_non_skill_major_one_vs_rest_coefficients.svg", "Major Coefficients: One-vs-Rest Non-Skill Logistic Models", major, "major")
    render_signed_bar(OUT_FIG / "ready_non_skill_user_country_one_vs_rest_coefficients.svg", "User Country Coefficients: One-vs-Rest Non-Skill Logistic Models", country, "user_country")

    print("Wrote one-vs-rest non-skill school, major, and user-country coefficient outputs.")


if __name__ == "__main__":
    main()
