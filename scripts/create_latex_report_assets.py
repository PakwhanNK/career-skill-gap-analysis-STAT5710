from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas


BASE = Path(r"C:\Users\knich\Projects\career-skill-gap-analysis-STAT5710")
TABLES = BASE / "outputs" / "tables" / "final"
OUT = BASE / "outputs" / "report_figures_pdf"

PALETTE = {
    "target": colors.HexColor("#0f766e"),
    "non_target": colors.HexColor("#c05621"),
    "blue": colors.HexColor("#2b6cb0"),
    "gold": colors.HexColor("#d69e2e"),
    "purple": colors.HexColor("#6b46c1"),
    "green": colors.HexColor("#2f855a"),
    "ink": colors.HexColor("#1f2937"),
    "grid": colors.HexColor("#d8d2c8"),
    "bg": colors.HexColor("#faf7f2"),
}


def pct(x: float) -> str:
    return f"{x:.1%}"


def make_canvas(path: Path, title: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    w, h = landscape(letter)
    c.setFillColor(PALETTE["bg"])
    c.rect(0, 0, w, h, fill=1, stroke=0)
    c.setFillColor(PALETTE["ink"])
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(w / 2, h - 34, title)
    return c, w, h


def vertical_bar(path: Path, title: str, labels: list[str], values: list[float], value_fmt=str, max_y: float | None = None):
    c, w, h = make_canvas(path, title)
    left, bottom, top, right = 65, 85, 70, 35
    plot_w, plot_h = w - left - right, h - top - bottom
    max_val = max_y if max_y is not None else max(values) * 1.15 if values else 1
    bar_gap = plot_w / max(1, len(labels))
    bar_w = bar_gap * 0.55
    c.setStrokeColor(PALETTE["grid"])
    c.line(left, bottom, w - right, bottom)
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + bar_gap * i + (bar_gap - bar_w) / 2
        bh = 0 if max_val == 0 else plot_h * value / max_val
        c.setFillColor(PALETTE["blue"] if i % 2 == 0 else PALETTE["gold"])
        c.roundRect(x, bottom, bar_w, bh, 4, fill=1, stroke=0)
        c.setFillColor(PALETTE["ink"])
        c.setFont("Helvetica", 9)
        c.drawCentredString(x + bar_w / 2, bottom + bh + 8, value_fmt(value))
        c.saveState()
        c.translate(x + bar_w / 2, bottom - 12)
        c.rotate(-28)
        c.drawCentredString(0, 0, label[:28])
        c.restoreState()
    c.save()


def horizontal_bar(path: Path, title: str, labels: list[str], values: list[float], value_fmt=str):
    c, w, h = make_canvas(path, title)
    left, right, top, bottom = 250, 70, 70, 45
    row_h = min(26, (h - top - bottom) / max(1, len(labels)))
    max_val = max(values) if values else 1
    plot_w = w - left - right
    c.setFont("Helvetica", 9)
    for i, (label, value) in enumerate(zip(labels, values)):
        y = h - top - row_h * (i + 1)
        bw = 0 if max_val == 0 else plot_w * value / max_val
        c.setFillColor(PALETTE["ink"])
        c.drawRightString(left - 8, y + 7, label[:38])
        c.setFillColor(PALETTE["purple"])
        c.roundRect(left, y + 2, bw, row_h * 0.65, 4, fill=1, stroke=0)
        c.setFillColor(PALETTE["ink"])
        c.drawString(left + bw + 6, y + 7, value_fmt(value))
    c.save()


def stacked_percent_bar(path: Path, title: str, labels: list[str], target_values: list[float], non_target_values: list[float]):
    c, w, h = make_canvas(path, title)
    left, right, top, bottom = 265, 105, 75, 45
    row_h = min(23, (h - top - bottom) / max(1, len(labels)))
    plot_w = w - left - right
    c.setFont("Helvetica", 9)
    c.setFillColor(PALETTE["target"])
    c.rect(w - 210, h - 58, 12, 12, fill=1, stroke=0)
    c.setFillColor(PALETTE["ink"])
    c.drawString(w - 193, h - 56, "Target")
    c.setFillColor(PALETTE["non_target"])
    c.rect(w - 120, h - 58, 12, 12, fill=1, stroke=0)
    c.setFillColor(PALETTE["ink"])
    c.drawString(w - 103, h - 56, "Non-target")
    for i, (label, target, non_target) in enumerate(zip(labels, target_values, non_target_values)):
        y = h - top - row_h * (i + 1)
        target_w = plot_w * target
        non_target_w = plot_w * non_target
        c.setFillColor(PALETTE["ink"])
        c.drawRightString(left - 8, y + 6, label[:38])
        c.setFillColor(PALETTE["target"])
        c.roundRect(left, y + 2, target_w, row_h * 0.65, 3, fill=1, stroke=0)
        c.setFillColor(PALETTE["non_target"])
        c.roundRect(left + target_w, y + 2, non_target_w, row_h * 0.65, 3, fill=1, stroke=0)
        c.setFillColor(PALETTE["ink"])
        c.drawString(left + plot_w + 8, y + 6, pct(target))
    c.save()


def grouped_bar(path: Path, title: str, labels: list[str], values_a: list[float], values_b: list[float], label_a: str, label_b: str, value_fmt=str, max_y: float | None = None):
    c, w, h = make_canvas(path, title)
    left, bottom, top, right = 65, 100, 78, 35
    plot_w, plot_h = w - left - right, h - top - bottom
    max_val = max_y if max_y is not None else max(values_a + values_b) * 1.15 if values_a else 1
    gap = plot_w / max(1, len(labels))
    bar_w = gap * 0.23
    c.setFont("Helvetica", 9)
    c.setFillColor(PALETTE["blue"])
    c.rect(w - 170, h - 58, 12, 12, fill=1, stroke=0)
    c.setFillColor(PALETTE["ink"])
    c.drawString(w - 153, h - 56, label_a)
    c.setFillColor(PALETTE["gold"])
    c.rect(w - 90, h - 58, 12, 12, fill=1, stroke=0)
    c.setFillColor(PALETTE["ink"])
    c.drawString(w - 73, h - 56, label_b)
    c.setStrokeColor(PALETTE["grid"])
    c.line(left, bottom, w - right, bottom)
    for i, label in enumerate(labels):
        cx = left + gap * (i + 0.5)
        for value, dx, color in [(values_a[i], -bar_w * 0.65, PALETTE["blue"]), (values_b[i], bar_w * 0.65, PALETTE["gold"])]:
            bh = 0 if max_val == 0 else plot_h * value / max_val
            x = cx + dx - bar_w / 2
            c.setFillColor(color)
            c.roundRect(x, bottom, bar_w, bh, 3, fill=1, stroke=0)
            c.setFillColor(PALETTE["ink"])
            c.drawCentredString(x + bar_w / 2, bottom + bh + 6, value_fmt(value))
        c.saveState()
        c.translate(cx, bottom - 12)
        c.rotate(-28)
        c.drawCentredString(0, 0, label[:24])
        c.restoreState()
    c.save()


def scatter(path: Path, title: str, scores_path: Path, group_col: str, group_names: dict[int, str]):
    scores = pd.read_csv(scores_path)
    if len(scores) > 5000:
        scores = scores.sample(5000, random_state=5710)
    c, w, h = make_canvas(path, title)
    left, right, top, bottom = 75, 140, 75, 70
    plot_w, plot_h = w - left - right, h - top - bottom
    x = scores["PC1"].astype(float)
    y = scores["PC2"].astype(float)
    x_min, x_max = x.quantile(0.01), x.quantile(0.99)
    y_min, y_max = y.quantile(0.01), y.quantile(0.99)
    if abs(x_max - x_min) < 1e-9:
        x_min, x_max = x_min - 1, x_max + 1
    if abs(y_max - y_min) < 1e-9:
        y_min, y_max = y_min - 1, y_max + 1
    palette = [PALETTE["non_target"], PALETTE["target"], PALETTE["blue"], PALETTE["gold"], PALETTE["purple"], PALETTE["green"]]
    c.setFillColor(colors.white)
    c.rect(left, bottom, plot_w, plot_h, fill=1, stroke=0)
    c.setStrokeColor(PALETTE["grid"])
    c.rect(left, bottom, plot_w, plot_h, fill=0, stroke=1)
    for _, row in scores.iterrows():
        xi = min(max(float(row["PC1"]), x_min), x_max)
        yi = min(max(float(row["PC2"]), y_min), y_max)
        px = left + (xi - x_min) / (x_max - x_min) * plot_w
        py = bottom + (yi - y_min) / (y_max - y_min) * plot_h
        group = int(row[group_col])
        c.setFillColor(palette[group % len(palette)])
        c.circle(px, py, 2.6, fill=1, stroke=0)
    c.setFillColor(PALETTE["ink"])
    c.setFont("Helvetica", 10)
    c.drawCentredString(left + plot_w / 2, 35, "PC1")
    c.saveState()
    c.translate(25, bottom + plot_h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "PC2")
    c.restoreState()
    for i, (group, label) in enumerate(group_names.items()):
        yy = h - top - 20 - i * 20
        c.setFillColor(palette[int(group) % len(palette)])
        c.circle(w - right + 30, yy + 4, 5, fill=1, stroke=0)
        c.setFillColor(PALETTE["ink"])
        c.drawString(w - right + 42, yy, label)
    c.save()


def line_chart(path: Path, title: str, labels: list[str], values: list[float], value_fmt=str):
    c, w, h = make_canvas(path, title)
    left, right, top, bottom = 75, 45, 75, 70
    plot_w, plot_h = w - left - right, h - top - bottom
    min_val, max_val = min(values), max(values)
    if abs(max_val - min_val) < 1e-9:
        min_val -= 1
        max_val += 1
    c.setFillColor(colors.white)
    c.rect(left, bottom, plot_w, plot_h, fill=1, stroke=0)
    c.setStrokeColor(PALETTE["grid"])
    c.rect(left, bottom, plot_w, plot_h, fill=0, stroke=1)
    points = []
    for i, value in enumerate(values):
        x = left + plot_w * i / max(1, len(values) - 1)
        y = bottom + plot_h * (value - min_val) / (max_val - min_val)
        points.append((x, y, value))
    c.setStrokeColor(PALETTE["blue"])
    c.setLineWidth(2)
    for (x1, y1, _), (x2, y2, _) in zip(points, points[1:]):
        c.line(x1, y1, x2, y2)
    c.setFillColor(PALETTE["blue"])
    c.setFont("Helvetica", 9)
    for label, (x, y, value) in zip(labels, points):
        c.circle(x, y, 4, fill=1, stroke=0)
        c.setFillColor(PALETTE["ink"])
        c.drawCentredString(x, bottom - 20, label)
        c.drawCentredString(x, y + 10, value_fmt(value))
        c.setFillColor(PALETTE["blue"])
    c.setFillColor(PALETTE["ink"])
    c.drawCentredString(left + plot_w / 2, 32, "Number of clusters (k)")
    c.save()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    counts = pd.read_csv(TABLES / "ready_eda_target_vs_nontarget_counts.csv")
    vertical_bar(OUT / "target_vs_non_target_counts.pdf", "Reviewed Target vs Non-Target Counts", counts["group"].tolist(), counts["users"].tolist(), lambda x: f"{int(x):,}")

    schools = pd.read_csv(TABLES / "ready_eda_top20_target_schools.csv").head(12)
    horizontal_bar(OUT / "top_target_schools.pdf", "Top Reviewed Target Schools", schools["school"].tolist(), schools["users"].tolist(), lambda x: f"{int(x):,}")

    school_pct = pd.read_csv(TABLES / "ready_eda_school_target_non_target_percentages.csv").head(20)
    stacked_percent_bar(
        OUT / "school_target_non_target_percentages.pdf",
        "Target vs Non-Target Share by School",
        school_pct["school"].tolist(),
        school_pct["target_share"].tolist(),
        school_pct["non_target_share"].tolist(),
    )

    majors = pd.read_csv(TABLES / "ready_eda_target_major_pie.csv")
    horizontal_bar(OUT / "target_major_mix.pdf", "Target Major Mix", majors["major"].tolist(), majors["share"].tolist(), pct)

    model_cmp = pd.read_csv(TABLES / "ready_classification_model_comparison.csv").dropna(subset=["test_auc"])
    vertical_bar(OUT / "model_comparison_auc.pdf", "Classification Model Comparison: AUC", model_cmp["model"].tolist(), model_cmp["test_auc"].tolist(), lambda x: f"{x:.3f}", max_y=1)

    cluster = pd.read_csv(TABLES / "ready_skill_cluster_profiles.csv").sort_values("cluster")
    horizontal_bar(OUT / "skill_cluster_target_rates.pdf", "Reviewed Target Rate by Skill Cluster", [f"Cluster {int(x)}" for x in cluster["cluster"]], cluster["target_rate"].tolist(), pct)

    elbow = pd.read_csv(TABLES / "ready_skill_kmeans_elbow.csv")
    line_chart(
        OUT / "skill_kmeans_elbow.pdf",
        "K-Means Elbow: Skill PCA Components",
        elbow["k"].astype(str).tolist(),
        elbow["inertia_pct_of_k2"].tolist(),
        pct,
    )

    yoy_roles = pd.read_csv(TABLES / "ready_yoy_2024_2025_top_roles_percentages.csv").head(10)
    grouped_bar(OUT / "top_role_percentages_2024_2025.pdf", "Top Role Categories as % of Each Year's Observed Cohort", yoy_roles["role_k17000_v3"].tolist(), yoy_roles["share_2024"].tolist(), yoy_roles["share_2025"].tolist(), "2024", "2025", pct)

    skill_shift = pd.read_csv(TABLES / "ready_yoy_2024_2025_target_skill_shift.csv")
    skill_shift = pd.concat([skill_shift.head(6), skill_shift.tail(6)]).drop_duplicates("skill")
    grouped_bar(OUT / "target_skill_shares_2024_2025.pdf", "Target Skill Shares: 2024 vs 2025", skill_shift["skill"].tolist(), skill_shift["share_2024"].tolist(), skill_shift["share_2025"].tolist(), "2024", "2025", pct)

    scatter(
        OUT / "skill_pca_target_vs_non_target.pdf",
        "Skill PCA: Reviewed Target vs Non-Target",
        TABLES / "ready_skill_pca_scores.csv",
        "target",
        {0: "Non-target", 1: "Target"},
    )
    cluster_profiles = pd.read_csv(TABLES / "ready_skill_cluster_profiles.csv").sort_values("cluster")
    cluster_names = {int(row["cluster"]): f"Cluster {int(row['cluster'])}" for _, row in cluster_profiles.iterrows()}
    scatter(
        OUT / "skill_clusters_in_pca_space.pdf",
        "Skill Clusters in PCA Space",
        TABLES / "ready_skill_pca_scores.csv",
        "skill_cluster",
        cluster_names,
    )

    print(f"Wrote PDF report figures to {OUT}")


if __name__ == "__main__":
    main()
