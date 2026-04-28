from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd


BASE = Path(r"C:\Users\knich\Projects\career-skill-gap-analysis-STAT5710")
OUT_TABLES = BASE / "outputs" / "tables"
OUT_FIG = BASE / "outputs" / "figures"

COLORS = {
    "target": "#0f766e",
    "non_target": "#c05621",
    "blue": "#2b6cb0",
    "gold": "#d69e2e",
    "purple": "#6b46c1",
    "slate": "#475569",
    "bg": "#faf7f2",
    "ink": "#1f2937",
    "grid": "#e5ded2",
    "light_target": "#c6f6d5",
    "light_non_target": "#fbd38d",
}


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def render_horizontal_bar_chart(title: str, labels: list[str], values: list[float], colors: list[str], out: Path, fmt: str = "float") -> None:
    width = 1080
    row_h = 34
    top = 90
    left = 320
    right = 80
    bottom = 40
    height = top + bottom + row_h * len(labels)
    plot_w = width - left - right
    max_val = max(values) if values else 1.0

    parts = [svg_header(width, height), f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>']
    parts.append(f'<text x="{width/2}" y="38" text-anchor="middle" font-size="24" font-weight="700" fill="{COLORS["ink"]}">{escape(title)}</text>')

    for idx, (label, val, color) in enumerate(zip(labels, values, colors)):
        y = top + idx * row_h
        bar_w = 0 if max_val == 0 else plot_w * val / max_val
        val_label = f"{val:.3f}" if fmt == "float" else f"{val:,.0f}"
        parts.append(f'<text x="{left-12}" y="{y+20}" text-anchor="end" font-size="13" fill="{COLORS["ink"]}">{escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y+6}" width="{bar_w:.1f}" height="20" rx="5" fill="{color}"/>')
        parts.append(f'<text x="{left + bar_w + 8:.1f}" y="{y+21}" font-size="13" fill="{COLORS["ink"]}">{val_label}</text>')

    parts.append("</svg>")
    write_text(out, "\n".join(parts))


def render_confusion_matrix(cm: pd.DataFrame, out: Path) -> None:
    width, height = 760, 620
    cell = 150
    left = 210
    top = 150
    labels_x = ["Pred Non-Target", "Pred Target"]
    labels_y = ["Actual Non-Target", "Actual Target"]
    vals = cm.to_numpy()
    max_val = vals.max() if vals.size else 1

    parts = [svg_header(width, height), f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>']
    parts.append(f'<text x="{width/2}" y="44" text-anchor="middle" font-size="24" font-weight="700" fill="{COLORS["ink"]}">Reviewed-Only Confusion Matrix</text>')

    # axis labels
    for i, label in enumerate(labels_x):
        x = left + i * cell + cell / 2
        parts.append(f'<text x="{x}" y="{top-24}" text-anchor="middle" font-size="15" fill="{COLORS["ink"]}">{escape(label)}</text>')
    for i, label in enumerate(labels_y):
        y = top + i * cell + cell / 2 + 6
        parts.append(f'<text x="{left-18}" y="{y}" text-anchor="end" font-size="15" fill="{COLORS["ink"]}">{escape(label)}</text>')

    for r in range(2):
        for c in range(2):
            x = left + c * cell
            y = top + r * cell
            val = int(vals[r, c])
            strength = val / max_val if max_val else 0
            if r == c:
                fill = COLORS["light_target"] if r == 1 else "#dbeafe"
            else:
                fill = COLORS["light_non_target"] if c == 1 else "#fee2e2"
            opacity = 0.35 + 0.55 * strength
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" fill-opacity="{opacity:.2f}" stroke="{COLORS["grid"]}" stroke-width="2"/>')
            parts.append(f'<text x="{x + cell/2}" y="{y + cell/2 + 8}" text-anchor="middle" font-size="30" font-weight="700" fill="{COLORS["ink"]}">{val:,}</text>')

    parts.append("</svg>")
    write_text(out, "\n".join(parts))


def render_metric_bars(metrics: pd.DataFrame, out: Path) -> None:
    test = metrics.loc[metrics["split"] == "test"].iloc[0]
    labels = ["AUC", "Accuracy", "Precision", "Recall", "F1"]
    values = [float(test["auc"]), float(test["accuracy"]), float(test["precision"]), float(test["recall"]), float(test["f1"])]

    width, height = 860, 560
    left, right, top, bottom = 80, 40, 80, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    bar_w = plot_w / len(values) * 0.55
    gap = plot_w / len(values)

    parts = [svg_header(width, height), f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>']
    parts.append(f'<text x="{width/2}" y="38" text-anchor="middle" font-size="24" font-weight="700" fill="{COLORS["ink"]}">Reviewed-Only Test Metrics</text>')

    for i in range(6):
        y = top + plot_h * i / 5
        val = 1 - i / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="{COLORS["ink"]}">{val:.1f}</text>')

    palette = [COLORS["target"], COLORS["blue"], COLORS["gold"], COLORS["purple"], COLORS["slate"]]
    for idx, (label, val, color) in enumerate(zip(labels, values, palette)):
        x = left + gap * idx + (gap - bar_w) / 2
        h = plot_h * val
        y = top + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="6" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-size="14" fill="{COLORS["ink"]}">{val:.3f}</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{height-40}" text-anchor="middle" font-size="14" fill="{COLORS["ink"]}">{label}</text>')

    parts.append("</svg>")
    write_text(out, "\n".join(parts))


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    coef = pd.read_csv(OUT_TABLES / "reviewed_classification_feature_coefficients.csv")
    cm = pd.read_csv(OUT_TABLES / "reviewed_classification_confusion_matrix.csv", index_col=0)
    metrics = pd.read_csv(OUT_TABLES / "reviewed_classification_metrics.csv")

    top_coef = coef.head(8).copy()
    top_coef["direction"] = top_coef["coefficient"].apply(lambda x: "target" if x >= 0 else "non_target")
    top_colors = [COLORS["target"] if d == "target" else COLORS["non_target"] for d in top_coef["direction"]]
    render_horizontal_bar_chart(
        "Top Feature Importance (|Coefficient|)",
        top_coef["feature"].tolist(),
        top_coef["abs_coefficient"].tolist(),
        top_colors,
        OUT_FIG / "reviewed_classification_feature_importance.svg",
        fmt="float",
    )

    signed_coef = coef.head(8).copy()
    signed_coef["signed_magnitude"] = signed_coef["coefficient"].abs()
    signed_colors = [COLORS["target"] if x >= 0 else COLORS["non_target"] for x in signed_coef["coefficient"]]
    render_horizontal_bar_chart(
        "Top Signed Coefficients",
        signed_coef["feature"].tolist(),
        signed_coef["signed_magnitude"].tolist(),
        signed_colors,
        OUT_FIG / "reviewed_classification_signed_coefficients.svg",
        fmt="float",
    )

    render_confusion_matrix(cm, OUT_FIG / "reviewed_classification_confusion_matrix.svg")
    render_metric_bars(metrics, OUT_FIG / "reviewed_classification_test_metrics.svg")
    print("Wrote reviewed-only classification visuals.")


if __name__ == "__main__":
    main()
