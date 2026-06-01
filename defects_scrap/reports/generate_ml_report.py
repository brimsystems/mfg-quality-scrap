"""
generate_ml_report.py
Generates ml_report.html — monthly model performance report.
Pulls from MLflow artifacts and ml/data/scoring/ outputs.

Run from the reports/ directory:
    python3 generate_ml_report.py

Output: ml_report.html
"""

from pathlib import Path
import sys
import base64
import io
import json
from datetime import datetime

import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

# ── Paths ──────────────────────────────────────────────────────────────────
ML_DIR       = Path("../ml").resolve()
SCORING_DIR  = ML_DIR / "data" / "scoring"
OUTPUT       = Path("ml_report.html")
PERIOD_LABEL = "202601"
PERIOD_NAME  = "January – March 2026"

MLFLOW_TRACKING = str(ML_DIR / "mlruns")

# ── Palette — matches generate_report.py exactly ──────────────────────────
BRAND_BLUE = "#3D5166"
ACCENT     = "#6B8FA8"
LIGHT_BLUE = "#A8C0D1"
AMBER      = "#D4881E"
RED        = "#CC0000"
GREEN      = "#1A7A3A"
GREY       = "#AAAAAA"
DARK_GREY  = "#555555"
TEXT       = "#222222"
BOX_GREY   = "#DDDDDD"

# ── Chart sizing — matches generate_report.py ─────────────────────────────
CHART_W   = 8.2
CHART_H   = 3.8
CHART_H_T = 4.5
CHART_DPI = 130
BODY_FS   = 11
TITLE_FS  = 13

plt.rcParams.update({
    "figure.facecolor": "white",  "axes.facecolor": "white",
    "axes.edgecolor":   "#DDDDDD","axes.grid":       False,
    "font.family":      "sans-serif",
    "font.size":        BODY_FS,
    "axes.titlesize":   TITLE_FS, "axes.titleweight": "bold",
    "axes.labelsize":   BODY_FS,  "xtick.labelsize":  BODY_FS,
    "ytick.labelsize":  BODY_FS,  "legend.fontsize":  BODY_FS,
    "figure.dpi":       CHART_DPI,
})

def chart_style(ax):
    ax.yaxis.grid(True, color="#EEEEEE", linestyle="-", linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD")
    ax.spines["bottom"].set_color("#DDDDDD")

def make_fig(h=None):
    return plt.subplots(figsize=(CHART_W, h or CHART_H))

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=CHART_DPI)
    buf.seek(0)
    b = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b

def fmt_pct(x): return f"{x:.1%}"
def fmt_num(x): return f"{x:,.0f}"

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

print("Loading data...")

# MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING)
client = MlflowClient()

mv       = client.get_latest_versions("defect_risk_scorer", stages=["Production"])[0]
run      = client.get_run(mv.run_id)
run_id   = mv.run_id
model_type = run.data.tags.get("model_type", "XGBoost").title()
val_auc    = run.data.metrics.get("val_roc_auc", 0)
test_auc   = run.data.metrics.get("test_roc_auc")

# Scoring outputs
preds_path   = SCORING_DIR / f"predictions_{PERIOD_LABEL}.parquet"
summary_path = SCORING_DIR / f"accuracy_summary_{PERIOD_LABEL}.csv"

preds   = pd.read_parquet(preds_path)
summary = pd.read_csv(summary_path)

preds["actual_start"] = pd.to_datetime(preds["actual_start"])

# Parse SHAP drivers (stored as list of dicts)
def parse_drivers(drivers):
    if isinstance(drivers, str):
        try:
            return json.loads(drivers.replace("'", '"'))
        except Exception:
            return []
    return drivers if isinstance(drivers, list) else []

preds["shap_drivers_parsed"] = preds["shap_drivers"].apply(parse_drivers)

# Human-readable driver labels
DRIVER_LABELS = {
    "is_bending_shift_b":      "Bending machine on Shift B",
    "is_high_complexity":      "High complexity part",
    "is_supplier_c_thin_gauge":"Supplier C thin gauge material",
    "is_lapsed_cert_op":       "Operator with lapsed certification",
    "machine_type":            "Machine type",
    "shift_code":              "Shift assignment",
    "complexity":              "Part complexity",
    "supplier":                "Material supplier",
    "operator_id":             "Operator assignment",
    "machine_id":              "Machine assignment",
    "quantity_ordered":        "Production quantity",
    "machine_age_years":       "Machine age",
    "std_labor_hrs":           "Estimated labor hours",
    "lot_cert_status":         "Lot certification status",
    "material_type":           "Material type",
    "requires_welding":        "Requires welding",
    "schedule_variance_hrs":   "Schedule variance",
}

def friendly_driver(feature, direction):
    label = DRIVER_LABELS.get(feature, feature.replace("_", " ").title())
    arrow = "↑ increases risk" if direction == "increases_risk" else "↓ decreases risk"
    return f"{label} ({arrow})"

# ── Key stats ──────────────────────────────────────────────────────────────
total_scored    = len(preds)
actual_defect_rate = preds["actual_defect_flag"].mean()

high_row   = summary[summary["risk_tier"] == "High"].iloc[0]
med_row    = summary[summary["risk_tier"] == "Medium"].iloc[0]

high_flagged   = int(high_row["jobs_flagged"])
high_precision = float(high_row["precision"])
high_recall    = float(high_row["recall"])
high_tp        = int(high_row["true_positives"])
high_fp        = int(high_row["false_positives"])

med_flagged    = int(med_row["jobs_flagged"])
med_precision  = float(med_row["precision"])
med_recall     = float(med_row["recall"])
med_tp         = int(med_row["true_positives"])
med_fp         = int(med_row["false_positives"])

combined_flagged = high_flagged + med_flagged - sum(
    (preds["defect_probability"] >= float(high_row["threshold"])) &
    (preds["defect_probability"] >= float(med_row["threshold"]))
)

# ── Top flagged jobs for detail table ─────────────────────────────────────
top_flagged = (
    preds[preds["risk_tier"].isin(["High", "Medium"])]
    .sort_values("defect_probability", ascending=False)
    .head(25)
)

# ── Top 5 highest-risk jobs for plain-English SHAP panel ──────────────────
top5 = preds.nlargest(5, "defect_probability")

print("Data loaded.")

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def chart_risk_distribution():
    """Bar chart: job count by risk tier."""
    tiers  = ["High", "Medium", "Low"]
    colors = [RED, AMBER, GREEN]
    counts = [
        len(preds[preds["risk_tier"] == t]) for t in tiers
    ]
    fig, ax = make_fig()
    bars = ax.bar(tiers, counts, color=colors, width=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f"{count:,}\n({count/total_scored:.1%})",
                ha="center", va="bottom", fontsize=BODY_FS)
    ax.set_ylabel("Work Orders")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:,.0f}"))
    chart_style(ax)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_probability_distribution():
    """Histogram of predicted defect probabilities."""
    fig, ax = make_fig()
    ax.hist(preds["defect_probability"], bins=30,
            color=BRAND_BLUE, edgecolor="white", linewidth=0.5)
    ax.axvline(float(high_row["threshold"]), color=RED,
               linestyle="--", linewidth=1.5,
               label=f"High threshold ({high_row['threshold']})")
    ax.axvline(float(med_row["threshold"]), color=AMBER,
               linestyle="--", linewidth=1.5,
               label=f"Medium threshold ({med_row['threshold']})")
    ax.set_xlabel("Predicted Defect Probability")
    ax.set_ylabel("Work Orders")
    ax.legend()
    chart_style(ax)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_accuracy_by_tier():
    """Grouped bar: precision and recall by risk tier."""
    tiers     = ["High", "Medium"]
    precision = [high_precision, med_precision]
    recall    = [high_recall,    med_recall]
    x = np.arange(len(tiers))
    w = 0.35
    fig, ax = make_fig()
    bars_p = ax.bar(x - w/2, [v*100 for v in precision],
                    width=w, color=BRAND_BLUE, label="Precision")
    bars_r = ax.bar(x + w/2, [v*100 for v in recall],
                    width=w, color=ACCENT, label="Recall")
    for bars, vals in [(bars_p, precision), (bars_r, recall)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f"{val:.1%}", ha="center", va="bottom",
                    fontsize=BODY_FS, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("(%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))
    ax.set_ylim(0, 115)
    ax.legend()
    chart_style(ax)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_top_drivers():
    """Horizontal bar: most common top SHAP driver across flagged jobs."""
    flagged = preds[preds["risk_tier"].isin(["High","Medium"])]
    driver_counts = flagged["top_driver_feature"].value_counts().head(8)
    labels = [DRIVER_LABELS.get(f, f.replace("_"," ").title())
              for f in driver_counts.index]
    fig, ax = make_fig(h=CHART_H_T)
    y = np.arange(len(labels))
    ax.barh(y, driver_counts.values, color=BRAND_BLUE, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels[::-1] if False else labels)
    ax.set_xlabel("Number of Flagged Jobs")
    ax.xaxis.grid(True, color="#EEEEEE", linestyle="-", linewidth=0.8)
    ax.yaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD")
    ax.spines["bottom"].set_color("#DDDDDD")
    ax.set_axisbelow(True)
    # Invert so highest count is at top
    ax.invert_yaxis()
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_weekly_flags():
    """Stacked bar: High/Medium/Low job counts by week."""
    preds["week"] = preds["actual_start"].dt.to_period("W").apply(
        lambda p: p.start_time
    )
    weekly = preds.groupby(["week","risk_tier"]).size().unstack(fill_value=0)
    for t in ["High","Medium","Low"]:
        if t not in weekly.columns:
            weekly[t] = 0
    weekly = weekly[["High","Medium","Low"]].sort_index()
    weeks  = [str(w.date()) for w in weekly.index]
    x      = np.arange(len(weeks))
    fig, ax = make_fig(h=CHART_H_T)
    bottoms = np.zeros(len(weekly))
    for tier, color in [("Low",GREEN),("Medium",AMBER),("High",RED)]:
        vals = weekly[tier].values
        ax.bar(x, vals, bottom=bottoms, color=color, width=0.7, label=tier)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(weeks, rotation=30, ha="right", fontsize=BODY_FS-1)
    ax.set_ylabel("Work Orders")
    ax.legend(loc="upper right")
    chart_style(ax)
    plt.tight_layout()
    return fig_to_b64(fig)


print("Generating charts...")
charts = {
    "risk_distribution":   chart_risk_distribution(),
    "prob_distribution":   chart_probability_distribution(),
    "accuracy_by_tier":    chart_accuracy_by_tier(),
    "top_drivers":         chart_top_drivers(),
    "weekly_flags":        chart_weekly_flags(),
}
print("Charts complete.")

# ══════════════════════════════════════════════════════════════════════════════
# HTML HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def wrap(key, title="", caption=""):
    title_html   = f'<div class="chart-title">{title}</div>' if title else ""
    caption_html = f'<div class="chart-caption">{caption}</div>' if caption else ""
    return (f'<div class="chart-wrap">{title_html}'
            f'<img src="data:image/png;base64,{charts[key]}" '
            f'style="width:100%;height:auto;display:block;">'
            f'{caption_html}</div>')

def section_title(id, label, title):
    return f'''<div class="section-title-block" id="{id}">
      <div class="section-label">{label}</div>
      <h2 class="section-title">{title}</h2>
    </div>'''

def kpi_card(value, label, sub="", color=None):
    color = color or BRAND_BLUE
    return f'''<div class="kpi-card">
      <div class="kpi-value" style="color:{color};">{value}</div>
      <div class="kpi-label">{label}</div>
      {f'<div class="kpi-sub">{sub}</div>' if sub else ""}
    </div>'''

def kpi_row(*cards):
    inner = "".join(cards)
    return f'<div class="kpi-row">{inner}</div>'

def accuracy_table():
    rows = ""
    for _, row in summary.iterrows():
        tier        = row["risk_tier"]
        color       = RED if tier == "High" else AMBER
        flagged_pct = f"{row['pct_of_total']:.1%}"
        rows += f'''<tr>
          <td><span class="tier-badge" style="background:{color};">{tier}</span></td>
          <td style="text-align:right;">{row["threshold"]}</td>
          <td style="text-align:right;">{int(row["jobs_flagged"]):,}
              <span style="color:#999;font-size:12px;">({flagged_pct})</span></td>
          <td style="text-align:right;">{int(row["true_positives"]):,}</td>
          <td style="text-align:right;">{int(row["false_positives"]):,}</td>
          <td style="text-align:right;font-weight:700;">{row["precision"]:.1%}</td>
          <td style="text-align:right;">{row["recall"]:.1%}</td>
        </tr>'''
    return f'''<table class="data-table">
      <thead><tr>
        <th>Risk Tier</th><th>Threshold</th><th>Jobs Flagged</th>
        <th>True Positives</th><th>False Positives</th>
        <th>Precision</th><th>Recall</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>'''

def flagged_jobs_table():
    rows = ""
    for _, row in top_flagged.iterrows():
        tier    = row["risk_tier"]
        color   = RED if tier == "High" else AMBER
        drivers = parse_drivers(row["shap_drivers"])
        top_d   = drivers[0] if drivers else {}
        driver_label = DRIVER_LABELS.get(
            top_d.get("feature",""), 
            str(top_d.get("feature","—")).replace("_"," ").title()
        )
        outcome = row.get("actual_defect_flag", None)
        if outcome is None:
            outcome_html = '<span style="color:#999;">—</span>'
        elif outcome:
            outcome_html = f'<span style="color:{RED};font-weight:700;">Defective</span>'
        else:
            outcome_html = f'<span style="color:{GREEN};font-weight:700;">Clean</span>'

        rows += f'''<tr>
          <td style="font-family:monospace;font-size:13px;">{row[ID_COL]}</td>
          <td>{row["actual_start"].strftime("%b %d")}</td>
          <td><span class="tier-badge" style="background:{color};">{tier}</span></td>
          <td style="text-align:right;font-weight:700;">{row["defect_probability"]:.1%}</td>
          <td>{driver_label}</td>
          <td>{outcome_html}</td>
        </tr>'''
    return f'''<table class="data-table" style="font-size:14px;">
      <thead><tr>
        <th>Work Order</th><th>Date</th><th>Risk Tier</th>
        <th>Probability</th><th>Top Driver</th><th>Actual Outcome</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>'''

def shap_explanation_panel():
    """Plain-English explanation for the top 5 highest-risk jobs."""
    panels = ""
    for _, row in top5.iterrows():
        drivers = parse_drivers(row["shap_drivers"])
        tier    = row["risk_tier"]
        color   = RED if tier == "High" else AMBER

        driver_sentences = []
        for d in drivers[:3]:
            label = DRIVER_LABELS.get(
                d.get("feature",""),
                str(d.get("feature","")).replace("_"," ").title()
            )
            direction = "increases" if d.get("direction") == "increases_risk" else "decreases"
            driver_sentences.append(f"<strong>{label}</strong> {direction} the risk score")

        driver_text = "; ".join(driver_sentences) + "."

        outcome = row.get("actual_defect_flag", None)
        if outcome is None:
            outcome_html = ""
        elif outcome:
            outcome_html = f'<span style="color:{RED};font-size:13px;font-weight:600;">✗ Outcome: Defective</span>'
        else:
            outcome_html = f'<span style="color:{GREEN};font-size:13px;font-weight:600;">✓ Outcome: Clean</span>'

        panels += f'''<div class="shap-panel">
          <div class="shap-header">
            <span class="shap-wo">{row[ID_COL]}</span>
            <span class="tier-badge" style="background:{color};">{tier}</span>
            <span class="shap-prob">{row["defect_probability"]:.1%} probability</span>
            {outcome_html}
          </div>
          <p class="shap-text">{driver_text}</p>
        </div>'''
    return panels


# ── Column name for work order ID ──────────────────────────────────────────
ID_COL = "work_order_id"

# ══════════════════════════════════════════════════════════════════════════════
# HTML
# ══════════════════════════════════════════════════════════════════════════════

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Model Performance Report — {PERIOD_NAME}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
      background: #FFFFFF; color: {TEXT}; font-size: 16px; line-height: 1.7;
    }}
    .page-header {{ background: {BRAND_BLUE}; color: white; padding: 20px 40px; }}
    .page-header h1 {{ font-size: 22px; font-weight: 700; letter-spacing: -0.3px; }}
    .page-header .sub {{ font-size: 14px; opacity: 0.8; margin-top: 2px; }}
    .layout {{ display: flex; max-width: 1200px; margin: 0 auto; padding: 0 40px; }}

    .toc {{
      width: 200px; flex-shrink: 0; padding: 40px 20px 40px 0;
      position: sticky; top: 0; height: 100vh; overflow-y: auto;
      border-right: 1px solid #EEEEEE;
    }}
    .toc-title {{
      font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
      color: #AAAAAA; margin-bottom: 14px; font-weight: 600;
    }}
    .toc a {{
      display: block; font-size: 13px; color: #666; text-decoration: none;
      padding: 4px 0 4px 10px; border-left: 2px solid transparent; line-height: 1.4;
    }}
    .toc a:hover {{ color: {BRAND_BLUE}; border-left-color: {BRAND_BLUE}; }}
    .toc a.sub {{ font-size: 12px; padding-left: 20px; color: #AAAAAA; }}
    .toc a.sub:hover {{ color: {BRAND_BLUE}; border-left-color: {BRAND_BLUE}; }}
    .toc hr {{ border: none; border-top: 1px solid #EEEEEE; margin: 8px 0; }}

    .content {{ flex: 1; padding: 40px 0 80px 52px; max-width: 880px; }}

    .section-title-block {{
      margin: 48px 0 24px 0; padding-bottom: 12px;
      border-bottom: 2px solid {BRAND_BLUE};
    }}
    .content > .section-title-block:first-child {{ margin-top: 12px; }}
    .section-label {{
      font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
      color: {BRAND_BLUE}; font-weight: 600; margin-bottom: 4px;
    }}
    .section-title {{ font-size: 22px; font-weight: 700; color: {TEXT}; }}

    p {{ margin-bottom: 16px; color: #333; font-size: 16px; }}

    /* ── KPI cards ── */
    .kpi-row {{
      display: flex; gap: 16px; margin: 24px 0;
    }}
    .kpi-card {{
      flex: 1; border: 1px solid #DDDDDD; border-radius: 8px;
      padding: 20px 24px 16px 24px; background: white;
      border-bottom: 4px solid {BRAND_BLUE};
    }}
    .kpi-value {{ font-size: 32px; font-weight: 700; line-height: 1; margin-bottom: 6px; }}
    .kpi-label {{ font-size: 13px; color: #666; font-weight: 600;
                  text-transform: uppercase; letter-spacing: 0.5px; }}
    .kpi-sub   {{ font-size: 13px; color: #999; margin-top: 4px; }}

    /* ── Charts ── */
    .chart-title {{
      font-size: 17px; font-weight: 700; color: {TEXT};
      text-align: center; margin-bottom: 8px;
    }}
    .chart-wrap {{
      margin: 20px 0; border: 1px solid #EEEEEE; border-radius: 4px; padding: 12px;
    }}
    .chart-caption {{
      font-size: 12px; color: #888; margin-top: 8px;
      text-align: center; font-style: italic;
    }}

    /* ── Tables ── */
    .data-table {{
      width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 14px;
    }}
    .data-table th {{
      background: #F7F8FA; padding: 10px 12px; text-align: left;
      font-size: 12px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.5px; color: #666; border-bottom: 2px solid #EEEEEE;
    }}
    .data-table td {{
      padding: 9px 12px; border-bottom: 1px solid #F0F0F0; color: #333;
    }}
    .data-table tr:hover td {{ background: #FAFAFA; }}

    /* ── Risk tier badge ── */
    .tier-badge {{
      display: inline-block; padding: 2px 8px; border-radius: 3px;
      color: white; font-size: 11px; font-weight: 700;
      text-transform: uppercase; letter-spacing: 0.5px;
    }}

    /* ── SHAP panels ── */
    .shap-panel {{
      background: #F7F8FA; border-left: 3px solid {BRAND_BLUE};
      padding: 14px 18px; margin: 12px 0; border-radius: 0 4px 4px 0;
    }}
    .shap-header {{
      display: flex; align-items: center; gap: 10px; margin-bottom: 6px;
      flex-wrap: wrap;
    }}
    .shap-wo   {{ font-family: monospace; font-size: 13px; color: {DARK_GREY}; }}
    .shap-prob {{ font-size: 14px; font-weight: 700; color: {BRAND_BLUE}; }}
    .shap-text {{ font-size: 14px; color: #444; line-height: 1.6; margin: 0; }}

    /* ── Callout ── */
    .callout {{
      background: #F0F4F8; border-left: 3px solid {ACCENT};
      padding: 16px 22px; margin: 20px 0; font-size: 15px; color: #555;
    }}
    .callout strong {{ color: {TEXT}; }}
  </style>
</head>
<body>

<div class="page-header">
  <h1>Model Performance Report — Pre-Production Defect Risk Scorer</h1>
  <div class="sub">Scoring period: {PERIOD_NAME} &nbsp;·&nbsp;
       Model version {mv.version} &nbsp;·&nbsp;
       Generated {datetime.now().strftime("%B %d, %Y")}</div>
</div>

<div class="layout">
  <nav class="toc">
    <div class="toc-title">Contents</div>
    <a href="#summary">Executive Summary</a>
    <hr>
    <a href="#scoring">Scoring Summary</a>
    <a href="#flags" class="sub">Flagged Jobs</a>
    <a href="#drivers" class="sub">Why Jobs Were Flagged</a>
    <hr>
    <a href="#accuracy">Accuracy Retrospective</a>
  </nav>

  <main class="content">

    {section_title("summary", "Section 1", "Executive Summary")}

    <p>During {PERIOD_NAME}, the pre-production defect risk scorer evaluated
    <strong>{fmt_num(total_scored)}</strong> production work orders before they ran.
    The model flagged <strong>{fmt_num(high_flagged)}</strong> jobs as High risk
    ({fmt_pct(high_flagged/total_scored)} of all jobs) and
    <strong>{fmt_num(med_flagged)}</strong> as Medium risk
    ({fmt_pct(med_flagged/total_scored)} of all jobs). The actual defect rate
    across all scored jobs was <strong>{fmt_pct(actual_defect_rate)}</strong>.</p>

    <p>Of the {fmt_num(high_flagged)} High-risk flags,
    <strong>{high_tp}</strong> genuinely produced defective output —
    a precision rate of <strong>{fmt_pct(high_precision)}</strong>.
    {high_fp} flags were false alarms. At the Medium tier,
    <strong>{fmt_pct(med_precision)}</strong> of flags were correct.
    Every flag includes an explanation of the specific conditions that drove
    the risk score, enabling targeted pre-production intervention.</p>

    <p>The financial impact of quality failures extends beyond the direct cost
    of scrapped parts. Each defective run also consumes machine time that
    produced no good output, requires rework labor often at premium rates,
    disrupts downstream scheduling when jobs must be remade, and — where
    defects reach the customer — carries relationship costs that do not appear
    in any scrap log. The {fmt_num(high_tp)} High-risk jobs correctly identified
    by the model represent quality events where pre-production intervention
    could have reduced setup risk, allowed material substitution, or prompted
    operator reassignment before the first part was cut.</p>

    {kpi_row(
        kpi_card(fmt_num(total_scored), "Jobs Scored", PERIOD_NAME),
        kpi_card(fmt_num(high_flagged),
                 "High Risk Flags",
                 f"{fmt_pct(high_flagged/total_scored)} of jobs &nbsp;·&nbsp; {fmt_pct(high_precision)} precision",
                 RED),
        kpi_card(fmt_num(med_flagged),
                 "Medium Risk Flags",
                 f"{fmt_pct(med_flagged/total_scored)} of jobs &nbsp;·&nbsp; {fmt_pct(med_precision)} precision",
                 AMBER),
    )}

    {section_title("scoring", "Section 2", "Scoring Summary")}

    {wrap("risk_distribution", "Work Orders by Risk Tier",
          "Jobs are scored before production runs. High and Medium tier jobs are flagged for pre-production review.")}

    {wrap("prob_distribution", "Distribution of Predicted Defect Probabilities",
          "Vertical lines mark the High and Medium risk thresholds. Jobs to the right of each line are flagged at that tier.")}

    {wrap("weekly_flags", "Weekly Risk Tier Distribution",
          "Flagged job volume across the scoring period. Stable distribution indicates consistent model operation.")}

    {section_title("flags", "Section 2.1", "Flagged Jobs Detail")}

    <p>The table below shows the top {len(top_flagged)} highest-risk jobs from the scoring
    period, sorted by predicted defect probability. The Top Driver column identifies
    the single most influential factor in the model's risk assessment for each job.</p>

    {flagged_jobs_table()}

    {section_title("drivers", "Section 2.2", "Why Jobs Were Flagged")}

    {wrap("top_drivers", "Most Common Top Risk Driver Across Flagged Jobs",
          "The feature most responsible for each job's risk score, counted across all High and Medium tier flags.")}

    <p>The five highest-risk jobs this period and the factors driving their scores:</p>

    {shap_explanation_panel()}

    {section_title("accuracy", "Section 3", "Accuracy Retrospective")}

    <p>Because this scoring period covers historical data with known outcomes,
    we can evaluate how accurately the model's flags mapped to actual defect events.
    Precision measures what fraction of flags were correct; recall measures what
    fraction of all defective jobs were successfully flagged.</p>

    {accuracy_table()}

    {wrap("accuracy_by_tier", "Precision and Recall by Risk Tier")}

    <div class="callout">
      <strong>How to read these numbers:</strong> Precision answers "when the model
      raised a flag, how often was it right?" Recall answers "of all the jobs that
      actually failed, how many did the model catch?" High precision with lower recall
      means the model is conservative — it only flags jobs it is confident about,
      and misses some defective jobs that don't match the patterns it has learned.
      As more scoring periods accumulate, these metrics will be tracked over time
      to monitor whether model performance is stable or degrading.
    </div>

  </main>
</div>
</body>
</html>'''

OUTPUT.write_text(html, encoding="utf-8")
print(f"Report written to {OUTPUT.resolve()}")
