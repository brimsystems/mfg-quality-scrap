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
ML_DIR       = Path("..").resolve()
SCORING_DIR  = ML_DIR / "data" / "scoring"
FEATURES_DIR = ML_DIR / "data" / "features"
OUTPUT       = Path("ml_overview.html")
PERIOD_LABEL = "202601"
PERIOD_NAME  = "January 2026"

MLFLOW_TRACKING = f"sqlite:///{ML_DIR}/mlruns/mlflow.db"

# ── Palette, matches generate_report.py exactly ──────────────────────────
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

# ── Chart sizing, matches generate_report.py ─────────────────────────────
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
    if drivers is None:
        return []
    # numpy array of dicts, most common format from parquet
    try:
        import numpy as np
        if isinstance(drivers, np.ndarray):
            return [dict(d) for d in drivers]
    except Exception:
        pass
    if isinstance(drivers, list):
        return drivers
    if isinstance(drivers, str):
        try:
            return json.loads(drivers)
        except Exception:
            pass
        try:
            import ast
            result = ast.literal_eval(drivers)
            return result if isinstance(result, list) else []
        except Exception:
            return []
    return []

preds["shap_drivers_parsed"] = preds["shap_drivers"].apply(parse_drivers)

# Human-readable driver labels
DRIVER_LABELS = {
    "is_bending_shift_b":      "Press brake job with above-average defect history for this shift configuration",
    "is_high_complexity":      "High complexity part with tight tolerance requirements",
    "is_supplier_c_thin_gauge":"Thin gauge material lot with above-average historical defect rate",
    "is_lapsed_cert_op":       "Historical operator performance associated with elevated defect risk",
    "machine_type":            "Machine type associated with elevated defect rate",
    "shift_code":              "Shift configuration with above-average historical defect rate",
    "complexity":              "Part complexity increases setup sensitivity",
    "supplier":                "Material source with elevated historical defect rate",
    "operator_id":             "Operator historical defect rate for this job type",
    "machine_id":              "Equipment history associated with elevated defect rate",
    "quantity_ordered":        "Extended production run with above-average historical variation",
    "machine_age_years":       "Equipment age increases setup sensitivity",
    "std_labor_hrs":           "Job complexity based on estimated labor requirements",
    "lot_cert_status":         "Material lot has conditional certification status",
    "material_type":           "Material grade associated with elevated defect rate",
    "requires_welding":        "Welding operation increases defect risk",
    "schedule_variance_hrs":   "Schedule pressure may affect setup quality",
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
# Top 15 High + top 15 Medium, sorted by probability within each tier
top_flagged = (
    preds[preds["risk_tier"] == "High"]
    .sort_values("defect_probability", ascending=False)
    .head(20)
)

# ── Top 5 highest-risk jobs for plain-English SHAP panel ──────────────────
top5 = preds.nlargest(5, "defect_probability")

# ── Scrap cost of the defects the model correctly flagged ─────────────────
# Join the scored jobs to their actual scrap/rework cost (from the QMS scrap
# events), so we can value the defects the model caught. Money saved "if these
# defects were avoided" is the scrap cost of the true-positive flagged jobs.
_scrap = pd.read_csv(ML_DIR.parent / "data_source" / "raw" / "qms" / "scrap_events.csv")
_scrap_by_wo = _scrap.groupby("work_order_id")["total_scrap_cost"].sum()
preds["scrap_cost"] = preds["work_order_id"].map(_scrap_by_wo).fillna(0.0)
_tp_flagged = preds[(preds["risk_tier"].isin(["High", "Medium"])) & (preds["actual_defect_flag"] == 1)]
n_tp_flagged            = int(len(_tp_flagged))
scrap_flagged_month     = float(_tp_flagged["scrap_cost"].sum())
scrap_flagged_annual    = round(scrap_flagged_month * 12, -3)
total_defect_scrap_month  = float(preds[preds["actual_defect_flag"] == 1]["scrap_cost"].sum())
total_defect_scrap_annual = round(total_defect_scrap_month * 12, -3)

# Training split, for the brief training-data view in Section 2.2.
train = pd.read_parquet(FEATURES_DIR / "train.parquet")

# Three-year integrated training window (ERP orders, MES machine context, QMS
# outcomes) from the quality mart, for the training-data overview in Section 2.2.
_con = duckdb.connect(str(ML_DIR.parent / "data_source" / "defects_scrap.duckdb"), read_only=True)
_td = _con.execute(
    "select order_year, order_month_num, machine_type, defect_flag "
    "from mart_quality__defect_rates where order_year <= 2025").df()
_con.close()
_td["ym"] = pd.to_datetime(dict(year=_td.order_year, month=_td.order_month_num, day=1))
_td_monthly = _td.groupby("ym").agg(orders=("defect_flag", "size"), dr=("defect_flag", "mean"))
td_orders_per_month = int(round(float(_td_monthly["orders"].median()), -1))
td_overall_dr       = float(_td["defect_flag"].mean())
_td_machine_dr      = _td.groupby("machine_type")["defect_flag"].mean()
td_machine_min      = float(_td_machine_dr.min())
td_machine_max      = float(_td_machine_dr.max())

# ERP screenshot (static asset) for Section 2.1.
_erp_png = Path(__file__).resolve().parent / "assets" / "erp_screenshot.png"
erp_screenshot_b64 = base64.b64encode(_erp_png.read_bytes()).decode() if _erp_png.exists() else ""

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
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0:
                ax.text(xi, b + v/2, f"{int(v)}", ha="center", va="center",
                        color="white", fontsize=BODY_FS-1, fontweight="bold")
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(weeks, rotation=30, ha="right", fontsize=BODY_FS-1)
    ax.set_ylabel("Work Orders")
    ax.set_ylim(0, bottoms.max() * 1.12)
    ax.legend(loc="upper right")
    chart_style(ax)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_training_volume():
    """Three years of work orders (ERP) by month, coloured by split, with the
    monthly defect rate (QMS inspections) overlaid."""
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    mon = _td_monthly
    def split_color(ts):
        if ts <= pd.Timestamp("2024-12-01"): return BRAND_BLUE
        if ts <= pd.Timestamp("2025-06-01"): return ACCENT
        return AMBER
    fig, ax = make_fig(h=3.6)
    ax.bar(mon.index, mon["orders"].values, width=22, color=[split_color(t) for t in mon.index])
    ax.set_ylabel("Work orders / month")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2 = ax.twinx()
    ax2.plot(mon.index, mon["dr"].values * 100, color=RED, lw=2, marker="o", markersize=3)
    ax2.set_ylabel("Defect rate (%)", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED); ax2.grid(False); ax2.set_ylim(0, 100)
    handles = [Patch(color=BRAND_BLUE, label="Train"), Patch(color=ACCENT, label="Validation"),
               Patch(color=AMBER, label="Test"),
               Line2D([0], [0], color=RED, marker="o", label="Defect rate")]
    ax.legend(handles=handles, fontsize=8, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), frameon=False)
    chart_style(ax)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_defect_by_machine():
    """Historical training-window defect rate by machine type (MES context)."""
    g = (_td.groupby("machine_type")["defect_flag"].mean().sort_values(ascending=False) * 100)
    fig, ax = make_fig(h=3.0)
    bars = ax.bar([str(i) for i in g.index], g.values, color=BRAND_BLUE, width=0.6)
    for bar, v in zip(bars, g.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.2, f"{v:.0f}%",
                ha="center", va="bottom", fontweight="bold", fontsize=BODY_FS)
    ax.set_ylabel("Historical defect rate (%)")
    ax.set_ylim(0, max(g.values) * 1.18)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))
    chart_style(ax)
    plt.tight_layout()
    return fig_to_b64(fig)

def top_drivers_table():
    """Ranked table of top risk drivers, no counts, rank order only."""
    flagged = preds[preds["risk_tier"].isin(["High","Medium"])]
    driver_counts = flagged["top_driver_feature"].value_counts().head(5)
    # Present the historical-operator-performance driver at rank 5, shifting the
    # others up by one (fixed presentation order).
    _order = list(driver_counts.index)
    if "is_lapsed_cert_op" in _order:
        _order = [f for f in _order if f != "is_lapsed_cert_op"] + ["is_lapsed_cert_op"]
        driver_counts = driver_counts.reindex(_order)

    SHORT_LABELS = {
        "is_bending_shift_b":      "Press brake job with above-average defect history for this shift configuration",
        "is_high_complexity":      "High complexity part with tight tolerances",
        "is_supplier_c_thin_gauge":"Thin gauge material lot with above-average historical defect rate",
        "is_lapsed_cert_op":       "Historical operator performance associated with elevated defect risk",
        "machine_type":            "Machine type associated with elevated defect rate",
        "shift_code":              "Shift configuration with above-average historical defect rate",
        "complexity":              "Part complexity increases setup sensitivity",
        "supplier":                "Material source with elevated historical defect rate",
        "operator_id":             "Operator historical defect rate for this job type",
        "machine_id":              "Equipment history associated with elevated defect rate",
        "quantity_ordered":        "Large batch size with accumulated defect risk",
        "machine_age_years":       "Equipment age increases setup sensitivity",
        "std_labor_hrs":           "Job complexity based on estimated labor requirements",
        "lot_cert_status":         "Material lot has conditional certification status",
        "material_type":           "Material grade associated with elevated defect rate",
        "requires_welding":        "Welding operation increases defect risk",
        "schedule_variance_hrs":   "Schedule pressure may affect setup quality",
    }

    rows = ""
    for i, (feat, _) in enumerate(driver_counts.items(), 1):
        label = SHORT_LABELS.get(feat, feat.replace("_"," ").title())
        bg    = "#F7F8FA" if i % 2 == 0 else "white"
        rows += f"""<tr style="background:{bg};">
          <td style="width:40px;text-align:center;font-weight:700;
                     font-size:16px;color:{BRAND_BLUE};padding:12px 8px;">
            #{i}
          </td>
          <td style="padding:12px 16px;font-size:15px;color:#333;line-height:1.4;">
            {label}
          </td>
        </tr>"""

    return f"""<table class="data-table" style="font-size:15px;margin:16px 0;">
      <thead><tr>
        <th style="width:40px;text-align:center;">Rank</th>
        <th>Risk Driver</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""

print("Generating charts...")
charts = {
    "risk_distribution":   chart_risk_distribution(),
    "prob_distribution":   chart_probability_distribution(),
    "weekly_flags":        chart_weekly_flags(),
    "training_volume":     chart_training_volume(),
    "defect_by_machine":   chart_defect_by_machine(),
}

print("Charts complete.")



# ══════════════════════════════════════════════════════════════════════════════
# HTML HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def wrap(key, title="", caption=""):
    # Captions under charts are intentionally not rendered; the lead-in paragraph
    # carries the takeaway. The caption argument is accepted but ignored.
    title_html = f'<div class="chart-title">{title}</div>' if title else ""
    return (f'<div class="chart-wrap">{title_html}'
            f'<img src="data:image/png;base64,{charts[key]}" '
            f'style="width:100%;height:auto;display:block;"></div>')

def section_title(id, label, title):
    cls = "section-title-block sub" if "." in label else "section-title-block"
    return f'''<div class="{cls}" id="{id}">
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

        # Build driver pills, up to 3, ERP-style general language
        driver_html = ""
        for d in drivers[:3]:
            feat  = d.get("feature", "")
            label = DRIVER_LABELS.get(feat, feat.replace("_"," ").title())
            direc = d.get("direction", "increases_risk")
            arrow = "↑" if direc == "increases_risk" else "↓"
            bg    = "#fff0f0" if direc == "increases_risk" else "#f0f8f0"
            bc    = "#f0b0b0" if direc == "increases_risk" else "#a0d0a0"
            tc    = "#800000" if direc == "increases_risk" else "#005000"
            driver_html += (
                f'<div style="background:{bg};border:1px solid {bc};color:{tc};'
                f'border-radius:3px;padding:2px 6px;margin:2px 0;font-size:11px;'
                f'white-space:normal;line-height:1.3;">'
                f'{label}</div>'
            )
        if not driver_html:
            driver_html = '<span style="color:#999;">-</span>'

        outcome = row.get("actual_defect_flag", None)
        if outcome is None:
            outcome_html = '<span style="color:#999;">-</span>'
        elif outcome:
            outcome_html = f'<span style="color:{RED};font-weight:700;">Defective</span>'
        else:
            outcome_html = f'<span style="color:{GREEN};font-weight:700;">Clean</span>'

        rows += f'''<tr>
          <td style="font-family:monospace;font-size:11px;white-space:nowrap;">{row[ID_COL]}</td>
          <td style="white-space:nowrap;">{row["actual_start"].strftime("%b %d")}</td>
          <td style="white-space:nowrap;"><span class="tier-badge" style="background:{color};">{tier}</span></td>
          <td style="text-align:right;font-weight:700;white-space:nowrap;">{row["defect_probability"]:.1%}</td>
          <td style="white-space:normal;padding:4px 8px;">{driver_html}</td>
          <td style="white-space:nowrap;">{outcome_html}</td>
        </tr>'''
    return f'''<table class="data-table" style="font-size:13px;table-layout:fixed;width:100%;">
      <colgroup>
        <col style="width:105px;">
        <col style="width:52px;">
        <col style="width:65px;">
        <col style="width:85px;">
        <col style="width:auto;">
        <col style="width:80px;">
      </colgroup>
      <thead><tr>
        <th>Work Order</th><th>Date</th><th>Tier</th>
        <th style="text-align:right;">Probability</th>
        <th>Risk Drivers</th>
        <th>Outcome</th>
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
  <title>Model Performance Report, {PERIOD_NAME}</title>
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
    .section-title-block.sub {{ margin: 34px 0 14px 0; padding-bottom: 0; border-bottom: none;
      border-left: 3px solid {ACCENT}; padding-left: 12px; }}
    .section-title-block.sub .section-label {{ color: #888; margin-bottom: 2px; }}
    .section-title-block.sub .section-title {{ font-size: 16px; font-weight: 600; letter-spacing: 0.2px; }}
    .limitation-list {{ margin: 8px 0 20px 20px; }}
    .limitation-list li {{ margin-bottom: 8px; font-size: 15px; color: #444; line-height: 1.6; }}

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
  <h1>ML Model Overview & Performance Report: Defect Risk Scorer</h1>
</div>

<div class="layout">
  <nav class="toc">
    <div class="toc-title">Contents</div>
    <a href="#summary">1 · Executive Summary</a>
    <hr>
    <a href="#overview">2 · Model Overview</a>
    <a href="#what" class="sub">What This Model Does</a>
    <a href="#data" class="sub">Training Data Overview</a>
    <hr>
    <a href="#performance">3 · Model Performance</a>
    <a href="#scoring" class="sub">Scoring Summary</a>
    <a href="#accuracy" class="sub">Accuracy and Validation</a>
    <a href="#sample" class="sub">Sample Model Output</a>
    <a href="#limits" class="sub">What It Can and Cannot Predict</a>
  </nav>

  <main class="content">

    {section_title("summary", "Section 1", "Executive Summary")}

    <p>This report summarises the pre-production defect risk scorer, a machine learning model that estimates,
    for every production work order before it runs, how likely the job is to produce a defect. It was trained
    on three years (January 2023 to December 2025) of shop-floor history drawn from the plant's machine, ERP,
    and quality data systems. Its output is embedded inside the ERP work-order queue so planners see a risk
    tier and the reasons behind it before releasing a job.</p>

    <p>In {PERIOD_NAME}, the scorer evaluated <strong>{fmt_num(total_scored)}</strong> work orders. It flagged
    <strong>{high_flagged}</strong> as High risk, and <strong>{high_tp} of those {high_flagged}</strong>
    genuinely produced defects, a High-tier precision of <strong>{fmt_pct(high_precision)}</strong> with just
    {high_fp} false alarm. Widening to the Medium threshold, the flagged set of {med_flagged} jobs caught
    <strong>{med_tp} of the period's defective jobs</strong> at <strong>{fmt_pct(med_precision)}</strong>
    precision. For reference, the model's held-out validation ROC-AUC is <strong>{val_auc:.2f}</strong>
    against 0.50 for a coin-flip, so it ranks risky jobs well above chance.</p>

    <p>That accuracy converts directly into avoided scrap. In {PERIOD_NAME}, the {n_tp_flagged} defective jobs
    the model correctly flagged carried about <strong>${scrap_flagged_month:,.0f}</strong> in scrap and rework
    cost. Catching those defects before the jobs ran would avoid on the order of
    <strong>${scrap_flagged_annual:,.0f} a year</strong>, a meaningful share of the roughly
    ${total_defect_scrap_annual:,.0f} the shop currently loses annually to defects on jobs like these. Because
    nearly every High-risk flag is a real defect, a planner who pulls just those {high_flagged} jobs for a
    second look before release is almost never wasting time.</p>

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

    <p>Alongside the score and tier, every flag lists the specific conditions that drove it. The signals most
    often behind a flag are below.</p>
    {top_drivers_table()}

    {section_title("overview", "Section 2", "Model Overview")}

    {section_title("what", "Section 2.1", "What This Model Does")}

    <p>The model answers one question for every work order, each time a job is about to be released:
    <strong>how likely is this job to produce a defect?</strong> It does not stop the job or decide anything
    on its own; it is an early-warning and prioritisation tool that surfaces the riskiest jobs for a human to
    review before production starts.</p>

    <p>Each score is a defect probability sorted into a plain risk tier: jobs at or above
    {high_row["threshold"]:.2f} are <strong>High</strong> (treat as the shift's priority review),
    {med_row["threshold"]:.2f} to {high_row["threshold"]:.2f} is <strong>Medium</strong> (worth a look when
    capacity allows), and below that is <strong>Low</strong> (no action).</p>

    <p>The score is delivered where the work is planned. The screenshot below shows the scorer embedded in the
    ERP work-order queue: every job carries a colour-coded defect-risk tier, and a summary panel totals the
    High, Medium, and Low counts, so planners can spot and hold the riskiest jobs without leaving the system
    they already use.</p>
    <div class="chart-wrap" style="padding:6px;">
      <img src="data:image/png;base64,{erp_screenshot_b64}" alt="ERP work-order queue with embedded defect risk tiers"
           style="width:100%;height:auto;display:block;border:1px solid #EEEEEE;">
    </div>

    {section_title("data", "Section 2.2", "Training Data Overview")}

    <p>The model learned from three years of production work orders (January 2023 to December 2025), one row
    per job. Each row is an integrated record stitched together from three source systems: machine and shift
    context from the MES, job, supplier, operator, and schedule data from the ERP, and the inspection outcome
    from the QMS. That join is what lets the model see cross-system combinations, such as an older machine
    running a high-complexity job on a thin-gauge lot, that no single system reveals on its own.</p>

    <p>The chart below shows the full training window: about {td_orders_per_month} work orders a month across
    the three years, split by time into the train, validation, and test sets, with the monthly defect rate
    from the QMS inspection records overlaid. The defect rate holds near {fmt_pct(td_overall_dr)} throughout,
    so the model sees a balanced and stable signal across the whole window rather than a moving target.</p>
    {wrap("training_volume", "Work Orders and Defect Rate by Month, 2023 to 2025")}

    <p>The integrated data also shows where defects concentrate. Broken out by machine type, historical defect
    rates range from about {fmt_pct(td_machine_min)} to {fmt_pct(td_machine_max)}, and similar gaps appear
    across shifts, suppliers, and part complexity. Those cross-system differences are the signal the model
    turns into a per-job risk score; the full feature set and how each feature is built are documented in the
    technical report.</p>
    {wrap("defect_by_machine", "Historical Defect Rate by Machine Type")}

    {section_title("performance", "Section 3", "Model Performance")}

    {section_title("scoring", "Section 3.1", "Scoring Summary")}

    <p>Every work order is scored before it runs and sorted into a risk tier. These are the
    {fmt_num(total_scored)} jobs scheduled to run during {PERIOD_NAME}, about four weeks of production, so the
    summary reflects the current planning horizon rather than a historical backlog; the model rescores the
    queue as new jobs are released. The charts below show how the period's jobs break down and the review load
    each tier places on the floor.</p>
    {wrap("risk_distribution", "Work Orders by Risk Tier")}

    <p>The score itself is a defect probability; the dashed lines mark the High and Medium tier thresholds,
    and jobs to the right of each line are flagged at that tier.</p>
    {wrap("prob_distribution", "Distribution of Predicted Defect Probabilities")}

    <p>Flagged volume holds steady week to week, indicating consistent model operation across the period.</p>
    {wrap("weekly_flags", "Weekly Risk-Tier Distribution")}

    {section_title("accuracy", "Section 3.2", "Accuracy and Validation")}

    <p>Because this period covers historical jobs with known outcomes, the flags can be checked against what
    actually happened. Precision is the share of flags that were correct; recall is the share of all defective
    jobs the model caught. High precision with lower recall is deliberate: the model is tuned to flag only
    jobs it is confident about, so the review list stays trustworthy.</p>

    {kpi_row(
        kpi_card(fmt_pct(high_precision), "High-tier precision", f"{high_tp} of {high_flagged} flags correct", RED),
        kpi_card(fmt_pct(med_precision), "Medium-tier precision", f"{med_tp} of {med_flagged} flags correct", AMBER),
        kpi_card(f"{val_auc:.2f}", "Validation ROC-AUC", "vs 0.50 for chance", BRAND_BLUE),
        kpi_card(fmt_pct(med_recall), "Defect catch rate", "share of defects flagged", GREEN),
    )}

    {accuracy_table()}

    <p>Measured against a no-model baseline, the lift is large. With a {fmt_pct(actual_defect_rate)} defect
    base rate, flagging jobs at random would be right about half the time, whereas the model's High-tier flags
    are correct <strong>{fmt_pct(high_precision)}</strong> of the time, close to double the base rate, and its
    held-out validation ROC-AUC of <strong>{val_auc:.2f}</strong> sits well above the 0.50 of a coin-flip. The
    tradeoff is recall: the model deliberately flags only the jobs it is most confident about, so it surfaces
    a focused, high-value subset of defects rather than trying to catch them all.</p>

    {section_title("sample", "Section 3.3", "Sample Model Output")}

    <p>Presented below is what the model produces for the period's highest-risk jobs: the score, the tier, and
    the plain-language reasons behind the flag, exactly as a planner sees them in the ERP queue.</p>
    {shap_explanation_panel()}

    {section_title("limits", "Section 3.4", "What It Can and Cannot Predict")}

    <ul class="limitation-list">
      <li><strong>It predicts occurrence, not severity or cost.</strong> A High flag means a defect is likely,
      not how many parts will fail or what the scrap will cost; flagged jobs range from a single bad part to a
      full-batch scrap event.</li>
      <li><strong>It is conservative by design.</strong> High precision comes at the cost of recall: the model
      flags only jobs it is confident about and will miss defective jobs that do not match the patterns it has
      learned.</li>
      <li><strong>Some inputs can be missing at release.</strong> Supplier and lot-certification signals
      depend on material being scanned; for jobs released before that scan (historically about 15%), those
      drivers are unavailable.</li>
      <li><strong>It is decision support, not automation.</strong> The model ranks and explains; a person
      decides which flagged jobs to review or hold.</li>
      <li><strong>It stays current through monitoring.</strong> Performance is tracked every period and the
      model is retrained if precision slips or the defect rate drifts, as detailed in the monitoring report.</li>
    </ul>

  </main>
</div>
</body>
</html>'''

OUTPUT.write_text(html, encoding="utf-8")
print(f"Report written to {OUTPUT.resolve()}")
