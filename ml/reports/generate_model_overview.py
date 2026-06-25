"""
generate_model_overview.py
Generates model_overview.html — technical model overview for the
C01 pre-production defect risk scorer.

Intended audience: technical reviewers, data-savvy client contacts,
PE operating partners. Covers model card, training data, model selection,
performance metrics, learning curve, calibration, precision-recall curve,
confusion matrix, and SHAP feature importance.

Run from the ml/reports/ directory:
    python3 generate_model_overview.py

Output: model_overview.html
"""

from pathlib import Path
import sys
import base64
import io
import tempfile
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, average_precision_score,
)

# ── Paths ──────────────────────────────────────────────────────────────────
ML_DIR      = Path("..").resolve()
FEATURES_DIR= ML_DIR / "data" / "features"
OUTPUT      = Path("model_overview.html")
MLFLOW_TRACKING = f"sqlite:///{ML_DIR}/mlruns/mlflow.db"

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

print("Loading MLflow artifacts...")
mlflow.set_tracking_uri(MLFLOW_TRACKING)
client = MlflowClient()

mv       = client.get_latest_versions("defect_risk_scorer", stages=["Production"])[0]
run      = client.get_run(mv.run_id)
run_id   = mv.run_id
model_type = run.data.tags.get("model_type", "xgboost").title()
val_auc    = run.data.metrics.get("val_roc_auc", 0)

# Load feature splits
print("Loading feature splits...")
train = pd.read_parquet(FEATURES_DIR / "train.parquet")
val   = pd.read_parquet(FEATURES_DIR / "validation.parquet")
test  = pd.read_parquet(FEATURES_DIR / "test.parquet")

TARGET = "defect_flag"
ID_COL = "work_order_id"
feature_cols = [c for c in train.columns if c not in [ID_COL, TARGET]]

X_train, y_train = train[feature_cols], train[TARGET].astype(int)
X_val,   y_val   = val[feature_cols],   val[TARGET].astype(int)
X_test,  y_test  = test[feature_cols],  test[TARGET].astype(int)

# Load registered model
print("Loading registered model...")
pipeline = mlflow.sklearn.load_model(f"models:/defect_risk_scorer/Production")

# Compute predictions
y_prob_val  = pipeline.predict_proba(X_val)[:, 1]
y_prob_test = pipeline.predict_proba(X_test)[:, 1]
y_pred_test = (y_prob_test >= 0.5).astype(int)

# Metrics
val_metrics = {
    "ROC-AUC":         roc_auc_score(y_val, y_prob_val),
    "Avg Precision":   average_precision_score(y_val, y_prob_val),
    "F1":              f1_score(y_val, (y_prob_val >= 0.5).astype(int), zero_division=0),
    "Precision":       precision_score(y_val, (y_prob_val >= 0.5).astype(int), zero_division=0),
    "Recall":          recall_score(y_val, (y_prob_val >= 0.5).astype(int), zero_division=0),
}
test_metrics = {
    "ROC-AUC":         roc_auc_score(y_test, y_prob_test),
    "Avg Precision":   average_precision_score(y_test, y_prob_test),
    "F1":              f1_score(y_test, y_pred_test, zero_division=0),
    "Precision":       precision_score(y_test, y_pred_test, zero_division=0),
    "Recall":          recall_score(y_test, y_pred_test, zero_division=0),
}

# Download MLflow artifacts
print("Downloading MLflow artifacts...")
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    artifacts = {}
    # Find parent run — model_comparison.csv is logged at parent level
    child_run    = client.get_run(run_id)
    parent_run_id = child_run.data.tags.get("mlflow.parentRunId", run_id)

    # Per-model artifacts in child run, parent-level artifacts in parent run
    per_model = ["shap_importance.csv", "pr_curve.csv", "roc_curve.csv",
                 "calibration.csv", "learning_curve.csv"]
    parent_level = ["model_comparison.csv"]

    for name in per_model:
        loaded = False
        for path in [f"{model_type.lower()}/{name}", f"xgboost/{name}",
                     f"random_forest/{name}", f"logistic_regression/{name}", name]:
            try:
                local = client.download_artifacts(run_id, path, str(tmp))
                artifacts[name] = pd.read_csv(local)
                print(f"  ✓ {name}")
                loaded = True
                break
            except Exception:
                continue
        if not loaded:
            print(f"  ✗ {name}: not found")
            artifacts[name] = None

    for name in parent_level:
        loaded = False
        for search_id in [parent_run_id, run_id]:
            try:
                local = client.download_artifacts(search_id, name, str(tmp))
                artifacts[name] = pd.read_csv(local)
                print(f"  ✓ {name} (parent run)")
                loaded = True
                break
            except Exception:
                continue
        if not loaded:
            print(f"  ✗ {name}: not found in parent or child run")
            artifacts[name] = None

    shap_df      = artifacts.get("shap_importance.csv")
    pr_df        = artifacts.get("pr_curve.csv")
    roc_df       = artifacts.get("roc_curve.csv")
    cal_df       = artifacts.get("calibration.csv")
    lc_df        = artifacts.get("learning_curve.csv")
    comparison_df= artifacts.get("model_comparison.csv")

print("Data loaded.")

# ── Feature categories ─────────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "machine_type","machine_id","shift_code","operator_id",
    "complexity","material_type","supplier","lot_cert_status","requires_welding",
]
NUMERICAL_FEATURES = [
    "machine_age_years","std_labor_hrs","quantity_ordered","schedule_variance_hrs",
]
INTERACTION_FEATURES = [
    "is_bending_shift_b","is_high_complexity",
    "is_supplier_c_thin_gauge","is_lapsed_cert_op",
]

RISK_HIGH   = 0.75
RISK_MEDIUM = 0.65

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def chart_class_balance():
    splits = ["Train", "Validation", "Test"]
    pos    = [y_train.mean(), y_val.mean(), y_test.mean()]
    neg    = [1 - p for p in pos]
    x = np.arange(len(splits))
    fig, ax = make_fig(h=3.2)
    ax.bar(x, [p*100 for p in pos], color=BRAND_BLUE, width=0.5, label="Defective")
    ax.bar(x, [n*100 for n in neg], bottom=[p*100 for p in pos],
           color=LIGHT_BLUE, width=0.5, label="Clean")
    for i, (p, n) in enumerate(zip(pos, neg)):
        ax.text(i, p*100/2, f"{p:.1%}", ha="center", va="center",
                color="white", fontweight="bold", fontsize=BODY_FS)
        ax.text(i, p*100 + n*100/2, f"{n:.1%}", ha="center", va="center",
                color=BRAND_BLUE, fontweight="bold", fontsize=BODY_FS)
    ax.set_xticks(x); ax.set_xticklabels(splits)
    ax.set_ylabel("Share of Split (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))
    ax.legend(loc="upper right")
    chart_style(ax); plt.tight_layout()
    return fig_to_b64(fig)


def chart_confusion_matrix():
    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    # Row-normalized percentages
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Predicted\nClean","Predicted\nDefective"])
    ax.set_yticklabels(["Actual\nClean","Actual\nDefective"])
    labels  = [[f"TN\n{tn:,}", f"FP\n{fp:,}"],
                [f"FN\n{fn:,}", f"TP\n{tp:,}"]]
    pcts    = [[cm_pct[0,0], cm_pct[0,1]],
               [cm_pct[1,0], cm_pct[1,1]]]
    for i in range(2):
        for j in range(2):
            dark = cm[i,j] > cm.max() / 2
            ax.text(j, i - 0.12, labels[i][j],
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if dark else "black")
            ax.text(j, i + 0.22, f"({pcts[i][j]:.1%})",
                    ha="center", va="center", fontsize=10,
                    color="white" if dark else DARK_GREY)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_learning_curve():
    if lc_df is None:
        return None
    fig, ax = make_fig()
    ax.plot(lc_df["train_size"], lc_df["train_roc_auc"]*100,
            color=BRAND_BLUE, linewidth=2, marker="o", markersize=4,
            label="Training AUC")
    ax.fill_between(lc_df["train_size"],
                    (lc_df["train_roc_auc"] - lc_df["train_roc_auc_std"])*100,
                    (lc_df["train_roc_auc"] + lc_df["train_roc_auc_std"])*100,
                    color=BRAND_BLUE, alpha=0.1)
    ax.plot(lc_df["train_size"], lc_df["val_roc_auc"]*100,
            color=ACCENT, linewidth=2, marker="s", markersize=4,
            label="Validation AUC")
    ax.fill_between(lc_df["train_size"],
                    (lc_df["val_roc_auc"] - lc_df["val_roc_auc_std"])*100,
                    (lc_df["val_roc_auc"] + lc_df["val_roc_auc_std"])*100,
                    color=ACCENT, alpha=0.1)
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("ROC-AUC (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))
    ax.legend()
    chart_style(ax); plt.tight_layout()
    return fig_to_b64(fig)


def chart_calibration():
    if cal_df is None:
        return None
    fig, ax = make_fig()
    ax.plot([0,1],[0,1], color=GREY, linestyle="--",
            linewidth=1.5, label="Perfectly calibrated")
    ax.plot(cal_df["mean_predicted_probability"],
            cal_df["fraction_positive"],
            color=BRAND_BLUE, linewidth=2,
            marker="o", markersize=5, label="Model calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()
    chart_style(ax)
    ax.xaxis.grid(True, color="#EEEEEE", linestyle="-", linewidth=0.8)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_pr_curve():
    if pr_df is None:
        return None
    fig, ax = make_fig()
    ax.plot(pr_df["recall"], pr_df["precision"],
            color=BRAND_BLUE, linewidth=2, label="Precision-Recall curve")
    # Mark operating thresholds
    for thresh, color, label in [
        (RISK_HIGH,   RED,   f"High threshold ({RISK_HIGH})"),
        (RISK_MEDIUM, AMBER, f"Medium threshold ({RISK_MEDIUM})"),
    ]:
        closest = pr_df.iloc[(pr_df["threshold"] - thresh).abs().argsort()[:1]]
        if not closest.empty:
            ax.scatter(closest["recall"], closest["precision"],
                       color=color, s=80, zorder=5, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=BODY_FS - 1)
    chart_style(ax)
    ax.xaxis.grid(True, color="#EEEEEE", linestyle="-", linewidth=0.8)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_roc_curve():
    if roc_df is None:
        return None
    auc = roc_auc_score(y_test, y_prob_test)
    fig, ax = make_fig()
    ax.plot(roc_df["fpr"], roc_df["tpr"],
            color=BRAND_BLUE, linewidth=2,
            label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0,1],[0,1], color=GREY, linestyle="--",
            linewidth=1.5, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()
    chart_style(ax)
    ax.xaxis.grid(True, color="#EEEEEE", linestyle="-", linewidth=0.8)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_shap():
    if shap_df is None:
        return None
    SHAP_LABELS = {
        "is_bending_shift_b":       "Press brake shift configuration",
        "is_high_complexity":       "High complexity part",
        "is_supplier_c_thin_gauge": "Thin gauge material source",
        "is_lapsed_cert_op":        "Historical operator performance",
        "machine_type":             "Machine type",
        "shift_code":               "Shift code",
        "complexity":               "Part complexity",
        "supplier":                 "Supplier",
        "operator_id":              "Operator",
        "machine_id":               "Machine ID",
        "quantity_ordered":         "Quantity ordered",
        "machine_age_years":        "Machine age (years)",
        "std_labor_hrs":            "Std labor hours",
        "lot_cert_status":          "Lot cert status",
        "material_type":            "Material type",
        "requires_welding":         "Requires welding",
        "schedule_variance_hrs":    "Schedule variance",
    }
    top = shap_df.head(12).copy()
    top["label"] = top["feature"].map(lambda f: SHAP_LABELS.get(f, f))
    top = top.sort_values("mean_abs_shap")
    fig, ax = make_fig(h=CHART_H_T)
    bars = ax.barh(top["label"], top["mean_abs_shap"],
                   color=BRAND_BLUE, height=0.65)
    for bar, val in zip(bars, top["mean_abs_shap"]):
        ax.text(bar.get_width() + top["mean_abs_shap"].max()*0.01,
                bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=BODY_FS-1, color=DARK_GREY)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.xaxis.grid(True, color="#EEEEEE", linestyle="-", linewidth=0.8)
    ax.yaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD")
    ax.spines["bottom"].set_color("#DDDDDD")
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig_to_b64(fig)


print("Generating charts...")
charts = {
    "class_balance":  chart_class_balance(),
    "confusion":      chart_confusion_matrix(),
    "learning_curve": chart_learning_curve(),
    "calibration":    chart_calibration(),
    "pr_curve":       chart_pr_curve(),
    "roc_curve":      chart_roc_curve(),
    "shap":           chart_shap(),
}
print("Charts complete.")

# ══════════════════════════════════════════════════════════════════════════════
# HTML HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def wrap(key, title="", caption=""):
    if charts.get(key) is None:
        return f'<div class="chart-wrap"><p style="color:#999;text-align:center;padding:20px;">Chart not available — MLflow artifact not found.</p></div>'
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

def metric_row(label, val_val, test_val):
    return f'''<tr>
      <td>{label}</td>
      <td style="text-align:right;font-weight:700;">{val_val:.4f}</td>
      <td style="text-align:right;font-weight:700;color:{BRAND_BLUE};">{test_val:.4f}</td>
    </tr>'''

def metrics_table():
    rows = "".join(
        metric_row(k, val_metrics[k], test_metrics[k])
        for k in ["ROC-AUC","Avg Precision","F1","Precision","Recall"]
    )
    return f'''<table class="data-table">
      <thead><tr>
        <th>Metric</th>
        <th style="text-align:right;">Validation</th>
        <th style="text-align:right;">Test (held-out)</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>'''

def model_comparison_table():
    if comparison_df is None:
        return '<p style="color:#999;">Model comparison artifact not found.</p>'
    rows = ""
    best_auc = comparison_df["val_roc_auc"].max()
    for _, row in comparison_df.iterrows():
        is_best = row["val_roc_auc"] == best_auc
        bg      = f'background:#EEF4F8;' if is_best else ''
        winner  = ' <span style="color:#1A7A3A;font-weight:700;">✓ Selected</span>' if is_best else ''
        rows += f'''<tr style="{bg}">
          <td style="font-weight:{'700' if is_best else '400'};">
            {row["model_type"].replace("_"," ").title()}{winner}
          </td>
          <td style="text-align:right;">{row["val_roc_auc"]:.4f}</td>
          <td style="text-align:right;">{row["val_f1"]:.4f}</td>
          <td style="text-align:right;">{row["val_precision"]:.4f}</td>
          <td style="text-align:right;">{row["val_recall"]:.4f}</td>
        </tr>'''
    return f'''<table class="data-table">
      <thead><tr>
        <th>Model</th>
        <th style="text-align:right;">Val ROC-AUC</th>
        <th style="text-align:right;">Val F1</th>
        <th style="text-align:right;">Val Precision</th>
        <th style="text-align:right;">Val Recall</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>'''

def feature_table():
    cats  = [(f, "Categorical") for f in CATEGORICAL_FEATURES]
    nums  = [(f, "Numerical")   for f in NUMERICAL_FEATURES]
    ints  = [(f, "Interaction") for f in INTERACTION_FEATURES]
    rows  = ""
    for feat, ftype in cats + nums + ints:
        is_numeric = pd.api.types.is_numeric_dtype(train[feat]) if feat in train.columns else False
        corr = train[feat].corr(train[TARGET].astype(float)) if (feat in train.columns and is_numeric) else None
        color = {"Categorical": ACCENT, "Numerical": BRAND_BLUE,
                 "Interaction": AMBER}.get(ftype, GREY)
        rows += f'''<tr>
          <td style="font-family:monospace;font-size:13px;">{feat}</td>
          <td><span style="background:{color};color:white;padding:1px 7px;
              border-radius:3px;font-size:11px;font-weight:600;">{ftype}</span></td>
          <td style="text-align:right;">{f"{corr:.3f}" if corr is not None else "—"}</td>
        </tr>'''
    return f'''<table class="data-table">
      <thead><tr>
        <th>Feature</th><th>Type</th>
        <th style="text-align:right;">Corr. with Target</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>'''

def training_data_table():
    splits = [
        ("Train",      train, "Jan 2023 – Dec 2024"),
        ("Validation", val,   "Jan 2025 – Jun 2025"),
        ("Test",       test,  "Jul 2025 – Dec 2025"),
    ]
    rows = ""
    for label, df, period in splits:
        rows += f'''<tr>
          <td><strong>{label}</strong></td>
          <td>{period}</td>
          <td style="text-align:right;">{len(df):,}</td>
          <td style="text-align:right;">{df[TARGET].mean():.1%}</td>
        </tr>'''
    return f'''<table class="data-table">
      <thead><tr>
        <th>Split</th><th>Period</th>
        <th style="text-align:right;">Work Orders</th>
        <th style="text-align:right;">Defect Flag Rate</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>'''

# ══════════════════════════════════════════════════════════════════════════════
# HTML
# ══════════════════════════════════════════════════════════════════════════════

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Technical Model Overview — Defect Risk Scorer</title>
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
    .model-card {{
      background: #F7F8FA; border-top: 3px solid {BRAND_BLUE};
      padding: 24px 28px; margin-bottom: 24px;
    }}
    .model-card-grid {{
      display: grid; grid-template-columns: 1fr 1fr; gap: 16px 32px;
      font-size: 14px;
    }}
    .mc-label {{ color: #888; font-weight: 600; font-size: 12px;
                 text-transform: uppercase; letter-spacing: 0.5px; }}
    .mc-value {{ color: {TEXT}; font-weight: 500; margin-top: 2px; }}
    .limitation-list {{ margin: 8px 0 20px 20px; }}
    .limitation-list li {{
      margin-bottom: 8px; font-size: 15px; color: #444; line-height: 1.6;
    }}
    .chart-pair {{
      display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 20px 0;
    }}
    .chart-pair .chart-wrap {{ margin: 0; }}
    .callout {{
      background: #F0F4F8; border-left: 3px solid {ACCENT};
      padding: 16px 22px; margin: 20px 0; font-size: 15px; color: #555;
    }}
  </style>
</head>
<body>

<div class="page-header">
  <h1>Technical Model Overview — Pre-Production Defect Risk Scorer</h1>
  <div class="sub">C01 Quality &amp; Scrap &nbsp;·&nbsp;
       Model version {mv.version} ({model_type}) &nbsp;·&nbsp;
       Generated {datetime.now().strftime("%B %d, %Y")}</div>
</div>

<div class="layout">
  <nav class="toc">
    <div class="toc-title">Contents</div>
    <a href="#model-card">Model Card</a>
    <hr>
    <a href="#training-data">Training Data</a>
    <a href="#features" class="sub">Feature Set</a>
    <a href="#class-balance" class="sub">Class Balance</a>
    <hr>
    <a href="#model-selection">Model Selection</a>
    <hr>
    <a href="#performance">Performance</a>
    <a href="#learning-curve" class="sub">Learning Curve</a>
    <a href="#calibration" class="sub">Calibration</a>
    <a href="#pr-curve" class="sub">Precision-Recall</a>
    <a href="#confusion" class="sub">Confusion Matrix</a>
    <hr>
    <a href="#shap">Feature Importance</a>
    <hr>
    <a href="#limitations">Known Limitations</a>
  </nav>

  <main class="content">

    {section_title("model-card", "Section 1", "Model Card")}

    <div class="model-card">
      <div class="model-card-grid">
        <div>
          <div class="mc-label">Model Name</div>
          <div class="mc-value">defect_risk_scorer</div>
        </div>
        <div>
          <div class="mc-label">Model Type</div>
          <div class="mc-value">{model_type} (via scikit-learn Pipeline)</div>
        </div>
        <div>
          <div class="mc-label">Version</div>
          <div class="mc-value">{mv.version} — Production</div>
        </div>
        <div>
          <div class="mc-label">Registry</div>
          <div class="mc-value">MLflow Model Registry</div>
        </div>
        <div>
          <div class="mc-label">Target Variable</div>
          <div class="mc-value">defect_flag (quantity_failed &gt; 0)</div>
        </div>
        <div>
          <div class="mc-label">Prediction Type</div>
          <div class="mc-value">Binary classification — defect probability [0, 1]</div>
        </div>
        <div>
          <div class="mc-label">Operating Thresholds</div>
          <div class="mc-value">High ≥ {RISK_HIGH} &nbsp;·&nbsp; Medium ≥ {RISK_MEDIUM}</div>
        </div>
        <div>
          <div class="mc-label">Hyperparameter Tuning</div>
          <div class="mc-value">Optuna (150 trials, validation ROC-AUC objective)</div>
        </div>
        <div style="grid-column:1/-1;">
          <div class="mc-label">Purpose</div>
          <div class="mc-value">Embedded in ERP; scores each production work order for defect risk before job
          release, enabling pre-production intervention. Not intended for post-production
          diagnosis or scrap cost prediction.</div>
        </div>
      </div>
    </div>

    {section_title("training-data", "Section 2", "Training Data")}

    <p>The model was trained on production work order data from a sheet metal fabrication
    operation covering January 2023 through December 2025. A time-based split was used to mirror the real deployment scenario where the model scores future jobs
    it has never seen.</p>

    {training_data_table()}

    <div class="callout">
      <strong>Target definition note:</strong> defect_flag = 1 when any parts in a work
      order failed inspection (quantity_failed &gt; 0). With a 55% positive rate, the
      target captures a broad range of defect severity. The model predicts whether any
      defect will occur, not the magnitude of the defect event.
    </div>

    {section_title("features", "Section 2.1", "Feature Set")}

    <p>Features are engineered in <code>src/features.py</code> and applied identically
    at training and scoring time. Interaction features encode domain-informed cross-system
    patterns identified in the diagnostic analysis.</p>

    {feature_table()}

    {section_title("class-balance", "Section 2.2", "Class Balance by Split")}

    <p>Class balance is consistent across all three splits, confirming the time-based
    split did not introduce distributional shift in the target variable.</p>

    {wrap("class_balance", "Defective vs Clean Work Orders by Split")}

    {section_title("model-selection", "Section 3", "Model Selection")}

    <p>Three candidate classifiers were trained and evaluated on the validation set using
    Optuna hyperparameter optimization (150 trials per model). The best-performing model
    on validation ROC-AUC was selected and evaluated once on the held-out test set.</p>

    {model_comparison_table()}

    {section_title("performance", "Section 4", "Performance Metrics")}

    <p>Metrics for the selected model (Xgboost) are reported on both the validation set (used for model selection and
    hyperparameter tuning) and the test set (touched once, after model selection was
    complete). The test set is the honest estimate of real-world performance.</p>

    {metrics_table()}

    {section_title("learning-curve", "Section 4.1", "Learning Curve")}

    <p>Training and validation AUC converge as training size increases, confirming the
    model is learning generalizable signal. A gap between the two curves is expected —
    XGBoost fits training data tightly, and the validation period reflects slightly
    different operating conditions from a later time window.</p>

    {wrap("learning_curve", "Training vs Validation ROC-AUC by Training Set Size")}

    <p>The ROC curve shows the tradeoff between true positive rate and false positive rate
    across all possible thresholds on the held-out test set.</p>

    {wrap("roc_curve", "ROC Curve — Test Set")}

    {section_title("calibration", "Section 4.2", "Calibration")}

    <p>A well-calibrated model tracks the diagonal — when it predicts 70% probability,
    approximately 70% of those jobs actually produce defects. Deviations from the diagonal
    indicate the raw probability outputs should be interpreted with caution.</p>

    {wrap("calibration", "Predicted Probability vs Actual Defect Rate")}

    {section_title("pr-curve", "Section 4.3", "Precision-Recall Curve")}

    <p>Markers show the operating points for the High and Medium risk tiers.
    Moving left along the curve increases precision at the cost of recall.</p>

    {wrap("pr_curve", "Precision-Recall Curve with Operating Thresholds")}

    {section_title("confusion", "Section 4.4", "Confusion Matrix")}

    <p>True positives, false positives, true negatives, and false negatives
    at a 0.5 operating threshold on the held-out test set.</p>

    <div style="max-width:480px;margin:20px auto;">
      {wrap("confusion", "Confusion Matrix — Test Set")}
    </div>

    {section_title("shap", "Section 5", "Feature Importance (SHAP)")}

    <p>SHAP (SHapley Additive exPlanations) values measure each feature's average
    contribution to the model's predictions across the validation set. Features with
    higher mean absolute SHAP values have greater influence on the model's output.
    SHAP values are computed on the validation set using TreeExplainer.</p>

    <p>Features ranked by average impact on model output across the validation set.
    Interaction features appear prominently because they encode the strongest
    cross-system patterns identified in the diagnostic analysis.</p>

    {wrap("shap", "Mean Absolute SHAP Value by Feature")}

    {section_title("limitations", "Section 6", "Known Limitations")}

    <ul class="limitation-list">
      <li><strong>Target definition:</strong> defect_flag = (quantity_failed &gt; 0) captures
      any failure, including minor single-part events in large batches. The 55% positive rate
      reflects this broad definition and limits the model's discrimination ceiling.</li>
      <li><strong>Simulated training data:</strong> This model was trained on synthetic data
      with embedded patterns. Real-world performance will depend on the signal strength of
      actual shop floor data and may differ materially.</li>
      <li><strong>Feature availability at scoring time:</strong> Features requiring lot
      assignment (supplier, lot_cert_status) may be missing for jobs where material has not
      been scanned at release time (~15% of orders historically).</li>
      <li><strong>Retraining cadence:</strong> The model should be retrained when the
      Evidently drift monitoring report flags statistically significant feature drift, or
      when High-tier precision drops below 85% on a new scoring period.</li>
      <li><strong>Scope:</strong> The model predicts defect occurrence, not defect severity
      or scrap cost. Jobs flagged as High risk may range from minor single-part failures
      to full-batch scrap events.</li>
    </ul>

  </main>
</div>
</body>
</html>'''

OUTPUT.write_text(html, encoding="utf-8")
print(f"\nModel overview written to {OUTPUT.resolve()}")
