from pathlib import Path
import base64
import io
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mlflow
from mlflow import MlflowClient

# ── Paths ──────────────────────────────────────────────────────────────────
ML_DIR         = Path("..").resolve()
FEATURES_DIR   = ML_DIR / "data" / "features"
SCORING_DIR    = ML_DIR / "data" / "scoring"
MONITORING_DIR = ML_DIR / "data" / "monitoring"
OUTPUT         = Path("monitoring_report.html")

MLFLOW_TRACKING = f"sqlite:///{ML_DIR}/mlruns/mlflow.db"

# Periods covered (label, display name) — order defines the trend axis
PERIODS = [
    ("202601", "Jan 2026"),
    ("202602", "Feb 2026"),
    ("202603", "Mar 2026"),
]
REFERENCE_NAME = "January 2023 – December 2024"

# Decision thresholds (mirror monitoring.py)
RISK_HIGH            = 0.75
RISK_MEDIUM          = 0.65
MIN_HIGH_PRECISION   = 0.85
TARGET_RATE_DELTA    = 0.10
MAX_DRIFTED_FEATURES = 3
DRIFT_SCORE_THRESHOLD = 0.1

# ── Palette ────────────────────────────────────────────────────────────────
BRAND_BLUE = "#3D5166"
ACCENT     = "#6B8FA8"
LIGHT_BLUE = "#A8C0D1"
AMBER      = "#D4881E"
RED        = "#CC0000"
GREEN      = "#1A7A3A"
GREY       = "#AAAAAA"
DARK_GREY  = "#555555"
TEXT       = "#222222"

CHART_W   = 8.2
CHART_H   = 3.8
CHART_DPI = 130
BODY_FS   = 11
TITLE_FS  = 13

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor":   "#DDDDDD","axes.grid": False,
    "font.family":      "sans-serif",
    "font.size":        BODY_FS, "axes.titlesize": TITLE_FS,
    "axes.titleweight": "bold",  "axes.labelsize": BODY_FS,
    "xtick.labelsize":  BODY_FS, "ytick.labelsize": BODY_FS,
    "legend.fontsize":  BODY_FS, "figure.dpi": CHART_DPI,
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

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

print("Loading data...")
mlflow.set_tracking_uri(MLFLOW_TRACKING)
client = MlflowClient()

mv         = client.get_latest_versions("defect_risk_scorer", stages=["Production"])[0]
run        = client.get_run(mv.run_id)
run_id     = mv.run_id
model_ver  = mv.version
model_type = run.data.tags.get("model_type", run.data.tags.get("best_model_type", "xgboost")).title()
train_date = datetime.fromtimestamp(run.info.start_time / 1000).strftime("%B %d, %Y")

all_versions = client.search_model_versions("name='defect_risk_scorer'")
version_history = []
for v in sorted(all_versions, key=lambda x: int(x.version)):
    vrun = client.get_run(v.run_id)
    version_history.append({
        "version": v.version,
        "stage":   v.current_stage,
        "type":    vrun.data.tags.get("model_type",
                   vrun.data.tags.get("best_model_type", "—")).title(),
        "trained": datetime.fromtimestamp(vrun.info.start_time/1000).strftime("%Y-%m-%d"),
        "val_auc": vrun.data.metrics.get("val_roc_auc", None),
    })

TARGET = "defect_flag"
train  = pd.read_parquet(FEATURES_DIR / "train.parquet")
val_ref = pd.read_parquet(FEATURES_DIR / "validation_predictions.parquet")

labels = [lbl for lbl, _ in PERIODS]
names  = [nm  for _, nm in PERIODS]

# Consolidated longitudinal table — the backbone for all trends
pm = pd.read_csv(MONITORING_DIR / "period_monitoring.csv",
                 dtype={"period_label": str}).set_index("period_label")
pm = pm.loc[[l for l in labels if l in pm.index]]

# Per-period detail
preds_by, drift_by, feat_by = {}, {}, {}
for lbl in labels:
    p = SCORING_DIR / f"predictions_{lbl}.parquet"
    preds_by[lbl] = pd.read_parquet(p) if p.exists() else None
    d = MONITORING_DIR / f"drift_summary_{lbl}.csv"
    drift_by[lbl] = pd.read_csv(d) if d.exists() else None
    f = MONITORING_DIR / f"current_features_{lbl}.parquet"
    feat_by[lbl] = pd.read_parquet(f) if f.exists() else None

latest_label = labels[-1]
latest_name  = names[-1]
latest = pm.loc[latest_label]

print(f"  Periods loaded: {', '.join(labels)}")
print(f"  Latest-period status: {latest['status']}")

# ── Status styling ───────────────────────────────────────────────────────────
STATUS_STYLE = {
    "HEALTHY":     (GREEN, "✓", "NO ACTION REQUIRED"),
    "INVESTIGATE": (AMBER, "◐", "INVESTIGATE"),
    "RETRAIN":     (RED,   "⚠", "RETRAIN RECOMMENDED"),
}
status_color, status_icon, status_label = STATUS_STYLE.get(
    str(latest["status"]), (GREY, "•", str(latest["status"])))
retrain_reasons = [r for r in str(latest.get("reasons", "")).split(" | ") if r]

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def chart_performance_trend():
    """Precision/recall by tier across the three periods."""
    x = list(range(len(pm)))
    fig, ax = make_fig(h=4.0)
    series = [
        ("high_precision",   "High Precision",   BRAND_BLUE, "-",  "o"),
        ("high_recall",      "High Recall",      ACCENT,     "-",  "s"),
        ("medium_precision", "Medium Precision", AMBER,      "--", "o"),
        ("medium_recall",    "Medium Recall",    LIGHT_BLUE, "--", "s"),
    ]
    for col, lbl, color, ls, mk in series:
        ax.plot(x, (pm[col] * 100).values, ls, marker=mk, color=color,
                linewidth=2, markersize=7, label=lbl)
    ax.axhline(MIN_HIGH_PRECISION * 100, color=RED, linestyle=":", linewidth=1.3,
               label=f"Precision floor ({MIN_HIGH_PRECISION:.0%})")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("(%)"); ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(ncol=2, fontsize=9, loc="lower center")
    chart_style(ax); plt.tight_layout()
    return fig_to_b64(fig)

def chart_flag_rate_trend():
    """Share of jobs by risk tier across periods."""
    tiers = ["High", "Medium", "Low"]
    colors = {"High": RED, "Medium": AMBER, "Low": GREEN}
    data = {t: [] for t in tiers}
    for lbl in labels:
        p = preds_by[lbl]
        vc = p["risk_tier"].value_counts() if p is not None else pd.Series(dtype=int)
        n  = len(p) if p is not None else 0
        for t in tiers:
            data[t].append((vc.get(t, 0) / n * 100) if n else 0)
    x = np.arange(len(labels)); w = 0.25
    fig, ax = make_fig(h=3.6)
    for i, t in enumerate(tiers):
        ax.bar(x + (i - 1) * w, data[t], w, color=colors[t], label=f"{t} risk")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Share of Scored Jobs (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(fontsize=9)
    chart_style(ax); plt.tight_layout()
    return fig_to_b64(fig)

def chart_target_drift():
    """Actual defect rate per period vs the training baseline."""
    base = float(pm["train_defect_rate"].iloc[0]) * 100
    vals = (pm["period_defect_rate"] * 100).values
    colors = [RED if abs(v/100 - base/100) > TARGET_RATE_DELTA else GREEN for v in vals]
    fig, ax = make_fig(h=3.6)
    bars = ax.bar(names, vals, color=colors, width=0.5)
    ax.axhline(base, color=BRAND_BLUE, linestyle="--", linewidth=1.5,
               label=f"Training baseline ({base:.1f}%)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=BODY_FS)
    ax.set_ylabel("Defect Flag Rate (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(fontsize=9)
    chart_style(ax); plt.tight_layout()
    return fig_to_b64(fig)

def chart_prediction_drift_trend():
    """Prediction-drift distance per period vs the validation reference."""
    vals = pm["prediction_drift_score"].values
    thr  = float(pm["prediction_drift_threshold"].iloc[0])
    colors = [RED if v >= thr else GREEN for v in vals]
    fig, ax = make_fig(h=3.6)
    bars = ax.bar(names, vals, color=colors, width=0.5)
    ax.axhline(thr, color=RED, linestyle="--", linewidth=1.5,
               label=f"Drift threshold ({thr})")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=BODY_FS)
    ax.set_ylabel("Distance vs validation reference")
    ax.set_ylim(0, max(thr * 1.25, float(np.max(vals)) * 1.4 + 1e-6))
    ax.legend(fontsize=9)
    chart_style(ax); plt.tight_layout()
    return fig_to_b64(fig)

def chart_prediction_distribution():
    """Score distribution: validation reference vs the latest period."""
    fig, ax = make_fig()
    cur = preds_by[latest_label]
    bins = np.linspace(0, 1, 31)
    ax.hist(val_ref["defect_probability"], bins=bins, alpha=0.55, density=True,
            color=GREY, label="Validation reference", edgecolor="white", linewidth=0.4)
    if cur is not None:
        ax.hist(cur["defect_probability"], bins=bins, alpha=0.65, density=True,
                color=BRAND_BLUE, label=f"Current ({latest_name})",
                edgecolor="white", linewidth=0.4)
    ax.axvline(RISK_HIGH, color=RED, linestyle="--", linewidth=1.3,
               label=f"High ({RISK_HIGH})")
    ax.axvline(RISK_MEDIUM, color=AMBER, linestyle="--", linewidth=1.3,
               label=f"Medium ({RISK_MEDIUM})")
    ax.set_xlabel("Predicted Defect Probability"); ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    chart_style(ax); plt.tight_layout()
    return fig_to_b64(fig)

def chart_feature_heatmap():
    """Per-feature drift distance across periods (feature × month)."""
    present = [l for l in labels if drift_by[l] is not None]
    if not present:
        return None
    base = drift_by[present[0]][["feature"]].copy()
    mat  = base.set_index("feature")
    for l in present:
        s = drift_by[l].set_index("feature")["drift_score"]
        mat[l] = s
    # order by mean distance descending (most-moved at top)
    mat["_m"] = mat.mean(axis=1)
    mat = mat.sort_values("_m", ascending=True).drop(columns="_m")  # ascending -> top of axis = largest
    cols = present
    M = mat[cols].values
    fig, ax = plt.subplots(figsize=(CHART_W, max(3.6, len(mat) * 0.34)))
    im = ax.imshow(M, aspect="auto", cmap="Blues",
                   vmin=0, vmax=max(DRIFT_SCORE_THRESHOLD, float(np.nanmax(M))))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([dict(PERIODS).get(c, c) for c in cols])
    ax.set_yticks(range(len(mat))); ax.set_yticklabels(mat.index, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color=(RED if v >= DRIFT_SCORE_THRESHOLD else "#333333"))
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Drift distance", fontsize=9)
    ax.set_title("")
    plt.tight_layout()
    return fig_to_b64(fig)

def chart_data_quality():
    """Null rates per feature for the latest period."""
    cf = feat_by[latest_label]
    if cf is None:
        return None
    feat_cols = [c for c in cf.columns if c not in ["work_order_id", TARGET]]
    null_rates = (cf[feat_cols].isna().mean() * 100)
    null_rates = null_rates[null_rates > 0].sort_values(ascending=False)
    if null_rates.empty:
        null_rates = pd.Series([0.0], index=["No nulls detected"])
    fig, ax = make_fig(h=max(2.8, len(null_rates) * 0.4))
    colors = [RED if v > 10 else AMBER if v > 5 else BRAND_BLUE for v in null_rates.values]
    ax.barh(null_rates.index, null_rates.values, color=colors, height=0.6)
    ax.set_xlabel("Null Rate (%)")
    ax.xaxis.grid(True, color="#EEEEEE", linewidth=0.8); ax.yaxis.grid(False)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD"); ax.spines["bottom"].set_color("#DDDDDD")
    ax.set_axisbelow(True); plt.tight_layout()
    return fig_to_b64(fig)

print("Generating charts...")
charts = {
    "perf_trend":      chart_performance_trend(),
    "flag_rate":       chart_flag_rate_trend(),
    "target_drift":    chart_target_drift(),
    "pred_drift":      chart_prediction_drift_trend(),
    "pred_dist":       chart_prediction_distribution(),
    "feature_heatmap": chart_feature_heatmap(),
    "data_quality":    chart_data_quality(),
}
print("Charts complete.")

# ══════════════════════════════════════════════════════════════════════════════
# HTML HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def wrap(key, title="", caption=""):
    if charts.get(key) is None:
        return ('<div class="chart-wrap"><p style="color:#999;text-align:center;'
                'padding:20px;">Chart not available.</p></div>')
    title_html   = f'<div class="chart-title">{title}</div>' if title else ""
    caption_html = f'<div class="chart-caption">{caption}</div>' if caption else ""
    return (f'<div class="chart-wrap">{title_html}'
            f'<img src="data:image/png;base64,{charts[key]}" '
            f'style="width:100%;height:auto;display:block;">{caption_html}</div>')

def section_title(id, label, title):
    return (f'<div class="section-title-block" id="{id}">'
            f'<div class="section-label">{label}</div>'
            f'<h2 class="section-title">{title}</h2></div>')

def status_block():
    if retrain_reasons:
        items = "".join(f"<li>{r}</li>" for r in retrain_reasons)
        reason_html = f'<ul class="trigger-list">{items}</ul>'
    else:
        reason_html = '<p style="margin:8px 0 0 0;color:#555;">No triggers met across the monitored periods.</p>'
    return f'''<div class="status-block" style="border-color:{status_color};">
      <div class="status-header" style="background:{status_color};">
        <span class="status-icon">{status_icon}</span>
        <span class="status-label">{status_label}</span>
        <span style="margin-left:auto;font-size:13px;opacity:0.9;">As of {latest_name}</span>
      </div>
      <div class="status-body">
        <div class="status-meta">
          <div><span class="meta-label">Model Version</span>
               <span class="meta-val">v{model_ver} ({model_type})</span></div>
          <div><span class="meta-label">Trained</span>
               <span class="meta-val">{train_date}</span></div>
          <div><span class="meta-label">Periods Monitored</span>
               <span class="meta-val">{names[0]} – {names[-1]}</span></div>
          <div><span class="meta-label">Reference</span>
               <span class="meta-val">{REFERENCE_NAME}</span></div>
          <div><span class="meta-label">High-Tier Precision</span>
               <span class="meta-val" style="color:{'#CC0000' if latest['high_precision'] < MIN_HIGH_PRECISION else '#1A7A3A'};">
                 {latest['high_precision']:.1%}</span></div>
          <div><span class="meta-label">Defect Rate Shift</span>
               <span class="meta-val" style="color:{'#CC0000' if latest['target_rate_delta'] > TARGET_RATE_DELTA else '#1A7A3A'};">
                 {latest['target_rate_delta']:.1%}</span></div>
          <div><span class="meta-label">Features Drifted</span>
               <span class="meta-val" style="color:{'#CC0000' if latest['n_features_drifted'] > 0 else '#1A7A3A'};">
                 {int(latest['n_features_drifted'])} / {int(latest['n_features_monitored'])}</span></div>
        </div>
        <div>
          <div style="font-size:12px;font-weight:700;color:#555;text-transform:uppercase;
                      letter-spacing:0.5px;margin-bottom:6px;">Trigger Reasons</div>
          {reason_html}
        </div>
      </div>
    </div>'''

def retraining_rules():
    rules = [
        ("Primary", f"High-tier precision below {MIN_HIGH_PRECISION:.0%} (measured performance)",
         bool(latest["high_precision"] < MIN_HIGH_PRECISION)),
        ("Primary", f"Defect rate shifts more than {TARGET_RATE_DELTA:.0%} from training baseline",
         bool(latest["target_rate_delta"] > TARGET_RATE_DELTA)),
        ("Secondary", "Prediction (output) distribution drifts vs validation reference",
         bool(latest["prediction_drift_detected"])),
        ("Secondary", f"More than {MAX_DRIFTED_FEATURES} input features drift vs training",
         bool(latest["n_features_drifted"] > MAX_DRIFTED_FEATURES)),
    ]
    rows = ""
    for tier, rule, triggered in rules:
        icon  = "⚠" if triggered else "✓"
        color = RED if triggered else GREEN
        badge = (f'<span style="font-size:11px;font-weight:700;color:{BRAND_BLUE};">{tier}</span>')
        rows += f'''<tr>
          <td style="width:30px;text-align:center;color:{color};font-size:16px;">{icon}</td>
          <td>{badge}</td>
          <td style="color:#333;">{rule}</td>
          <td style="text-align:center;color:{color};font-weight:700;">
            {"TRIGGERED" if triggered else "OK"}</td>
        </tr>'''
    return f'''<table class="data-table">
      <thead><tr><th></th><th>Tier</th><th>Rule</th><th style="text-align:center;">Status</th></tr></thead>
      <tbody>{rows}</tbody></table>'''

def performance_table():
    rows = ""
    for lbl, nm in PERIODS:
        if lbl not in pm.index: continue
        r = pm.loc[lbl]
        rows += f'''<tr>
          <td style="font-weight:600;">{nm}</td>
          <td style="text-align:right;">{int(r['n_scored']):,}</td>
          <td style="text-align:right;">{r['high_precision']:.1%}</td>
          <td style="text-align:right;">{r['high_recall']:.1%}</td>
          <td style="text-align:right;">{r['medium_precision']:.1%}</td>
          <td style="text-align:right;">{r['medium_recall']:.1%}</td>
        </tr>'''
    return f'''<table class="data-table">
      <thead><tr><th>Period</th><th style="text-align:right;">Scored</th>
        <th style="text-align:right;">High Prec.</th><th style="text-align:right;">High Rec.</th>
        <th style="text-align:right;">Med Prec.</th><th style="text-align:right;">Med Rec.</th></tr></thead>
      <tbody>{rows}</tbody></table>'''

def target_table():
    base = pm["train_defect_rate"].iloc[0]
    rows = ""
    for lbl, nm in PERIODS:
        if lbl not in pm.index: continue
        r = pm.loc[lbl]
        drifted = bool(r["target_drift_detected"])
        color = RED if drifted else GREEN
        rows += f'''<tr>
          <td style="font-weight:600;">{nm}</td>
          <td style="text-align:right;">{r['period_defect_rate']:.1%}</td>
          <td style="text-align:right;">{base:.1%}</td>
          <td style="text-align:right;">{r['target_rate_delta']:.1%}</td>
          <td style="text-align:center;color:{color};font-weight:700;">
            {"⚠ Shift" if drifted else "✓ Stable"}</td>
        </tr>'''
    return f'''<table class="data-table">
      <thead><tr><th>Period</th><th style="text-align:right;">Actual Rate</th>
        <th style="text-align:right;">Baseline</th><th style="text-align:right;">Δ</th>
        <th style="text-align:center;">Status</th></tr></thead>
      <tbody>{rows}</tbody></table>'''

def prediction_drift_table():
    rows = ""
    for lbl, nm in PERIODS:
        if lbl not in pm.index: continue
        r = pm.loc[lbl]
        drifted = bool(r["prediction_drift_detected"])
        color = RED if drifted else GREEN
        rows += f'''<tr>
          <td style="font-weight:600;">{nm}</td>
          <td style="text-align:right;font-variant-numeric:tabular-nums;">{r['prediction_drift_score']:.4f}</td>
          <td style="text-align:right;">{r['prediction_drift_threshold']:.2f}</td>
          <td style="text-align:right;">{r['ref_pred_mean']:.3f} → {r['cur_pred_mean']:.3f}</td>
          <td style="text-align:center;color:{color};font-weight:700;">
            {"⚠ Drift" if drifted else "✓ Stable"}</td>
        </tr>'''
    return f'''<table class="data-table">
      <thead><tr><th>Period</th><th style="text-align:right;">Distance</th>
        <th style="text-align:right;">Threshold</th><th style="text-align:right;">Mean Score (ref→cur)</th>
        <th style="text-align:center;">Status</th></tr></thead>
      <tbody>{rows}</tbody></table>'''

def feature_drift_table():
    """Latest-period per-feature detail."""
    df = drift_by[latest_label]
    if df is None:
        return '<p style="color:#999;">Feature drift summary not found.</p>'
    rows = ""
    for _, row in df.sort_values("drift_score", ascending=False).iterrows():
        drifted = bool(row["drift_detected"])
        color = RED if drifted else GREEN
        icon  = "⚠" if drifted else "✓"
        rows += f'''<tr>
          <td style="font-family:monospace;font-size:13px;">{row["feature"]}</td>
          <td>{row.get("feature_type","—")}</td>
          <td>{row.get("test_method","—")}</td>
          <td style="text-align:right;font-variant-numeric:tabular-nums;">{float(row["drift_score"]):.4f}</td>
          <td style="text-align:right;">{float(row["threshold"]):.2f}</td>
          <td style="text-align:center;color:{color};font-weight:700;">{icon} {"Drift" if drifted else "Stable"}</td>
        </tr>'''
    return f'''<table class="data-table">
      <thead><tr><th>Feature</th><th>Type</th><th>Test</th>
        <th style="text-align:right;">Drift Distance</th><th style="text-align:right;">Threshold</th>
        <th style="text-align:center;">Status</th></tr></thead>
      <tbody>{rows}</tbody></table>'''

def data_quality_table():
    cf = feat_by[latest_label]
    if cf is None:
        return '<p style="color:#999;">Current period features not found.</p>'
    feat_cols = [c for c in cf.columns if c not in ["work_order_id", TARGET]]
    rows = ""
    for col in feat_cols:
        null_pct = cf[col].isna().mean() * 100
        n_unique = cf[col].nunique()
        if cf[col].dtype == object and col in train.columns:
            new_cats = set(cf[col].dropna().unique()) - set(train[col].dropna().unique())
            new_cat_str = f"{len(new_cats)} new" if new_cats else "—"
        else:
            new_cat_str = "—"
        null_color = RED if null_pct > 10 else AMBER if null_pct > 5 else TEXT
        rows += f'''<tr>
          <td style="font-family:monospace;font-size:13px;">{col}</td>
          <td style="text-align:right;color:{null_color};font-weight:{'700' if null_pct > 5 else '400'};">{null_pct:.1f}%</td>
          <td style="text-align:right;">{n_unique:,}</td>
          <td style="text-align:center;">{new_cat_str}</td>
        </tr>'''
    return f'''<table class="data-table">
      <thead><tr><th>Feature</th><th style="text-align:right;">Null Rate</th>
        <th style="text-align:right;">Unique Values</th><th style="text-align:center;">New Categories</th></tr></thead>
      <tbody>{rows}</tbody></table>'''

def version_history_table():
    rows = ""
    for v in version_history:
        is_current = str(v["version"]) == str(model_ver)
        bg = "background:#EEF4F8;" if is_current else ""
        auc_str = f"{v['val_auc']:.4f}" if v["val_auc"] else "—"
        stage_color = BRAND_BLUE if v["stage"] == "Production" else GREY
        rows += f'''<tr style="{bg}">
          <td style="font-weight:{'700' if is_current else '400'};">v{v["version"]} {"← current" if is_current else ""}</td>
          <td>{v["trained"]}</td><td>{v["type"]}</td>
          <td style="text-align:right;">{auc_str}</td>
          <td><span style="color:{stage_color};font-weight:600;">{v["stage"]}</span></td>
        </tr>'''
    return f'''<table class="data-table">
      <thead><tr><th>Version</th><th>Trained</th><th>Type</th>
        <th style="text-align:right;">Val ROC-AUC</th><th>Stage</th></tr></thead>
      <tbody>{rows}</tbody></table>'''

# Pre-render chart blocks (avoids brace-interpolation issues in the template)
perf_block   = wrap("perf_trend",  f"Precision &amp; Recall by Tier — {names[0]}–{names[-1]}")
flag_block   = wrap("flag_rate",   f"Risk-Tier Mix by Period")
target_block = wrap("target_drift","Actual Defect Rate vs Training Baseline")
pdrift_block = wrap("pred_drift",   "Prediction-Drift Distance by Period")
pdist_block  = wrap("pred_dist",    f"Score Distribution — Validation Reference vs {latest_name}")
heat_block   = wrap("feature_heatmap", "Per-Feature Drift Distance (feature × period)")
dq_block     = wrap("data_quality", f"Feature Null Rates — {latest_name}")

# ══════════════════════════════════════════════════════════════════════════════
# HTML
# ══════════════════════════════════════════════════════════════════════════════

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title> MLOps Monitoring Report — {names[0]}–{names[-1]}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #FFFFFF; color: {TEXT}; font-size: 16px; line-height: 1.7; }}
    .page-header {{ background: {BRAND_BLUE}; color: white; padding: 20px 40px; }}
    .page-header h1 {{ font-size: 22px; font-weight: 700; }}
    .page-header .sub {{ font-size: 14px; opacity: 0.8; margin-top: 2px; }}
    .layout {{ display: flex; max-width: 1200px; margin: 0 auto; padding: 0 40px; }}
    .toc {{ width: 210px; flex-shrink: 0; padding: 40px 20px 40px 0;
      position: sticky; top: 0; height: 100vh; overflow-y: auto; border-right: 1px solid #EEEEEE; }}
    .toc-title {{ font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
      color: #AAAAAA; margin-bottom: 14px; font-weight: 600; }}
    .toc a {{ display: block; font-size: 13px; color: #666; text-decoration: none;
      padding: 4px 0 4px 10px; border-left: 2px solid transparent; }}
    .toc a:hover {{ color: {BRAND_BLUE}; border-left-color: {BRAND_BLUE}; }}
    .toc hr {{ border: none; border-top: 1px solid #EEEEEE; margin: 8px 0; }}
    .content {{ flex: 1; padding: 40px 0 80px 52px; max-width: 880px; }}
    .section-title-block {{ margin: 48px 0 24px 0; padding-bottom: 12px; border-bottom: 2px solid {BRAND_BLUE}; }}
    .content > .section-title-block:first-child {{ margin-top: 12px; }}
    .section-label {{ font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
      color: {BRAND_BLUE}; font-weight: 600; margin-bottom: 4px; }}
    .section-title {{ font-size: 22px; font-weight: 700; }}
    p {{ margin-bottom: 16px; color: #333; font-size: 16px; }}
    .chart-title {{ font-size: 17px; font-weight: 700; color: {TEXT}; text-align: center; margin-bottom: 8px; }}
    .chart-wrap {{ margin: 20px 0; border: 1px solid #EEEEEE; border-radius: 4px; padding: 12px; }}
    .chart-caption {{ font-size: 12px; color: #888; margin-top: 8px; text-align: center; font-style: italic; }}
    .data-table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 14px; }}
    .data-table th {{ background: #F7F8FA; padding: 10px 12px; text-align: left; font-size: 12px;
      font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #666; border-bottom: 2px solid #EEEEEE; }}
    .data-table td {{ padding: 9px 12px; border-bottom: 1px solid #F0F0F0; color: #333; }}
    .data-table tr:hover td {{ background: #FAFAFA; }}
    .status-block {{ border: 2px solid; border-radius: 6px; overflow: hidden; margin: 20px 0; }}
    .status-header {{ display: flex; align-items: center; gap: 12px; padding: 14px 20px; color: white; }}
    .status-icon {{ font-size: 22px; }}
    .status-label {{ font-size: 17px; font-weight: 700; letter-spacing: 0.3px; }}
    .status-body {{ padding: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
    .status-meta {{ display: flex; flex-direction: column; gap: 8px; }}
    .status-meta > div {{ display: flex; gap: 10px; font-size: 14px; }}
    .meta-label {{ color: #888; font-weight: 600; font-size: 12px; text-transform: uppercase;
      letter-spacing: 0.5px; width: 150px; flex-shrink: 0; padding-top: 1px; }}
    .meta-val {{ color: {TEXT}; font-weight: 500; }}
    .trigger-list {{ margin: 6px 0 0 18px; }}
    .trigger-list li {{ font-size: 14px; color: #333; margin-bottom: 5px; line-height: 1.5; }}
    .callout {{ background:#F7F8FA; border-left:3px solid {ACCENT}; padding:14px 18px;
      margin:20px 0; font-size:14px; color:#555; }}
  </style>
</head>
<body>

<div class="page-header">
  <h1> MLOps Monitoring Report — Pre-Production Defect Risk Scorer</h1>
  <div class="sub">Periods: {names[0]} – {names[-1]} &nbsp;·&nbsp;
       Model v{model_ver} ({model_type}) &nbsp;</div>
</div>

<div class="layout">
  <nav class="toc">
    <div class="toc-title">Contents</div>
    <a href="#status">1 · Status &amp; Decision</a>
    <hr>
    <a href="#performance">2 · Performance</a>
    <a href="#target">3 · Target Drift</a>
    <a href="#prediction">4 · Prediction Drift</a>
    <a href="#feature">5 · Feature Drift</a>
    <a href="#quality">6 · Data Quality</a>
    <hr>
    <a href="#log">7 · Monitoring Log</a>
  </nav>

  <main class="content">

    {section_title("status", "Section 1", "Status &amp; Retraining Decision")}
    <p>The retraining decision is led by directly-measured signals (model
    performance and defect-rate drift) because ground-truth outcomes
    for this shop land within the scoring period. Prediction and feature drift
    act as early proxies and diagnostic context, not primary triggers. The
    verdict below reflects the most recent period ({latest_name}); the sections
    that follow show the full trend.</p>

    {status_block()}

    <p>Retraining rules are evaluated every period. Primary rules measure harm
    directly; secondary rules are leading proxies that warrant investigation
    rather than immediate retraining.</p>

    {retraining_rules()}

    {section_title("performance", "Section 2", "Performance")}
    <p>Precision and recall by risk tier against actual outcomes. These are the primary
    evidence for retraining decisions. Because labels arrive quickly here,
    measured performance is the strongest signal rather than a lagging one.</p>

    {perf_block}
    {performance_table()}

    {flag_block}
    <p>Risk-tier mix shows the operational load each period places on the floor
    team. Useful for calibrating review capacity against the High-risk count.</p>

    {section_title("target", "Section 3", "Target Drift")}
    <p>Actual defect rate per period versus the training baseline, as a rate
    comparison rather than a distribution distance, which is more interpretable for a
    binary outcome. A material shift signals the shop's underlying quality
    profile has changed, which can affect calibration even when inputs are stable.</p>

    {target_block}
    {target_table()}

    {section_title("prediction", "Section 4", "Prediction Drift")}
    <p>Distribution of the model's predicted probabilities versus the
    validation-set reference, measured with the same distance method used for
    features. This is a label-free early indicator: it catches the model
    behaving differently regardless of which input moved. Here it corroborates
    the performance read rather than leading it.</p>

    {pdrift_block}
    {prediction_drift_table()}

    {pdist_block}

    {section_title("feature", "Section 5", "Feature Drift")}
    <p>Per-feature distance between each period's input distribution and the
    training reference (Jensen-Shannon for categoricals, normed Wasserstein for
    numericals). Values at or above {DRIFT_SCORE_THRESHOLD} are flagged. Feature
    drift is diagnostic because it helps explain a performance change if one occurs,
    but on its own does not establish that the model is wrong.</p>

    {heat_block}

    <p>Latest-period detail ({latest_name}), ordered by distance:</p>
    {feature_drift_table()}

    {section_title("quality", "Section 6", "Data Quality")}
    <p>Null rates, cardinality, and unseen categories for the latest scoring
    period. Unseen categories are absorbed by the model's
    <code>unknown_value=-1</code> encoding but reduce prediction quality for
    affected jobs.</p>

    {dq_block}
    {data_quality_table()}

    {section_title("log", "Section 7", "Monitoring Log")}
    <p>Model version history and run metadata for traceability.</p>

    {version_history_table()}

    <div class="callout">
      <strong>Note on data.</strong> These periods are generated from a single
      simulation seed, so genuine drift is expected to read near zero across all
      layers and performance is expected to be roughly stable. The report
      demonstrates the monitoring apparatus and decision logic; it is not a
      claim that this dataset is drifting.
    </div>

    <div class="callout">
      <strong>MLflow run:</strong> {run_id} &nbsp;·&nbsp;
      <strong>Registry:</strong> defect_risk_scorer v{model_ver} (Production) &nbsp;·&nbsp;
      <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")}
    </div>

  </main>
</div>
</body>
</html>'''

OUTPUT.write_text(html, encoding="utf-8")
print(f"\nMonitoring report written to {OUTPUT.resolve()}")
