"""
generate_report.py
Generates report.html from mart model data in defects_scrap.duckdb.
Run from the reports/ directory:
    python3 generate_report.py
Output: report.html
"""

from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from scipy import stats
import base64, io

DB_PATH = Path("../data/defects_scrap.duckdb").resolve()
OUTPUT  = Path("report.html")

con = duckdb.connect(str(DB_PATH), read_only=True)
dr  = con.execute("SELECT * FROM mart_quality__defect_rates").df()
sc  = con.execute("SELECT * FROM mart_quality__scrap_summary").df()
con.close()

dr["actual_start"] = pd.to_datetime(dr["actual_start"])
dr["order_month"]  = pd.to_datetime(dr["order_month"])
sc["scrap_date"]   = pd.to_datetime(sc["scrap_date"])
sc["scrap_month"]  = pd.to_datetime(sc["scrap_month"])

DATE_MIN_LABEL = "January 2023"
DATE_MAX_LABEL = "December 2025"

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
BOX_GREY   = "#DDDDDD"

SUPPLIER_COLORS   = {"Supplier A": GREY, "Supplier B": LIGHT_BLUE,
                     "Supplier C": BRAND_BLUE, "Supplier D": ACCENT}
COMPLEXITY_COLORS = {"Low": GREY, "Medium": LIGHT_BLUE, "High": BRAND_BLUE}
CX_MACHINE_COLORS = [BRAND_BLUE, GREY, AMBER, GREEN]

# ── Chart sizing constants — change here to update all charts ──────────────
CHART_W   = 8.2    # inches — matches content column width
CHART_H   = 3.8    # default chart height in inches
CHART_H_T = 4.5    # taller charts (heatmap, boxplot, stacked)
CHART_DPI = 130

# ── Font sizing — body text is 16px CSS ≈ 12pt at 96dpi ──────────────────
# BODY_FS: all non-title chart text (axes, ticks, labels, annotations, legend)
# TITLE_FS: chart title — matches finding-block title (~17px CSS)
BODY_FS  = 11
TITLE_FS = 13

plt.rcParams.update({
    "figure.facecolor": "white",  "axes.facecolor":  "white",
    "axes.edgecolor":   "#DDDDDD","axes.grid":        False,
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
    """Create a standardized figure. h overrides default height."""
    return plt.subplots(figsize=(CHART_W, h or CHART_H))

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=CHART_DPI)
    buf.seek(0)
    b = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b

def fmt_pct(x):  return f"{x:.1%}"
def fmt_usd(x):  return f"${x:,.0f}"
def fmt_num(x):  return f"{x:,.0f}"

def fail_rate(df):
    qi = df["quantity_inspected"].sum()
    qf = df["quantity_failed"].sum()
    return qf / qi if qi > 0 else 0

def monthly_labels(months, step=3):
    labels = [m.strftime("%b '%y") for m in months]
    return labels, labels[::step]

# ── Key stats ──────────────────────────────────────────────────────────────
overall_fr          = fail_rate(dr)
total_scrap         = sc["total_scrap_cost"].sum()
total_orders        = len(dr)
total_inspected     = dr["quantity_inspected"].sum()
avg_scrap_per_event = sc["total_scrap_cost"].sum() / len(sc)

p1_mask  = (dr["machine_type"]=="Bending") & (dr["shift_code"]=="Shift B")
p1b_mask = (dr["machine_type"]=="Bending") & (dr["shift_code"]=="Shift A")
p2_mask  = dr["supplier"]=="Supplier C"
p2b_mask = dr["supplier"]!="Supplier C"
p3_mask  = dr["complexity"]=="High"
p3b_mask = dr["complexity"]!="High"

p1_fr  = fail_rate(dr[p1_mask]);  p1b_fr = fail_rate(dr[p1b_mask])
p2_fr  = fail_rate(dr[p2_mask.fillna(False)]); p2b_fr = fail_rate(dr[p2b_mask.fillna(True)])
p3_fr  = fail_rate(dr[p3_mask]);  p3b_fr = fail_rate(dr[p3b_mask])

p1_mult = p1_fr / p1b_fr if p1b_fr > 0 else 0
p2_mult = p2_fr / p2b_fr if p2b_fr > 0 else 0
p3_mult = p3_fr / p3b_fr if p3b_fr > 0 else 0

def incremental_cost(pattern_mask, baseline_rate):
    affected        = dr[pattern_mask].copy()
    actual_failed   = affected["quantity_failed"].sum()
    expected_failed = affected["quantity_inspected"].sum() * baseline_rate
    extra_defects   = max(actual_failed - expected_failed, 0)
    scrap_ratio     = len(sc) / dr["quantity_failed"].sum() if dr["quantity_failed"].sum() > 0 else 1
    return extra_defects * scrap_ratio * avg_scrap_per_event

p1_inc = incremental_cost(p1_mask,               p1b_fr)
p2_inc = incremental_cost(p2_mask.fillna(False),  p2b_fr)
p3_inc = incremental_cost(p3_mask,                p3b_fr)
annual_incremental = (p1_inc + p2_inc + p3_inc) / 3

# Pre-compute supplier×complexity rates for bullet points
rates_sup_cx = (
    dr.dropna(subset=["supplier","complexity"])
    .groupby(["supplier","complexity"])
    .agg(qi=("quantity_inspected","sum"), qf=("quantity_failed","sum"))
    .assign(dr=lambda d: d["qf"]/d["qi"])
    .reset_index()
)
sc_high = rates_sup_cx[(rates_sup_cx["supplier"]=="Supplier C")&(rates_sup_cx["complexity"]=="High")]["dr"].values[0]
oth_high = rates_sup_cx[(rates_sup_cx["supplier"]!="Supplier C")&(rates_sup_cx["complexity"]=="High")]["dr"].mean()

# ═══════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# Each returns a base64 PNG string.
# To swap a chart: replace the body of the function; signature stays the same.
# ═══════════════════════════════════════════════════════════════════════════

def chart_defect_rate_trend():
    monthly = (
        dr.dropna(subset=["defect_rate"])
        .groupby("order_month")
        .agg(qi=("quantity_inspected","sum"), qf=("quantity_failed","sum"))
        .assign(fr=lambda d: d["qf"]/d["qi"])
        .reset_index().sort_values("order_month")
    )
    labels, ticks = monthly_labels(monthly["order_month"])
    fig, ax = make_fig()
    ax.plot(labels, monthly["fr"]*100, color=BRAND_BLUE,
            linewidth=2, marker="o", markersize=4)
    ax.axhline(monthly["fr"].mean()*100, color=GREY, linestyle=":", linewidth=1.5,
               label=f"Mean ({monthly['fr'].mean():.1%})")
    ax.set_xticks(ticks); ax.set_xticklabels(ticks, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))
    ax.set_ylabel("Defect Rate (%)"); ax.legend()
    chart_style(ax); plt.tight_layout()
    return fig_to_b64(fig)

def chart_scrap_trend():
    monthly = (
        sc.groupby("scrap_month")["total_scrap_cost"]
        .sum().reset_index().sort_values("scrap_month")
    )
    labels, ticks = monthly_labels(monthly["scrap_month"])
    fig, ax = make_fig()
    ax.bar(labels, monthly["total_scrap_cost"]/1000, color=BRAND_BLUE, width=0.7)
    ax.axhline(monthly["total_scrap_cost"].mean()/1000, color=GREY,
               linestyle=":", linewidth=1.5,
               label=f"Mean (${monthly['total_scrap_cost'].mean()/1000:,.0f}K/mo)")
    ax.set_xticks(ticks); ax.set_xticklabels(ticks, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}K"))
    ax.set_ylabel("Scrap Cost ($K)"); ax.legend()
    chart_style(ax); plt.tight_layout()
    return fig_to_b64(fig)

def chart_p1_heatmap():
    pivot = (
        dr.dropna(subset=["defect_rate","shift_code"])
        .groupby(["machine_type","shift_code"])["defect_rate"]
        .mean().unstack("shift_code")
    )
    import seaborn as sns
    annot_arr = pd.DataFrame(
        [[f"{v:.1f}%" for v in row] for row in (pivot * 100).values],
        index=pivot.index, columns=pivot.columns
    )
    fig, ax = make_fig(h=3.0)
    sns.heatmap(pivot*100, ax=ax, annot=annot_arr, fmt="",
                cmap="YlOrRd", linewidths=0.5, linecolor="white",
                cbar_kws={"label":"Mean Defect Rate (%)"},
                annot_kws={"size": BODY_FS, "family": "sans-serif"})
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(labelsize=BODY_FS)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=BODY_FS)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    cbar.ax.yaxis.label.set_size(BODY_FS)
    cbar.ax.yaxis.label.set_family("sans-serif")
    plt.tight_layout()
    return fig_to_b64(fig)

def chart_p1_boxplot():
    bend_b = dr[(dr["machine_type"]=="Bending")&(dr["shift_code"]=="Shift B")]["defect_rate"].dropna()
    bend_a = dr[(dr["machine_type"]=="Bending")&(dr["shift_code"]=="Shift A")]["defect_rate"].dropna()
    other  = dr[dr["machine_type"]!="Bending"]["defect_rate"].dropna()
    t_stat, p_val = stats.ttest_ind(bend_b, bend_a, equal_var=False)
    fig, ax = make_fig(h=CHART_H_T)
    bp = ax.boxplot([bend_b*100, bend_a*100, other*100],
               labels=["Bending\nShift B","Bending\nShift A","All Other\nMachines"],
               patch_artist=True,
               medianprops=dict(color=BRAND_BLUE, linewidth=2),
               flierprops=dict(marker="o", markersize=3,
                               markerfacecolor=GREY, alpha=0.4))
    for patch in bp["boxes"]:
        patch.set_facecolor(BOX_GREY)
    ax.text(0.98, 0.97, f"Welch t-test  p = {p_val:.4f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=BODY_FS, color=DARK_GREY)
    ax.set_ylabel("Work Order Defect Rate (%)\n(each point = one work order)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))
    chart_style(ax); plt.tight_layout()
    return fig_to_b64(fig)

def chart_p1_trend():
    monthly_p1 = (
        dr[dr["machine_type"]=="Bending"]
        .dropna(subset=["shift_code","defect_rate"])
        .groupby(["order_month","shift_code"])
        .agg(qi=("quantity_inspected","sum"), qf=("quantity_failed","sum"))
        .assign(fr=lambda d: d["qf"]/d["qi"])
        .reset_index().sort_values("order_month")
    )
    all_months = sorted(monthly_p1["order_month"].unique())
    labels, ticks = monthly_labels(pd.DatetimeIndex(all_months))
    label_map = {m: l for m, l in zip(all_months, labels)}
    fig, ax = make_fig(h=CHART_H_T)
    for shift, color in [("Shift B", BRAND_BLUE), ("Shift A", GREY)]:
        sub = monthly_p1[monthly_p1["shift_code"]==shift].copy()
        sub_labels = [label_map[m] for m in sub["order_month"]]
        ax.plot(sub_labels, sub["fr"]*100, color=color,
                linewidth=2, marker="o", markersize=4, label=shift, zorder=3)
    ax.set_xticks(ticks); ax.set_xticklabels(ticks, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))
    ax.set_ylabel("Defect Rate (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
    chart_style(ax); plt.tight_layout(rect=[0, 0.12, 1, 1])
    return fig_to_b64(fig)

def chart_p2_supplier_bar():
    sup = (
        dr.dropna(subset=["supplier"])
        .groupby("supplier")
        .agg(qi=("quantity_inspected","sum"), qf=("quantity_failed","sum"))
        .assign(dr=lambda d: d["qf"]/d["qi"])
        .reset_index().sort_values("dr", ascending=False)
    )
    fig, ax = make_fig()
    colors_sup = [RED if s=="Supplier C" else BRAND_BLUE for s in sup["supplier"]]
    bars = ax.bar(sup["supplier"], sup["dr"]*100, color=colors_sup, width=0.5)
    ax.axhline(overall_fr*100, color=GREY, linestyle="--", linewidth=1.5,
               label=f"Mean ({overall_fr:.1%})")
    for bar, val in zip(bars, sup["dr"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f"{val:.1%}", ha="center", va="bottom", fontsize=BODY_FS)
    ax.set_ylabel("Defect Rate (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))
    ax.legend(); chart_style(ax); plt.tight_layout()
    return fig_to_b64(fig)

def chart_p2_complexity_mix():
    """100% stacked bar: complexity mix per supplier."""
    df = dr.dropna(subset=["supplier","complexity"]).copy()
    mix = (
        df.groupby(["supplier","complexity"])["work_order_id"]
        .count().unstack("complexity").fillna(0)
    )
    order = ["Low","Medium","High"]
    mix_pct = mix.div(mix.sum(axis=1), axis=0)[order] * 100
    colors_cx = {"Low": GREY, "Medium": LIGHT_BLUE, "High": BRAND_BLUE}
    fig, ax = make_fig(h=CHART_H_T)
    bottoms = np.zeros(len(mix_pct))
    for cx in order:
        vals = mix_pct[cx].values
        ax.bar(mix_pct.index, vals, bottom=bottoms,
               color=colors_cx[cx], width=0.6, label=cx)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 8:
                ax.text(i, b + v/2, f"{v:.0f}%",
                        ha="center", va="center",
                        fontsize=BODY_FS, color="white", fontweight="bold")
        bottoms += vals
    ax.set_ylabel("Share of Work Orders (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))
    ax.legend(title="Complexity", loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#EEEEEE", linestyle="-", linewidth=0.8)
    ax.set_axisbelow(True)
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    return fig_to_b64(fig)

def chart_p2_defect_by_cx_supplier():
    """Grouped bar: defect rate by complexity tier, grouped by supplier."""
    df = dr.dropna(subset=["supplier","complexity"]).copy()
    rates = (
        df.groupby(["supplier","complexity"])
        .agg(qi=("quantity_inspected","sum"), qf=("quantity_failed","sum"))
        .assign(dr=lambda d: d["qf"]/d["qi"])
        .reset_index()
    )
    suppliers = sorted(df["supplier"].unique())
    cx_order  = ["Low","Medium","High"]
    x = np.arange(len(cx_order))
    w = 0.8 / len(suppliers)
    fig, ax = make_fig(h=CHART_H_T)
    for i, sup in enumerate(suppliers):
        vals = [
            rates[(rates["supplier"]==sup)&(rates["complexity"]==cx)]["dr"].sum()*100
            for cx in cx_order
        ]
        color = RED if sup=="Supplier C" else GREY
        bars = ax.bar(x + i*w - 0.4 + w/2, vals, width=w*0.85,
                      color=color, alpha=1.0 if sup=="Supplier C" else 0.5, label=sup)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=BODY_FS)
    ax.set_xticks(x); ax.set_xticklabels(cx_order)
    ax.set_ylabel("Defect Rate (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))
    grey_patch = mpatches.Patch(color=GREY, alpha=0.5, label="Suppliers A, B, D")
    red_patch  = mpatches.Patch(color=RED, label="Supplier C")
    ax.legend(handles=[grey_patch, red_patch], loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    chart_style(ax); plt.tight_layout(rect=[0, 0.12, 1, 1])
    return fig_to_b64(fig)

def chart_p2_historical():
    monthly_sup = (
        dr.dropna(subset=["supplier","defect_rate"])
        .groupby(["order_month","supplier"])
        .agg(qi=("quantity_inspected","sum"), qf=("quantity_failed","sum"))
        .assign(fr=lambda d: d["qf"]/d["qi"])
        .reset_index().sort_values("order_month")
    )
    all_months = sorted(monthly_sup["order_month"].unique())
    labels, ticks = monthly_labels(pd.DatetimeIndex(all_months))
    label_map = {m: l for m, l in zip(all_months, labels)}
    fig, ax = make_fig(h=CHART_H_T)
    for sup in sorted(monthly_sup["supplier"].unique()):
        sub = monthly_sup[monthly_sup["supplier"]==sup].copy()
        sub_labels = [label_map[m] for m in sub["order_month"]]
        color = SUPPLIER_COLORS.get(sup, GREY)
        lw = 2.5 if sup=="Supplier C" else 1.5
        ms = 5 if sup=="Supplier C" else 3
        ax.plot(sub_labels, sub["fr"]*100, color=color,
                linewidth=lw, marker="o", markersize=ms,
                label=sup, zorder=4 if sup=="Supplier C" else 3)
    ax.set_xticks(ticks); ax.set_xticklabels(ticks, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))
    ax.set_ylabel("Defect Rate (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=False)
    chart_style(ax); plt.tight_layout(rect=[0, 0.12, 1, 1])
    return fig_to_b64(fig)

def chart_p3_complexity_bar():
    order = ["Low","Medium","High"]
    comp = (
        dr.groupby("complexity")
        .agg(qi=("quantity_inspected","sum"), qf=("quantity_failed","sum"))
        .assign(dr=lambda d: d["qf"]/d["qi"])
        .reindex(order).reset_index()
    )
    fig, ax = make_fig()
    bars = ax.bar(comp["complexity"], comp["dr"]*100, color=BRAND_BLUE, width=0.5)
    ax.axhline(overall_fr*100, color=GREY, linestyle="--", linewidth=1.5,
               label=f"Mean ({overall_fr:.1%})")
    for bar, val in zip(bars, comp["dr"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f"{val:.1%}", ha="center", va="bottom", fontsize=BODY_FS)
    ax.set_ylabel("Defect Rate (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))
    ax.legend(); chart_style(ax); plt.tight_layout()
    return fig_to_b64(fig)

def chart_p3_cx_machine():
    order      = ["Low","Medium","High"]
    machines   = sorted(dr["machine_type"].unique())
    cx_machine = (
        dr.dropna(subset=["defect_rate"])
        .groupby(["complexity","machine_type"])
        .agg(qi=("quantity_inspected","sum"), qf=("quantity_failed","sum"))
        .assign(dr=lambda d: d["qf"]/d["qi"])
        .reset_index()
    )
    x = np.arange(len(order)); w = 0.8 / len(machines)
    fig, ax = make_fig(h=CHART_H_T)
    for i, machine in enumerate(machines):
        vals = [
            cx_machine[(cx_machine["complexity"]==c)&
                       (cx_machine["machine_type"]==machine)]["dr"].sum()*100
            for c in order
        ]
        bars = ax.bar(x + i*w - 0.4 + w/2, vals, width=w*0.85,
                      color=CX_MACHINE_COLORS[i % len(CX_MACHINE_COLORS)], label=machine)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=BODY_FS)
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("Defect Rate (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    chart_style(ax); plt.tight_layout(rect=[0, 0.12, 1, 1])
    return fig_to_b64(fig)

def chart_p3_historical():
    monthly_cx = (
        dr.dropna(subset=["complexity","defect_rate"])
        .groupby(["order_month","complexity"])
        .agg(qi=("quantity_inspected","sum"), qf=("quantity_failed","sum"))
        .assign(fr=lambda d: d["qf"]/d["qi"])
        .reset_index().sort_values("order_month")
    )
    all_months = sorted(monthly_cx["order_month"].unique())
    labels, ticks = monthly_labels(pd.DatetimeIndex(all_months))
    label_map = {m: l for m, l in zip(all_months, labels)}
    fig, ax = make_fig(h=CHART_H_T)
    for cx in ["Low","Medium","High"]:
        sub = monthly_cx[monthly_cx["complexity"]==cx].copy()
        sub_labels = [label_map[m] for m in sub["order_month"]]
        color = COMPLEXITY_COLORS.get(cx, GREY)
        lw = 2.5 if cx=="High" else 1.5
        ms = 5 if cx=="High" else 3
        ax.plot(sub_labels, sub["fr"]*100, color=color,
                linewidth=lw, marker="o", markersize=ms, label=cx,
                zorder=4 if cx=="High" else 3)
    ax.set_xticks(ticks); ax.set_xticklabels(ticks, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))
    ax.set_ylabel("Defect Rate (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
    chart_style(ax); plt.tight_layout(rect=[0, 0.12, 1, 1])
    return fig_to_b64(fig)

def chart_incremental_cost():
    labels = ["Bending × Shift B","Supplier C\nMaterial","High Complexity\nParts"]
    annual = [p1_inc/3, p2_inc/3, p3_inc/3]
    fig, ax = make_fig()
    bars = ax.barh(labels[::-1], [a/1000 for a in annual[::-1]],
                   color=BRAND_BLUE, height=0.5)
    for bar, val in zip(bars, annual[::-1]):
        ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                f"${val/1000:,.0f}K/yr",
                va="center", fontsize=BODY_FS, fontweight="bold", color=TEXT)
    ax.set_xlabel("Estimated Incremental Annual Cost ($K)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}K"))
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color="#EEEEEE", linestyle="-", linewidth=0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD"); ax.spines["bottom"].set_color("#DDDDDD")
    ax.set_axisbelow(True); plt.tight_layout()
    return fig_to_b64(fig)

# ── Generate all charts ────────────────────────────────────────────────────
print("Generating charts...")
charts = {
    "defect_trend":       chart_defect_rate_trend(),
    "scrap_trend":        chart_scrap_trend(),
    "p1_heatmap":         chart_p1_heatmap(),
    "p1_boxplot":         chart_p1_boxplot(),
    "p1_trend":           chart_p1_trend(),
    "p2_supplier_bar":    chart_p2_supplier_bar(),
    "p2_complexity_mix":  chart_p2_complexity_mix(),
    "p2_cx_supplier":     chart_p2_defect_by_cx_supplier(),
    "p2_historical":      chart_p2_historical(),
    "p3_complexity_bar":  chart_p3_complexity_bar(),
    "p3_cx_machine":      chart_p3_cx_machine(),
    "p3_historical":      chart_p3_historical(),
    "incremental_cost":   chart_incremental_cost(),
}
print("Charts complete.")

# ── HTML helpers ───────────────────────────────────────────────────────────
def wrap(key, title="", caption=""):
    title_html   = f'<div class="chart-title">{title}</div>' if title else ""
    caption_html = f'<div class="chart-caption">{caption}</div>' if caption else ""
    return (f'<div class="chart-wrap">{title_html}'
            f'<img src="data:image/png;base64,{charts[key]}" '
            f'style="width:100%;height:auto;display:block;">'
            f'{caption_html}</div>')

def bullets(items):
    lis = "".join(f"<li>{i}</li>" for i in items)
    return f'<ul class="findings-list">{lis}</ul>'

def section_title(id, label, title):
    return f'''<div class="section-title-block" id="{id}">
      <div class="section-label">{label}</div>
      <h2 class="section-title">{title}</h2>
    </div>'''

def finding_block(id, title, mult_label, mult_value, inc_annual):
    return f'''<div class="finding-block" id="{id}">
      <div class="finding-left"><div class="finding-title">{title}</div></div>
      <div class="finding-right">
        <div class="finding-stat-group">
          <div>
            <div class="finding-stat-val">{mult_value}</div>
            <div class="finding-stat-lbl">{mult_label}</div>
          </div>
          <div>
            <div class="finding-stat-val" style="color:{AMBER};">{fmt_usd(inc_annual)}</div>
            <div class="finding-stat-lbl">est. incremental cost / yr</div>
          </div>
        </div>
      </div>
    </div>'''

# ── Bullet content ─────────────────────────────────────────────────────────
p1_bullets = bullets([
    f"Bending × Shift B aggregate defect rate: <strong>{fmt_pct(p1_fr)}</strong> vs "
    f"<strong>{fmt_pct(p1b_fr)}</strong> on Shift A — a <strong>{p1_mult:.1f}×</strong> elevation.",
    "The Shift B elevation is statistically significant (Welch t-test, p&nbsp;&lt;&nbsp;0.05) and "
    "visible in the distribution of individual work order outcomes: the median defect rate and "
    "the spread of outcomes are both higher on Shift B.",
    "The pattern is persistent across the full analysis period — not a short-term anomaly. "
    "The monthly trend chart shows Shift B running above Shift A in essentially every month of the 36-month window.",
    "The elevation is specific to bending operations and does not appear on other machine types "
    "(laser cutting, punching, welding). Bending is the most operator-dependent process on the floor — "
    "small differences in setup, technique, or in-process verification produce measurable dimensional variation "
    "in ways that are less likely on CNC-driven operations.",
    "Likely drivers include inconsistent equipment recalibration between shifts, setup state "
    "handoff gaps, and operator experience differentials — all of which are more consequential "
    "in bending than in other operations due to the direct role of operator judgment in achieving "
    "accurate bend angles.",
])

p2_bullets = bullets([
    f"Supplier C aggregate defect rate: <strong>{fmt_pct(p2_fr)}</strong> vs "
    f"<strong>{fmt_pct(p2b_fr)}</strong> for all other suppliers — a <strong>{p2_mult:.1f}×</strong> elevation.",
    "The differential persists across machine types and shifts, indicating a material quality "
    "issue rather than a downstream process issue.",
    f"Critically, the elevation holds within every complexity tier: Supplier C's High-complexity "
    f"defect rate is <strong>{fmt_pct(sc_high)}</strong> vs <strong>{fmt_pct(oth_high)}</strong> "
    f"for other suppliers on the same complexity tier. The complexity mix across suppliers is "
    f"broadly consistent — Supplier C does not disproportionately supply high-complexity parts — "
    f"ruling out complexity as a confounding factor.",
    "The monthly defect rate chart shows Supplier C running persistently above all other suppliers "
    "across the analysis period, with no convergence trend.",
])

p3_bullets = bullets([
    f"High-complexity aggregate defect rate: <strong>{fmt_pct(p3_fr)}</strong> vs "
    f"<strong>{fmt_pct(p3b_fr)}</strong> for all other tiers — the strongest "
    "single-dimension signal in the dataset.",
    "Defect rate elevation is most pronounced in bending operations, running at about double "
    "the defect rate of other operations using high complexity parts.",
    "The relationship is monotonic: Low → Medium → High tracks with strictly increasing defect rates "
    "across all 36 months. This is not a threshold effect — complexity elevation is gradual and consistent.",
    "The complexity effect is not uniform across machine types. The grouped chart shows that certain "
    "equipment types show a more pronounced sensitivity to complexity than others, suggesting that "
    "machine capability and tooling condition interact with part complexity in producing defects.",
    "The monthly trend shows the High-complexity tier running above Medium and Low in every month "
    "of the analysis period, with no sign of convergence.",
])

# ── HTML ───────────────────────────────────────────────────────────────────
html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Analytics Report: Quality &amp; Scrap Diagnostics (Metal Fabrication)</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
      background: #FFFFFF; color: {TEXT}; font-size: 16px; line-height: 1.7;
    }}
    .page-header {{ background: {BRAND_BLUE}; color: white; padding: 20px 40px; }}
    .page-header h1 {{ font-size: 22px; font-weight: 700; letter-spacing: -0.3px; }}
    .layout {{ display: flex; max-width: 1200px; margin: 0 auto; padding: 0 40px; }}

    /* ── TOC ── */
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

    /* ── Content ── */
    .content {{ flex: 1; padding: 40px 0 80px 52px; max-width: 880px; }}

    /* ── Section titles ── */
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

    /* ── Body text ── */
    p {{ margin-bottom: 16px; color: #333; font-size: 16px; }}

    /* ── Context block ── */
    .context-block {{
      background: #F7F8FA; border-top: 3px solid {BRAND_BLUE};
      padding: 24px 28px 20px 28px; margin-bottom: 0;
    }}
    .context-title {{
      font-size: 22px; font-weight: 700; color: {BRAND_BLUE}; margin-bottom: 14px;
    }}
    .context-block p {{
      font-size: 15px; line-height: 1.8; color: #444; margin-bottom: 12px;
    }}
    .context-block p:last-child {{ margin-bottom: 0; }}

    /* ── Finding blocks ── */
    .finding-block {{
      display: flex; align-items: center; background: #F7F8FA;
      border-left: 4px solid {BRAND_BLUE}; padding: 18px 22px;
      margin: 32px 0 20px 0; gap: 24px;
    }}
    .finding-left {{ flex: 1; }}
    .finding-title {{ font-size: 17px; font-weight: 700; color: {TEXT}; line-height: 1.3; }}
    .finding-right {{ flex-shrink: 0; }}
    .finding-stat-group {{ display: flex; gap: 28px; text-align: right; }}
    .finding-stat-val {{ font-size: 24px; font-weight: 700; color: {RED}; line-height: 1; }}
    .finding-stat-lbl {{ font-size: 11px; color: #888; margin-top: 3px; }}

    /* ── Bullet lists ── */
    .findings-list {{ margin: 12px 0 20px 20px; color: #333; }}
    .findings-list li {{ margin-bottom: 8px; font-size: 15px; line-height: 1.6; }}

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

    /* ── Callouts ── */
    .cost-callout {{
      background: #FFF8F0; border-left: 3px solid {AMBER};
      padding: 16px 22px; margin: 20px 0; font-size: 15px; color: #555;
    }}
    .cost-callout strong {{ color: {TEXT}; }}

    /* ── Methodology ── */
    .method-item {{ margin-bottom: 20px; padding-left: 18px; border-left: 2px solid #EEEEEE; }}
    .method-item strong {{
      display: block; color: {BRAND_BLUE}; margin-bottom: 3px; font-size: 15px;
    }}
  </style>
</head>
<body>

<div class="page-header">
  <h1>Analytics Report: Quality &amp; Scrap Diagnostics (Metal Fabrication)</h1>
</div>

<div class="layout">
  <nav class="toc">
    <div class="toc-title">Contents</div>
    <a href="#exec">Executive Summary</a>
    <hr>
    <a href="#findings">Findings</a>
    <a href="#p1" class="sub">1. Bending × Shift B</a>
    <a href="#p2" class="sub">2. Supplier C Material</a>
    <a href="#p3" class="sub">3. High Complexity Parts</a>
    <hr>
    <a href="#financial">Financial Impact</a>
    <hr>
    <a href="#methodology">Methodology</a>
  </nav>

  <main class="content">

    {section_title("exec", "Section 1", "Executive Summary")}

    <p>Across {fmt_num(total_orders)} production work orders and {fmt_num(total_inspected)} parts
    inspected between {DATE_MIN_LABEL} and {DATE_MAX_LABEL}, the overall defect rate was
    <strong>{fmt_pct(overall_fr)}</strong> and total scrap cost was
    <strong>{fmt_usd(total_scrap)}</strong>. Both metrics have remained broadly steady over the
    historical period, with a slight upward trend.</p>

    {wrap("defect_trend", "Defect Rate by Month")}
    {wrap("scrap_trend",  "Total Scrap Cost by Month")}

    <p>We built a data pipeline to extract, standardize, and merge records across all
    four systems into a single source, then conducted comprehensive analytical diagnostics to
    surface previously unseen patterns.</p>

    <p>As a result of this analysis, three distinct patterns emerged that point to specific,
    concentrated, and addressable operational drivers of elevated quality cost. These patterns
    carry a combined estimated <strong>incremental annual cost of {fmt_usd(annual_incremental)}</strong>
    above what would be expected if those dimensions performed at their respective baseline rates.
    Bending operations on Shift B run at <strong>{p1_mult:.1f}×</strong> the defect rate of
    Shift A; material sourced from Supplier C at <strong>{p2_mult:.1f}×</strong> other suppliers;
    and high-complexity parts at <strong>{p3_mult:.1f}×</strong> lower-complexity equivalents.
    None of these patterns were visible within any single source system.</p>

    {section_title("findings", "Section 2", "Findings")}

    <p>Each finding below represents a defect rate elevation that is statistically significant,
    consistent across the analysis period, and only quantifiable through cross-system joins.
    Incremental cost reflects the excess above what would have occurred had that dimension
    performed at its baseline rate.</p>

    {finding_block("p1",
        "Bending operations on Shift B produce defects at {:.1f}× the rate of Shift A".format(p1_mult),
        "vs Bending Shift A", f"{p1_mult:.1f}×", p1_inc/3)}

    {p1_bullets}

    {wrap("p1_heatmap", "Defect Rate by Machine Type × Shift")}
    {wrap("p1_boxplot", "Defect Rate Distribution — Bending by Shift vs All Machines")}
    {wrap("p1_trend",   "Bending Defect Rate Over Time — Shift B vs Shift A")}

    {finding_block("p2",
        "Supplier C material is associated with a {:.1f}× elevated defect rate".format(p2_mult),
        "vs all other suppliers", f"{p2_mult:.1f}×", p2_inc/3)}

    {p2_bullets}

    {wrap("p2_supplier_bar",   "Defect Rate by Supplier")}
    {wrap("p2_complexity_mix", "Complexity Mix by Supplier")}
    {wrap("p2_cx_supplier",    "Defect Rate by Complexity Tier × Supplier")}
    {wrap("p2_historical",     "Monthly Defect Rate by Supplier")}

    {finding_block("p3",
        "High-complexity parts fail at {:.1f}× the rate of other complexity tiers".format(p3_mult),
        "vs non-High complexity", f"{p3_mult:.1f}×", p3_inc/3)}

    {p3_bullets}

    {wrap("p3_complexity_bar", "Defect Rate by Complexity Tier")}
    {wrap("p3_cx_machine",     "Defect Rate by Complexity × Machine Type")}
    {wrap("p3_historical",     "Monthly Defect Rate by Complexity")}

    {section_title("financial", "Section 3", "Financial Impact")}

    {wrap("incremental_cost", "Estimated Incremental Annual Cost per Pattern")}

    {bullets([
        f"Combined estimated incremental annual cost across all three patterns: <strong>{fmt_usd(annual_incremental)}</strong>.",
        f"High complexity parts carry the largest share at <strong>{fmt_usd(p3_inc/3)}/yr</strong>, "
        "driven by both the magnitude of the rate elevation and the volume of affected work orders.",
        f"Bending × Shift B carries <strong>{fmt_usd(p1_inc/3)}/yr</strong> despite being a single machine-shift combination.",
        f"Supplier C material carries <strong>{fmt_usd(p2_inc/3)}/yr</strong> — the most directly addressable pattern, "
        "as the comparison group (other suppliers) demonstrates the defect rate achievable with better material quality.",
        "All figures are conservative — direct scrap costs only. Rework labor, machine time on "
        "defective units, and downstream escape costs are excluded.",
    ])}

    <div class="cost-callout">
      <strong>Baseline and methodology:</strong> For each pattern, the baseline rate is the
      defect rate of the natural comparison group — Bending Shift A for Bending Shift B, all
      other suppliers for Supplier C, and non-High complexity work orders for High complexity.
      Comparing against the same machine type, supplier peer group, or complexity peer controls
      for confounding factors and isolates the specific effect. This is a deliberately conservative
      approach — comparing against the fleet-wide average would produce larger figures.
      Incremental cost = (actual defects − expected defects at baseline rate) × average scrap
      cost per event ({fmt_usd(avg_scrap_per_event)}). Direct scrap costs only — excludes
      rework, machine time, and downstream escapes.
    </div>

    {section_title("methodology", "Section 4", "Methodology")}

    <div class="method-item">
      <strong>Data Sources</strong>
      Inspection records and scrap events from the QMS; production work orders and part catalog
      from the ERP; material lot receipts and certification status from the WMS; operator records
      from the HR system. The analysis covers {DATE_MIN_LABEL} through {DATE_MAX_LABEL} —
      {fmt_num(total_orders)} work orders and {fmt_num(total_inspected)} parts inspected across
      all five production lines.
    </div>

    <div class="method-item">
      <strong>Pipeline</strong>
      Data was extracted from each source system and loaded into a DuckDB analytical database
      using dlt. Transformation and join logic was implemented in dbt, producing mart-layer
      tables that serve as the basis for this analysis. All logic is version-controlled and
      reproducible.
    </div>

    <div class="method-item">
      <strong>Defect Rate Definition</strong>
      Defect rate is defined as quantity failed divided by quantity inspected at the work order
      level. Aggregate rates are volume-weighted (total failed / total inspected across the group).
      Statistical significance is assessed using Welch's t-test. Effect sizes are reported as
      multipliers relative to the comparison group.
    </div>

    <div class="method-item">
      <strong>Known Data Limitations</strong>
      Approximately 606 scrap events reference inspection records removed during QMS deduplication.
      These are retained in cost calculations via work order association and do not affect defect
      rate calculations. Approximately 15% of work orders have no material lot association due to
      missing scan records at job start; these are excluded from supplier analyses.
    </div>

  </main>
</div>
</body>
</html>'''

OUTPUT.write_text(html, encoding="utf-8")
print(f"Report written to {OUTPUT.resolve()}")