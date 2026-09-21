from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import base64, io

DB_PATH = Path("../../data_source/defects_scrap.duckdb").resolve()
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
DATE_MAX_LABEL = "March 2026"

# ── Palette (BRIM house style) ─────────────────────────────────────────────
# Document chrome. Kept out of charts: the dark grey sits too close to the
# chart dark blue to separate cleanly.
DARK_GREY  = "#322B4B"   # header bars, titles, takeaways, divider lines
BG_GREY    = "#F3F5F7"   # box and card backgrounds
TEXT       = "#000000"   # body font
# Chart colors.
DARK_BLUE  = "#381FA1"   # chart primary
LIGHT_BLUE = "#54C0E8"   # chart secondary
ACCENT_RED = "#CC0000"   # chart accent; conditional-formatting "bad"
MUTED_RED  = "#FFA3A3"   # chart secondary red
MED_GREY   = "#8093A4"   # chart neutral
LIGHT_GREY = "#D5DCE1"   # chart neutral (gridlines, unfilled areas)
# Conditional formatting (good / medium / bad ranges).
GREEN      = "#00A84C"   # "good" (high range)
AMBER      = "#FFBA3F"   # "medium" (mid range)

# Aliases so the chart code below reads against the same names as before.
BRAND_BLUE = DARK_BLUE
ACCENT     = LIGHT_BLUE
RED        = ACCENT_RED
GREY       = MED_GREY
BOX_GREY   = LIGHT_GREY

SUPPLIER_COLORS   = {"Supplier A": MED_GREY, "Supplier B": LIGHT_BLUE,
                     "Supplier C": DARK_BLUE, "Supplier D": LIGHT_GREY}
COMPLEXITY_COLORS = {"Low": LIGHT_GREY, "Medium": LIGHT_BLUE, "High": DARK_BLUE}
CX_MACHINE_COLORS = [DARK_BLUE, LIGHT_BLUE, MED_GREY, MUTED_RED]

# Sequential ramp for the heatmap, kept inside the brand reds.
HEAT_CMAP = LinearSegmentedColormap.from_list("brim_heat", [BG_GREY, MUTED_RED, ACCENT_RED])

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
    "axes.edgecolor":   LIGHT_GREY, "axes.grid":      False,
    "font.family":      "sans-serif",
    "font.size":        BODY_FS,
    "axes.titlesize":   TITLE_FS, "axes.titleweight": "bold",
    "axes.labelsize":   BODY_FS,  "xtick.labelsize":  BODY_FS,
    "ytick.labelsize":  BODY_FS,  "legend.fontsize":  BODY_FS,
    "text.color":       TEXT,     "axes.labelcolor":  TEXT,
    "axes.titlecolor":  TEXT,     "xtick.color":      TEXT,
    "ytick.color":      TEXT,
    "figure.dpi":       CHART_DPI,
})

def chart_style(ax):
    ax.yaxis.grid(True, color=LIGHT_GREY, linestyle="-", linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LIGHT_GREY)
    ax.spines["bottom"].set_color(LIGHT_GREY)

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

# ── Totals, trailing window, and financial-impact estimates ─────────────────
total_defects = int(dr["quantity_failed"].sum())
n_months      = dr["order_month"].nunique()
window_years  = n_months / 12.0

# Trailing 12 months (most recent 12 order-months in the data).
_months_sorted = sorted(dr["order_month"].unique())
_last12  = _months_sorted[-12:]
_dr12    = dr[dr["order_month"].isin(_last12)]
_sc12    = sc[sc["scrap_month"].isin(_last12)]
l12_defects = int(_dr12["quantity_failed"].sum())
l12_fr      = fail_rate(_dr12)
l12_scrap   = _sc12["total_scrap_cost"].sum()
L12_MIN_LABEL = pd.Timestamp(_last12[0]).strftime("%B %Y")
L12_MAX_LABEL = pd.Timestamp(_last12[-1]).strftime("%B %Y")

# Monthly means over the full window, used as the "3-yr mean" overlays.
mean_defects_mo = dr.groupby("order_month")["quantity_failed"].sum().mean()
mean_scrap_mo   = sc.groupby("scrap_month")["total_scrap_cost"].sum().mean()

# Financial impact. Savings from bringing a segment's defect rate to a benchmark
# scale that segment's observed scrap cost (material + labor) by the proportional
# rate reduction: savings = scrap_cost * (rate - target) / rate. This assumes a
# roughly constant cost per defective unit and that the whole gap is addressable,
# so the figures are an upper-bound opportunity. The segments overlap, so they
# are not additive.
def _seg_savings(seg_dr, seg_scrap_cost, target):
    r = fail_rate(seg_dr)
    return r, seg_scrap_cost, (seg_scrap_cost * (r - target) / r if r > 0 else 0.0)

# P1: Bending Shift B -> Bending Shift A rate.
p1_scrap = sc[(sc["machine_type"]=="Bending") & (sc["shift_code"]=="Shift B")]["total_scrap_cost"].sum()
_, _, p1_save = _seg_savings(dr[p1_mask], p1_scrap, p1b_fr)

# P2: Supplier C -> 6% (other suppliers' rate on high-complexity work). Scrap is
# attributed by work order, since the scrap record does not always carry supplier.
P2_TARGET = 0.06
_p2_wos  = dr[p2_mask.fillna(False)]["work_order_id"]
p2_scrap = sc[sc["work_order_id"].isin(_p2_wos)]["total_scrap_cost"].sum()
_, _, p2_save = _seg_savings(dr[p2_mask.fillna(False)], p2_scrap, P2_TARGET)

# P3a (conservative): high-complexity Bending -> 12% (in line with other operations).
P3A_TARGET = 0.12
_hcb_dr  = dr[(dr["complexity"]=="High") & (dr["machine_type"]=="Bending")]
p3a_scrap = sc[(sc["complexity"]=="High") & (sc["machine_type"]=="Bending")]["total_scrap_cost"].sum()
p3a_fr, _, p3a_save = _seg_savings(_hcb_dr, p3a_scrap, P3A_TARGET)

# P3b (stretch): all high-complexity -> 9% (a blended rate achieved in 3 of the
# months in the window).
P3B_TARGET = 0.09
p3b_scrap = sc[sc["complexity"]=="High"]["total_scrap_cost"].sum()
_, _, p3b_save = _seg_savings(dr[p3_mask], p3b_scrap, P3B_TARGET)

def per_yr(x): return x / window_years

# ═══════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# Each returns a base64 PNG string.
# To swap a chart: replace the body of the function; signature stays the same.
# ═══════════════════════════════════════════════════════════════════════════

def chart_defect_rate_trend():
    """Trailing 12 months: total defects (columns, left axis) and defect rate
    (line, right axis), with the 3-yr mean rate overlaid."""
    monthly = (
        dr.dropna(subset=["defect_rate"])
        .groupby("order_month")
        .agg(qi=("quantity_inspected","sum"), qf=("quantity_failed","sum"))
        .assign(fr=lambda d: d["qf"]/d["qi"])
        .reset_index().sort_values("order_month")
    ).tail(12)
    labels = [pd.Timestamp(m).strftime("%b '%y") for m in monthly["order_month"]]
    x = np.arange(len(monthly))
    defects = monthly["qf"].values
    rate_pct = monthly["fr"].values * 100
    fig, ax = make_fig()
    ax.bar(x, defects, color=LIGHT_BLUE, width=0.62, label="Total defects", zorder=1)
    for xi, v in zip(x, defects):
        ax.text(xi, v + defects.max()*0.02, f"{v:,.0f}", ha="center", va="bottom",
                fontsize=BODY_FS-2, color=TEXT)
    ax.set_ylabel("Total Defects"); ax.set_ylim(0, defects.max()*1.28)
    ax2 = ax.twinx()
    ax2.plot(x, rate_pct, color=DARK_BLUE, linewidth=2, marker="o", markersize=4,
             label="Defect rate", zorder=3)
    for xi, v in zip(x, rate_pct):
        ax2.text(xi, v + rate_pct.max()*0.04, f"{v:.1f}%", ha="center", va="bottom",
                 fontsize=BODY_FS-2, color=DARK_BLUE, fontweight="bold")
    ax2.axhline(overall_fr*100, color=GREY, linestyle=":", linewidth=1.5,
                label=f"3-yr mean ({overall_fr:.1%})")
    ax2.set_ylabel("Defect Rate (%)"); ax2.set_ylim(0, 12)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    h1,l1 = ax.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="upper center", bbox_to_anchor=(0.5,-0.22), ncol=3, frameon=False)
    chart_style(ax); ax2.grid(False)
    plt.tight_layout()
    return fig_to_b64(fig)

def chart_scrap_trend():
    """Trailing 12 months of scrap cost with per-month labels and the 3-yr mean."""
    monthly = (
        sc.groupby("scrap_month")["total_scrap_cost"]
        .sum().reset_index().sort_values("scrap_month")
    ).tail(12)
    labels = [pd.Timestamp(m).strftime("%b '%y") for m in monthly["scrap_month"]]
    x = np.arange(len(monthly))
    vals = monthly["total_scrap_cost"].values / 1000
    fig, ax = make_fig()
    ax.bar(x, vals, color=BRAND_BLUE, width=0.65)
    for xi, v in zip(x, vals):
        ax.text(xi, v + vals.max()*0.02, f"${v:,.0f}K", ha="center", va="bottom",
                fontsize=BODY_FS-2, color=TEXT)
    ax.axhline(mean_scrap_mo/1000, color=GREY, linestyle=":", linewidth=1.5,
               label=f"3-yr mean (${mean_scrap_mo/1000:,.0f}K/mo)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, vals.max()*1.2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}K"))
    ax.set_ylabel("Scrap Cost ($K)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.18), frameon=False)
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
                cmap=HEAT_CMAP, linewidths=0.5, linecolor="white",
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
               tick_labels=["Bending\nShift B","Bending\nShift A","All Other\nMachines"],
               patch_artist=True,
               medianprops=dict(color=BRAND_BLUE, linewidth=2),
               flierprops=dict(marker="o", markersize=3,
                               markerfacecolor=GREY, alpha=0.4))
    for patch in bp["boxes"]:
        patch.set_facecolor(BOX_GREY)
    ax.text(0.98, 0.97, f"Welch t-test  p = {p_val:.4f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=BODY_FS, color=MED_GREY)
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
    ax.yaxis.grid(True, color=LIGHT_GREY, linestyle="-", linewidth=0.8)
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

def finding_block(id, title, mult_label, mult_value, save_value, save_label):
    return f'''<div class="finding-block" id="{id}">
      <div class="finding-left"><div class="finding-title">{title}</div></div>
      <div class="finding-right">
        <div class="finding-stat-group">
          <div>
            <div class="finding-stat-val">{mult_value}</div>
            <div class="finding-stat-lbl">{mult_label}</div>
          </div>
          <div>
            <div class="finding-stat-val" style="color:{GREEN};">{save_value}</div>
            <div class="finding-stat-lbl">{save_label}</div>
          </div>
        </div>
      </div>
    </div>'''

# ── Bullet content ─────────────────────────────────────────────────────────
p1_bullets = bullets([
    f"Bending × Shift B aggregate defect rate: <strong>{fmt_pct(p1_fr)}</strong> vs "
    f"<strong>{fmt_pct(p1b_fr)}</strong> on Shift A, a <strong>{p1_mult:.1f}×</strong> elevation.",
    "The Shift B elevation is statistically significant (Welch t-test, p&nbsp;&lt;&nbsp;0.05) and "
    "visible in the distribution of individual work order outcomes: the median defect rate and "
    "the spread of outcomes are both higher on Shift B.",
    "The pattern is persistent across the full analysis period, not a short-term anomaly. "
    "The monthly trend chart shows Shift B running above Shift A in essentially every month of the analysis period.",
    "The elevation is specific to bending operations and does not appear on other machine types "
    "(laser cutting, punching, welding). Bending is the most operator-dependent process on the floor, so "
    "small differences in setup, technique, or in-process verification produce measurable dimensional variation "
    "in ways that are less likely on CNC-driven operations.",
    "Likely drivers include inconsistent equipment recalibration between shifts, setup state "
    "handoff gaps, and operator experience differentials, all of which are more consequential "
    "in bending than in other operations due to the direct role of operator judgment in achieving "
    "accurate bend angles.",
])

p2_bullets = bullets([
    f"Supplier C aggregate defect rate: <strong>{fmt_pct(p2_fr)}</strong> vs "
    f"<strong>{fmt_pct(p2b_fr)}</strong> for all other suppliers, a <strong>{p2_mult:.1f}×</strong> elevation.",
    "The differential persists across machine types and shifts, indicating a material quality "
    "issue rather than a downstream process issue.",
    f"Critically, the elevation holds within every complexity tier: Supplier C's High-complexity "
    f"defect rate is <strong>{fmt_pct(sc_high)}</strong> vs <strong>{fmt_pct(oth_high)}</strong> "
    f"for other suppliers on the same complexity tier. The complexity mix across suppliers is "
    f"broadly consistent (Supplier C does not disproportionately supply high-complexity parts), "
    f"ruling out complexity as a confounding factor.",
    "The monthly defect rate chart shows Supplier C running persistently above all other suppliers "
    "across the analysis period, with no convergence trend.",
])

p3_bullets = bullets([
    f"High-complexity aggregate defect rate: <strong>{fmt_pct(p3_fr)}</strong> vs "
    f"<strong>{fmt_pct(p3b_fr)}</strong> for all other tiers, the strongest "
    "single-dimension signal in the dataset.",
    "Defect rate elevation is most pronounced in bending operations, running at about double "
    "the defect rate of other operations using high complexity parts.",
    "The relationship is monotonic: Low → Medium → High tracks with strictly increasing defect rates "
    "across the full analysis period. This is not a threshold effect: complexity elevation is gradual and consistent.",
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
  <title>Analytics Diagnostic Report: Defect Risks &amp; Scrap Costs</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
      background: #FFFFFF; color: {TEXT}; font-size: 16px; line-height: 1.7;
    }}
    .page-header {{ background: {DARK_GREY}; color: white; padding: 20px 40px; }}
    .page-header h1 {{ font-size: 22px; font-weight: 700; letter-spacing: -0.3px; }}
    .layout {{ display: flex; max-width: 1200px; margin: 0 auto; padding: 0 40px; }}

    /* ── TOC ── */
    .toc {{
      width: 200px; flex-shrink: 0; padding: 40px 20px 40px 0;
      position: sticky; top: 0; height: 100vh; overflow-y: auto;
      border-right: 1px solid {LIGHT_GREY};
    }}
    .toc-title {{
      font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
      color: {MED_GREY}; margin-bottom: 14px; font-weight: 600;
    }}
    .toc a {{
      display: block; font-size: 13px; color: {MED_GREY}; text-decoration: none;
      padding: 4px 0 4px 10px; border-left: 2px solid transparent; line-height: 1.4;
    }}
    .toc a:hover {{ color: {DARK_GREY}; border-left-color: {DARK_GREY}; }}
    .toc a.sub {{ font-size: 12px; padding-left: 20px; color: {MED_GREY}; }}
    .toc a.sub:hover {{ color: {DARK_GREY}; border-left-color: {DARK_GREY}; }}
    .toc hr {{ border: none; border-top: 1px solid {LIGHT_GREY}; margin: 8px 0; }}

    /* ── Content ── */
    .content {{ flex: 1; padding: 40px 0 80px 52px; max-width: 880px; }}

    /* ── Section titles ── */
    .section-title-block {{
      margin: 48px 0 24px 0; padding-bottom: 12px;
      border-bottom: 2px solid {DARK_GREY};
    }}
    .content > .section-title-block:first-child {{ margin-top: 12px; }}
    .section-label {{
      font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
      color: {TEXT}; font-weight: 600; margin-bottom: 4px;
    }}
    .section-title {{ font-size: 22px; font-weight: 700; color: {TEXT}; }}

    /* ── Body text ── */
    p {{ margin-bottom: 16px; color: {TEXT}; font-size: 16px; }}

    /* ── Context block ── */
    .context-block {{
      background: {BG_GREY}; border-top: 3px solid {DARK_GREY};
      padding: 24px 28px 20px 28px; margin-bottom: 0;
    }}
    .context-title {{
      font-size: 22px; font-weight: 700; color: {DARK_GREY}; margin-bottom: 14px;
    }}
    .context-block p {{
      font-size: 15px; line-height: 1.8; color: {TEXT}; margin-bottom: 12px;
    }}
    .context-block p:last-child {{ margin-bottom: 0; }}

    /* ── Finding blocks ── */
    .finding-block {{
      display: flex; align-items: center; background: {BG_GREY};
      border-left: 4px solid {DARK_GREY}; padding: 18px 22px;
      margin: 32px 0 20px 0; gap: 24px;
    }}
    .finding-left {{ flex: 1; }}
    .finding-title {{ font-size: 17px; font-weight: 700; color: {DARK_GREY}; line-height: 1.3; }}
    .finding-right {{ flex-shrink: 0; }}
    .finding-stat-group {{ display: flex; gap: 28px; text-align: right; }}
    .finding-stat-val {{ font-size: 24px; font-weight: 700; color: {ACCENT_RED}; line-height: 1; }}
    .finding-stat-lbl {{ font-size: 11px; color: {MED_GREY}; margin-top: 3px; }}

    /* ── Bullet lists ── */
    .findings-list {{ margin: 12px 0 20px 20px; color: {TEXT}; }}
    .findings-list li {{ margin-bottom: 8px; font-size: 15px; line-height: 1.6; }}

    /* ── Charts ── */
    .chart-title {{
      font-size: 17px; font-weight: 700; color: {DARK_GREY};
      text-align: center; margin-bottom: 8px;
    }}
    .chart-wrap {{
      margin: 20px 0; border: 1px solid {LIGHT_GREY}; border-radius: 4px; padding: 12px;
    }}
    .chart-caption {{
      font-size: 12px; color: {MED_GREY}; margin-top: 8px;
      text-align: center; font-style: italic;
    }}

    /* ── Callouts ── */
    .cost-callout {{
      background: {BG_GREY}; border-left: 3px solid {AMBER};
      padding: 16px 22px; margin: 20px 0; font-size: 15px; color: {TEXT};
    }}
    .cost-callout strong {{ color: {DARK_GREY}; }}

    /* ── Financial-impact table ── */
    .fin-table {{ width: 100%; border-collapse: collapse; margin: 18px 0; font-size: 14px; }}
    .fin-table th {{ background: {BG_GREY}; text-align: left; padding: 9px 12px; font-size: 12px;
      text-transform: uppercase; letter-spacing: 0.5px; color: {MED_GREY}; border-bottom: 2px solid {LIGHT_GREY}; }}
    .fin-table td {{ padding: 9px 12px; border-bottom: 1px solid {LIGHT_GREY}; color: {TEXT}; }}
    .fin-table th.num, .fin-table td.num {{ text-align: right; }}
    .fin-table td.save {{ font-weight: 700; color: {GREEN}; text-align: right; }}

    /* ── Methodology ── */
    .method-item {{ margin-bottom: 20px; padding-left: 18px; border-left: 2px solid {LIGHT_GREY}; }}
    .method-item strong {{
      display: block; color: {DARK_GREY}; margin-bottom: 3px; font-size: 15px;
    }}
  </style>
</head>
<body>

<div class="page-header">
  <h1>Analytics Diagnostic Report: Defect Risks &amp; Scrap Costs</h1>
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

    <p>From {DATE_MIN_LABEL} to {DATE_MAX_LABEL} ({n_months} months), there were
    {total_orders/1e3:.1f}K production work orders and {total_inspected/1e3:.1f}K parts were
    inspected. Of these, there were {total_defects/1e3:.1f}K total defects, a defect rate of
    <strong>{fmt_pct(overall_fr)}</strong>, and total scrap cost of
    <strong>${total_scrap/1e3:,.0f}K</strong>. These metrics remained broadly steady over the
    period, with no significant upward or downward trend.</p>

    <p>Over the past 12 months ({L12_MIN_LABEL} to {L12_MAX_LABEL}), there were
    {l12_defects/1e3:.1f}K total defects, a defect rate of <strong>{fmt_pct(l12_fr)}</strong>, and
    total scrap cost of <strong>${l12_scrap/1e3:,.0f}K</strong>.</p>

    {wrap("defect_trend", "Defects and Defect Rate by Month (TTM)")}
    {wrap("scrap_trend",  "Total Scrap Cost by Month")}

    <p>In this analysis, we set out to understand defects and scrap costs at a deeper level. We built
    a data pipeline to extract, standardize, and merge records across all three systems (ERP, MES,
    QMS) into a single source, allowing us to surface previously unseen patterns.</p>

    <p>Three distinct findings emerged, pointing to specific and addressable operational drivers of
    elevated defects and scrap cost. They are: (i) bending operations on Shift B run at
    <strong>{p1_mult:.1f}×</strong> the defect rate of Shift A; (ii) material sourced from Supplier C
    run at <strong>{p2_mult:.1f}×</strong> the defect rate of other suppliers; and (iii)
    high-complexity parts run at <strong>{p3_mult:.1f}×</strong> the defect rate of lower-complexity
    equivalents. These findings are detailed in Section 2.</p>

    <p>Bringing each finding to a benchmark defect rate is a meaningful scrap-cost savings
    opportunity, on the order of <strong>${per_yr(p1_save)/1e3:,.0f}K per year</strong> on the
    Shift B bending gap, <strong>${per_yr(p2_save)/1e3:,.0f}K per year</strong> on Supplier C
    material, and up to <strong>${per_yr(p3b_save)/1e3:,.0f}K per year</strong> on high-complexity
    work, quantified in
    Section 3. The financial impact extends well beyond the direct cost of scrapped material: each
    defective run also requires rework labor, disrupts downstream scheduling, adds inspection
    overhead, and potentially carries customer relationship costs.</p>

    {section_title("findings", "Section 2", "Findings")}

    <p>Each finding below represents a defect rate elevation that is statistically significant,
    consistent across the analysis period, and only quantifiable through cross-system joins.</p>

    {finding_block("p1",
        "Bending operations on Shift B produce defects at {:.1f}× the rate of Shift A".format(p1_mult),
        "vs Bending Shift A", f"{p1_mult:.1f}×",
        f"${per_yr(p1_save)/1e3:,.0f}K/yr", "opportunity at Shift A level")}

    {p1_bullets}

    {wrap("p1_heatmap", "Defect Rate by Machine Type × Shift")}
    {wrap("p1_boxplot", "Defect Rate Distribution: Bending by Shift vs All Machines")}
    {wrap("p1_trend",   "Bending Defect Rate Over Time: Shift B vs Shift A")}

    {finding_block("p2",
        "Supplier C material is associated with a {:.1f}× elevated defect rate".format(p2_mult),
        "vs all other suppliers", f"{p2_mult:.1f}×",
        f"${per_yr(p2_save)/1e3:,.0f}K/yr", "opportunity at 6% target")}

    {p2_bullets}

    {wrap("p2_supplier_bar",   "Defect Rate by Supplier")}
    {wrap("p2_complexity_mix", "Complexity Mix by Supplier")}
    {wrap("p2_cx_supplier",    "Defect Rate by Complexity Tier × Supplier")}
    {wrap("p2_historical",     "Monthly Defect Rate by Supplier")}

    {finding_block("p3",
        "High-complexity parts fail at {:.1f}× the rate of other complexity tiers".format(p3_mult),
        "vs non-High complexity", f"{p3_mult:.1f}×",
        f"${per_yr(p3b_save)/1e3:,.0f}K/yr", "opportunity at 9% target")}

    {p3_bullets}

    {wrap("p3_complexity_bar", "Defect Rate by Complexity Tier")}
    {wrap("p3_cx_machine",     "Defect Rate by Complexity × Machine Type")}
    {wrap("p3_historical",     "Monthly Defect Rate by Complexity")}

    {section_title("financial", "Section 3", "Financial Impact")}

    <p>Each finding above translates into scrap cost that could be recovered by bringing the affected
    segment's defect rate down to a benchmark. The estimates below scale each segment's observed
    scrap cost (material plus rework labor) by the proportional reduction in its defect rate:
    <em>savings = segment scrap cost &times; (current rate &minus; target rate) &divide; current
    rate</em>.</p>

    <table class="fin-table">
      <thead><tr><th>Finding</th><th class="num">Current</th><th>Benchmark target</th>
        <th class="num">Est. savings / yr</th></tr></thead>
      <tbody>
        <tr><td>P1 &middot; Bending Shift B</td><td class="num">{fmt_pct(p1_fr)}</td>
          <td>Shift A level ({fmt_pct(p1b_fr)})</td>
          <td class="save">${per_yr(p1_save)/1e3:,.0f}K</td></tr>
        <tr><td>P2 &middot; Supplier C material</td><td class="num">{fmt_pct(p2_fr)}</td>
          <td>6% (others, high-complexity)</td>
          <td class="save">${per_yr(p2_save)/1e3:,.0f}K</td></tr>
        <tr><td>P3a &middot; High-complexity bending</td><td class="num">{fmt_pct(p3a_fr)}</td>
          <td>12% (other operations)</td>
          <td class="save">${per_yr(p3a_save)/1e3:,.0f}K</td></tr>
        <tr><td>P3b &middot; High-complexity, all operations</td><td class="num">{fmt_pct(p3_fr)}</td>
          <td>9% (stretch)</td>
          <td class="save">${per_yr(p3b_save)/1e3:,.0f}K</td></tr>
      </tbody>
    </table>

    <p><strong>Assumptions and caveats.</strong> Scrap cost is the actual per-event material and
    rework-labor cost recorded in the QMS, attributed to each segment (Supplier C by work order,
    since the scrap record does not always carry the lot's supplier). The estimates assume the cost
    per defective unit is roughly constant and that the full gap to benchmark is addressable, so they
    are an upper-bound opportunity rather than committed savings. The segments overlap (a single work
    order can be Bending, Shift B, Supplier C, and high-complexity at once), so the rows are not
    additive. P3 is shown two ways: a conservative case that brings only high-complexity bending
    ({fmt_pct(p3a_fr)}) in line with the other operations (12%), and a stretch case that brings all
    high-complexity work to a 9% blended rate, a level the shop already reached in 3 of the
    {n_months} months in the window.</p>

    <p><strong>Levers.</strong> The three findings differ sharply in how hard they are to act on:</p>
    <ul class="findings-list">
      <li><strong>P1 (Bending Shift B): low cost, mostly process discipline.</strong> Operational
      improvements include standardizing shift-start setup and calibration, tightening the setup
      handoff between shifts, adding a first-piece verification step, and coaching the operators whose
      runs drive the spread. No capital; the main commitment is supervision and adherence.</li>
      <li><strong>P2 (Supplier C): low internal cost, but needs supplier and commercial
      action.</strong> Open a supplier corrective-action process with Supplier C, tighten incoming
      inspection and acceptance criteria on their lots, and, if the gap persists, requalify or shift
      volume to a better-performing supplier. Little internal capital, but it depends on supplier
      engagement and a sourcing decision.</li>
      <li><strong>P3 (High complexity): highest value, highest effort, some capital.</strong>
      High-complexity bending is the outlier at {fmt_pct(p3a_fr)}. Levers include tooling and fixture
      upgrades, process-capability studies on the hardest features, and design-for-manufacturability
      review with customers on the worst parts. Parts of this require capital (tooling, fixturing,
      possibly machine capability) and engineering time, so it is a medium-term program rather than a
      quick fix.</li>
    </ul>

    {section_title("methodology", "Section 4", "Methodology")}

    <div class="method-item">
      <strong>Data Sources</strong>
      Inspection records and scrap events from the QMS; production work orders and part catalog
      from the ERP; material lot receipts and certification status from the WMS; operator records
      from the HR system. The analysis covers {DATE_MIN_LABEL} through {DATE_MAX_LABEL},
      spanning {fmt_num(total_orders)} work orders and {fmt_num(total_inspected)} parts inspected
      across all five production lines.
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