from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import base64, io

DB_PATH = Path("../../data_source/defects_scrap.duckdb").resolve()
OUTPUT  = Path("dashboard.html")

con = duckdb.connect(str(DB_PATH), read_only=True)
dr  = con.execute("SELECT * FROM mart_quality__defect_rates").df()
sc  = con.execute("SELECT * FROM mart_quality__scrap_summary").df()
con.close()

dr["actual_start"] = pd.to_datetime(dr["actual_start"])
dr["order_month"]  = pd.to_datetime(dr["order_month"])
sc["scrap_date"]   = pd.to_datetime(sc["scrap_date"])
sc["scrap_month"]  = pd.to_datetime(sc["scrap_month"])

CURRENT_MONTH      = dr["order_month"].max()
PRIOR_MONTH        = CURRENT_MONTH - pd.DateOffset(months=1)
TTM_START          = CURRENT_MONTH - pd.DateOffset(months=11)
TTM_PRIOR_END      = CURRENT_MONTH - pd.DateOffset(years=1)
TTM_PRIOR_START    = TTM_PRIOR_END  - pd.DateOffset(months=11)
CURRENT_WEEK_END   = dr["actual_start"].max().normalize()
CURRENT_WEEK_START = CURRENT_WEEK_END - pd.DateOffset(weeks=1)
PRIOR_WEEK_END     = CURRENT_WEEK_START
PRIOR_WEEK_START   = PRIOR_WEEK_END - pd.DateOffset(weeks=1)
T12_CHART_START    = CURRENT_MONTH - pd.DateOffset(months=11)

BRAND_BLUE   = "#3D5166"
DARK_GREY    = "#555555"
GOOD_GREEN   = "#1A7A3A"
BAD_RED      = "#CC0000"
NEUTRAL_GREY = "#AAAAAA"
MONTHLY_BAR  = "#AAAAAA"
AVG_LINE     = "#CC0000"

MACHINE_COLORS   = ["#3D5166","#6B8FA8","#D4881E","#4A7C59","#8E6BAF","#B94040","#A8C0D1"]
REASON_COLORS    = ["#3D5166","#6B8FA8","#D4881E","#4A7C59","#8E6BAF","#B94040"]
SUPPLIER_COLORS  = {"Supplier A": "#AAAAAA", "Supplier B": "#6B8FA8",
                    "Supplier C": "#3D5166", "Supplier D": "#A8C0D1"}
COMPLEXITY_COLORS = {"Low": "#AAAAAA", "Medium": "#6B8FA8", "High": "#3D5166"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor":   "#DDDDDD", "axes.grid": False,
    "font.family":      "sans-serif", "font.size": 18,
    "axes.titlesize":   20, "axes.titleweight": "bold",
    "axes.labelsize":   18, "xtick.labelsize":  15,
    "ytick.labelsize":  15, "legend.fontsize":  15,
    "figure.dpi":       130,
})

def date_filter(df, col, start, end):
    d = pd.to_datetime(df[col])
    return df[(d >= start) & (d < end)]

def fail_rate(df):
    qi = df["quantity_inspected"].sum()
    qf = df["quantity_failed"].sum()
    return qf / qi if qi > 0 else 0

def scrap_cost(df):      return df["total_scrap_cost"].sum()
def parts_scrapped(df):  return df["quantity_scrapped"].sum()
def cost_per_event(df):  return df["total_scrap_cost"].sum() / max(len(df), 1)
def parts_inspected(df): return df["quantity_inspected"].sum()
def parts_failed(df):    return df["quantity_failed"].sum()

def fmt_pct(x):   return f"{x:.1%}"
def fmt_usd(x):   return f"${x:,.0f}"
def fmt_usd_k(x): return f"${x/1000:,.0f}K"
def fmt_num(x):   return f"{x:,.0f}"

cw_dr   = date_filter(dr,"actual_start",CURRENT_WEEK_START,CURRENT_WEEK_END)
cw_sc   = date_filter(sc,"scrap_date",  CURRENT_WEEK_START,CURRENT_WEEK_END)
pw_dr   = date_filter(dr,"actual_start",PRIOR_WEEK_START,  PRIOR_WEEK_END)
pw_sc   = date_filter(sc,"scrap_date",  PRIOR_WEEK_START,  PRIOR_WEEK_END)
cm_dr   = dr[dr["order_month"].dt.to_period("M")==CURRENT_MONTH.to_period("M")]
cm_sc   = sc[sc["scrap_month"].dt.to_period("M")==CURRENT_MONTH.to_period("M")]
pm_dr   = dr[dr["order_month"].dt.to_period("M")==PRIOR_MONTH.to_period("M")]
pm_sc   = sc[sc["scrap_month"].dt.to_period("M")==PRIOR_MONTH.to_period("M")]
ttm_dr  = dr[dr["order_month"] >= TTM_START]
ttm_sc  = sc[sc["scrap_month"] >= TTM_START]
ttmp_dr = dr[(dr["order_month"]>=TTM_PRIOR_START)&(dr["order_month"]<=TTM_PRIOR_END)]
ttmp_sc = sc[(sc["scrap_month"]>=TTM_PRIOR_START)&(sc["scrap_month"]<=TTM_PRIOR_END)]

# ── KPI HTML ───────────────────────────────────────────────────────────────
def direction(current, prior, lower_is_better, neutral=False):
    if neutral or prior == 0:
        return "", NEUTRAL_GREY, ""
    delta  = (current - prior) / abs(prior)
    better = delta < 0 if lower_is_better else delta > 0
    arrow  = "▲" if delta > 0 else "▼"
    color  = GOOD_GREEN if better else BAD_RED
    return arrow, color, f"{abs(delta):.1%}"

def kpi_card(cur_val, pri_val, arrow, arrow_color, pct_str, bar_color):
    indicator = f'<span style="font-size:18px;color:{arrow_color};font-weight:bold;margin-left:4px;">{arrow} {pct_str}</span>' \
                if arrow else ""
    return f'''<div style="border:1px solid #DDDDDD;border-radius:8px;
                padding:16px 8px 0 28px;background:white;flex:1;min-width:0;overflow:hidden;">
      <div style="display:flex;align-items:baseline;flex-wrap:wrap;margin-bottom:2px;">
        <span style="font-size:28px;font-weight:700;color:{BRAND_BLUE};line-height:1.1;">{cur_val}</span>
        {indicator}
      </div>
      <div style="font-size:18px;color:#333333;margin-bottom:12px;">Current</div>
      <div style="font-size:22px;font-weight:600;color:#999;">{pri_val}</div>
      <div style="font-size:14px;color:#555555;margin-bottom:0;">Prior</div>
      <div style="height:8px;background:{bar_color};border-radius:0 0 8px 8px;margin-top:12px;margin-left:-28px;margin-right:-8px;"></div>
    </div>'''

def col_header(label):
    return f'<div style="flex:1;min-width:0;text-align:center;font-size:18px;font-weight:700;color:#111111;padding-bottom:8px;">{label}</div>'

def qi_cards_for(dr_cur, dr_pri):
    qi_c, qi_p = parts_inspected(dr_cur), parts_inspected(dr_pri)
    qf_c, qf_p = parts_failed(dr_cur),    parts_failed(dr_pri)
    fr_c, fr_p = fail_rate(dr_cur),        fail_rate(dr_pri)
    ar_qi,co_qi,pct_qi = direction(qi_c,qi_p,False,neutral=False)
    ar_qf,co_qf,pct_qf = direction(qf_c,qf_p,True)
    ar_fr,co_fr,pct_fr = direction(fr_c,fr_p,True)
    return [
        kpi_card(fmt_num(qi_c), fmt_num(qi_p), ar_qi, NEUTRAL_GREY, pct_qi, NEUTRAL_GREY),
        kpi_card(fmt_num(qf_c), fmt_num(qf_p), ar_qf, co_qf, pct_qf, co_qf),
        kpi_card(fmt_pct(fr_c), fmt_pct(fr_p), ar_fr, co_fr, pct_fr, co_fr),
    ]

def sc_cards_for(sc_cur, sc_pri):
    ps_c,ps_p = parts_scrapped(sc_cur), parts_scrapped(sc_pri)
    co_c,co_p = scrap_cost(sc_cur),     scrap_cost(sc_pri)
    cp_c,cp_p = cost_per_event(sc_cur), cost_per_event(sc_pri)
    ar_ps,c_ps,pct_ps = direction(ps_c,ps_p,True)
    ar_co,c_co,pct_co = direction(co_c,co_p,True)
    ar_cp,c_cp,pct_cp = direction(cp_c,cp_p,True)
    return [
        kpi_card(fmt_num(ps_c),   fmt_num(ps_p),   ar_ps,c_ps,pct_ps,c_ps),
        kpi_card(fmt_usd_k(co_c), fmt_usd_k(co_p), ar_co,c_co,pct_co,c_co),
        kpi_card(fmt_usd(cp_c),   fmt_usd(cp_p),   ar_cp,c_cp,pct_cp,c_cp),
    ]

ROW_LABEL_W = "120px"

# Vertical divider — full height spanning from section headers through all rows
# Achieved by wrapping left and right groups in a flex container with
# a 1px border-right on the left group
def kpi_row(label, qi_cards, sc_cards):
    qi_html = "".join(qi_cards)
    sc_html = "".join(sc_cards)
    return f'''
    <div style="display:flex;margin-bottom:10px;align-items:stretch;">
      <div style="width:{ROW_LABEL_W};flex-shrink:0;font-weight:700;color:#111111;
                  font-size:18px;display:flex;align-items:center;padding-right:8px;">
        {label}
      </div>
      <div style="display:flex;gap:4px;flex:1;padding-right:9px;">{qi_html}</div>
      <div style="display:flex;gap:4px;flex:1;padding-left:9px;">{sc_html}</div>
    </div>'''

# ── KPI section header — full-width blue box ───────────────────────────────
kpi_header = f'''
<div style="background:{DARK_GREY};color:white;border-radius:8px;padding:12px 20px;
            font-size:20px;font-weight:700;margin-bottom:12px;text-align:center;">
  KPIs
</div>'''

# ── Column group headers — no background, black text, bottom border ────────
section_headers = f'''
<div style="display:flex;margin-bottom:4px;">
  <div style="width:{ROW_LABEL_W};flex-shrink:0;"></div>
  <div style="flex:1;padding-right:9px;">
    <div style="font-size:18px;font-weight:700;color:#111111;
                border-bottom:2px solid #111111;padding-bottom:8px;text-align:center;">
      Defects
    </div>
  </div>
  <div style="flex:1;padding-left:9px;">
    <div style="font-size:18px;font-weight:700;color:#111111;
                border-bottom:2px solid #111111;padding-bottom:8px;text-align:center;">
      Scrap Costs
    </div>
  </div>
</div>'''

col_headers = f'''
<div style="display:flex;margin-bottom:8px;margin-top:10px;">
  <div style="width:{ROW_LABEL_W};flex-shrink:0;"></div>
  <div style="display:flex;gap:4px;flex:1;padding-right:9px;">
    {col_header("Parts Inspected")}
    {col_header("Defects")}
    {col_header("Defect Rate")}
  </div>
  <div style="display:flex;gap:4px;flex:1;padding-left:9px;">
    {col_header("Scrapped Parts")}
    {col_header("Total Scrap Cost")}
    {col_header("Cost / Scrap Event")}
  </div>
</div>'''

weekly_row  = kpi_row("Weekly",       qi_cards_for(cw_dr, pw_dr),    sc_cards_for(cw_sc, pw_sc))
monthly_row = kpi_row("Monthly",      qi_cards_for(cm_dr, pm_dr),    sc_cards_for(cm_sc, pm_sc))
ttm_row     = kpi_row("Trailing 12M", qi_cards_for(ttm_dr, ttmp_dr), sc_cards_for(ttm_sc, ttmp_sc))

kpi_section = kpi_header + section_headers + col_headers + weekly_row + monthly_row + ttm_row

# ── Chart helpers ──────────────────────────────────────────────────────────
def chart_style(ax):
    ax.yaxis.grid(True, color="#EEEEEE", linestyle="-", linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD")
    ax.spines["bottom"].set_color("#DDDDDD")

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def add_segment_pct_labels(ax, x_pos, segment_vals, bottoms, bar_totals,
                            threshold=0.05, fontsize=15):
    for xi, val, bot, total in zip(x_pos, segment_vals, bottoms, bar_totals):
        if total > 0 and val / total >= threshold:
            mid = bot + val / 2
            ax.text(xi, mid, f"{val/total:.0%}", ha="center", va="center",
                    fontsize=fontsize, color="white", fontweight="bold")

def stacked_legend(ax, avg_label, fontsize=15):
    handles, labels = ax.get_legend_handles_labels()
    avg_h  = [h for h, l in zip(handles, labels) if "TTM avg" in l]
    avg_l  = [l for l in labels if "TTM avg" in l]
    bar_h  = [h for h, l in zip(handles, labels) if "TTM avg" not in l]
    bar_l  = [l for l in labels if "TTM avg" not in l]
    leg1 = ax.legend(bar_h, bar_l, loc="upper center",
                     bbox_to_anchor=(0.5, -0.18), ncol=3,
                     fontsize=fontsize, frameon=False)
    ax.add_artist(leg1)
    if avg_h:
        ax.legend(avg_h, avg_l, loc="upper right", fontsize=fontsize, frameon=True)

# ── Monthly data prep ──────────────────────────────────────────────────────
monthly_dr = (
    dr[dr["order_month"] >= T12_CHART_START]
    .groupby("order_month")
    .agg(qi=("quantity_inspected","sum"), qf=("quantity_failed","sum"))
    .assign(fail_rate=lambda d: d["qf"]/d["qi"])
    .reset_index().sort_values("order_month")
)
monthly_dr["label"] = monthly_dr["order_month"].dt.strftime("%b '%y")

monthly_sc_agg = (
    sc[sc["scrap_month"] >= T12_CHART_START]
    .groupby("scrap_month")
    .agg(total_cost=("total_scrap_cost","sum"),
         material=("material_cost_total","sum"),
         labor=("labor_cost_total","sum"))
    .reset_index().sort_values("scrap_month")
)
monthly_sc_agg["label"] = monthly_sc_agg["scrap_month"].dt.strftime("%b '%y")

n_months   = len(monthly_dr)
x_months   = np.arange(n_months)
mo_labels  = list(monthly_dr["label"])
all_months = sorted(monthly_sc_agg["scrap_month"].unique())

avg_fail_rate  = monthly_dr["fail_rate"].mean()
avg_scrap_cost = monthly_sc_agg["total_cost"].mean()

reason_monthly = (
    sc[sc["scrap_month"] >= T12_CHART_START]
    .groupby(["scrap_month","scrap_reason"])["total_scrap_cost"]
    .sum().reset_index()
)
all_reasons = sorted(reason_monthly["scrap_reason"].unique())

machine_monthly_dr = (
    dr[dr["order_month"] >= T12_CHART_START]
    .groupby(["order_month","machine_type"])
    .agg(qi=("quantity_inspected","sum"), qf=("quantity_failed","sum"))
    .assign(fail_rate=lambda d: d["qf"]/d["qi"])
    .reset_index()
)
all_machines = sorted(machine_monthly_dr["machine_type"].unique())

machine_monthly_sc = (
    sc[sc["scrap_month"] >= T12_CHART_START]
    .groupby(["scrap_month","machine_type"])["total_scrap_cost"]
    .sum().reset_index()
)

charts_b64 = {}

# ── Chart 1: Defect Rate (line) ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5.5))
vals    = monthly_dr["fail_rate"].values * 100
avg_pct = avg_fail_rate * 100
ax.plot(x_months, vals, color=MONTHLY_BAR, linewidth=2.5,
        marker="o", markersize=7)
ax.axhline(avg_pct, color=AVG_LINE, linestyle="--",
           linewidth=2, label=f"TTM avg ({avg_fail_rate:.1%})")
y_range = vals.max() - vals.min() if vals.max() > 0 else 1
for xi, val in zip(x_months, vals):
    ax.text(xi, val + y_range*0.05, f"{val:.1f}%", ha="center",
            va="bottom", fontsize=15, color="#555")
ax.set_xticks(x_months)
ax.set_xticklabels(mo_labels, rotation=45, ha="right")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))
ax.set_ylabel("Fail Rate (%)")
ax.legend(fontsize=15)
chart_style(ax)
plt.tight_layout()
charts_b64["fail_rate"] = fig_to_b64(fig)
plt.close()

# ── Chart 2: Total Scrap Cost ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5.5))
vals_k  = monthly_sc_agg["total_cost"].values / 1000
avg_k   = avg_scrap_cost / 1000
ax.bar(x_months, vals_k, color=MONTHLY_BAR, width=0.7)
ax.axhline(avg_k, color=AVG_LINE, linestyle="--",
           linewidth=2, label=f"TTM avg (${avg_k:,.0f}K)")
y_range = vals_k.max() - vals_k.min() if vals_k.max() > 0 else 1
for xi, val in zip(x_months, vals_k):
    ax.text(xi, val + y_range*0.02, f"${val:,.0f}", ha="center",
            va="bottom", fontsize=15, color="#333")
ax.set_xticks(x_months)
ax.set_xticklabels(mo_labels, rotation=45, ha="right")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}K"))
ax.set_ylabel("Scrap Cost ($K)")
ax.legend(fontsize=15)
chart_style(ax)
plt.tight_layout()
charts_b64["scrap_cost"] = fig_to_b64(fig)
plt.close()

# ── Chart 3: Defects by Machine Type (stacked) ────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7.5))
bottoms  = np.zeros(n_months)
seg_data = {}
for i, machine in enumerate(all_machines):
    vals = np.array([
        machine_monthly_dr[(machine_monthly_dr["order_month"]==m) &
                           (machine_monthly_dr["machine_type"]==machine)]["qf"].sum()
        for m in all_months
    ])
    seg_data[machine] = (vals, bottoms.copy())
    ax.bar(x_months, vals, width=0.7, bottom=bottoms,
           color=MACHINE_COLORS[i % len(MACHINE_COLORS)], label=machine)
    bottoms += vals
for machine, (vals, bots) in seg_data.items():
    add_segment_pct_labels(ax, x_months, vals, bots, bottoms)
y_range = bottoms.max() - bottoms.min() if bottoms.max() > 0 else 1
for xi, bar_top in zip(x_months, bottoms):
    if bar_top > 0:
        ax.text(xi, bar_top + y_range*0.03, fmt_num(bar_top),
                ha="center", va="bottom", fontsize=15,
                fontweight="bold", color="#333")
ax.set_xticks(x_months)
ax.set_xticklabels(mo_labels, rotation=45, ha="right")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:,.0f}"))
ax.set_ylabel("Number of Defects")
stacked_legend(ax, "")
chart_style(ax)
plt.tight_layout(rect=[0, 0.15, 1, 1])
charts_b64["defect_machine"] = fig_to_b64(fig)
plt.close()

# ── Chart 4: Scrap Cost by Reason ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7.5))
bottoms  = np.zeros(n_months)
seg_data = {}
for i, reason in enumerate(all_reasons):
    vals = np.array([
        reason_monthly[(reason_monthly["scrap_month"]==m) &
                       (reason_monthly["scrap_reason"]==reason)]["total_scrap_cost"].sum()
        for m in all_months
    ]) / 1000
    seg_data[reason] = (vals, bottoms.copy())
    ax.bar(x_months, vals, width=0.7, bottom=bottoms,
           color=REASON_COLORS[i % len(REASON_COLORS)],
           label=reason.replace("_"," ").title())
    bottoms += vals
for reason, (vals, bots) in seg_data.items():
    add_segment_pct_labels(ax, x_months, vals, bots, bottoms)
y_range = bottoms.max() - bottoms.min() if bottoms.max() > 0 else 1
for xi, tot in zip(x_months, bottoms):
    ax.text(xi, tot + y_range*0.02, f"${tot:,.0f}", ha="center",
            va="bottom", fontsize=15, fontweight="bold", color="#333")
ax.set_xticks(x_months)
ax.set_xticklabels(mo_labels, rotation=45, ha="right")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
ax.set_ylabel("Scrap Cost ($K)")
stacked_legend(ax, f"TTM avg (${bottoms.mean():,.1f}K)")
chart_style(ax)
plt.tight_layout(rect=[0, 0.15, 1, 1])
charts_b64["scrap_reason"] = fig_to_b64(fig)
plt.close()

# ── Chart 5: Defect Rate by Complexity (line) ─────────────────────────────
monthly_cx = (
    dr[dr["order_month"] >= T12_CHART_START]
    .dropna(subset=["complexity"])
    .groupby(["order_month","complexity"])
    .agg(qi=("quantity_inspected","sum"), qf=("quantity_failed","sum"))
    .assign(fr=lambda d: d["qf"]/d["qi"])
    .reset_index().sort_values("order_month")
)
fig, ax = plt.subplots(figsize=(13, 5.5))
for cx in ["Low","Medium","High"]:
    sub = monthly_cx[monthly_cx["complexity"]==cx]
    sub_labels = [mo_labels[i] for i, m in enumerate(sorted(monthly_dr["order_month"].unique()))
                  if m in sub["order_month"].values]
    sub_vals   = sub.set_index("order_month").reindex(
        sorted(monthly_dr["order_month"].unique())).dropna()
    ax.plot(mo_labels[:len(sub_vals)], sub_vals["fr"].values*100,
            color=COMPLEXITY_COLORS[cx], linewidth=2.5,
            marker="o", markersize=6, label=cx)
ax.axhline(avg_fail_rate*100, color=AVG_LINE, linestyle="--",
           linewidth=2, label=f"TTM avg ({avg_fail_rate:.1%})")
ax.set_xticks(x_months)
ax.set_xticklabels(mo_labels, rotation=45, ha="right")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))
ax.set_ylabel("Defect Rate (%)")
ax.legend(fontsize=15)
chart_style(ax)
plt.tight_layout()
charts_b64["defect_complexity"] = fig_to_b64(fig)
plt.close()

# ── Chart 6: Scrap Cost by Machine Type (stacked) ─────────────────────────
fig, ax = plt.subplots(figsize=(13, 7.5))
bottoms  = np.zeros(n_months)
seg_data = {}
for i, machine in enumerate(all_machines):
    vals = np.array([
        machine_monthly_sc[(machine_monthly_sc["scrap_month"]==m) &
                           (machine_monthly_sc["machine_type"]==machine)]["total_scrap_cost"].sum()
        for m in all_months
    ]) / 1000
    seg_data[machine] = (vals, bottoms.copy())
    ax.bar(x_months, vals, width=0.7, bottom=bottoms,
           color=MACHINE_COLORS[i % len(MACHINE_COLORS)], label=machine)
    bottoms += vals
for machine, (vals, bots) in seg_data.items():
    add_segment_pct_labels(ax, x_months, vals, bots, bottoms)
y_range = bottoms.max() - bottoms.min() if bottoms.max() > 0 else 1
for xi, tot in zip(x_months, bottoms):
    ax.text(xi, tot + y_range*0.02, f"${tot:,.0f}", ha="center",
            va="bottom", fontsize=15, fontweight="bold", color="#333")
ax.set_xticks(x_months)
ax.set_xticklabels(mo_labels, rotation=45, ha="right")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
ax.set_ylabel("Scrap Cost ($K)")
stacked_legend(ax, f"TTM avg (${bottoms.mean():,.1f}K)")
chart_style(ax)
plt.tight_layout(rect=[0, 0.15, 1, 1])
charts_b64["cost_machine"] = fig_to_b64(fig)
plt.close()

# ── Chart 7: Defect Rate by Supplier (line) ───────────────────────────────
monthly_sup = (
    dr[dr["order_month"] >= T12_CHART_START]
    .dropna(subset=["supplier"])
    .groupby(["order_month","supplier"])
    .agg(qi=("quantity_inspected","sum"), qf=("quantity_failed","sum"))
    .assign(fr=lambda d: d["qf"]/d["qi"])
    .reset_index().sort_values("order_month")
)
all_sup_months = sorted(monthly_dr["order_month"].unique())
fig, ax = plt.subplots(figsize=(13, 5.5))
for sup in sorted(monthly_sup["supplier"].unique()):
    sub = monthly_sup[monthly_sup["supplier"]==sup].set_index("order_month").reindex(all_sup_months)
    color = SUPPLIER_COLORS.get(sup, MONTHLY_BAR)
    lw = 2.5 if sup=="Supplier C" else 1.8
    ms = 7 if sup=="Supplier C" else 5
    ax.plot(mo_labels, sub["fr"].values*100,
            color=color, linewidth=lw, marker="o", markersize=ms, label=sup)
ax.axhline(avg_fail_rate*100, color=AVG_LINE, linestyle="--",
           linewidth=2, label=f"TTM avg ({avg_fail_rate:.1%})")
ax.set_xticks(x_months)
ax.set_xticklabels(mo_labels, rotation=45, ha="right")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))
ax.set_ylabel("Defect Rate (%)")
ax.legend(fontsize=15)
chart_style(ax)
plt.tight_layout()
charts_b64["defect_supplier"] = fig_to_b64(fig)
plt.close()

# ── Chart 8: Material vs Labor (100% stacked) ─────────────────────────────
fig, ax = plt.subplots(figsize=(13, 8.5))
mat_k    = monthly_sc_agg["material"].values / 1000
labor_k  = monthly_sc_agg["labor"].values    / 1000
totals_k = mat_k + labor_k
mat_pct   = mat_k   / totals_k * 100
labor_pct = labor_k / totals_k * 100
ax.bar(x_months, mat_pct,   width=0.7, color="#3D5166", label="Material")
ax.bar(x_months, labor_pct, width=0.7, color="#6B8FA8", label="Labor", bottom=mat_pct)
for xi, mp, lp in zip(x_months, mat_pct, labor_pct):
    if mp >= 5:
        ax.text(xi, mp/2, f"{mp:.0f}%", ha="center", va="center",
                fontsize=15, color="white", fontweight="bold")
    if lp >= 5:
        ax.text(xi, mp + lp/2, f"{lp:.0f}%", ha="center", va="center",
                fontsize=15, color="white", fontweight="bold")
ax.set_xticks(x_months)
ax.set_xticklabels(mo_labels, rotation=45, ha="right")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))
ax.set_ylim(0, 110)
ax.set_ylabel("% of Scrap Cost")
ax.legend(["Material", "Labor"], loc="upper center",
          bbox_to_anchor=(0.5, -0.12), ncol=2,
          fontsize=15, frameon=False)
chart_style(ax)
plt.tight_layout(rect=[0, 0.15, 1, 1])
charts_b64["mat_labor"] = fig_to_b64(fig)
plt.close()

# ── Chart card ─────────────────────────────────────────────────────────────
def chart_card(key, title):
    return f'''<div style="background:white;border:1px solid #DDDDDD;border-radius:8px;padding:20px;">
      <div style="font-size:18px;font-weight:700;color:#111111;margin-bottom:14px;">{title}</div>
      <img src="data:image/png;base64,{charts_b64[key]}" style="width:100%;height:auto;display:block;">
    </div>'''

chart_grid = f'''
<div style="margin-top:36px;">
  <div style="background:{DARK_GREY};color:white;border-radius:8px;padding:12px 20px;
              font-size:20px;font-weight:700;margin-bottom:16px;text-align:center;">
    Trend Charts (Trailing 12 Months)
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
    {chart_card("fail_rate",        "Defect Rate")}
    {chart_card("scrap_cost",       "Total Scrap Cost ($K)")}
    {chart_card("defect_complexity","Defect Rate by Part Complexity")}
    {chart_card("scrap_reason",     "Scrap Cost by Reason ($K)")}
    {chart_card("defect_supplier",  "Defect Rate by Supplier")}
    {chart_card("cost_machine",     "Scrap Cost by Machine Type ($K)")}
    {chart_card("defect_machine",   "Defects by Machine Type")}
    {chart_card("mat_labor",        "Material & Labor as % of Scrap Cost")}
  </div>
</div>'''

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Analytics Dashboard: Defect Risks &amp; Scrap Costs</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #F5F6F8; margin: 0; padding: 0; color: #333;
    }}
    .page-header {{ background: {BRAND_BLUE}; color: white; padding: 20px 40px; }}
    .page-header h1 {{ font-size: 22px; font-weight: 700; letter-spacing: -0.3px; }}
    .container {{
      max-width: 1600px; margin: 0 auto; padding: 28px 32px 64px 32px;
    }}
  </style>
</head>
<body>
  <div class="page-header">
    <h1>Analytics Dashboard: Defect Risks &amp; Scrap Costs</h1>
  </div>
  <div class="container">
    {kpi_section}
    {chart_grid}
  </div>
</body>
</html>'''

OUTPUT.write_text(html, encoding="utf-8")
print(f"Dashboard written to {OUTPUT.resolve()}")