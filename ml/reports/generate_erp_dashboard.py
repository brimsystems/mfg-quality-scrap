"""
generate_erp_dashboard.py
Generates erp_dashboard.html — simulated ERP work order queue
with embedded pre-production defect risk scores.

Shows April 13, 2026 active work orders as they would appear in a
live manufacturing ERP with the defect risk scorer integrated.

Run from the ml/reports/ directory:
    python3 generate_erp_dashboard.py

Output: erp_dashboard.html
"""

from pathlib import Path
import json
import duckdb
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
DB_PATH      = Path("../../data_source/defects_scrap.duckdb").resolve()
SCORING_DIR  = Path("../data/scoring").resolve()
OUTPUT       = Path("erp_dashboard.html")
SCORE_DATE   = "2026-04-13"

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading work orders...")
con = duckdb.connect(str(DB_PATH), read_only=True)

wo = con.execute(f"""
    SELECT
        d.work_order_id,
        d.actual_start,
        d.machine_id,
        d.machine_type,
        d.machine_age_years,
        d.operator_id,
        d.shift_code,
        d.part_number,
        d.customer,
        d.quantity_ordered,
        d.complexity,
        d.material_type,
        d.std_labor_hrs,
        d.requires_welding,
        d.supplier,
        d.lot_cert_status,
        d.defect_flag,
        d.defect_rate
    FROM mart_quality__defect_rates d
    WHERE DATE(d.actual_start) = '{SCORE_DATE}'
    ORDER BY d.actual_start
""").df()
con.close()

print(f"  Found {len(wo)} work orders for {SCORE_DATE}")

# If no data for that exact date, find the closest available date
if len(wo) == 0:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    wo = con.execute("""
        SELECT
            d.work_order_id, d.actual_start, d.machine_id, d.machine_type,
            d.machine_age_years, d.operator_id, d.shift_code,
            d.part_number, d.customer,
            d.quantity_ordered, d.complexity, d.material_type,
            d.std_labor_hrs, d.requires_welding, d.supplier,
            d.lot_cert_status, d.defect_flag, d.defect_rate
        FROM mart_quality__defect_rates d
        WHERE d.actual_start >= '2026-01-01'
        ORDER BY d.actual_start
        LIMIT 35
    """).df()
    con.close()
    SCORE_DATE = str(pd.to_datetime(wo["actual_start"].iloc[0]).date())
    print(f"  No data for April 13 — using {SCORE_DATE} ({len(wo)} orders)")

wo["actual_start"] = pd.to_datetime(wo["actual_start"])

# ── Load scoring predictions ───────────────────────────────────────────────
pred_path = SCORING_DIR / "predictions_202601.parquet"
if pred_path.exists():
    preds = pd.read_parquet(pred_path)
    wo = wo.merge(
        preds[["work_order_id","defect_probability","risk_tier","shap_drivers","top_driver_feature"]],
        on="work_order_id", how="left"
    )
    print(f"  Merged {preds['work_order_id'].isin(wo['work_order_id']).sum()} risk scores")
else:
    # Generate placeholder scores from features if no predictions file
    print("  No predictions file — generating placeholder scores")
    np.random.seed(42)
    wo["defect_probability"] = np.random.beta(2, 5, len(wo))
    wo["risk_tier"] = pd.cut(
        wo["defect_probability"],
        bins=[0, 0.55, 0.75, 1.0],
        labels=["Low","Medium","High"]
    ).astype(str)
    wo["top_driver_feature"] = "is_high_complexity"
    wo["shap_drivers"] = None

# Fill missing risk scores
wo["defect_probability"] = wo["defect_probability"].fillna(0.1)
wo["risk_tier"] = wo["risk_tier"].fillna("Low")

# ── Machine name lookup ────────────────────────────────────────────────────
MACHINE_NAMES = {
    "M01": "Laser Cutter 1",
    "M02": "Laser Cutter 2",
    "M03": "Press Brake 1",
    "M04": "Press Brake 2",
    "M05": "Welding Station 1",
    "M06": "Welding Station 2",
    "M07": "Punch Press 1",
}

# ── Part descriptions (realistic for sheet metal fab) ─────────────────────
PART_DESCRIPTIONS = {
    "P-1000": "Mounting Bracket, 14ga",
    "P-1001": "Side Panel Assembly",
    "P-1002": "Base Plate, 1/4\" HR",
    "P-1003": "Enclosure Cover",
    "P-1004": "Reinforcement Rib",
    "P-1005": "Door Frame, 16ga",
    "P-1006": "Equipment Stand",
    "P-1007": "Gusset Plate",
    "P-1008": "Support Channel",
    "P-1009": "Terminal Box Lid",
    "P-1010": "Hinge Bracket",
    "P-1011": "Control Panel Face",
    "P-1012": "Formed Angle, 12ga",
    "P-1013": "Weld Nut Plate",
    "P-1014": "Pivot Arm",
    "P-1015": "Guard Shield",
    "P-1016": "Cable Tray Section",
    "P-1017": "End Cap Assembly",
    "P-1018": "Stiffener Plate",
    "P-1019": "Cover Plate, 3/8\"",
}

def get_desc(part_num):
    return PART_DESCRIPTIONS.get(str(part_num), "Fabricated Component")

# ── Risk factor translation ────────────────────────────────────────────────
# Converts model signals into plain operational language
# Deliberately general — no specific operator or supplier names

def get_risk_factors(row):
    """
    Returns up to 3 plain-English risk factors for a work order.
    Language is deliberately general and operational.
    """
    factors = []
    prob    = row.get("defect_probability", 0)
    tier    = row.get("risk_tier", "Low")

    if tier not in ("High", "Medium"):
        return []

    top_driver = str(row.get("top_driver_feature", ""))
    age        = row.get("machine_age_years", 0)
    complexity = str(row.get("complexity", ""))
    shift      = str(row.get("shift_code", ""))
    machine_t  = str(row.get("machine_type", ""))
    cert       = str(row.get("lot_cert_status", ""))
    material   = str(row.get("material_type", ""))
    supplier   = str(row.get("supplier", ""))

    # Build factors based on conditions — general language only
    if "bending" in top_driver or (machine_t == "Bending" and shift == "Shift B"):
        factors.append("Bending setup conditions associated with higher-than-average rework rate")

    if "complexity" in top_driver or complexity == "High":
        if age and age > 8:
            factors.append(f"High complexity part on aging equipment ({int(age)} yrs old)")
        else:
            factors.append("High complexity part requiring tight tolerances")

    if "supplier" in top_driver or "cert" in top_driver:
        if cert in ("Conditional", "Missing"):
            factors.append("Material lot has conditional certification status")
        else:
            factors.append("Material supply source has elevated historical defect rate")

    if "lapsed" in top_driver or "op007" in top_driver.lower():
        factors.append("Assigned operator has compliance certification gap")

    # Fill remaining slots with secondary signals
    if len(factors) < 2:
        if machine_t == "Bending" and complexity == "High":
            factors.append("High complexity bend — tight angle tolerance required")
        if shift == "Shift B" and age and age > 6:
            factors.append("Equipment age increases setup sensitivity on second shift")
        if "Steel" in material and "ga" in material and int(material.split("ga")[0].split(" ")[-1]) <= 16:
            factors.append("Thin gauge material requires precise feed pressure")

    if len(factors) < 1:
        # Generic fallback based on probability
        if prob > 0.80:
            factors.append("Multiple concurrent risk factors detected")
        elif prob > 0.65:
            factors.append("Setup conditions warrant pre-run supervisor check")

    # Recommended action
    if tier == "High":
        action = "Supervisor setup review required before job release"
    else:
        action = "Review setup conditions before starting first piece"

    return factors[:3], action


# ── Operator name lookup (anonymized for realism) ─────────────────────────
OP_NAMES = {
    f"OP{str(i).zfill(3)}": name for i, name in enumerate([
        "R. Martinez", "T. Johnson", "S. Williams", "D. Brown",
        "M. Davis",    "J. Wilson",  "C. Moore",   "A. Taylor",
        "L. Anderson", "K. Thomas",  "B. Jackson", "E. White",
        "F. Harris",   "G. Martin",  "H. Garcia",  "I. Lee",
        "N. Rodriguez","O. Lewis",   "P. Walker",  "Q. Hall",
    ], 1)
}

def op_name(op_id):
    return OP_NAMES.get(str(op_id), str(op_id))

# ── Status assignment ──────────────────────────────────────────────────────
def get_status(row):
    hour = row["actual_start"].hour
    if hour < 8:
        return "Open"
    elif hour < 14:
        return "In Progress"
    else:
        return "Open"

wo["status"]      = wo.apply(get_status, axis=1)
wo["machine_name"]= wo["machine_id"].map(MACHINE_NAMES).fillna(wo["machine_id"])
wo["op_name"]     = wo["operator_id"].map(lambda x: op_name(x))
wo["part_desc"]   = wo["part_number"].map(get_desc)

# Sort: High risk first, then Medium, then Low, then by start time
tier_order = {"High": 0, "Medium": 1, "Low": 2}
wo["tier_sort"] = wo["risk_tier"].map(tier_order).fillna(2)
wo = wo.sort_values(["tier_sort", "actual_start"]).reset_index(drop=True)

n_high   = (wo["risk_tier"] == "High").sum()
n_medium = (wo["risk_tier"] == "Medium").sum()
n_total  = len(wo)

print(f"  High risk: {n_high}  Medium: {n_medium}  Low: {n_total - n_high - n_medium}")

# ── Build table rows ───────────────────────────────────────────────────────
DATE_DISPLAY = pd.to_datetime(SCORE_DATE).strftime("%m/%d/%Y")

def risk_badge(tier):
    if tier == "High":
        return '<span class="risk-badge risk-high">&#9679; HIGH</span>'
    elif tier == "Medium":
        return '<span class="risk-badge risk-med">&#9679; MED</span>'
    else:
        return '<span class="risk-badge risk-low">&#9679; LOW</span>'

def status_badge(status):
    if status == "In Progress":
        return '<span class="status-badge status-inprog">In Progress</span>'
    elif status == "On Hold":
        return '<span class="status-badge status-hold">On Hold</span>'
    else:
        return '<span class="status-badge status-open">Open</span>'

def complexity_cell(c):
    color = {"High": "#b94040", "Medium": "#888", "Low": "#4a7c59"}.get(c, "#888")
    return f'<span style="color:{color};font-weight:600;">{c}</span>'

rows_html = ""
# panels now written directly to rows_html

for i, row in wo.iterrows():
    tier = str(row["risk_tier"])
    row_class = "row-high" if tier == "High" else ("row-med" if tier == "Medium" else "")
    expandable = tier in ("High", "Medium")
    expand_icon = "&#9660;" if expandable else ""
    onclick = f'onclick="togglePanel({i})"' if expandable else ""
    cursor  = "cursor:pointer;" if expandable else ""

    rows_html += f"""
    <tr class="wo-row {row_class}" id="row-{i}" {onclick} style="{cursor}">
      <td class="mono">{row['work_order_id']}</td>
      <td>{status_badge(row['status'])}</td>
      <td class="mono">{row['part_number']}</td>
      <td>{row['part_desc']}</td>
      <td>{row['customer']}</td>
      <td class="num">{int(row['quantity_ordered']):,}</td>
      <td>{row['machine_name']}</td>
      <td>{row['op_name']}</td>
      <td>{row.get('shift_code','—')}</td>
      <td class="mono">{row['actual_start'].strftime('%I:%M %p')}</td>
      <td class="num">{row['std_labor_hrs']:.1f}</td>
      <td>{complexity_cell(row['complexity'])}</td>
      <td>{row['material_type']}</td>
      <td>{risk_badge(tier)} {f'<span class="expand-icon">{expand_icon}</span>' if expandable else ''}</td>
    </tr>"""

    if expandable:
        factors, action = get_risk_factors(row)
        prob_pct = f"{row['defect_probability']:.0%}" if pd.notna(row.get('defect_probability')) else "—"

        factor_items = ""
        for j, f in enumerate(factors):
            factor_items += f'<div class="risk-factor"><span class="factor-num">{j+1}</span>{f}</div>'

        if not factor_items:
            factor_items = '<div class="risk-factor"><span class="factor-num">1</span>Elevated risk based on current job configuration</div>'

        tier_color = "#b94040" if tier == "High" else "#c87941"
        tier_label = "HIGH RISK" if tier == "High" else "MEDIUM RISK"

        rows_html += f"""
    <tr class="panel-row" id="panel-{i}" style="display:none;">
      <td colspan="14">
        <div class="risk-panel">
          <div class="panel-header" style="border-left:4px solid {tier_color};">
            <div class="panel-title">
              <span style="color:{tier_color};font-weight:700;">&#9873; {tier_label}</span>
              &nbsp;&mdash;&nbsp; {row['work_order_id']} &nbsp;&middot;&nbsp;
              {row['part_desc']} &nbsp;&middot;&nbsp;
              <span style="color:#555;">Predicted defect probability: <strong>{prob_pct}</strong></span>
            </div>
          </div>
          <div class="panel-body">
            <div class="panel-col">
              <div class="panel-section-label">Risk Factors</div>
              {factor_items}
            </div>
            <div class="panel-col">
              <div class="panel-section-label">Recommended Action</div>
              <div class="action-box">{action}</div>
              <div class="panel-section-label" style="margin-top:12px;">Job Details</div>
              <div class="detail-grid">
                <span class="detail-label">Machine:</span><span>{row['machine_name']} ({row['machine_id']})</span>
                <span class="detail-label">Operator:</span><span>{row['op_name']}</span>
                <span class="detail-label">Shift:</span><span>{row.get('shift_code','—')}</span>
                <span class="detail-label">Material:</span><span>{row['material_type']}</span>
                <span class="detail-label">Lot Status:</span><span>{row.get('lot_cert_status','—')}</span>
                <span class="detail-label">Complexity:</span><span>{row['complexity']}</span>
              </div>
            </div>
            <div class="panel-actions">
              <button class="erp-btn">Acknowledge</button>
              <button class="erp-btn">Add Note</button>
              <button class="erp-btn erp-btn-primary">Override &amp; Release</button>
            </div>
          </div>
        </div>
      </td>
    </tr>"""

# ══════════════════════════════════════════════════════════════════════════════
# HTML
# ══════════════════════════════════════════════════════════════════════════════

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MetroFab Industries — Work Orders</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: "Segoe UI", Tahoma, Geneva, sans-serif;
      font-size: 12px;
      background: #E8E9EC;
      color: #1a1a1a;
    }}

    /* ── Application header ── */
    .app-header {{
      background: linear-gradient(to bottom, #1c3a5e, #163050);
      color: white;
      padding: 0;
      border-bottom: 2px solid #0d1f35;
    }}
    .app-header-top {{
      display: flex; align-items: center; justify-content: space-between;
      padding: 6px 16px;
      border-bottom: 1px solid rgba(255,255,255,0.1);
    }}
    .app-logo {{
      font-size: 15px; font-weight: 700; letter-spacing: 0.5px;
      color: white;
    }}
    .app-logo span {{
      font-weight: 300; font-size: 12px; opacity: 0.7; margin-left: 8px;
    }}
    .app-user {{
      font-size: 11px; opacity: 0.8;
      display: flex; align-items: center; gap: 12px;
    }}
    .app-user .date {{ opacity: 0.6; }}

    /* ── Module navigation tabs ── */
    .app-nav {{
      display: flex; align-items: flex-end;
      padding: 0 8px;
      background: #1c3a5e;
    }}
    .nav-tab {{
      padding: 7px 16px 6px 16px;
      font-size: 11px; font-weight: 500;
      color: rgba(255,255,255,0.65);
      cursor: pointer;
      border-bottom: 2px solid transparent;
      white-space: nowrap;
      transition: color 0.1s;
      user-select: none;
    }}
    .nav-tab:hover {{ color: rgba(255,255,255,0.9); }}
    .nav-tab.active {{
      color: white;
      border-bottom: 2px solid #5b9bd5;
      background: rgba(255,255,255,0.06);
    }}

    /* ── Breadcrumb ── */
    .breadcrumb {{
      background: #f5f5f5;
      border-bottom: 1px solid #d0d0d0;
      padding: 5px 16px;
      font-size: 11px;
      color: #666;
    }}
    .breadcrumb a {{ color: #1c5fa0; text-decoration: none; }}
    .breadcrumb a:hover {{ text-decoration: underline; }}
    .breadcrumb .sep {{ margin: 0 5px; color: #aaa; }}

    /* ── Main content ── */
    .main {{ padding: 10px 14px 24px 14px; }}

    /* ── Page title bar ── */
    .page-title-bar {{
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 8px;
    }}
    .page-title {{
      font-size: 14px; font-weight: 700; color: #1a1a1a;
    }}
    .page-subtitle {{
      font-size: 11px; color: #666; margin-top: 1px;
    }}
    .risk-summary {{
      display: flex; align-items: center; gap: 12px;
      background: white;
      border: 1px solid #d0d0d0;
      border-radius: 3px;
      padding: 6px 14px;
      font-size: 11px;
    }}
    .risk-summary-item {{ display: flex; align-items: center; gap: 5px; }}
    .risk-summary-dot {{
      width: 8px; height: 8px; border-radius: 50%;
      display: inline-block;
    }}

    /* ── Toolbar ── */
    .toolbar {{
      display: flex; align-items: center; gap: 2px;
      background: #f0f0f0;
      border: 1px solid #c8c8c8;
      border-bottom: none;
      padding: 4px 6px;
      border-radius: 3px 3px 0 0;
    }}
    .tb-btn {{
      padding: 3px 10px;
      font-size: 11px;
      background: linear-gradient(to bottom, #ffffff, #e8e8e8);
      border: 1px solid #b0b0b0;
      border-radius: 2px;
      cursor: pointer;
      color: #1a1a1a;
      white-space: nowrap;
    }}
    .tb-btn:hover {{ background: linear-gradient(to bottom, #e8f0fc, #d0e0f8); border-color: #7aabde; }}
    .tb-btn:disabled, .tb-btn.disabled {{
      color: #aaa; cursor: default;
      background: linear-gradient(to bottom, #f5f5f5, #ebebeb);
    }}
    .tb-sep {{ width: 1px; background: #c0c0c0; height: 20px; margin: 0 4px; }}
    .tb-right {{ margin-left: auto; }}

    /* ── Filter bar ── */
    .filter-bar {{
      display: flex; align-items: center; gap: 8px;
      background: #fafafa;
      border: 1px solid #c8c8c8;
      border-bottom: none;
      padding: 5px 8px;
    }}
    .filter-label {{ font-size: 11px; color: #555; white-space: nowrap; }}
    .filter-select {{
      font-size: 11px; padding: 2px 18px 2px 4px;
      border: 1px solid #b0b0b0; border-radius: 2px;
      background: white; color: #1a1a1a;
      appearance: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='5'%3E%3Cpath d='M0 0l4 5 4-5z' fill='%23666'/%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 4px center;
      padding-right: 16px;
    }}
    .filter-search {{
      font-size: 11px; padding: 2px 6px;
      border: 1px solid #b0b0b0; border-radius: 2px;
      width: 200px;
    }}
    .filter-search:focus {{ outline: none; border-color: #5b9bd5; }}

    /* ── Table ── */
    .table-wrap {{
      border: 1px solid #c8c8c8;
      border-radius: 0 0 3px 3px;
      overflow-x: auto;
      background: white;
    }}
    table {{
      width: 100%; border-collapse: collapse;
      font-size: 11.5px;
    }}
    thead tr {{
      background: linear-gradient(to bottom, #f0f0f0, #e4e4e4);
      border-bottom: 2px solid #b8b8b8;
    }}
    th {{
      padding: 6px 8px; text-align: left;
      font-size: 11px; font-weight: 600; color: #333;
      white-space: nowrap;
      border-right: 1px solid #d8d8d8;
      cursor: pointer;
      user-select: none;
    }}
    th:last-child {{ border-right: none; }}
    th .sort-arrow {{ color: #999; margin-left: 3px; font-size: 9px; }}
    th.risk-col {{
      background: linear-gradient(to bottom, #e8eef6, #dce5f0);
      color: #1c3a5e;
    }}

    .wo-row td {{
      padding: 5px 8px;
      border-bottom: 1px solid #eeeeee;
      border-right: 1px solid #f0f0f0;
      vertical-align: middle;
      white-space: nowrap;
    }}
    .wo-row td:last-child {{ border-right: none; }}
    .wo-row:nth-child(even) td {{ background: #f8f8f8; }}
    .wo-row:nth-child(odd) td {{ background: white; }}
    .wo-row:hover td {{ background: #e8f0fb !important; }}
    .wo-row.row-high td {{ background: #fff5f5 !important; }}
    .wo-row.row-high:hover td {{ background: #ffe8e8 !important; }}
    .wo-row.row-med td {{ background: #fffbf0 !important; }}
    .wo-row.row-med:hover td {{ background: #fff3d6 !important; }}

    .mono {{ font-family: "Courier New", Courier, monospace; font-size: 11px; }}
    .num  {{ text-align: right; font-variant-numeric: tabular-nums; }}

    /* ── Status badges ── */
    .status-badge {{
      display: inline-block; padding: 1px 6px;
      border-radius: 2px; font-size: 10px; font-weight: 600;
    }}
    .status-open   {{ background: #e8f0e8; color: #2a6a2a; border: 1px solid #b8d8b8; }}
    .status-inprog {{ background: #e0ecf8; color: #1a4a80; border: 1px solid #aaccee; }}
    .status-hold   {{ background: #f8e8d0; color: #804010; border: 1px solid #e0c090; }}

    /* ── Risk badges ── */
    .risk-badge {{
      display: inline-block; padding: 2px 7px;
      border-radius: 2px; font-size: 10px; font-weight: 700;
      letter-spacing: 0.3px;
    }}
    .risk-high {{ background: #b94040; color: white; }}
    .risk-med  {{ background: #c87941; color: white; }}
    .risk-low  {{ background: #4a7c59; color: white; }}
    .expand-icon {{ color: #666; font-size: 9px; margin-left: 4px; }}

    /* ── Expanded panel ── */
    .panel-row td {{ padding: 0; border-bottom: 2px solid #c8c8c8; }}
    .risk-panel {{
      background: #f8f8f8;
      border-top: 1px solid #e0e0e0;
    }}
    .panel-header {{
      padding: 8px 14px;
      background: white;
      border-bottom: 1px solid #e8e8e8;
    }}
    .panel-title {{
      font-size: 12px; color: #333;
    }}
    .panel-body {{
      display: flex; gap: 0; padding: 0;
    }}
    .panel-col {{
      flex: 1; padding: 12px 16px;
      border-right: 1px solid #e0e0e0;
    }}
    .panel-col:last-of-type {{ border-right: none; }}
    .panel-section-label {{
      font-size: 10px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.8px; color: #888; margin-bottom: 8px;
    }}
    .risk-factor {{
      display: flex; align-items: flex-start; gap: 8px;
      margin-bottom: 8px; font-size: 12px; color: #333; line-height: 1.4;
    }}
    .factor-num {{
      display: inline-flex; align-items: center; justify-content: center;
      width: 18px; height: 18px; border-radius: 50%;
      background: #1c3a5e; color: white;
      font-size: 10px; font-weight: 700; flex-shrink: 0; margin-top: 1px;
    }}
    .action-box {{
      background: #fffbe6; border: 1px solid #f0d060;
      border-radius: 2px; padding: 7px 10px;
      font-size: 11.5px; color: #5a4000;
    }}
    .detail-grid {{
      display: grid; grid-template-columns: auto 1fr;
      gap: 4px 10px; font-size: 11px;
    }}
    .detail-label {{ color: #888; font-weight: 600; }}
    .panel-actions {{
      display: flex; flex-direction: column;
      justify-content: center; gap: 6px;
      padding: 12px 16px;
      background: #f0f0f0;
      border-left: 1px solid #e0e0e0;
      min-width: 160px;
    }}
    .erp-btn {{
      padding: 5px 12px; font-size: 11px;
      background: linear-gradient(to bottom, #ffffff, #e8e8e8);
      border: 1px solid #b0b0b0; border-radius: 2px;
      cursor: pointer; color: #1a1a1a; text-align: center;
    }}
    .erp-btn:hover {{ background: linear-gradient(to bottom, #e8f0fc, #d0e0f8); border-color: #7aabde; }}
    .erp-btn-primary {{
      background: linear-gradient(to bottom, #2060b0, #1a4a90);
      color: white; border-color: #1a4a90;
    }}
    .erp-btn-primary:hover {{ background: linear-gradient(to bottom, #2870c0, #1e54a0); }}

    /* ── ML badge in column header ── */
    .ml-tag {{
      display: inline-block;
      font-size: 8px; font-weight: 700;
      background: #1c3a5e; color: white;
      padding: 1px 4px; border-radius: 2px;
      margin-left: 4px; vertical-align: middle;
      letter-spacing: 0.3px;
    }}
    .info-icon {{
      color: #5b9bd5; font-size: 10px;
      cursor: help; margin-left: 2px;
    }}

    /* ── Status bar ── */
    .status-bar {{
      background: #e0e0e0;
      border-top: 1px solid #c0c0c0;
      padding: 3px 14px;
      font-size: 10px; color: #555;
      display: flex; gap: 16px; align-items: center;
    }}
    .status-bar .sep {{ color: #bbb; }}
  </style>
</head>
<body>

<!-- ── Application Header ── -->
<div class="app-header">
  <div class="app-header-top">
    <div class="app-logo">
      MetroFab Industries
      <span>Enterprise Resource Planning</span>
    </div>
    <div class="app-user">
      <span>&#128100; J. Martinez (Production Supervisor)</span>
      <span class="date">&#128197; {pd.to_datetime(SCORE_DATE).strftime("%A, %B %d, %Y")}</span>
    </div>
  </div>
  <div class="app-nav">
    <div class="nav-tab">Dashboard</div>
    <div class="nav-tab active">Work Orders</div>
    <div class="nav-tab">Scheduling</div>
    <div class="nav-tab">Inventory</div>
    <div class="nav-tab">Quality</div>
    <div class="nav-tab">Purchasing</div>
    <div class="nav-tab">Reports</div>
    <div class="nav-tab">Admin</div>
  </div>
</div>

<!-- ── Breadcrumb ── -->
<div class="breadcrumb">
  <a href="#">Production</a>
  <span class="sep">&rsaquo;</span>
  <a href="#">Work Orders</a>
  <span class="sep">&rsaquo;</span>
  Active Queue
</div>

<!-- ── Main ── -->
<div class="main">

  <div class="page-title-bar">
    <div>
      <div class="page-title">Work Order Queue — Active</div>
      <div class="page-subtitle">All open and in-progress work orders &middot; {DATE_DISPLAY}</div>
    </div>
    <div class="risk-summary">
      <span style="font-size:10px;color:#888;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">Defect Risk Summary</span>
      <span class="sep" style="color:#ddd;">&nbsp;|&nbsp;</span>
      <div class="risk-summary-item">
        <span class="risk-summary-dot" style="background:#b94040;"></span>
        <strong style="color:#b94040;">{n_high}</strong>&nbsp;High
      </div>
      <div class="risk-summary-item">
        <span class="risk-summary-dot" style="background:#c87941;"></span>
        <strong style="color:#c87941;">{n_medium}</strong>&nbsp;Medium
      </div>
      <div class="risk-summary-item">
        <span class="risk-summary-dot" style="background:#4a7c59;"></span>
        <strong style="color:#4a7c59;">{n_total - n_high - n_medium}</strong>&nbsp;Low
      </div>
    </div>
  </div>

  <!-- ── Toolbar ── -->
  <div class="toolbar">
    <button class="tb-btn">&#43; New WO</button>
    <button class="tb-btn disabled" disabled>Edit</button>
    <button class="tb-btn disabled" disabled>Delete</button>
    <span class="tb-sep"></span>
    <button class="tb-btn">Release</button>
    <button class="tb-btn">Hold</button>
    <button class="tb-btn">Print Traveler</button>
    <span class="tb-sep"></span>
    <button class="tb-btn">Export</button>
    <button class="tb-btn tb-right">&#8635; Refresh</button>
  </div>

  <!-- ── Filter bar ── -->
  <div class="filter-bar">
    <span class="filter-label">Date:</span>
    <select class="filter-select"><option>{DATE_DISPLAY}</option></select>
    <span class="filter-label">Machine:</span>
    <select class="filter-select"><option>All</option></select>
    <span class="filter-label">Shift:</span>
    <select class="filter-select"><option>All</option></select>
    <span class="filter-label">Status:</span>
    <select class="filter-select"><option>Active</option></select>
    <span class="filter-label">Risk:</span>
    <select class="filter-select"><option>All</option><option>High</option><option>Medium</option></select>
    <input type="text" class="filter-search" placeholder="&#128269; Search work orders...">
  </div>

  <!-- ── Table ── -->
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>WO # <span class="sort-arrow">&#9650;</span></th>
          <th>Status</th>
          <th>Part #</th>
          <th>Description</th>
          <th>Customer</th>
          <th class="num">Qty</th>
          <th>Machine</th>
          <th>Operator</th>
          <th>Shift</th>
          <th>Sched Start</th>
          <th class="num">Est Hrs</th>
          <th>Complexity</th>
          <th>Material</th>
          <th class="risk-col">
            &#9873; Defect Risk
            <span class="ml-tag">ML</span>
            <span class="info-icon" title="Pre-production risk score. Powered by BRIM Defect Risk Scorer v1. Click flagged rows for detail.">&#9432;</span>
          </th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>

  <!-- ── Status bar ── -->
  <div class="status-bar">
    <span>Showing <strong>{n_total}</strong> work orders</span>
    <span class="sep">&middot;</span>
    <span style="color:#b94040;font-weight:600;">&#9873; {n_high} High Risk</span>
    <span class="sep">&middot;</span>
    <span style="color:#c87941;font-weight:600;">&#9873; {n_medium} Medium Risk</span>
    <span class="sep">&middot;</span>
    <span>Last updated: {DATE_DISPLAY} 6:02 AM</span>
    <span class="sep">&middot;</span>
    <span style="color:#888;">Risk scores powered by BRIM Defect Risk Scorer</span>
  </div>

</div>

<script>
  function togglePanel(i) {{
    const panel = document.getElementById('panel-' + i);
    const row   = document.getElementById('row-' + i);
    const icon  = row.querySelector('.expand-icon');
    if (panel.style.display === 'none' || panel.style.display === '') {{
      panel.style.display = 'table-row';
      if (icon) icon.innerHTML = '&#9650;';
    }} else {{
      panel.style.display = 'none';
      if (icon) icon.innerHTML = '&#9660;';
    }}
  }}

  // Make toolbar buttons interactive (visual feedback only)
  document.querySelectorAll('.tb-btn:not(.disabled)').forEach(btn => {{
    btn.addEventListener('click', function() {{
      if (this.textContent.includes('Refresh')) {{
        this.textContent = '&#8635; Refreshing...';
        setTimeout(() => {{ this.innerHTML = '&#8635; Refresh'; }}, 800);
      }}
    }});
  }});

  // Filter by risk dropdown (functional)
  document.querySelectorAll('.filter-select').forEach(sel => {{
    sel.addEventListener('change', function() {{
      const val = this.value;
      if (this.previousElementSibling && this.previousElementSibling.textContent === 'Risk:') {{
        document.querySelectorAll('.wo-row').forEach(row => {{
          const badge = row.querySelector('.risk-badge');
          if (!badge) return;
          const tier = badge.textContent.trim();
          if (val === 'All') {{
            row.style.display = '';
          }} else if (val === 'High' && !tier.includes('HIGH')) {{
            row.style.display = 'none';
          }} else if (val === 'Medium' && !tier.includes('MED')) {{
            row.style.display = 'none';
          }} else {{
            row.style.display = '';
          }}
        }});
      }}
    }});
  }});

  // Search filter (functional)
  document.querySelector('.filter-search').addEventListener('input', function() {{
    const q = this.value.toLowerCase();
    document.querySelectorAll('.wo-row').forEach(row => {{
      const text = row.textContent.toLowerCase();
      row.style.display = text.includes(q) ? '' : 'none';
    }});
  }});
</script>
</body>
</html>"""

OUTPUT.write_text(html, encoding="utf-8")
print(f"\nERP dashboard written to {OUTPUT.resolve()}")
print(f"  {n_total} work orders  |  {n_high} High risk  |  {n_medium} Medium risk")
