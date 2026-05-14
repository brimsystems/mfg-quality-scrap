"""
Generator: scrap_events
Source system: QMS (same system as inspections, or a separate scrap log)
Reality: Scrapped parts logged by quality technician or supervisor.
         Reason codes applied inconsistently — QA staff use structured codes,
         operators filling in for QA use free text.
         Cost fields are estimated, not pulled from actual job cost data.
Cross-system note: Links QMS scrap outcomes to:
  - ERP job cost (machine_id, operator — for labor cost attribution)
  - Materials receiving (lot_id → unit_cost_per_lb — for material cost)
  - HR (operator_id — for cert status cross-reference)
  These cost connections don't exist in the QMS natively — they require
  joining to ERP and Materials, which is a core dbt pipeline task.
Dirty characteristics:
  - scrap_reason_raw: structured codes mixed with free-text variants
  - total_scrap_cost: estimated from partial data, sometimes inconsistent
"""
import random
import numpy as np
import pandas as pd
from ..config import RANDOM_SEED
from ..dirty.transformations import dirty_scrap_reason

SCRAP_REASONS = [
    "OPERATOR_ERROR", "MATERIAL_DEFECT", "MACHINE_ISSUE",
    "SETUP_ERROR", "DESIGN_ISSUE", "UNKNOWN"
]

# Reason code probability weighted by defect type (realistic causal logic)
_REASON_BY_DEFECT = {
    "Dimensional":        [0.20, 0.10, 0.20, 0.35, 0.05, 0.10],
    "Surface Scratch":    [0.10, 0.45, 0.30, 0.05, 0.00, 0.10],
    "Burr":               [0.15, 0.10, 0.45, 0.20, 0.00, 0.10],
    "Weld Defect":        [0.45, 0.15, 0.25, 0.10, 0.00, 0.05],
    "Porosity":           [0.60, 0.10, 0.20, 0.05, 0.00, 0.05],
    "Incorrect Material": [0.05, 0.75, 0.00, 0.05, 0.05, 0.10],
    "Bend Angle":         [0.20, 0.05, 0.30, 0.35, 0.05, 0.05],
    "None":               [1/6,  1/6,  1/6,  1/6,  1/6,  1/6 ],
}


def generate_scrap_events(
    inspection_records_df: pd.DataFrame,
    production_orders_df: pd.DataFrame,
    material_lots_df: pd.DataFrame,
) -> pd.DataFrame:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    order_lookup = production_orders_df.set_index("work_order_id").to_dict("index")
    lot_cost_map = material_lots_df.set_index("lot_id_clean")["unit_cost_per_lb"].to_dict()

    rows    = []
    counter = 1

    for _, insp in inspection_records_df.iterrows():
        if insp["quantity_failed"] <= 0:
            continue
        if insp["disposition"] not in ("Scrap", "Rework"):
            continue

        order_d    = order_lookup.get(insp["work_order_id"], {})
        machine_id = order_d.get("machine_id", "UNK")
        op_id      = order_d.get("operator_id_clean", "UNK")
        shift      = order_d.get("shift_code")
        mat_type   = order_d.get("material_type", "UNK")
        lot_id     = order_d.get("lot_id_clean")

        defect  = insp["defect_code_clean"]
        weights = _REASON_BY_DEFECT.get(defect, [1/6] * 6)
        reason  = random.choices(SCRAP_REASONS, weights=weights)[0]

        # Cost estimation (imprecise — mirrors reality)
        base_cost     = lot_cost_map.get(lot_id, 1.0)
        lbs_per_part  = random.uniform(0.5, 8.0)
        mat_cost_unit = round(base_cost * lbs_per_part, 2)
        labor_cost    = round(random.uniform(5.0, 60.0), 2)

        scrapped = int(insp["quantity_failed"] * random.uniform(0.35, 1.0))
        reworked = max(0, insp["quantity_failed"] - scrapped)

        try:
            scrap_dt = str(pd.to_datetime(insp["inspection_date"]))
        except Exception:
            scrap_dt = str(insp["inspection_date"])

        rows.append({
            "scrap_id":               f"SCRAP-{counter}",
            "work_order_id":          insp["work_order_id"],
            "inspection_id":          insp["inspection_id"],
            "scrap_date":             scrap_dt,
            "machine_id":             machine_id,
            "operator_id":            op_id,
            "shift_code":             shift,
            "material_type":          mat_type,
            "lot_id":                 lot_id,
            "defect_code_clean":      defect,
            "quantity_scrapped":      scrapped,
            "quantity_reworked":      reworked,
            "scrap_reason_raw":       dirty_scrap_reason(reason),  # dirty
            "scrap_reason_clean":     reason,
            "material_cost_per_unit": mat_cost_unit,
            "labor_cost_per_unit":    labor_cost,
            "total_scrap_cost":       round(
                (scrapped * mat_cost_unit) + (scrapped * labor_cost), 2
            ),
        })
        counter += 1

    return pd.DataFrame(rows)