"""
Generator: inspection_records
Source system: QMS (standalone quality system — ETQ, MasterControl, or spreadsheet)
Reality: Inspectors record results manually during or after production.
         System timeouts cause duplicate entries (~8% of records).
         Timestamps sometimes filled in retroactively — 3% fall outside
         the shift window they belong to.
Cross-system note: The QMS stores inspection_id and defect codes.
         To find root causes, you must join to:
           - ERP (work_order_id → machine_id, operator, complexity)
           - Materials (lot_id_clean → supplier, material_type)
           - HR (operator_id → welding_cert_current)
         None of these joins exist in the QMS natively.
         The four hidden patterns only appear when all systems are connected.

Hidden patterns embedded:
  P1  Shift B × Press Brake 1 (M03) → 3.4× base defect rate
      [ERP shift_code] × [MES machine_id] → [QMS defect rate]
  P2  Supplier C × thin-gauge lots → 1.9× base defect rate
      [Materials supplier] × [Materials material_type] → [QMS defect rate]
  P3  High-complexity parts → 1.6× base defect rate
      [ERP complexity] → [QMS defect rate]
  P4  OP007 on welding stations → 2.2× rate, biased toward porosity
      [HR welding_cert_current=No] → [QMS defect rate, defect_code]

Dirty characteristics:
  - defect_code_raw: case variants, typos, abbreviations
  - inspection_date: 3% anomalous timestamps (outside shift window)
  - 8% duplicate records (system double-entry after timeout)
"""
import random
import numpy as np
import pandas as pd
from datetime import timedelta
from faker import Faker
from ..config import (
    RANDOM_SEED, BASE_DEFECT_RATE, THIN_GAUGE, PATTERN_MULTIPLIERS
)
from ..dirty.transformations import dirty_defect, jitter_timestamp

fake = Faker()

DEFECT_CODES = [
    "Dimensional", "Surface Scratch", "Burr", "Weld Defect",
    "Incorrect Material", "Bend Angle", "Porosity", "None"
]

# Defect code probability by machine type (realistic causal logic)
_DEFECT_BY_MACHINE = {
    "M01": {"Surface Scratch": 0.40, "Dimensional": 0.30, "Burr": 0.20, "Incorrect Material": 0.10},
    "M02": {"Surface Scratch": 0.40, "Dimensional": 0.30, "Burr": 0.20, "Incorrect Material": 0.10},
    "M03": {"Bend Angle": 0.45, "Dimensional": 0.30, "Burr": 0.15, "Surface Scratch": 0.10},
    "M04": {"Bend Angle": 0.40, "Dimensional": 0.35, "Burr": 0.15, "Surface Scratch": 0.10},
    "M05": {"Weld Defect": 0.40, "Porosity": 0.30, "Dimensional": 0.20, "Surface Scratch": 0.10},
    "M06": {"Weld Defect": 0.40, "Porosity": 0.30, "Dimensional": 0.20, "Surface Scratch": 0.10},
    "M07": {"Burr": 0.45, "Dimensional": 0.30, "Surface Scratch": 0.15, "Bend Angle": 0.10},
}


def _compute_defect_rate(
    shift_code: str,
    machine_id: str,
    op_id: str,
    complexity: str,
    lot_id: str,
    lot_supplier_map: dict,
    lot_material_map: dict,
) -> float:
    """
    Compute defect rate for this job by applying pattern multipliers.
    Each pattern requires data from a different source system —
    this is what makes the root cause invisible without integration.
    """
    rate = BASE_DEFECT_RATE

    # P1: ERP shift_code × MES machine_id
    if shift_code == "Shift B" and machine_id == "M03":
        rate *= PATTERN_MULTIPLIERS["shift_b_m03"]

    # P2: Materials supplier × Materials material_type
    if lot_id:
        if lot_supplier_map.get(lot_id) == "Supplier C":
            if lot_material_map.get(lot_id) in THIN_GAUGE:
                rate *= PATTERN_MULTIPLIERS["supplier_c_thin_gauge"]

    # P3: ERP complexity
    if complexity == "High":
        rate *= PATTERN_MULTIPLIERS["high_complexity"]

    # P4: HR operator cert status (implicit via OP007 ID)
    if op_id == "OP007" and machine_id in ("M05", "M06"):
        rate *= PATTERN_MULTIPLIERS["op007_welding"]

    return min(rate, 0.52)


def _pick_defect_code(machine_id: str, op_id: str, has_defects: bool) -> str:
    if not has_defects:
        return "None"
    # P4: OP007 → heavy porosity bias (lapsed cert → gas contamination)
    if op_id == "OP007" and machine_id in ("M05", "M06"):
        return random.choices(
            ["Porosity", "Weld Defect", "Dimensional"],
            weights=[0.65, 0.25, 0.10]
        )[0]
    dist = _DEFECT_BY_MACHINE.get(machine_id, {})
    if dist:
        return random.choices(list(dist.keys()), weights=list(dist.values()))[0]
    return random.choice(DEFECT_CODES[:-1])


def generate_inspection_records(
    production_orders_df: pd.DataFrame,
    material_lots_df: pd.DataFrame,
    operators_df: pd.DataFrame,
) -> pd.DataFrame:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    lot_supplier_map = material_lots_df.set_index("lot_id_clean")["supplier"].to_dict()
    lot_material_map = material_lots_df.set_index("lot_id_clean")["material_type"].to_dict()
    all_op_ids       = operators_df["operator_id"].tolist()

    rows    = []
    counter = 1

    for _, order in production_orders_df.iterrows():
        qty  = int(order["quantity_ordered"])
        rate = _compute_defect_rate(
            shift_code       = order.get("shift_code"),
            machine_id       = order["machine_id"],
            op_id            = order["operator_id_clean"],
            complexity       = order["complexity"],
            lot_id           = order.get("lot_id_clean"),
            lot_supplier_map = lot_supplier_map,
            lot_material_map = lot_material_map,
        )
        qty_fail = int(np.random.binomial(qty, rate))
        qty_pass = qty - qty_fail
        defect   = _pick_defect_code(order["machine_id"], order["operator_id_clean"], qty_fail > 0)

        try:
            base_ts = pd.to_datetime(order["actual_start"])
        except Exception:
            base_ts = pd.Timestamp("2023-01-02 08:00:00")

        insp_ts = base_ts + pd.Timedelta(hours=random.uniform(1.0, 5.5))
        if random.random() < 0.03:          # 3% anomalous timestamp
            insp_ts = jitter_timestamp(insp_ts)

        record = {
            "inspection_id":      f"INSP-{counter}",
            "work_order_id":      order["work_order_id"],
            "inspection_date":    str(insp_ts),
            "inspector_id":       random.choice(all_op_ids),
            "quantity_inspected": qty,
            "quantity_passed":    qty_pass,
            "quantity_failed":    qty_fail,
            "defect_code_raw":    dirty_defect(defect),   # dirty
            "defect_code_clean":  defect,
            "disposition":        (
                random.choices(
                    ["Scrap", "Rework", "Use-As-Is"],
                    weights=[0.45, 0.42, 0.13]
                )[0] if qty_fail > 0 else "Pass"
            ),
            "notes": fake.sentence() if random.random() < 0.22 else None,
        }
        rows.append(record)
        counter += 1

        # 8% duplicate records (QMS system double-entry after timeout)
        if random.random() < 0.08:
            dup = record.copy()
            dup["inspection_id"]   = f"INSP-{counter}"
            dup["inspection_date"] = str(insp_ts + pd.Timedelta(minutes=random.randint(2, 45)))
            if random.random() < 0.30:   # minor quantity discrepancy on re-entry
                offset = random.randint(-2, 2)
                dup["quantity_failed"] = max(0, dup["quantity_failed"] + offset)
                dup["quantity_passed"] = qty - dup["quantity_failed"]
            rows.append(dup)
            counter += 1

    return pd.DataFrame(rows)