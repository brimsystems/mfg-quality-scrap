"""
Generator: production_orders
Source system: ERP (work order / job management module)
Reality: Work orders created by production planner. Machine and operator
         assigned at job start by the supervisor — sometimes updated later.
         Lot ID entered manually when material is pulled — frequently skipped.
         Shift code logged retroactively by supervisors at end of week.
Cross-system note: This is the central linking table. It connects:
  - ERP part/customer data (part_number, customer)
  - MES machine data (machine_id)
  - HR operator data (operator_id — dirty 5% of the time)
  - Materials lot data (lot_id — missing 15% of the time)
  None of these links are clean or complete in the source system.
Dirty characteristics:
  - part_number_raw:  5 format variants across entry points
  - operator_id_raw:  5% stored as name string (manual override in tablet UI)
  - shift_code:       10% null (retroactive logging abandoned)
  - lot_id_raw:       15% null (material not scanned at job start)
"""
import random
import numpy as np
import pandas as pd
from datetime import timedelta
from faker import Faker
from ..config import RANDOM_SEED, START_DATE, END_DATE, SHIFT_HOURS
from ..dirty.transformations import dirty_part

fake = Faker()


def generate_production_orders(
    operators_df: pd.DataFrame,
    material_lots_df: pd.DataFrame,
    part_catalog_df: pd.DataFrame,
) -> pd.DataFrame:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Build lookups
    shift_a_ops = operators_df[operators_df["shift"] == "Shift A"]["operator_id"].tolist()
    shift_b_ops = operators_df[operators_df["shift"] == "Shift B"]["operator_id"].tolist()
    op_name_map = operators_df.set_index("operator_id")["operator_name"].to_dict()
    parts       = part_catalog_df.to_dict("records")

    lots = material_lots_df.copy()
    lots["receipt_date_dt"] = pd.to_datetime(lots["receipt_date"])

    rows       = []
    wo_counter = 10000
    d          = START_DATE

    while d < END_DATE:
        is_weekday  = d.weekday() < 5
        is_saturday = d.weekday() == 5
        if not is_weekday and not (is_saturday and random.random() < 0.20):
            d += timedelta(days=1)
            continue

        for _ in range(random.randint(6, 15)):   # 6-15 orders/day for ML volume
            part_row   = random.choice(parts)
            shift      = random.choice(["Shift A", "Shift B"])
            sh_start   = SHIFT_HOURS[shift][0]
            op_pool    = shift_a_ops if shift == "Shift A" else shift_b_ops
            op_id      = random.choice(op_pool)
            machine_id = part_row["primary_machine"]

            # Bias welding parts toward OP007 on welding machines (sets up P4)
            if part_row["requires_welding"] and random.random() < 0.15:
                machine_id = random.choice(["M05", "M06"])
                if shift == "Shift B" and "OP007" in shift_b_ops and random.random() < 0.35:
                    op_id = "OP007"

            # 5% of operator fields stored as name string (tablet manual override)
            op_field = op_name_map[op_id] if random.random() < 0.05 else op_id

            # Find available lot for this part's material
            avail = lots[
                (lots["receipt_date_dt"].dt.date <= d.date()) &
                (lots["material_type"] == part_row["material_type"])
            ]
            lot_clean = None
            lot_raw   = None
            if not avail.empty:
                lot_row   = avail.tail(60).sample(1).iloc[0]
                lot_clean = lot_row["lot_id_clean"]
                lot_raw   = lot_row["lot_id_raw"] if random.random() > 0.15 else None

            actual_start = d.replace(
                hour=sh_start + random.randint(0, 1),
                minute=random.randint(0, 59),
                second=0, microsecond=0
            )

            rows.append({
                "work_order_id":     f"WO-{wo_counter}",
                "part_number_raw":   dirty_part(part_row["part_number"]),  # dirty
                "part_number_clean": part_row["part_number"],
                "customer":          part_row["customer"],
                "quantity_ordered":  random.randint(10, 500),
                "machine_id":        machine_id,
                "operator_id_raw":   op_field,                              # dirty
                "operator_id_clean": op_id,
                "shift_code":        shift if random.random() > 0.10 else None,
                "lot_id_raw":        lot_raw,                               # dirty
                "lot_id_clean":      lot_clean,
                "order_date":        str(d.date()),
                "scheduled_start":   str(actual_start - timedelta(minutes=random.randint(0, 90))),
                "actual_start":      str(actual_start),
                "complexity":        part_row["complexity"],
                "material_type":     part_row["material_type"],
                "requires_welding":  part_row["requires_welding"],
                "std_labor_hrs":     part_row["std_labor_hrs"],
            })
            wo_counter += 1

        d += timedelta(days=1)

    return pd.DataFrame(rows)