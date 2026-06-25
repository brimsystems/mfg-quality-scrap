"""
Generator: part_catalog
Source system: ERP (item master / part master)
Reality: Part definitions maintained by engineering or production planning.
         Complexity is an informal field — in reality it often lives in the
         estimating spreadsheet rather than the ERP, and must be mapped manually.
         Standard labor hours are set at time of quoting and rarely updated.
Cross-system note: part_number is the join key to production_orders but appears
                   in 5 different formats across systems — a core cleaning task.
"""
import random
import numpy as np
import pandas as pd
from ..config import RANDOM_SEED, PART_IDS, CUSTOMERS, MAT_TYPES, MACHINES_DATA


def generate_part_catalog() -> pd.DataFrame:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    machine_ids = [m[0] for m in MACHINES_DATA]

    return pd.DataFrame({
        "part_number":      PART_IDS,
        "customer":         [random.choice(CUSTOMERS) for _ in PART_IDS],
        "material_type":    [random.choice(MAT_TYPES)  for _ in PART_IDS],
        "complexity":       random.choices(
                                ["Low", "Medium", "High"],
                                weights=[0.35, 0.45, 0.20],
                                k=len(PART_IDS)
                            ),
        "primary_machine":  random.choices(machine_ids, k=len(PART_IDS)),
        "std_labor_hrs":    [round(random.uniform(0.25, 5.0), 2) for _ in PART_IDS],
        "unit_price":       [round(random.uniform(15, 900), 2)   for _ in PART_IDS],
        "requires_welding": [random.choices([True, False], weights=[0.30, 0.70])[0]
                             for _ in PART_IDS],
    })