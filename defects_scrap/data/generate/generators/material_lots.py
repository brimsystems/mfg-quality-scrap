"""
Generator: material_lots
Source system: Materials / Receiving System (standalone WMS or ERP purchasing module)
Reality: Lot receipts logged by receiving clerk at time of delivery.
         Cert status tracked per lot — but this data lives in the receiving
         system and is never joined to production or quality data.
         Supplier C's higher conditional/missing cert rate (Pattern 2) is
         visible only here, never surfaced alongside defect rates.
Cross-system note: lot_id appears in ERP production orders (when scanned at
                   job start, which happens only 85% of the time) and in QMS
                   scrap events. The supplier → defect rate link requires
                   joining these three systems together.
Dirty characteristics:
  - lot_id_raw: 40% chance of non-canonical format (LOT1234, 1234, L-1234)
  - cert_status: Supplier C has higher conditional/missing rate
"""
import random
import pandas as pd
from datetime import timedelta
from faker import Faker
from ..config import (
    RANDOM_SEED, START_DATE, END_DATE,
    SUPPLIERS, SUPPLIER_WEIGHTS, MAT_TYPES
)
from ..dirty.transformations import dirty_lot

fake = Faker()


def generate_material_lots() -> pd.DataFrame:
    random.seed(RANDOM_SEED)

    rows    = []
    counter = 1000
    d       = START_DATE

    while d < END_DATE:
        for _ in range(random.randint(4, 8)):
            supplier = random.choices(SUPPLIERS, weights=SUPPLIER_WEIGHTS)[0]
            lot_base = f"LOT-{counter}"
            mat_type = random.choice(MAT_TYPES)

            # Supplier C has worse cert compliance — hidden until joined to defects
            if supplier == "Supplier C":
                cert = random.choices(
                    ["Certified", "Conditional", "Missing"],
                    weights=[0.55, 0.30, 0.15]
                )[0]
            else:
                cert = random.choices(
                    ["Certified", "Conditional", "Missing"],
                    weights=[0.80, 0.14, 0.06]
                )[0]

            rows.append({
                "lot_id_clean":     lot_base,
                "lot_id_raw":       dirty_lot(lot_base) if random.random() < 0.40 else lot_base,
                "supplier":         supplier,
                "material_type":    mat_type,
                "receipt_date":     str((d + timedelta(days=random.randint(0, 5))).date()),
                "cert_status":      cert,
                "quantity_lbs":     random.randint(500, 6000),
                "unit_cost_per_lb": round(random.uniform(1.5, 5), 3),
            })
            counter += 1
        d += timedelta(weeks=1)

    return pd.DataFrame(rows)