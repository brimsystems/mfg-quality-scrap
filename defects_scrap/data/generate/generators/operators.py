"""
Generator: operators
Source system: HR System (ADP / Paycom spreadsheet export)
Reality: Basic employee records maintained by HR admin. Certification levels
         tracked here as a yes/no field updated annually — but this data
         is never connected to production or quality systems.
         OP007's lapsed welding cert (Pattern 4) is visible only here.
Cross-system note: operator_id appears in ERP work orders and QMS inspection
                   records, but HR data is never joined to either in practice.
         The cert_current field crossing to defect rate is the hidden insight.
"""
import random
import numpy as np
import pandas as pd
from faker import Faker
from ..config import RANDOM_SEED

fake = Faker()
Faker.seed(RANDOM_SEED)


def generate_operators(n: int = 20) -> pd.DataFrame:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    names  = [fake.name() for _ in range(n)]
    shifts = ["Shift A"] * (n // 2) + ["Shift B"] * (n // 2)
    random.shuffle(shifts)

    specs = random.choices(
        ["Laser", "Bending", "Welding", "General"],
        weights=[0.25, 0.25, 0.20, 0.30],
        k=n
    )
    specs[6] = "Welding"   # OP007 is the problem welder

    # welding_cert_current: only OP007's is "No" — lapsed 14 months ago
    cert_current = ["Yes"] * n
    cert_current[6] = "No"
    cert_current[8] = "No"
    cert_current[11] = "No"
    cert_current[14] = "No"

    return pd.DataFrame({
        "operator_id":          [f"OP{str(i + 1).zfill(3)}" for i in range(n)],
        "operator_name":        names,
        "shift":                shifts,
        "hire_date":            [str(fake.date_between(start_date="-12y", end_date="-6m"))
                                 for _ in range(n)],
        "cert_level":           np.random.choice(
                                    ["Level 1", "Level 2", "Level 3"],
                                    size=n, p=[0.35, 0.45, 0.20]
                                ),
        "specialization":       specs,
        "welding_cert_current": cert_current,
    })