"""
features.py
Shared feature engineering logic for the C01 defect risk scorer.
Manually maintained — update here when feature definitions change.
Imported by src/training.py and src/scoring.py only.

Feature decisions are documented and analytically justified in
ml/notebooks/ml_prep.ipynb. This file is the production implementation
of those decisions. Keep both in sync when features change.
"""

import pandas as pd


CATEGORICAL_FEATURES = [
    "machine_type",
    "machine_id",
    "shift_code",
    "operator_id",
    "complexity",
    "material_type",
    "supplier",
    "lot_cert_status",
    "requires_welding",
]

NUMERICAL_FEATURES = [
    "machine_age_years",
    "std_labor_hrs",
    "quantity_ordered",
    "schedule_variance_hrs",
]

INTERACTION_FEATURES = [
    "is_bending_shift_b",
    "is_high_complexity",
    "is_supplier_c_thin_gauge",
    "is_lapsed_cert_op",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + INTERACTION_FEATURES

TARGET = "defect_flag"
ID_COL = "work_order_id"

# Operators with lapsed certifications — elevated defect rates across all job types
LAPSED_CERT_OPS = {"OP007", "OP009", "OP012", "OP015"}

# Thin gauge material types — Supplier C quality issues concentrated here
THIN_GAUGE = {"16ga Steel", "14ga Steel"}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives domain-informed interaction features from pre-production mart columns.
    Called identically on train, validation, test, and scoring data.
    Null handling is performed upstream in the dbt pipeline.
    """
    df = df.copy()

    # Bending × Shift B — cross-system pattern requiring MES + QMS join
    df["is_bending_shift_b"] = (
        (df["machine_type"] == "Bending") &
        (df["shift_code"] == "Shift B")
    ).astype(int)

    # High complexity — monotonic signal, binary threshold at High tier
    df["is_high_complexity"] = (
        df["complexity"] == "High"
    ).astype(int)

    # Supplier C thin gauge — quality issues concentrated in 14ga/16ga Steel
    df["is_supplier_c_thin_gauge"] = (
        (df["supplier"] == "Supplier C") &
        (df["material_type"].isin(THIN_GAUGE))
    ).astype(int)

    # Lapsed certification operators — elevated defect rates across all job types
    df["is_lapsed_cert_op"] = (
        df["operator_id"].isin(LAPSED_CERT_OPS)
    ).astype(int)

    return df
