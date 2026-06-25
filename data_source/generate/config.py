"""
Central configuration for Case 01 — Quality Escapes & Scrap.
All parameters live here so regenerating with different settings
requires changing only this file.
"""
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
# File lives at data_source/generate/config.py
# parent.parent.parent resolves to the repo root
MODULE_DIR  = Path(__file__).parent.parent.parent
RAW_DIR     = MODULE_DIR / "data_source" / "raw"
SAMPLES_DIR = MODULE_DIR / "data_source" / "samples"

# ── Randomization ─────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Date range (36 months for ML volume) ──────────────────────────────────
START_DATE  = datetime(2023, 1, 2)   # first Monday of 2023
END_DATE    = datetime(2026, 3, 31)
SAMPLE_SIZE = 200                    # rows per sample file (committed to git)

# ── Business parameters ───────────────────────────────────────────────────
BASE_DEFECT_RATE = 0.03             # 2.5% baseline — typical job-shop sheet metal
THIN_GAUGE       = frozenset({"16ga Steel", "14ga Steel"})

SUPPLIERS        = ["Supplier A", "Supplier B", "Supplier C", "Supplier D"]
SUPPLIER_WEIGHTS = [0.35, 0.30, 0.25, 0.10]

MAT_TYPES = [
    "16ga Steel", "14ga Steel", "12ga Steel",
    '1/4" Plate', '3/8" Plate', "Aluminum 5052", "Stainless 304"
]

CUSTOMERS = [f"Customer {chr(65 + i)}" for i in range(8)]
PART_IDS  = [f"P-{1000 + i}" for i in range(35)]

SHIFT_HOURS = {
    "Shift A": (6, 14),    # 6am – 2pm
    "Shift B": (14, 22),   # 2pm – 10pm
}

# ── Machine definitions ───────────────────────────────────────────────────
# Columns: machine_id, machine_name, machine_type, age_years, location
MACHINES_DATA = [
    ("M01", "Laser Cutter 1",    "Laser Cutting", 3,  "Bay A"),
    ("M02", "Laser Cutter 2",    "Laser Cutting", 8,  "Bay A"),
    ("M03", "Press Brake 1",     "Bending",       12, "Bay B"),  # ← Pattern 1
    ("M04", "Press Brake 2",     "Bending",       2,  "Bay B"),
    ("M05", "Welding Station 1", "Welding",        6,  "Bay C"),  # ← Pattern 4
    ("M06", "Welding Station 2", "Welding",        4,  "Bay C"),  # ← Pattern 4
    ("M07", "Punch Press 1",     "Punching",       9,  "Bay A"),
]

# ── Hidden pattern multipliers (embedded in inspection generator) ──────────
PATTERN_MULTIPLIERS = {
    "shift_b_m03":           2.2,   # P1: Shift B × Press Brake 1
    "supplier_c_thin_gauge": 2.2,   # P2: Supplier C × thin-gauge material
    "high_complexity":       2.0,   # P3: High-complexity parts
    "operator_cert":         3.0,   # P4: Operator certification lapsed
}

# ── Source system → table mapping (drives output directory structure) ──────
TABLE_SYSTEM_MAP = {
    "machines":           "mes",
    "operators":          "hr",
    "material_lots":      "materials",
    "part_catalog":       "erp",
    "production_orders":  "erp",
    "inspection_records": "qms",
    "scrap_events":       "qms",
}