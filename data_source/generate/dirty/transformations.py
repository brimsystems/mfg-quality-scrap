"""
Dirty data transformations for Case 01.

Each function takes a clean value and returns a realistic dirty variant.
These mirror the actual data quality issues found in SMB manufacturing systems:
  - Multiple entry clerks using different formats for the same field
  - Fields copied between systems with different naming conventions
  - Manual entry typos and abbreviations
  - Timestamps recorded retroactively or by a different person
"""
import random
from datetime import timedelta

# ── Part number format variants ───────────────────────────────────────────
# Reality: ERP uses "P-1234", shop floor tablet uses "P1234",
#          older paper travelers say "PART-1234"
_PART_FORMATS = [
    lambda p: p,                              # P-1234    canonical  55%
    lambda p: p.replace("-", ""),             # P1234               20%
    lambda p: p.lower(),                      # p-1234              10%
    lambda p: "PART-" + p.split("-")[1],      # PART-1234           10%
    lambda p: p.replace("-", " "),            # P 1234               5%
]
_PART_WEIGHTS = [0.55, 0.20, 0.10, 0.10, 0.05]

# ── Defect code variants ──────────────────────────────────────────────────
# Reality: QMS has a dropdown but inspectors type in a notes field half the time
DEFECT_VARIANTS = {
    "Dimensional":        ["Dimensional", "DIMENSIONAL", "Dim", "dimensional",
                           "dim.", "Dimentional", "Dimension Error"],
    "Surface Scratch":    ["Surface Scratch", "Scratch", "SCRATCH", "surface scratch",
                           "scrach", "Scr", "surface scr"],
    "Burr":               ["Burr", "BURR", "burr", "Bur",
                           "Sharp Edge", "sharp edge", "Sharp Burr"],
    "Weld Defect":        ["Weld Defect", "WELD", "weld defect", "Weld",
                           "weld def.", "Welding Issue", "Weld Reject"],
    "Incorrect Material": ["Incorrect Material", "Wrong Material", "Mat Error",
                           "incorrect material", "MATERIAL", "Wrong Mat"],
    "Bend Angle":         ["Bend Angle", "BEND", "bend angle", "Angle Error",
                           "angle err", "Bend", "Out of Angle"],
    "Porosity":           ["Porosity", "POROSITY", "porosity", "Poros.",
                           "Void", "void", "Gas Pocket"],
    "None":               ["None", "NONE", "none", "No Defect",
                           "OK", "Pass", "", "PASS"],
}

# ── Scrap reason code variants ────────────────────────────────────────────
# Reality: structured codes used by QA, free text used by operators
SCRAP_REASON_VARIANTS = {
    "OPERATOR_ERROR":    ["OPERATOR_ERROR", "Operator Error", "Op Error",
                          "operator error", "op err", "human error", "Operator"],
    "MATERIAL_DEFECT":   ["MATERIAL_DEFECT", "Material Defect", "Mat Defect",
                          "material defect", "bad material", "Incoming Defect"],
    "MACHINE_ISSUE":     ["MACHINE_ISSUE", "Machine Issue", "Mach Issue",
                          "machine issue", "equipment failure", "Machine"],
    "SETUP_ERROR":       ["SETUP_ERROR", "Setup Error", "set up error",
                          "setup err", "Setup", "First Article Fail"],
    "DESIGN_ISSUE":      ["DESIGN_ISSUE", "Design Issue", "print error",
                          "design issue", "drawing error", "Print Rev Error"],
    "UNKNOWN":           ["UNKNOWN", "Unknown", "unknown", "N/A",
                          "TBD", "", "Not Recorded", "See Notes"],
}

# ── Lot ID format variants ────────────────────────────────────────────────
# Reality: receiving system uses "LOT-1234", shop floor drops the prefix
_LOT_FORMATS = [
    lambda l: l,                          # LOT-1234   canonical  60%
    lambda l: l.replace("-", ""),         # LOT1234               20%
    lambda l: l.split("-")[1],            # 1234                  15%
    lambda l: l.replace("LOT-", "L-"),   # L-1234                 5%
]
_LOT_WEIGHTS = [0.60, 0.20, 0.15, 0.05]


def dirty_part(part: str) -> str:
    """Return part number in a random non-canonical format."""
    fmt = random.choices(_PART_FORMATS, weights=_PART_WEIGHTS)[0]
    return fmt(part)


def dirty_defect(code: str) -> str:
    """Return defect code as a realistic variant (typo, abbreviation, etc.)."""
    return random.choice(DEFECT_VARIANTS.get(code, [code]))


def dirty_scrap_reason(code: str) -> str:
    """Return scrap reason as a realistic variant (structured or free-text)."""
    return random.choice(SCRAP_REASON_VARIANTS.get(code, [code]))


def dirty_lot(lot_id: str) -> str:
    """Return lot ID in a random format variant."""
    fmt = random.choices(_LOT_FORMATS, weights=_LOT_WEIGHTS)[0]
    return fmt(lot_id)


def jitter_timestamp(ts, max_minutes: int = 180):
    """
    Shift a timestamp randomly to simulate retroactive recording errors.
    Reality: inspector fills in paperwork at end of shift, not at time of event.
    """
    return ts + timedelta(minutes=random.randint(-max_minutes, max_minutes))