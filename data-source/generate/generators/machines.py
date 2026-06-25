"""
Generator: machines
Source system: MES / Shop Floor Data Collection System
Reality: MetroFab uses a basic shop floor tracking system that maintains
         an equipment register. Machine definitions are stable — rarely change.
         In smaller shops this might just be a tab in an ERP or a spreadsheet.
Cross-system note: machine_id appears in production_orders (ERP) and
                   inspection_records (QMS) but is stored inconsistently
                   across systems — a key join challenge in the dbt pipeline.
"""
import pandas as pd
from ..config import MACHINES_DATA


def generate_machines() -> pd.DataFrame:
    return pd.DataFrame(
        MACHINES_DATA,
        columns=["machine_id", "machine_name", "machine_type", "age_years", "location"]
    )