"""
BRIM Systems — Case 01: Quality Escapes & Scrap
Data Generator Entry Point

Manufacturer:  [redact]
Type:          Custom sheet metal fabricator, ~75 employees, Elizabeth NJ
Period:        Jan 2023 – Dec 2025 (36 months)

Usage (from repo root):
    python -m defects_scrap.data.generate.run_generator

Output structure:
    defects_scrap/data/raw/
    ├── erp/               ← JobBOSS-style ERP
    │   ├── production_orders.csv
    │   └── part_catalog.csv
    ├── mes/               ← Shop floor data collection
    │   └── machines.csv
    ├── qms/               ← Quality management system
    │   ├── inspection_records.csv
    │   └── scrap_events.csv
    ├── materials/         ← Receiving / WMS
    │   └── material_lots.csv
    └── hr/                ← HR system (ADP export)
        └── operators.csv

    defects_scrap/data/samples/
    └── (same structure — 200-row slices committed to git for demos)

Storage note:
    data/raw/ is gitignored. Files live in Codespace ephemeral storage.
    Regenerate at any time with the command above — output is deterministic
    (seeded) so you always get identical data. data/samples/ is committed
    to git and always available without regenerating.
"""

import pandas as pd
from pathlib import Path
from .config import RAW_DIR, SAMPLES_DIR, SAMPLE_SIZE, TABLE_SYSTEM_MAP
from .generators.machines           import generate_machines
from .generators.operators          import generate_operators
from .generators.material_lots      import generate_material_lots
from .generators.part_catalog       import generate_part_catalog
from .generators.production_orders  import generate_production_orders
from .generators.inspections        import generate_inspection_records
from .generators.scrap_events       import generate_scrap_events


def _save(df: pd.DataFrame, name: str, base_dir: Path, sample: bool = False) -> None:
    """Save a dataframe to the correct source-system subdirectory."""
    system  = TABLE_SYSTEM_MAP[name]
    out_dir = base_dir / system
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix   = "_sample" if sample else ""
    filepath = out_dir / f"{name}{suffix}.csv"
    df.to_csv(filepath, index=False)
    size_kb = filepath.stat().st_size / 1024
    size_str = f"{size_kb/1024:.2f} MB" if size_kb > 1024 else f"{size_kb:.0f} KB"
    print(f"  ✓  [{system:>10}]  {name + suffix:<40} {len(df):>7,} rows   {size_str}")


def run() -> None:
    print("=" * 70)
    print("BRIM Systems — Case 01: Quality Escapes & Scrap")
    print("Generating synthetic manufacturing data (36 months)")
    print("=" * 70)

    # ── Generate tables in dependency order ──────────────────────────────
    print("\n[1/7] Machines            (MES)")
    machines = generate_machines()

    print("[2/7] Operators           (HR)")
    operators = generate_operators()

    print("[3/7] Material lots       (Materials/Receiving)")
    material_lots = generate_material_lots()

    print("[4/7] Part catalog        (ERP)")
    part_catalog = generate_part_catalog()

    print("[5/7] Production orders   (ERP)  — may take ~15s...")
    production_orders = generate_production_orders(
        operators_df     = operators,
        material_lots_df = material_lots,
        part_catalog_df  = part_catalog,
    )

    print("[6/7] Inspection records  (QMS)  — may take ~30s...")
    inspection_records = generate_inspection_records(
        production_orders_df = production_orders,
        material_lots_df     = material_lots,
        operators_df         = operators,
    )

    print("[7/7] Scrap events        (QMS)")
    scrap_events = generate_scrap_events(
        inspection_records_df = inspection_records,
        production_orders_df  = production_orders,
        material_lots_df      = material_lots,
    )

    # ── Collect all tables ────────────────────────────────────────────────
    tables = {
        "machines":           machines,
        "operators":          operators,
        "material_lots":      material_lots,
        "part_catalog":       part_catalog,
        "production_orders":  production_orders,
        "inspection_records": inspection_records,
        "scrap_events":       scrap_events,
    }

    # ── Save full datasets (gitignored) ───────────────────────────────────
    print(f"\n── Full datasets → {RAW_DIR}")
    for name, df in tables.items():
        _save(df, name, RAW_DIR)

    # ── Save sample files (committed to git) ──────────────────────────────
    print(f"\n── Sample files ({SAMPLE_SIZE} rows) → {SAMPLES_DIR}")
    for name, df in tables.items():
        sample = df.head(SAMPLE_SIZE)
        _save(sample, name, SAMPLES_DIR, sample=True)

    # ── Pattern verification ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PATTERN VERIFICATION — confirm hidden signals are embedded")
    print("=" * 70)

    merged = production_orders.merge(
        inspection_records.drop_duplicates("work_order_id"),
        on="work_order_id", how="left"
    )
    merged["defect_rate"] = (
        merged["quantity_failed"] / merged["quantity_inspected"].replace(0, float("nan"))
    )

    # P1
    p1 = merged[merged["machine_id"] == "M03"].groupby("shift_code")["defect_rate"].mean()
    sa = p1.get("Shift A", 0.001)
    sb = p1.get("Shift B", 0.001)
    print(f"\n[P1] Shift B × M03 (ERP shift + MES machine → QMS defect rate)")
    print(f"     Shift A: {sa:.1%}   Shift B: {sb:.1%}   Multiplier: {sb/sa:.1f}×  (target 3.4×)")

    # P2
    THIN = {"16ga Steel", "14ga Steel"}
    m2   = merged.merge(
        material_lots[["lot_id_clean", "supplier", "material_type"]],
        left_on="lot_id_clean", right_on="lot_id_clean", how="left"
    )
    thin   = m2[m2["material_type_y"].isin(THIN)]
    p2     = thin.groupby("supplier")["defect_rate"].mean()
    others = p2[p2.index != "Supplier C"].mean()
    sc     = p2.get("Supplier C", 0)
    print(f"\n[P2] Supplier C thin gauge (Materials supplier → QMS defect rate)")
    print(f"     Supplier C: {sc:.1%}   Others avg: {others:.1%}   Multiplier: {sc/max(others,0.001):.1f}×  (target 1.9×)")

    # P4
    weld  = merged[merged["machine_id"].isin(["M05", "M06"])]
    p4    = weld.groupby("operator_id_clean")["defect_rate"].mean()
    op007 = p4.get("OP007", 0)
    rest  = p4[p4.index != "OP007"].mean()
    print(f"\n[P4] OP007 welding (HR cert lapse → QMS defect rate)")
    print(f"     OP007: {op007:.1%}   Others avg: {rest:.1%}   Multiplier: {op007/max(rest,0.001):.1f}×  (target 2.2×)")

    # Cost summary
    total = scrap_events["total_scrap_cost"].sum()
    print(f"\n── Scrap cost summary")
    print(f"   Total (36 months):   ${total:>12,.0f}")
    print(f"   Monthly average:     ${total/36:>12,.0f}")

    # Dirty data rates
    pn_dirty  = (production_orders["part_number_raw"] != production_orders["part_number_clean"]).mean()
    lot_miss  = production_orders["lot_id_raw"].isna().mean()
    shift_miss= production_orders["shift_code"].isna().mean()
    op_name   = production_orders["operator_id_raw"].apply(lambda x: not str(x).startswith("OP")).mean()
    dup_rate  = inspection_records.duplicated("work_order_id", keep=False).mean()
    print(f"\n── Dirty data rates")
    print(f"   Part number non-canonical:  {pn_dirty:.0%}")
    print(f"   Missing lot IDs:            {lot_miss:.0%}")
    print(f"   Missing shift codes:        {shift_miss:.0%}")
    print(f"   Operator field as name:     {op_name:.0%}")
    print(f"   Duplicate inspection WOs:   {dup_rate:.0%}")

    print("\n" + "=" * 70)
    print("Generation complete.")
    print()
    print("Next steps:")
    print("  git add defects_scrap/data/samples/")
    print("  git commit -m 'feat: add m01 sample data files'")
    print("  git push origin feature/m01-data-generator")
    print("=" * 70)


if __name__ == "__main__":
    run()