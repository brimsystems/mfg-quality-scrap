"""
inspect_drift.py
One-off diagnostic. Dumps the raw structure of an Evidently ValueDrift metric
so we can confirm exactly which fields it exposes — the distance score, the
threshold, and (the open question) whether Evidently provides its own per-column
drift verdict we can read through instead of comparing the distance ourselves.

Run from the ml/ directory:
    python3 src/inspect_drift.py

This calls the existing monitoring.py tasks as plain functions (via .fn) so it
runs outside any Prefect flow context — that way ordinary print() output is not
intercepted and we actually see the dump.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from monitoring import (
    load_reference_data,
    load_current_data,
    build_evidently_datasets,
    run_drift_report,
)

# Prefect @task objects expose the undecorated function as .fn — calling that
# runs the logic directly, with no flow/task context and no log interception.
ref = load_reference_data.fn()
cur = load_current_data.fn()
ref_ds, cur_ds, cols = build_evidently_datasets.fn(ref, cur)
result, _ = run_drift_report.fn(ref_ds, cur_ds)

d = result.dict()

print("\n" + "=" * 72)
print("TOP-LEVEL keys in result.dict():", list(d.keys()))
metrics = d.get("metrics", [])
print(f"metric count: {len(metrics)}")
print("=" * 72)

# Show the first TWO ValueDrift entries — ideally one categorical (Jensen-Shannon)
# and one numerical (Wasserstein), since their structure can differ.
shown = 0
for m in metrics:
    if str(m.get("metric_name", "")).startswith("ValueDrift"):
        print(f"\n--- ValueDrift entry #{shown + 1} ---")
        print("metric_name:", m.get("metric_name"))
        print("top-level keys:", list(m.keys()))
        print("full dict:")
        print(json.dumps(m, indent=2, default=str))
        shown += 1
        if shown >= 2:
            break

if shown == 0:
    print("\nNo ValueDrift metric found. All metric_names present:")
    for m in metrics:
        print("  -", m.get("metric_name"))

print("\n" + "=" * 72)
print("Done. Paste the two ValueDrift entries above.")
print("=" * 72)
