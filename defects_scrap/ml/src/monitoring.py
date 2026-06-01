"""
monitoring.py
Batch drift and data quality monitoring for the C01 defect risk scorer.
Compares the Jan–Mar 2026 scoring period against the training reference
dataset to detect feature drift and target drift.

Run from the ml/ directory:
    python3 src/monitoring.py

Outputs written to ml/data/monitoring/:
    drift_report_202601.html     — Evidently interactive HTML report
    drift_summary_202601.csv     — per-feature drift results for Part 4 report

MLflow: drift summary logged as artifact to a monitoring run under the
        same experiment as training.
"""

import logging
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import mlflow
from prefect import flow, task
import duckdb

from evidently.future.datasets import PandasDataset, DataDefinition
from evidently.future.report import Report
from evidently.future.presets import DataDriftPreset, DataSummaryPreset

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import (
    engineer_features,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, INTERACTION_FEATURES,
    ALL_FEATURES, TARGET, ID_COL,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

FEATURES_DIR    = Path("data/features").resolve()
SCORING_DIR     = Path("data/scoring").resolve()
MONITORING_DIR  = Path("data/monitoring")
MLFLOW_TRACKING = "mlruns"
EXPERIMENT_NAME = "c01_defect_risk_scorer"
PERIOD_LABEL    = "202601"

# Drift detection threshold — p-value below this flags a feature as drifted
DRIFT_THRESHOLD = 0.05

MONITORING_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# TASKS
# ══════════════════════════════════════════════════════════════════════════════

@task(name="load_reference_data")
def load_reference_data() -> pd.DataFrame:
    """
    Load training Parquet as the reference distribution.
    This is the data the model learned from — the baseline for comparison.
    """
    path = FEATURES_DIR / "train.parquet"
    if not path.exists():
        raise RuntimeError(
            f"Reference data not found at {path}. Run training.py first."
        )
    df = pd.read_parquet(path)
    log.info(f"Reference data loaded: {len(df):,} rows")
    log.info(f"  defect_flag rate: {df[TARGET].mean():.1%}")
    return df


@task(name="load_current_data")
def load_current_data(start: str = "2026-01-01", end: str = "2026-03-31") -> pd.DataFrame:
    """
    Load Jan–Mar 2026 work orders from the mart, apply feature engineering,
    and cache as Parquet for reproducibility.
    """
    cache_path = MONITORING_DIR / f"current_features_{PERIOD_LABEL}.parquet"

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        log.info(f"Current period loaded from cache: {len(df):,} rows")
        return df

    DB_PATH = Path("../data/defects_scrap.duckdb").resolve()
    base_cols = [ID_COL] + CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET]

    con = duckdb.connect(str(DB_PATH), read_only=True)
    df  = con.execute(f"""
        SELECT {', '.join(base_cols)}
        FROM mart_quality__defect_rates
        WHERE actual_start >= '{start}'
          AND actual_start <= '{end}'
        ORDER BY actual_start
    """).df()
    con.close()

    if len(df) == 0:
        raise RuntimeError(
            f"No work orders found between {start} and {end}. "
            "Verify data generation covered this period."
        )

    df = engineer_features(df)
    df.to_parquet(cache_path, index=False)

    log.info(f"Current period loaded from mart: {len(df):,} rows")
    log.info(f"  Period: {start} → {end}")
    log.info(f"  defect_flag rate: {df[TARGET].mean():.1%}")
    log.info(f"  Cached to: {cache_path}")
    return df


@task(name="build_evidently_datasets")
def build_evidently_datasets(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> tuple:
    monitor_cols = ALL_FEATURES + [TARGET]
    monitor_cols = [c for c in monitor_cols if c in reference.columns]

    ref_df = reference[monitor_cols].copy()
    cur_df = current[monitor_cols].copy()

    # requires_welding is bool — treat as numerical for Evidently compatibility
    cat_cols = [c for c in CATEGORICAL_FEATURES
                if c in ref_df.columns and c != "requires_welding"]
    num_cols = (NUMERICAL_FEATURES + INTERACTION_FEATURES +
                ["requires_welding", TARGET])
    num_cols = [c for c in num_cols if c in ref_df.columns]

    # Cast categoricals to string, numericals to float
    for col in cat_cols:
        ref_df[col] = ref_df[col].astype(str).fillna("UNKNOWN")
        cur_df[col] = cur_df[col].astype(str).fillna("UNKNOWN")
    for col in num_cols:
        ref_df[col] = ref_df[col].astype(float)
        cur_df[col] = cur_df[col].astype(float)

    dd = DataDefinition(
        numerical_columns=num_cols,
        categorical_columns=cat_cols,
    )

    ref_ds = PandasDataset(ref_df, data_definition=dd)
    cur_ds = PandasDataset(cur_df, data_definition=dd)

    log.info(f"Evidently datasets built")
    log.info(f"  Reference: {len(ref_df):,} rows  |  Current: {len(cur_df):,} rows")
    log.info(f"  Monitoring {len(monitor_cols)} features")

    return ref_ds, cur_ds, monitor_cols

@task(name="run_drift_report")
def run_drift_report(ref_ds, cur_ds) -> tuple:
    """
    Run Evidently Data Drift and Data Summary reports.
    Returns (result, html_path).
    """
    report  = Report([DataDriftPreset(), DataSummaryPreset()])
    result  = report.run(reference_data=ref_ds, current_data=cur_ds)

    html_path = MONITORING_DIR / f"drift_report_{PERIOD_LABEL}.html"
    result.save_html(str(html_path))
    log.info(f"Drift report saved: {html_path}")

    return result, html_path


@task(name="build_drift_summary")
def build_drift_summary(result, monitor_cols: list) -> pd.DataFrame:
    """
    Extract per-feature drift results from the Evidently report dict.
    Returns a tidy DataFrame with one row per feature.
    """
    metrics = result.dict().get("metrics", [])

    # Extract ValueDrift entries — one per feature
    drift_rows = []
    for m in metrics:
        name = m.get("metric_name", "")
        if not name.startswith("ValueDrift"):
            continue

        # Parse: ValueDrift(column=X,method=Y,threshold=Z)
        try:
            parts  = name.replace("ValueDrift(", "").rstrip(")")
            kv     = dict(p.split("=") for p in parts.split(","))
            col    = kv.get("column", "unknown")
            method = kv.get("method", "unknown")
            pvalue = float(m.get("value", 1.0))
            drifted = pvalue < DRIFT_THRESHOLD

            # Classify feature type
            if col in CATEGORICAL_FEATURES:
                feat_type = "categorical"
            elif col in INTERACTION_FEATURES:
                feat_type = "interaction"
            elif col == TARGET:
                feat_type = "target"
            else:
                feat_type = "numerical"

            drift_rows.append({
                "feature":       col,
                "feature_type":  feat_type,
                "drift_detected":drifted,
                "p_value":       round(pvalue, 4),
                "test_method":   method,
                "threshold":     DRIFT_THRESHOLD,
            })
        except Exception:
            continue

    summary = pd.DataFrame(drift_rows).sort_values(
        ["drift_detected", "p_value"], ascending=[False, True]
    ).reset_index(drop=True)

    n_drifted = summary["drift_detected"].sum()
    log.info(f"Drift summary: {n_drifted}/{len(summary)} features show drift "
             f"(p < {DRIFT_THRESHOLD})")

    if n_drifted > 0:
        drifted_feats = summary[summary["drift_detected"]]["feature"].tolist()
        log.info(f"  Drifted features: {drifted_feats}")
    else:
        log.info("  No statistically significant drift detected.")

    return summary


@task(name="write_drift_summary")
def write_drift_summary(summary: pd.DataFrame) -> Path:
    """Write drift summary CSV for consumption by generate_ml_report.py."""
    path = MONITORING_DIR / f"drift_summary_{PERIOD_LABEL}.csv"
    summary.to_csv(path, index=False)
    log.info(f"Drift summary written: {path}")
    return path


@task(name="log_monitoring_run_to_mlflow")
def log_monitoring_run_to_mlflow(
    summary: pd.DataFrame,
    html_path: Path,
    summary_path: Path,
) -> str:
    """Log monitoring run metadata and artifacts to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment(EXPERIMENT_NAME)

    n_drifted   = int(summary["drift_detected"].sum())
    n_features  = len(summary)
    drift_share = round(n_drifted / n_features, 4) if n_features > 0 else 0.0

    target_row  = summary[summary["feature"] == TARGET]
    target_drift = bool(target_row["drift_detected"].values[0]) \
                   if len(target_row) > 0 else False

    with mlflow.start_run(run_name=f"monitoring_{PERIOD_LABEL}") as run:
        mlflow.set_tags({
            "run_type":      "monitoring",
            "period_label":  PERIOD_LABEL,
            "reference":     "train split (Jan 2023 – Dec 2024)",
            "current":       "val + test splits (Jan – Dec 2025)",
            "drift_threshold": DRIFT_THRESHOLD,
        })
        mlflow.log_metrics({
            "n_features_monitored": n_features,
            "n_features_drifted":   n_drifted,
            "drift_share":          drift_share,
            "target_drift_detected":int(target_drift),
        })
        mlflow.log_artifact(str(html_path),    artifact_path="monitoring")
        mlflow.log_artifact(str(summary_path), artifact_path="monitoring")

        run_id = run.info.run_id
        log.info(f"Monitoring run logged to MLflow: {run_id}")
        log.info(f"  Features drifted: {n_drifted}/{n_features} "
                 f"({drift_share:.0%})")
        log.info(f"  Target drift:     {target_drift}")
        return run_id


# ══════════════════════════════════════════════════════════════════════════════
# FLOW
# ══════════════════════════════════════════════════════════════════════════════

@flow(name="defect_risk_monitoring", log_prints=True)
def monitoring_flow(period_label: str = PERIOD_LABEL) -> None:
    """
    Batch monitoring flow for the defect risk scorer.
    Compares the current scoring period against the training reference
    to detect feature drift and target drift.
    """
    log.info(f"Starting monitoring flow — period: {period_label}")

    reference = load_reference_data()
    current   = load_current_data()

    ref_ds, cur_ds, monitor_cols = build_evidently_datasets(reference, current)

    result, html_path = run_drift_report(ref_ds, cur_ds)
    summary           = build_drift_summary(result, monitor_cols)
    summary_path      = write_drift_summary(summary)

    log_monitoring_run_to_mlflow(summary, html_path, summary_path)

    log.info("Monitoring flow complete.")
    log.info(f"  HTML report: {html_path}")
    log.info(f"  CSV summary: {summary_path}")


if __name__ == "__main__":
    monitoring_flow()
