"""
monitoring.py
Batch monitoring for the C01 defect risk scorer.

Runs once per calendar month (Jan/Feb/Mar 2026), each month compared against the
training reference, producing a longitudinal view across four monitoring layers:

  1. Performance      — precision/recall vs reference, using actuals (direct,
                        and fast here because labels land within the period)
  2. Target drift     — actual defect rate vs the training baseline (rate
                        comparison, not a binary distance — more interpretable)
  3. Prediction drift — current score distribution vs the validation-set
                        prediction reference (early proxy, label-free)
  4. Feature drift    — input distributions vs training (diagnostic context)
  + Data quality      — nulls / cardinality / unseen categories (substrate)

Drift metric note:
  Evidently's ValueDrift returns a DISTANCE (Jensen-Shannon for categoricals,
  normed Wasserstein for numericals), with its own per-column threshold carried
  in metric['config']['threshold'] (default 0.1). We read that threshold through
  and flag drift as `distance >= threshold`. We do NOT treat the value as a
  p-value and we do NOT recompute a test by hand.

Run from the ml/ directory:
    python3 src/monitoring.py

Outputs to ml/data/monitoring/:
    drift_report_2026MM.html       — Evidently interactive feature-drift report (per month)
    drift_summary_2026MM.csv       — per-feature drift table (per month)
    current_features_2026MM.parquet— cached scored-period features (per month)
    period_monitoring.csv          — consolidated longitudinal table (all months)
"""

import logging
import warnings
from pathlib import Path

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
DB_PATH         = Path("../data_source/defects_scrap.duckdb").resolve()
MLFLOW_TRACKING = "sqlite:///mlruns/mlflow.db"
EXPERIMENT_NAME = "c01_defect_risk_scorer"

# Monitoring periods — (label, start, end, display name). One calendar month each.
PERIODS = [
    ("202601", "2026-01-01", "2026-01-31", "January 2026"),
    ("202602", "2026-02-01", "2026-02-28", "February 2026"),
    ("202603", "2026-03-01", "2026-03-31", "March 2026"),
]

# ── Thresholds ───────────────────────────────────────────────────────────────
# Distance threshold for feature and prediction drift. Evidently returns a
# distance (JS / normed Wasserstein); drift is flagged when distance >= threshold.
# We read Evidently's own per-column threshold from config and fall back to this.
DRIFT_SCORE_THRESHOLD = 0.1

# Retraining-decision thresholds (rebalanced — performance/target lead).
MIN_HIGH_PRECISION   = 0.85   # primary: High-tier precision floor
TARGET_RATE_DELTA    = 0.10   # primary: |period rate − training rate| ceiling
MAX_DRIFTED_FEATURES = 3      # secondary: feature-drift count that warrants review

MONITORING_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# DRIFT EXTRACTION (shared by feature and prediction drift)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_value_drift(result) -> list:
    """
    Pull per-column ValueDrift entries from an Evidently result.

    Reads column / method / threshold from metric['config'] (confirmed structure)
    and the distance from metric['value']. Flags drift as distance >= threshold,
    using Evidently's own per-column threshold.
    """
    rows = []
    for m in result.dict().get("metrics", []):
        if not str(m.get("metric_name", "")).startswith("ValueDrift"):
            continue
        cfg       = m.get("config", {}) or {}
        col       = cfg.get("column", "unknown")
        method    = cfg.get("method", "unknown")
        threshold = float(cfg.get("threshold", DRIFT_SCORE_THRESHOLD))
        score     = float(m.get("value", 0.0))
        rows.append({
            "feature":        col,
            "test_method":    method,
            "threshold":      threshold,
            "drift_score":    round(score, 4),
            "drift_detected": bool(score >= threshold),
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# REFERENCE DATA (loaded once, reused across periods)
# ══════════════════════════════════════════════════════════════════════════════

@task(name="load_reference_features")
def load_reference_features() -> pd.DataFrame:
    """Training feature set — the reference distribution for feature drift and
    the baseline defect rate for target drift."""
    path = FEATURES_DIR / "train.parquet"
    if not path.exists():
        raise RuntimeError(f"Reference not found at {path}. Run training.py first.")
    df = pd.read_parquet(path)
    print(f"Reference (train): {len(df):,} rows  |  defect_flag rate {df[TARGET].mean():.1%}")
    return df


@task(name="load_validation_reference")
def load_validation_reference() -> pd.DataFrame:
    """Validation-set predicted probabilities — the reference distribution for
    prediction drift (frozen at training time)."""
    path = FEATURES_DIR / "validation_predictions.parquet"
    if not path.exists():
        raise RuntimeError(
            f"Validation reference not found at {path}. "
            "Re-run training.py to emit validation_predictions.parquet."
        )
    df = pd.read_parquet(path)
    print(f"Prediction-drift reference (validation): {len(df):,} rows  |  "
          f"mean p {df['defect_probability'].mean():.4f}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PER-PERIOD INPUTS
# ══════════════════════════════════════════════════════════════════════════════

@task(name="load_current_features")
def load_current_features(start: str, end: str, label: str) -> pd.DataFrame:
    """Load the period's work orders from the mart, engineer features, cache."""
    cache_path = MONITORING_DIR / f"current_features_{label}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"[{label}] current features from cache: {len(df):,} rows")
        return df

    base_cols = [ID_COL] + CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET]
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df  = con.execute(f"""
        SELECT {', '.join(base_cols)}
        FROM mart_quality__defect_rates
        WHERE actual_start >= '{start}' AND actual_start <= '{end}'
        ORDER BY actual_start
    """).df()
    con.close()

    if len(df) == 0:
        raise RuntimeError(f"No work orders between {start} and {end} for {label}.")

    df = engineer_features(df)
    df.to_parquet(cache_path, index=False)
    print(f"[{label}] current features from mart: {len(df):,} rows  |  "
          f"defect_flag rate {df[TARGET].mean():.1%}")
    return df


@task(name="load_period_predictions")
def load_period_predictions(label: str) -> pd.DataFrame:
    """Scored predictions for the period (defect_probability + actual_defect_flag)."""
    path = SCORING_DIR / f"predictions_{label}.parquet"
    if not path.exists():
        raise RuntimeError(f"Predictions not found at {path}. Run scoring.py first.")
    return pd.read_parquet(path)


@task(name="load_period_accuracy")
def load_period_accuracy(label: str) -> pd.DataFrame:
    """Accuracy summary for the period (precision/recall by tier)."""
    path = SCORING_DIR / f"accuracy_summary_{label}.csv"
    if not path.exists():
        raise RuntimeError(f"Accuracy summary not found at {path}. Run scoring.py first.")
    return pd.read_csv(path)


# ══════════════════════════════════════════════════════════════════════════════
# DRIFT COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

@task(name="run_feature_drift")
def run_feature_drift(reference: pd.DataFrame, current: pd.DataFrame,
                      label: str) -> tuple:
    """
    Feature drift for one period: training reference vs current features.
    Monitors ALL_FEATURES only — the target is handled separately as a rate.
    Writes the per-feature CSV and the Evidently HTML. Returns (summary_df, html_path).
    """
    monitor_cols = [c for c in ALL_FEATURES if c in reference.columns]

    ref_df = reference[monitor_cols].copy()
    cur_df = current[monitor_cols].copy()

    # requires_welding is bool — treat as numerical for Evidently
    cat_cols = [c for c in CATEGORICAL_FEATURES
                if c in monitor_cols and c != "requires_welding"]
    num_cols = [c for c in (NUMERICAL_FEATURES + INTERACTION_FEATURES + ["requires_welding"])
                if c in monitor_cols]

    for c in cat_cols:
        ref_df[c] = ref_df[c].astype(str).fillna("UNKNOWN")
        cur_df[c] = cur_df[c].astype(str).fillna("UNKNOWN")
    for c in num_cols:
        ref_df[c] = ref_df[c].astype(float)
        cur_df[c] = cur_df[c].astype(float)

    dd = DataDefinition(numerical_columns=num_cols, categorical_columns=cat_cols)
    ref_ds = PandasDataset(ref_df, data_definition=dd)
    cur_ds = PandasDataset(cur_df, data_definition=dd)

    report = Report([DataDriftPreset(), DataSummaryPreset()])
    result = report.run(reference_data=ref_ds, current_data=cur_ds)

    html_path = MONITORING_DIR / f"drift_report_{label}.html"
    result.save_html(str(html_path))

    rows = _extract_value_drift(result)
    for r in rows:
        col = r["feature"]
        if col in CATEGORICAL_FEATURES:
            r["feature_type"] = "categorical"
        elif col in INTERACTION_FEATURES:
            r["feature_type"] = "interaction"
        else:
            r["feature_type"] = "numerical"

    summary = pd.DataFrame(rows)[
        ["feature", "feature_type", "drift_detected", "drift_score", "test_method", "threshold"]
    ].sort_values(["drift_detected", "drift_score"], ascending=[False, False]).reset_index(drop=True)

    summary_path = MONITORING_DIR / f"drift_summary_{label}.csv"
    summary.to_csv(summary_path, index=False)

    n_drift = int(summary["drift_detected"].sum())
    top = summary.iloc[0]
    print(f"[{label}] feature drift: {n_drift}/{len(summary)} drifted  |  "
          f"largest distance {top['feature']}={top['drift_score']:.4f}")
    return summary, html_path


@task(name="run_prediction_drift")
def run_prediction_drift(ref_probs: pd.Series, cur_probs: pd.Series,
                         label: str) -> dict:
    """
    Prediction (output) drift for one period: the model's score distribution
    this period vs the validation-set reference. Same distance method as
    feature drift, on the single defect_probability column.
    """
    ref_df = pd.DataFrame({"defect_probability": ref_probs.astype(float).values})
    cur_df = pd.DataFrame({"defect_probability": cur_probs.astype(float).values})

    dd = DataDefinition(numerical_columns=["defect_probability"], categorical_columns=[])
    ref_ds = PandasDataset(ref_df, data_definition=dd)
    cur_ds = PandasDataset(cur_df, data_definition=dd)

    result = Report([DataDriftPreset()]).run(reference_data=ref_ds, current_data=cur_ds)
    rows = _extract_value_drift(result)

    if rows:
        r = rows[0]
        out = {
            "prediction_drift_score":     r["drift_score"],
            "prediction_drift_threshold": r["threshold"],
            "prediction_drift_detected":  r["drift_detected"],
            "prediction_test_method":     r["test_method"],
        }
    else:
        out = {
            "prediction_drift_score": float("nan"),
            "prediction_drift_threshold": DRIFT_SCORE_THRESHOLD,
            "prediction_drift_detected": False,
            "prediction_test_method": "unavailable",
        }

    out["ref_pred_mean"] = round(float(ref_probs.mean()), 4)
    out["cur_pred_mean"] = round(float(cur_probs.mean()), 4)
    print(f"[{label}] prediction drift: distance {out['prediction_drift_score']:.4f} "
          f"(thr {out['prediction_drift_threshold']})  "
          f"ref p {out['ref_pred_mean']:.3f} → cur p {out['cur_pred_mean']:.3f}")
    return out


@task(name="log_period_to_mlflow")
def log_period_to_mlflow(row: dict, html_path: Path, summary_path: Path) -> None:
    """Log one period's monitoring run to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment(EXPERIMENT_NAME)
    label = row["period_label"]

    with mlflow.start_run(run_name=f"monitoring_{label}"):
        mlflow.set_tags({
            "run_type":      "monitoring",
            "period_label":  label,
            "period_name":   row["period_name"],
            "reference":     "train split (feature drift) + validation preds (prediction drift)",
            "status":        row["status"],
            "drift_metric":  "distance (Jensen-Shannon / Wasserstein)",
        })
        mlflow.log_metrics({
            "n_features_monitored":      row["n_features_monitored"],
            "n_features_drifted":        row["n_features_drifted"],
            "prediction_drift_score":    row["prediction_drift_score"],
            "prediction_drift_detected": int(row["prediction_drift_detected"]),
            "train_defect_rate":         row["train_defect_rate"],
            "period_defect_rate":        row["period_defect_rate"],
            "target_rate_delta":         row["target_rate_delta"],
            "target_drift_detected":     int(row["target_drift_detected"]),
            "high_precision":            row["high_precision"],
            "high_recall":               row["high_recall"],
            "medium_precision":          row["medium_precision"],
            "medium_recall":             row["medium_recall"],
            "retrain_recommended":       int(row["retrain_recommended"]),
        })
        mlflow.log_artifact(str(html_path),    artifact_path="monitoring")
        mlflow.log_artifact(str(summary_path), artifact_path="monitoring")


# ══════════════════════════════════════════════════════════════════════════════
# PER-PERIOD ORCHESTRATION (plain helper — no nested flows)
# ══════════════════════════════════════════════════════════════════════════════

def _decide(high_precision: float, target_rate_delta: float,
            prediction_drift_detected: bool, n_features_drifted: int) -> tuple:
    """
    Rebalanced retraining decision.
      Primary (lead, drives RETRAIN): measured performance and target drift.
      Secondary (proxy, drives INVESTIGATE): prediction and feature drift.
    Returns (status, retrain_recommended, reasons).
    """
    reasons = []
    primary = []
    if high_precision < MIN_HIGH_PRECISION:
        primary.append(f"High-tier precision {high_precision:.1%} below "
                       f"{MIN_HIGH_PRECISION:.0%} floor")
    if target_rate_delta > TARGET_RATE_DELTA:
        primary.append(f"Defect rate shifted {target_rate_delta:.1%} from training "
                       f"baseline (> {TARGET_RATE_DELTA:.0%})")

    secondary = []
    if prediction_drift_detected:
        secondary.append("Prediction (output) distribution drifted vs validation reference")
    if n_features_drifted > MAX_DRIFTED_FEATURES:
        secondary.append(f"{n_features_drifted} input features drifted "
                         f"(> {MAX_DRIFTED_FEATURES})")

    if primary:
        return "RETRAIN", True, primary + secondary
    if secondary:
        return "INVESTIGATE", False, secondary
    return "HEALTHY", False, []


def _monitor_period(reference, val_ref, train_rate, label, start, end, name) -> dict:
    """Run all monitoring layers for one period and assemble its summary row."""
    print(f"\n──────── {name} ({label}) ────────")

    current = load_current_features(start, end, label)
    preds   = load_period_predictions(label)
    acc     = load_period_accuracy(label)

    # Feature drift
    fsummary, html_path = run_feature_drift(reference, current, label)
    summary_path = MONITORING_DIR / f"drift_summary_{label}.csv"
    n_features  = len(fsummary)
    n_drifted   = int(fsummary["drift_detected"].sum())

    # Prediction drift
    pdrift = run_prediction_drift(
        val_ref["defect_probability"], preds["defect_probability"], label
    )

    # Target drift (rate comparison)
    period_rate = float(preds["actual_defect_flag"].mean())
    rate_delta  = abs(period_rate - train_rate)
    target_drift = rate_delta > TARGET_RATE_DELTA

    # Performance (from scoring's accuracy summary)
    high = acc[acc["risk_tier"] == "High"].iloc[0]
    med  = acc[acc["risk_tier"] == "Medium"].iloc[0]
    high_prec, high_rec = float(high["precision"]), float(high["recall"])
    med_prec,  med_rec  = float(med["precision"]),  float(med["recall"])

    # Decision
    status, retrain, reasons = _decide(high_prec, rate_delta,
                                       pdrift["prediction_drift_detected"], n_drifted)
    print(f"[{label}] performance: High P {high_prec:.1%} / R {high_rec:.1%}  |  "
          f"target Δ {rate_delta:.1%}  |  status {status}")

    return {
        "period_label":              label,
        "period_name":               name,
        "n_scored":                  len(preds),
        "high_flagged":              int(high["jobs_flagged"]),
        "n_features_monitored":      n_features,
        "n_features_drifted":        n_drifted,
        "prediction_drift_score":    pdrift["prediction_drift_score"],
        "prediction_drift_threshold":pdrift["prediction_drift_threshold"],
        "prediction_drift_detected": pdrift["prediction_drift_detected"],
        "ref_pred_mean":             pdrift["ref_pred_mean"],
        "cur_pred_mean":             pdrift["cur_pred_mean"],
        "train_defect_rate":         round(train_rate, 4),
        "period_defect_rate":        round(period_rate, 4),
        "target_rate_delta":         round(rate_delta, 4),
        "target_drift_detected":     bool(target_drift),
        "high_precision":            round(high_prec, 4),
        "high_recall":               round(high_rec, 4),
        "medium_precision":          round(med_prec, 4),
        "medium_recall":             round(med_rec, 4),
        "status":                    status,
        "retrain_recommended":       bool(retrain),
        "reasons":                   " | ".join(reasons) if reasons else "",
        "_html_path":                html_path,
        "_summary_path":             summary_path,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FLOW
# ══════════════════════════════════════════════════════════════════════════════

@flow(name="defect_risk_monitoring", log_prints=True)
def monitoring_flow(periods: list = PERIODS) -> None:
    """
    Longitudinal monitoring across the configured periods. Each month is
    compared against the training reference; results are consolidated into
    period_monitoring.csv for the report.
    """
    print(f"Monitoring {len(periods)} period(s): {', '.join(p[0] for p in periods)}")

    reference  = load_reference_features()
    val_ref    = load_validation_reference()
    train_rate = float(reference[TARGET].mean())

    rows = []
    for label, start, end, name in periods:
        row = _monitor_period(reference, val_ref, train_rate, label, start, end, name)
        log_period_to_mlflow(row, row["_html_path"], row["_summary_path"])
        rows.append(row)

    # Consolidated longitudinal table (drop internal path fields)
    out = pd.DataFrame(rows).drop(columns=["_html_path", "_summary_path"])
    out_path = MONITORING_DIR / "period_monitoring.csv"
    out.to_csv(out_path, index=False)

    print("\n════════ Monitoring complete ════════")
    print(out[["period_label", "status", "high_precision", "target_rate_delta",
               "prediction_drift_score", "n_features_drifted"]].to_string(index=False))
    print(f"\nConsolidated table: {out_path}")
    print(f"Latest-period status: {rows[-1]['status']}")


if __name__ == "__main__":
    monitoring_flow()