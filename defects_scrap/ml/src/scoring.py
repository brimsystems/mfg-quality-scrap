"""
scoring.py
Pre-production defect risk scorer — batch scoring pipeline.
Loads the registered production model from MLflow, scores all work orders
in the specified backfill period, computes per-job SHAP drivers, joins
actual outcomes, and writes prediction outputs for Part 4 reporting.

Run from the ml/ directory:
    python3 src/scoring.py

Outputs written to ml/data/scoring/:
    predictions_YYYYMM.parquet   — scored work orders with SHAP drivers
    accuracy_summary_YYYYMM.csv  — precision/recall by risk tier
"""

import logging
import warnings
from pathlib import Path
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
import shap
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from prefect import flow, task

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import (
    engineer_features, ALL_FEATURES, CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES, INTERACTION_FEATURES, TARGET, ID_COL
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DB_PATH         = Path("../data/defects_scrap.duckdb").resolve()
FEATURES_DIR    = Path("data/features").resolve()
SCORING_DIR     = Path("data/scoring")
MLFLOW_TRACKING = "mlruns"
EXPERIMENT_NAME = "c01_defect_risk_scorer"
MODEL_NAME      = "defect_risk_scorer"
MODEL_STAGE     = "Production"
N_SHAP_DRIVERS  = 3     # top N SHAP features to include per work order

# Backfill period — Jan–Mar 2026
SCORE_START = "2026-01-01"
SCORE_END   = "2026-03-31"

# Risk tier thresholds applied to predicted probability
RISK_THRESHOLDS = {"High": 0.75, "Medium": 0.65}

SCORING_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# TASKS
# ══════════════════════════════════════════════════════════════════════════════

@task(name="load_model", retries=2, retry_delay_seconds=10)
def load_model() -> tuple:
    """
    Load the Production model from MLflow Model Registry.
    Returns (pipeline, model_version, run_id).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    client = MlflowClient()

    versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    if not versions:
        raise RuntimeError(
            f"No model found in registry: '{MODEL_NAME}' stage='{MODEL_STAGE}'. "
            "Run training.py first."
        )
    mv          = versions[0]
    model_uri   = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    pipeline    = mlflow.sklearn.load_model(model_uri)

    log.info(f"Loaded '{MODEL_NAME}' version {mv.version} from {MODEL_STAGE}")
    log.info(f"  Source run: {mv.run_id}")
    return pipeline, mv.version, mv.run_id


@task(name="load_scoring_data")
def load_scoring_data(start: str, end: str) -> pd.DataFrame:
    """
    Load work orders from the mart for the scoring period.
    Includes ALL_FEATURES + TARGET (for actuals join) + ID_COL.
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    BASE_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    query = f"""
        SELECT
            {ID_COL},
            actual_start,
            {', '.join(BASE_FEATURES)},
            {TARGET}
        FROM mart_quality__defect_rates
        WHERE actual_start >= '{start}'
        AND actual_start <= '{end}'
        ORDER BY actual_start
    """
    df = con.execute(query).df()
    con.close()

    # Derive interaction features — mirrors ml_prep engineer_features()
    df = engineer_features(df)

    if len(df) == 0:
        raise RuntimeError(
            f"No work orders found between {start} and {end}. "
            "Verify data generation covered this period."
        )

    log.info(f"Loaded {len(df):,} work orders for scoring period {start} → {end}")
    log.info(f"  defect_flag rate (actuals): {df[TARGET].mean():.1%}")
    return df


@task(name="score_work_orders")
def score_work_orders(pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the model pipeline to produce defect probability and risk tier
    for each work order. Returns df with probability and tier columns added.
    """
    X = df[ALL_FEATURES].copy()

    probs       = pipeline.predict_proba(X)[:, 1]
    risk_tiers  = pd.cut(
        probs,
        bins  = [0.0, RISK_THRESHOLDS["Medium"], RISK_THRESHOLDS["High"], 1.0],
        labels= ["Low", "Medium", "High"],
        include_lowest=True,
    )

    df = df.copy()
    df["defect_probability"] = probs.round(4)
    df["risk_tier"]          = risk_tiers.astype(str)

    counts = df["risk_tier"].value_counts()
    log.info("Risk tier distribution:")
    for tier in ["High", "Medium", "Low"]:
        n   = counts.get(tier, 0)
        pct = n / len(df) * 100
        log.info(f"  {tier:<8} {n:>5,}  ({pct:.1f}%)")

    return df


@task(name="compute_shap_drivers")
def compute_shap_drivers(pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SHAP values for each work order and extract the top N
    feature drivers with their direction and magnitude.
    Adds a 'shap_drivers' column: list of (feature, value, shap_contribution).
    """
    X           = df[ALL_FEATURES].copy()
    prep_step   = pipeline.named_steps["prep"]
    model_step  = pipeline.named_steps["model"]

    X_transformed = prep_step.transform(X)

    # Feature names after preprocessing
    try:
        feature_names = prep_step.get_feature_names_out()
        feature_names = [n.split("__")[-1] if "__" in n else n
                         for n in feature_names]
        if len(feature_names) != X_transformed.shape[1]:
            feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]
    except Exception:
        feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    # Compute SHAP values
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier as RFC
    from sklearn.linear_model import LogisticRegression as LR

    if isinstance(model_step, (XGBClassifier, RFC)):
        explainer   = shap.TreeExplainer(model_step)
        shap_values = explainer.shap_values(X_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    else:
        explainer   = shap.LinearExplainer(model_step, X_df)
        shap_values = explainer.shap_values(X_df)

    # Extract top N drivers per work order
    drivers_list = []
    for i in range(len(df)):
        row_shap  = shap_values[i]
        top_idx   = np.argsort(np.abs(row_shap))[::-1][:N_SHAP_DRIVERS]
        drivers   = []
        for idx in top_idx:
            feat_name   = feature_names[idx]
            shap_val    = float(row_shap[idx])
            feat_val    = X_df.iloc[i, idx]
            drivers.append({
                "feature":      feat_name,
                "value":        feat_val,
                "contribution": round(shap_val, 4),
                "direction":    "increases_risk" if shap_val > 0 else "decreases_risk",
            })
        drivers_list.append(drivers)

    df = df.copy()
    df["shap_drivers"] = drivers_list

    # Also extract flat columns for the top driver — useful for quick filtering
    df["top_driver_feature"] = [d[0]["feature"] for d in drivers_list]
    df["top_driver_direction"] = [d[0]["direction"] for d in drivers_list]

    log.info(f"SHAP drivers computed for {len(df):,} work orders")
    log.info(f"  Most common top driver: "
             f"{df['top_driver_feature'].value_counts().index[0]}")
    return df


@task(name="build_accuracy_summary")
def build_accuracy_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute precision, recall, and counts by risk tier using actual outcomes.
    Precision: of jobs flagged at this tier, what fraction actually failed?
    Recall: of all actual failures, what fraction were flagged at this tier or above?
    """
    rows        = []
    all_positive = df[TARGET].sum()

    for tier, threshold in [
        ("High",   RISK_THRESHOLDS["High"]),
        ("Medium", RISK_THRESHOLDS["Medium"]),
    ]:
        flagged       = df[df["defect_probability"] >= threshold]
        true_pos      = flagged[TARGET].sum()
        false_pos     = len(flagged) - true_pos
        precision     = true_pos / len(flagged) if len(flagged) > 0 else 0.0
        recall        = true_pos / all_positive if all_positive > 0 else 0.0

        rows.append({
            "risk_tier":       tier,
            "threshold":       threshold,
            "jobs_flagged":    len(flagged),
            "true_positives":  int(true_pos),
            "false_positives": int(false_pos),
            "precision":       round(precision, 4),
            "recall":          round(recall, 4),
            "pct_of_total":    round(len(flagged) / len(df), 4),
        })

    summary = pd.DataFrame(rows)
    log.info("\nAccuracy summary:")
    log.info(summary.to_string(index=False))
    return summary


@task(name="write_outputs")
def write_outputs(df: pd.DataFrame,
                  summary: pd.DataFrame,
                  period_label: str) -> tuple:
    """
    Write predictions and accuracy summary to ml/data/scoring/.
    Returns file paths for MLflow logging.
    """
    # Predictions — drop raw target to keep output clean,
    # but keep actual_defect_flag as the ground truth column
    output_cols = [
        ID_COL, "actual_start", "defect_probability", "risk_tier",
        "shap_drivers", "top_driver_feature", "top_driver_direction",
        TARGET,
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    pred_df     = df[output_cols].rename(columns={TARGET: "actual_defect_flag"})

    pred_path    = SCORING_DIR / f"predictions_{period_label}.parquet"
    summary_path = SCORING_DIR / f"accuracy_summary_{period_label}.csv"

    pred_df.to_parquet(pred_path, index=False)
    summary.to_csv(summary_path, index=False)

    log.info(f"Predictions written to:     {pred_path}")
    log.info(f"Accuracy summary written to: {summary_path}")
    return pred_path, summary_path


@task(name="log_scoring_run_to_mlflow")
def log_scoring_run_to_mlflow(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    pred_path: Path,
    summary_path: Path,
    model_version: str,
    source_run_id: str,
    period_label: str,
) -> str:
    """
    Log the scoring run as a new MLflow run under the same experiment.
    Logs metadata, accuracy metrics, and output files as artifacts.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"scoring_{period_label}") as run:
        # Tags
        mlflow.set_tags({
            "run_type":        "scoring",
            "scoring_start":   SCORE_START,
            "scoring_end":     SCORE_END,
            "model_version":   str(model_version),
            "source_run_id":   source_run_id,
            "rows_scored":     len(df),
            "period_label":    period_label,
        })

        # Metrics
        high_summary = summary[summary["risk_tier"] == "High"]
        med_summary  = summary[summary["risk_tier"] == "Medium"]

        mlflow.log_metrics({
            "jobs_scored":              len(df),
            "actual_defect_rate":       round(df[TARGET].mean(), 4),
            "high_risk_count":          int(df["risk_tier"].eq("High").sum()),
            "medium_risk_count":        int(df["risk_tier"].eq("Medium").sum()),
            "low_risk_count":           int(df["risk_tier"].eq("Low").sum()),
            "high_tier_precision":      float(high_summary["precision"].values[0])
                                        if len(high_summary) > 0 else 0.0,
            "high_tier_recall":         float(high_summary["recall"].values[0])
                                        if len(high_summary) > 0 else 0.0,
            "medium_tier_precision":    float(med_summary["precision"].values[0])
                                        if len(med_summary) > 0 else 0.0,
            "medium_tier_recall":       float(med_summary["recall"].values[0])
                                        if len(med_summary) > 0 else 0.0,
        })

        # Artifacts
        mlflow.log_artifact(str(pred_path),    artifact_path="scoring_outputs")
        mlflow.log_artifact(str(summary_path), artifact_path="scoring_outputs")

        run_id = run.info.run_id
        log.info(f"Scoring run logged to MLflow: {run_id}")
        return run_id


# ══════════════════════════════════════════════════════════════════════════════
# FLOW
# ══════════════════════════════════════════════════════════════════════════════

@flow(name="defect_risk_scoring", log_prints=True)
def scoring_flow(
    start_date: str = SCORE_START,
    end_date:   str = SCORE_END,
) -> None:
    """
    Batch scoring flow for the defect risk scorer.
    Scores all work orders in the specified date range, computes SHAP
    drivers, joins actuals, and writes outputs for client reporting.
    """
    period_label = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m")

    log.info(f"Starting scoring flow: {start_date} → {end_date}")
    log.info(f"Model: {MODEL_NAME} ({MODEL_STAGE})")

    pipeline, model_version, source_run_id = load_model()
    df      = load_scoring_data(start_date, end_date)
    df      = score_work_orders(pipeline, df)
    df      = compute_shap_drivers(pipeline, df)
    summary = build_accuracy_summary(df)

    pred_path, summary_path = write_outputs(df, summary, period_label)

    log_scoring_run_to_mlflow(
        df, summary, pred_path, summary_path,
        model_version, source_run_id, period_label
    )

    log.info("Scoring flow complete.")
    log.info(f"  Scored:     {len(df):,} work orders")
    log.info(f"  High risk:  {df['risk_tier'].eq('High').sum():,} jobs flagged")
    log.info(f"  Period:     {period_label}")


if __name__ == "__main__":
    scoring_flow()
