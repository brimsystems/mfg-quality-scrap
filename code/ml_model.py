"""Defect-risk model: feature engineering, model training and selection, batch scoring, and monitoring. Reads the modeled marts and writes the registered model and scored outputs."""

import duckdb
import logging
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import shap
import sys
import warnings
from datetime import datetime
from evidently.future.datasets import PandasDataset, DataDefinition
from evidently.future.presets import DataDriftPreset, DataSummaryPreset
from evidently.future.report import Report
from mlflow import MlflowClient
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import learning_curve as sk_learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


# ==========================================================================
# Feature engineering
# ==========================================================================

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


# ==========================================================================
# Model training and selection
# ==========================================================================

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
FEATURES_DIR     = Path("data/features").resolve()
DB_PATH          = Path("../data_source/defects_scrap.duckdb").resolve()
# Split boundaries — must match ml_prep.ipynb documentation
TRAIN_END        = "2024-12-31"   # Train: Jan 2023 – Dec 2024
VAL_END          = "2025-06-30"   # Val:   Jan 2025 – Jun 2025
                                  # Test:  Jul 2025 – Dec 2025
MLFLOW_TRACKING  = "sqlite:///mlruns/mlflow.db"          # or "mlruns" for local
EXPERIMENT_NAME  = "defect_risk_scorer"
MODEL_NAME       = "defect_risk_scorer"
N_TRIALS         = 150          # Optuna trials per model
RANDOM_SEED      = 42
# Risk tier thresholds applied to predicted probability at scoring time.
# Populated here as placeholders — revisit after reviewing calibration curve.
RISK_THRESHOLDS  = {"High": 0.75, "Medium": 0.65}
# Palette — chart palette
BRAND_BLUE = "#3D5166"
ACCENT     = "#6B8FA8"
AMBER      = "#D4881E"
RED        = "#CC0000"
GREY       = "#AAAAAA"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor":   "#DDDDDD", "font.family": "sans-serif",
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "figure.dpi": 130,
})
def chart_style(ax):
    ax.yaxis.grid(True, color="#EEEEEE", linestyle="-", linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD")
    ax.spines["bottom"].set_color("#DDDDDD")
# DATA LOADING
def prepare_features() -> None:
    """
    Read from mart, apply feature engineering, split by date, export Parquet.
    Called once before training. Parquet files are the training pipeline's
    input — regenerate whenever the mart data or features.py changes.
    """
    log.info("Preparing feature splits from mart...")
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    df  = con.execute("SELECT * FROM mart_quality__defect_rates").df()
    con.close()

    df["actual_start"] = pd.to_datetime(df["actual_start"])
    log.info(f"  Loaded {len(df):,} work orders from mart")

    # Apply feature engineering — single source of truth in features.py
    df = engineer_features(df)

    # Time-based split — no shuffling, mirrors real deployment scenario
    train_mask = df["actual_start"] <= TRAIN_END
    val_mask   = (df["actual_start"] > TRAIN_END) & (df["actual_start"] <= VAL_END)
    test_mask  = df["actual_start"] > VAL_END

    export_cols = [ID_COL] + ALL_FEATURES + [TARGET]
    export_cols = [c for c in export_cols if c in df.columns]

    for label, mask in [("train", train_mask), ("validation", val_mask), ("test", test_mask)]:
        split = df[mask][export_cols]
        path  = FEATURES_DIR / f"{label}.parquet"
        split.to_parquet(path, index=False)
        log.info(f"  {label:<12} {len(split):>6,} rows  "
                 f"defect_flag rate: {split[TARGET].mean():.1%}  → {path.name}")

    log.info("Feature export complete.")
def load_splits() -> tuple:
    """
    Load train/validation/test Parquet files from FEATURES_DIR.
    Call prepare_features() first if Parquet files don't exist or are stale.
    """
    for split in ["train", "validation", "test"]:
        path = FEATURES_DIR / f"{split}.parquet"
        if not path.exists():
            log.info(f"  {split}.parquet not found — running prepare_features()...")
            prepare_features()
            break

    train = pd.read_parquet(FEATURES_DIR / "train.parquet")
    val   = pd.read_parquet(FEATURES_DIR / "validation.parquet")
    test  = pd.read_parquet(FEATURES_DIR / "test.parquet")

    assert TARGET in train.columns, f"Target column '{TARGET}' not found in features."

    feature_cols = [c for c in train.columns if c not in [ID_COL, TARGET]]

    X_train, y_train = train[feature_cols], train[TARGET].astype(int)
    X_val,   y_val   = val[feature_cols],   val[TARGET].astype(int)
    X_test,  y_test  = test[feature_cols],  test[TARGET].astype(int)

    log.info(f"Train:      {len(X_train):,} rows  |  defect_flag rate: {y_train.mean():.1%}")
    log.info(f"Validation: {len(X_val):,} rows  |  defect_flag rate: {y_val.mean():.1%}")
    log.info(f"Test:       {len(X_test):,} rows  |  defect_flag rate: {y_test.mean():.1%}")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
def export_validation_reference(pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> None:
    """
    Score the validation split with the selected production model and persist
    the predicted-probability distribution as a frozen reference.

    This is the baseline for prediction drift in monitoring: each scoring
    period's predicted-probability distribution is compared against the model's
    behaviour on held-out validation data. Validation (not the training set) is
    used as the reference so the baseline reflects generalisation rather than
    fit, which is the more honest comparison point.
    """
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    ref = pd.DataFrame({
        "defect_probability": np.round(val_probs, 4),
        TARGET:               y_val.astype(int).values,
    })
    path = FEATURES_DIR / "validation_predictions.parquet"
    ref.to_parquet(path, index=False)
    log.info(f"Validation reference (prediction-drift baseline): "
             f"{len(ref):,} rows  →  {path.name}")
    log.info(f"  mean predicted probability: {val_probs.mean():.4f}  |  "
             f"defect_flag rate: {y_val.mean():.1%}")
# PREPROCESSING
def build_preprocessors(X: pd.DataFrame) -> tuple:
    """
    Returns two preprocessors:
      - tree_prep:   OrdinalEncoder + passthrough numericals (XGB, RF)
      - linear_prep: OneHotEncoder + StandardScaler (Logistic Regression)
    """
    cat_cols = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    tree_prep = ColumnTransformer([
        ("cat", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ), cat_cols),
        ("num", "passthrough", num_cols),
    ], remainder="drop")

    linear_prep = ColumnTransformer([
        ("cat", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ), cat_cols),
        ("num", StandardScaler(), num_cols),
    ], remainder="drop")

    return tree_prep, linear_prep, cat_cols, num_cols
# METRICS
def compute_metrics(y_true, y_prob, threshold: float = 0.5, prefix: str = "") -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        f"{prefix}roc_auc":         roc_auc_score(y_true, y_prob),
        f"{prefix}avg_precision":   average_precision_score(y_true, y_prob),
        f"{prefix}f1":              f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}precision":       precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}recall":          recall_score(y_true, y_pred, zero_division=0),
    }
# ARTIFACT GENERATION
# Produces files saved locally then logged to MLflow as artifacts.
# Part 4 (generate_ml_overview.py) retrieves these by artifact name.
def save_confusion_matrix(y_true, y_prob, threshold: float,
                          path: Path, title: str = "Confusion Matrix") -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nNo Defect", "Predicted\nDefect"])
    ax.set_yticklabels(["Actual\nNo Defect", "Actual\nDefect"])
    labels = [[f"TN\n{tn:,}", f"FP\n{fp:,}"],
              [f"FN\n{fn:,}", f"TP\n{tp:,}"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j], ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
def save_pr_curve(y_true, y_prob, path: Path) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pd.DataFrame({
        "threshold": np.append(thresholds, 1.0),
        "precision": precision,
        "recall":    recall,
    }).to_csv(path, index=False)
def save_roc_curve(y_true, y_prob, path: Path) -> None:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    pd.DataFrame({
        "fpr":       fpr,
        "tpr":       tpr,
        "threshold": thresholds,
    }).to_csv(path, index=False)
def save_calibration(y_true, y_prob, path: Path, n_bins: int = 10) -> None:
    fraction_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    pd.DataFrame({
        "mean_predicted_probability": mean_pred,
        "fraction_positive":          fraction_pos,
    }).to_csv(path, index=False)
def save_learning_curve(pipeline, X_train, y_train, path: Path) -> None:
    from sklearn.base import clone

    # XGBoost with early_stopping_rounds requires eval_set at fit time,
    # which sklearn's learning_curve cannot provide. Remove it for this call.
    lc_pipeline = clone(pipeline)
    model_step  = lc_pipeline.named_steps["model"]
    if isinstance(model_step, XGBClassifier):
        model_step.set_params(early_stopping_rounds=None, n_estimators=200)

    train_sizes, train_scores, val_scores = sk_learning_curve(
        lc_pipeline, X_train, y_train,
        cv=3, scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1, random_state=RANDOM_SEED
    )
    pd.DataFrame({
        "train_size":        train_sizes,
        "train_roc_auc":     train_scores.mean(axis=1),
        "train_roc_auc_std": train_scores.std(axis=1),
        "val_roc_auc":       val_scores.mean(axis=1),
        "val_roc_auc_std":   val_scores.std(axis=1),
    }).to_csv(path, index=False)
def save_shap(pipeline, X_val: pd.DataFrame,
              importance_path: Path, summary_path: Path) -> None:
    """
    Compute SHAP values on validation set.
    Exports mean absolute SHAP importance as CSV and beeswarm plot as PNG.
    """
    model_step = pipeline.named_steps["model"]
    prep_step  = pipeline.named_steps["prep"]
    X_transformed = prep_step.transform(X_val)

    # Get transformed feature names — use column count as ground truth
    n_cols = X_transformed.shape[1]
    try:
        feature_names = prep_step.get_feature_names_out()
        # Ensure length matches — OrdinalEncoder keeps same column count
        if len(feature_names) != n_cols:
            feature_names = [f"f{i}" for i in range(n_cols)]
    except Exception:
        feature_names = [f"f{i}" for i in range(n_cols)]

    # Strip sklearn prefixes (e.g. "cat__supplier" → "supplier")
    feature_names = [
        n.split("__")[-1] if "__" in n else n
        for n in feature_names
    ]
    
    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    if isinstance(model_step, (XGBClassifier, RandomForestClassifier)):
        explainer   = shap.TreeExplainer(model_step)
        shap_values = explainer.shap_values(X_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]   # positive class for binary
        elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]  # positive class
    else:
        # Logistic Regression
        explainer   = shap.LinearExplainer(model_step, X_df)
        shap_values = explainer.shap_values(X_df)

    # Mean absolute importance
    importance = pd.DataFrame({
        "feature":    feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance.to_csv(importance_path, index=False)

    # Beeswarm plot
    shap.summary_plot(shap_values, X_df, show=False, plot_size=(10, 6))
    plt.tight_layout()
    plt.savefig(summary_path, bbox_inches="tight", dpi=130)
    plt.close()
# HYPERPARAMETER SEARCH
def tune_logistic_regression(X_train, y_train, X_val, y_val,
                              linear_prep) -> tuple:
    def objective(trial):
        C       = trial.suggest_float("C", 1e-3, 100.0, log=True)
        pipeline = Pipeline([
            ("prep",  linear_prep),
            ("model", LogisticRegression(
                C=C, max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1
            ))
        ])
        pipeline.fit(X_train, y_train)
        return roc_auc_score(y_val, pipeline.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best  = study.best_params
    log.info(f"  LR best params: {best}  val_auc={study.best_value:.4f}")
    pipeline = Pipeline([
        ("prep",  linear_prep),
        ("model", LogisticRegression(
            C=best["C"], max_iter=1000,
            random_state=RANDOM_SEED, n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline, best, study.best_value
def tune_random_forest(X_train, y_train, X_val, y_val,
                       tree_prep) -> tuple:
    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 500),
            "max_depth":       trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf":trial.suggest_int("min_samples_leaf", 1, 50),
            "max_features":    trial.suggest_categorical(
                                   "max_features", ["sqrt", "log2", 0.5]
                               ),
        }
        pipeline = Pipeline([
            ("prep",  tree_prep),
            ("model", RandomForestClassifier(
                **params, random_state=RANDOM_SEED, n_jobs=-1
            ))
        ])
        pipeline.fit(X_train, y_train)
        return roc_auc_score(y_val, pipeline.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = study.best_params
    log.info(f"  RF best params: {best}  val_auc={study.best_value:.4f}")
    pipeline = Pipeline([
        ("prep",  tree_prep),
        ("model", RandomForestClassifier(
            **best, random_state=RANDOM_SEED, n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline, best, study.best_value
def tune_xgboost(X_train, y_train, X_val, y_val,
                 tree_prep) -> tuple:
    # Preprocess once so XGBoost early stopping can use eval_set
    prep_fit   = tree_prep.__class__(
        transformers=tree_prep.transformers,
        remainder=tree_prep.remainder
    )
    X_tr_t = prep_fit.fit_transform(X_train)
    X_va_t = prep_fit.transform(X_val)

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    def objective(trial):
        params = {
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "n_estimators":     1000,   # early stopping controls actual count
        }
        model = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED,
            eval_metric="auc",
            early_stopping_rounds=30,
            verbosity=0,
        )
        model.fit(
            X_tr_t, y_train,
            eval_set=[(X_va_t, y_val)],
            verbose=False,
        )
        return roc_auc_score(y_val, model.predict_proba(X_va_t)[:, 1])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = study.best_params
    log.info(f"  XGB best params: {best}  val_auc={study.best_value:.4f}")

    # Refit full pipeline with best params
    best_pipeline = Pipeline([
        ("prep",  tree_prep),
        ("model", XGBClassifier(
            **best,
            n_estimators=1000,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED,
            eval_metric="auc",
            early_stopping_rounds=30,
            verbosity=0,
        ))
    ])
    # XGBoost early stopping needs eval_set — fit model step separately
    best_pipeline.named_steps["prep"].fit(X_train)
    X_tr_final = best_pipeline.named_steps["prep"].transform(X_train)
    X_va_final = best_pipeline.named_steps["prep"].transform(X_val)
    best_pipeline.named_steps["model"].fit(
        X_tr_final, y_train,
        eval_set=[(X_va_final, y_val)],
        verbose=False,
    )
    return best_pipeline, best, study.best_value
# SINGLE MODEL TRAINING RUN
# Trains one model, logs to MLflow child run, returns metrics + pipeline.
def train_and_log_model(
    model_type:  str,
    pipeline,
    best_params: dict,
    X_train, y_train,
    X_val,   y_val,
    parent_run_id: str,
    tmp_dir: Path,
) -> dict:
    with mlflow.start_run(run_name=model_type, nested=True) as child_run:

        # ── Log hyperparameters ────────────────────────────────────────────
        mlflow.log_params(best_params)
        mlflow.set_tag("model_type", model_type)

        # ── Validation metrics ─────────────────────────────────────────────
        y_prob_val = pipeline.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, y_prob_val, prefix="val_")
        mlflow.log_metrics(val_metrics)
        log.info(f"  {model_type} val_roc_auc={val_metrics['val_roc_auc']:.4f}")

        # ── Artifacts ─────────────────────────────────────────────────────
        model_dir = tmp_dir / model_type
        model_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix
        cm_path = model_dir / "confusion_matrix.png"
        save_confusion_matrix(
            y_val, y_prob_val,
            threshold=RISK_THRESHOLDS["High"],
            path=cm_path,
            title=f"Confusion Matrix — {model_type} (Validation)"
        )

        # Precision-recall curve
        pr_path = model_dir / "pr_curve.csv"
        save_pr_curve(y_val, y_prob_val, pr_path)

        # ROC curve
        roc_path = model_dir / "roc_curve.csv"
        save_roc_curve(y_val, y_prob_val, roc_path)

        # Calibration
        cal_path = model_dir / "calibration.csv"
        save_calibration(y_val, y_prob_val, cal_path)

        # Learning curve (slow for large datasets — uses CV on train split)
        lc_path = model_dir / "learning_curve.csv"
        save_learning_curve(pipeline, X_train, y_train, lc_path)

        # SHAP
        shap_imp_path     = model_dir / "shap_importance.csv"
        shap_summary_path = model_dir / "shap_summary.png"
        save_shap(pipeline, X_val, shap_imp_path, shap_summary_path)

        # Log all artifacts
        mlflow.log_artifacts(str(model_dir), artifact_path=model_type)

        # Log pipeline as MLflow model artifact (not registered yet)
        mlflow.sklearn.log_model(pipeline, name=f"{model_type}_pipeline",
                                 serialization_format="cloudpickle")

        return {
            "model_type":    model_type,
            "pipeline":      pipeline,
            "best_params":   best_params,
            "val_roc_auc":   val_metrics["val_roc_auc"],
            "val_f1":        val_metrics["val_f1"],
            "val_precision": val_metrics["val_precision"],
            "val_recall":    val_metrics["val_recall"],
            "child_run_id":  child_run.info.run_id,
        }
# MAIN
def main():
    import tempfile

    # ── MLflow setup ───────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── Prepare and load feature splits ───────────────────────────────────
    log.info("Preparing features from mart...")
    prepare_features()
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_splits()

    cat_cols = X_train.select_dtypes(include=["object","bool","category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

    # ── Build preprocessors ────────────────────────────────────────────────
    tree_prep, linear_prep, _, _ = build_preprocessors(X_train)

    # ── Parent MLflow run ──────────────────────────────────────────────────
    from datetime import datetime
    run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M')}"

    with mlflow.start_run(run_name=run_name) as parent_run:
        parent_run_id = parent_run.info.run_id

        # Log dataset metadata as tags
        mlflow.set_tags({
            "target":           TARGET,
            "train_rows":       len(X_train),
            "val_rows":         len(X_val),
            "test_rows":        len(X_test),
            "feature_count":    len(feature_cols),
            "cat_features":     str(cat_cols),
            "num_features":     str(num_cols),
            "class_balance":    f"{y_train.mean():.3f}",
            "n_optuna_trials":  N_TRIALS,
            "random_seed":      RANDOM_SEED,
        })

        results = []

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            # ── Logistic Regression ────────────────────────────────────────
            log.info("Training Logistic Regression...")
            lr_pipeline, lr_params, _ = tune_logistic_regression(
                X_train, y_train, X_val, y_val, linear_prep
            )
            results.append(train_and_log_model(
                "logistic_regression", lr_pipeline, lr_params,
                X_train, y_train, X_val, y_val,
                parent_run_id, tmp_dir
            ))

            # ── Random Forest ──────────────────────────────────────────────
            log.info("Training Random Forest...")
            rf_pipeline, rf_params, _ = tune_random_forest(
                X_train, y_train, X_val, y_val, tree_prep
            )
            results.append(train_and_log_model(
                "random_forest", rf_pipeline, rf_params,
                X_train, y_train, X_val, y_val,
                parent_run_id, tmp_dir
            ))

            # ── XGBoost ────────────────────────────────────────────────────
            log.info("Training XGBoost...")
            xgb_pipeline, xgb_params, _ = tune_xgboost(
                X_train, y_train, X_val, y_val, tree_prep
            )
            results.append(train_and_log_model(
                "xgboost", xgb_pipeline, xgb_params,
                X_train, y_train, X_val, y_val,
                parent_run_id, tmp_dir
            ))

            # ── Model comparison table ─────────────────────────────────────
            comparison_df = pd.DataFrame([{
                "model_type":    r["model_type"],
                "val_roc_auc":   round(r["val_roc_auc"], 4),
                "val_f1":        round(r["val_f1"], 4),
                "val_precision": round(r["val_precision"], 4),
                "val_recall":    round(r["val_recall"], 4),
            } for r in results]).sort_values("val_roc_auc", ascending=False)

            comp_path = tmp_dir / "model_comparison.csv"
            comparison_df.to_csv(comp_path, index=False)
            mlflow.log_artifact(str(comp_path))

            log.info("\nModel comparison (validation set):")
            log.info(comparison_df.to_string(index=False))

            # ── Select best model ──────────────────────────────────────────
            best_result = max(results, key=lambda r: r["val_roc_auc"])
            best_pipeline = best_result["pipeline"]
            best_model_type = best_result["model_type"]

            mlflow.set_tag("best_model_type", best_model_type)
            mlflow.set_tag("best_val_roc_auc", f"{best_result['val_roc_auc']:.4f}")
            log.info(f"\nBest model: {best_model_type} "
                     f"(val_roc_auc={best_result['val_roc_auc']:.4f})")

            # ── Persist validation-set predictions as the prediction-drift
            #    reference for monitoring (frozen at training time) ───────────
            export_validation_reference(best_pipeline, X_val, y_val)

            # ── Final evaluation on test set (touched once) ────────────────
            log.info("Evaluating best model on test set...")
            y_prob_test  = best_pipeline.predict_proba(X_test)[:, 1]
            test_metrics = compute_metrics(y_test, y_prob_test, prefix="test_")
            mlflow.log_metrics(test_metrics)

            log.info(f"  test_roc_auc={test_metrics['test_roc_auc']:.4f}  "
                     f"test_f1={test_metrics['test_f1']:.4f}  "
                     f"test_precision={test_metrics['test_precision']:.4f}  "
                     f"test_recall={test_metrics['test_recall']:.4f}")

            # Test confusion matrix
            test_cm_path = tmp_dir / "test_confusion_matrix.png"
            save_confusion_matrix(
                y_test, y_prob_test,
                threshold=RISK_THRESHOLDS["High"],
                path=test_cm_path,
                title=f"Confusion Matrix — {best_model_type} (Test Set)"
            )
            mlflow.log_artifact(str(test_cm_path))

            # ── Register best model ────────────────────────────────────────
            log.info(f"Registering {best_model_type} to Model Registry...")
            model_uri = f"runs:/{best_result['child_run_id']}/{best_model_type}_pipeline"
            mv = mlflow.register_model(model_uri, MODEL_NAME)

            client = MlflowClient()
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=mv.version,
                stage="Production",
                archive_existing_versions=True,
            )
            client.update_model_version(
                name=MODEL_NAME,
                version=mv.version,
                description=(
                    f"Best model from run {parent_run_id}. "
                    f"Type: {best_model_type}. "
                    f"Val ROC-AUC: {best_result['val_roc_auc']:.4f}. "
                    f"Test ROC-AUC: {test_metrics['test_roc_auc']:.4f}."
                )
            )
            mlflow.set_tag("model_version", mv.version)
            log.info(f"Registered as '{MODEL_NAME}' version {mv.version} → Production")

    log.info("Training complete.")
    log.info(f"MLflow run ID: {parent_run_id}")
    log.info(f"View at: {MLFLOW_TRACKING}/#/experiments")
if __name__ == "__main__":
    main()


# ==========================================================================
# Batch scoring
# ==========================================================================

try:
    from prefect import flow, task
except ModuleNotFoundError:
    # Prefect is the optional orchestration layer. When it is not installed, fall
    # back to no-op decorators so the flow runs as a plain local batch.
    def task(*args, **kwargs):
        def _wrap(fn):
            return fn
        return args[0] if len(args) == 1 and callable(args[0]) and not kwargs else _wrap
    flow = task
SCORING_DIR     = Path("data/scoring")
MODEL_STAGE     = "Production"
N_SHAP_DRIVERS  = 3     # top N SHAP features to include per work order
# Monitoring periods — one calendar month each. Scored and monitored
# independently so prediction/performance drift can be tracked month over month.
PERIODS = [
    ("2026-01-01", "2026-01-31"),
    ("2026-02-01", "2026-02-28"),
    ("2026-03-01", "2026-03-31"),
]
# Risk tier thresholds applied to predicted probability
SCORING_DIR.mkdir(parents=True, exist_ok=True)
# TASKS
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

    if len(df) == 0:
        raise RuntimeError(
            f"No work orders found between {start} and {end}. "
            "Verify data generation covered this period."
        )

    # Derive interaction features — mirrors ml_prep engineer_features()
    df = engineer_features(df)

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
def build_accuracy_summary(df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    """
    Compute precision, recall, and counts by risk tier using actual outcomes.
    Precision: of jobs flagged at this tier, what fraction actually failed?
    Recall: of all actual failures, what fraction were flagged at this tier or above?

    period_label is carried on every row so the three monthly summaries
    concatenate cleanly into a longitudinal performance view.
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
            "period_label":    period_label,
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
    log.info("Accuracy summary:")
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
    # Predictions — keep actual_defect_flag as the ground truth column
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

    log.info(f"Predictions written to:      {pred_path}")
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
    start_date: str,
    end_date: str,
) -> str:
    """
    Log the scoring run as a new MLflow run under the same experiment.
    Tags carry the actual period dates so each monthly run is self-describing.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"scoring_{period_label}") as run:
        # Tags — use the dates actually scored, not module-level defaults
        mlflow.set_tags({
            "run_type":        "scoring",
            "scoring_start":   start_date,
            "scoring_end":     end_date,
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
# FLOW
def _run_period(
    pipeline,
    model_version,
    source_run_id,
    start_date: str,
    end_date: str,
) -> None:
    """
    Score a single period end to end.

    Plain helper (not a flow) so it can be driven by either the single-period
    or all-periods flow without nesting flow runs or passing the model across
    a flow-run boundary. The @task calls below execute within whichever flow
    invokes this helper.
    """
    period_label = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m")
    log.info(f"Scoring period {period_label}: {start_date} → {end_date}")

    df      = load_scoring_data(start_date, end_date)
    df      = score_work_orders(pipeline, df)
    df      = compute_shap_drivers(pipeline, df)
    summary = build_accuracy_summary(df, period_label)

    pred_path, summary_path = write_outputs(df, summary, period_label)

    log_scoring_run_to_mlflow(
        df, summary, pred_path, summary_path,
        model_version, source_run_id, period_label,
        start_date, end_date,
    )

    log.info(f"  Period {period_label} complete — {len(df):,} scored, "
             f"{df['risk_tier'].eq('High').sum():,} High-risk")
@flow(name="defect_risk_scoring", log_prints=True)
def scoring_flow(start_date: str, end_date: str) -> None:
    """
    Batch scoring flow for a single period (standalone entry point).
    Loads the Production model and scores [start_date, end_date].
    """
    log.info(f"Starting single-period scoring: {start_date} → {end_date}")
    pipeline, model_version, source_run_id = load_model()
    log.info(f"Model: {MODEL_NAME} ({MODEL_STAGE})")

    _run_period(pipeline, model_version, source_run_id, start_date, end_date)

    log.info("Scoring flow complete.")
@flow(name="defect_risk_scoring_all_periods", log_prints=True)
def score_all_periods(periods: list = PERIODS) -> None:
    """
    Score every configured monitoring period in sequence, loading the
    Production model once and reusing it across months. No nested flows —
    each period runs through the shared _run_period helper within this flow.
    """
    log.info(f"Scoring {len(periods)} period(s): "
             f"{', '.join(s for s, _ in periods)}")

    pipeline, model_version, source_run_id = load_model()

    for start_date, end_date in periods:
        _run_period(pipeline, model_version, source_run_id, start_date, end_date)

    log.info("All scoring periods complete.")
if __name__ == "__main__":
    score_all_periods()


# ==========================================================================
# Monitoring and drift
# ==========================================================================

MONITORING_DIR  = Path("data/monitoring")
# Monitoring periods — (label, start, end, display name). One calendar month each.
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
# DRIFT EXTRACTION (shared by feature and prediction drift)
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
# REFERENCE DATA (loaded once, reused across periods)
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
# PER-PERIOD INPUTS
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
# DRIFT COMPUTATION
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
# PER-PERIOD ORCHESTRATION (plain helper — no nested flows)
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
