import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import shap
import optuna
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve as sk_learning_curve
from xgboost import XGBClassifier

import duckdb
import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import (
    engineer_features,
    ALL_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    INTERACTION_FEATURES, TARGET, ID_COL,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION 
# ══════════════════════════════════════════════════════════════════════════════

FEATURES_DIR     = Path("data/features").resolve()
DB_PATH          = Path("../data_source/defects_scrap.duckdb").resolve()

# Split boundaries — must match ml_prep.ipynb documentation
TRAIN_END        = "2024-12-31"   # Train: Jan 2023 – Dec 2024
VAL_END          = "2025-06-30"   # Val:   Jan 2025 – Jun 2025
                                  # Test:  Jul 2025 – Dec 2025
MLFLOW_TRACKING  = "sqlite:///mlruns/mlflow.db"          # or "mlruns" for local
EXPERIMENT_NAME  = "c01_defect_risk_scorer"
MODEL_NAME       = "defect_risk_scorer"
N_TRIALS         = 150          # Optuna trials per model
RANDOM_SEED      = 42

# Risk tier thresholds applied to predicted probability at scoring time.
# Populated here as placeholders — revisit after reviewing calibration curve.
RISK_THRESHOLDS  = {"High": 0.75, "Medium": 0.65}

# Palette — consistent with BRIM report styling
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


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_prob, threshold: float = 0.5, prefix: str = "") -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        f"{prefix}roc_auc":         roc_auc_score(y_true, y_prob),
        f"{prefix}avg_precision":   average_precision_score(y_true, y_prob),
        f"{prefix}f1":              f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}precision":       precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}recall":          recall_score(y_true, y_pred, zero_division=0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ARTIFACT GENERATION
# Produces files saved locally then logged to MLflow as artifacts.
# Part 4 (generate_ml_overview.py) retrieves these by artifact name.
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER SEARCH
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE MODEL TRAINING RUN
# Trains one model, logs to MLflow child run, returns metrics + pipeline.
# ══════════════════════════════════════════════════════════════════════════════

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
        mlflow.sklearn.log_model(pipeline, name=f"{model_type}_pipeline")

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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

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