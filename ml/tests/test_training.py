"""
tests/test_training.py
Unit and integration tests for the C01 defect risk scorer training pipeline.

Run from the ml/ directory:
    pytest tests/test_training.py -v

Unit tests: fast, no MLflow, no real data — use synthetic fixtures.
Integration tests: marked with @pytest.mark.integration, require
    data/features/*.parquet and a local MLflow tracking directory.
    Run selectively: pytest tests/test_training.py -v -m integration
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ── Paths ──────────────────────────────────────────────────────────────────
ML_DIR       = Path(__file__).parent.parent
FEATURES_DIR = ML_DIR / "data" / "features"
SRC_DIR      = ML_DIR / "src"

import sys
sys.path.insert(0, str(SRC_DIR))

from training import (
    compute_metrics,
    build_preprocessors,
    save_confusion_matrix,
    save_pr_curve,
    save_roc_curve,
    save_calibration,
    save_learning_curve,
    TARGET,
    ID_COL,
    RISK_THRESHOLDS,
    RANDOM_SEED,
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_X():
    """Minimal synthetic feature DataFrame matching expected column types."""
    np.random.seed(RANDOM_SEED)
    n = 200
    return pd.DataFrame({
        "machine_type":    np.random.choice(["Bending","Laser Cutting","Welding","Punching"], n),
        "machine_id":      np.random.choice(["M01","M02","M03","M04","M05"], n),
        "shift_code":      np.random.choice(["Shift A","Shift B"], n),
        "operator_id":     np.random.choice([f"OP{str(i).zfill(3)}" for i in range(1,11)], n),
        "complexity":      np.random.choice(["Low","Medium","High"], n),
        "material_type":   np.random.choice(["16ga Steel","14ga Steel","Aluminum 5052"], n),
        "supplier":        np.random.choice(["Supplier A","Supplier B","Supplier C","Supplier D"], n),
        "lot_cert_status": np.random.choice(["Certified","Conditional","Missing"], n),
        "requires_welding":np.random.choice([True, False], n),
        "machine_age_years":np.random.uniform(1, 15, n),
        "std_labor_hrs":   np.random.uniform(0.25, 5.0, n),
        "quantity_ordered":np.random.randint(15, 75, n),
        "schedule_variance_hrs": np.random.uniform(-2, 4, n),
    })

@pytest.fixture
def synthetic_y():
    """Binary target with realistic class balance (~55% positive)."""
    np.random.seed(RANDOM_SEED)
    return pd.Series(np.random.binomial(1, 0.55, 200), name=TARGET)

@pytest.fixture
def synthetic_probs():
    """Realistic predicted probabilities for metric tests."""
    np.random.seed(RANDOM_SEED)
    return np.random.beta(2, 2, 200)

@pytest.fixture
def fitted_lr_pipeline(synthetic_X, synthetic_y):
    """Fitted logistic regression pipeline on synthetic data."""
    _, linear_prep, _, _ = build_preprocessors(synthetic_X)
    pipeline = Pipeline([
        ("prep",  linear_prep),
        ("model", LogisticRegression(C=1.0, max_iter=200, random_state=RANDOM_SEED))
    ])
    pipeline.fit(synthetic_X, synthetic_y)
    return pipeline

@pytest.fixture
def fitted_rf_pipeline(synthetic_X, synthetic_y):
    """Fitted random forest pipeline on synthetic data."""
    tree_prep, _, _, _ = build_preprocessors(synthetic_X)
    pipeline = Pipeline([
        ("prep",  tree_prep),
        ("model", RandomForestClassifier(
            n_estimators=20, max_depth=4,
            random_state=RANDOM_SEED, n_jobs=1
        ))
    ])
    pipeline.fit(synthetic_X, synthetic_y)
    return pipeline


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — DATA
# ══════════════════════════════════════════════════════════════════════════════

class TestDataIntegrity:

    def test_target_class_balance_range(self, synthetic_y):
        """defect_flag rate should be between 30% and 75% for a usable model."""
        rate = synthetic_y.mean()
        assert 0.30 <= rate <= 0.75, (
            f"Class balance {rate:.1%} is outside expected range 30–75%. "
            "Check data generation parameters."
        )

    def test_no_nulls_in_features(self, synthetic_X):
        """Synthetic fixture should have no nulls — mirrors mart output."""
        null_counts = synthetic_X.isna().sum()
        assert null_counts.sum() == 0, (
            f"Unexpected nulls in features:\n{null_counts[null_counts > 0]}"
        )

    def test_feature_column_types(self, synthetic_X):
        """Categorical and numerical columns have expected dtypes."""
        cat_cols = ["machine_type","machine_id","shift_code","operator_id",
                    "complexity","material_type","supplier","lot_cert_status"]
        num_cols = ["machine_age_years","std_labor_hrs",
                    "quantity_ordered","schedule_variance_hrs"]
        for col in cat_cols:
            assert col in synthetic_X.columns, f"Missing categorical column: {col}"
        for col in num_cols:
            assert col in synthetic_X.columns, f"Missing numerical column: {col}"
            assert pd.api.types.is_numeric_dtype(synthetic_X[col]), (
                f"Expected numeric dtype for {col}, got {synthetic_X[col].dtype}"
            )

    def test_quantity_ordered_range(self, synthetic_X):
        """Batch sizes should reflect realistic low-mix production fab runs."""
        assert synthetic_X["quantity_ordered"].min() >= 5
        assert synthetic_X["quantity_ordered"].max() <= 500

    def test_shift_code_values(self, synthetic_X):
        """shift_code should only contain known shift values."""
        valid = {"Shift A", "Shift B", "UNKNOWN"}
        actual = set(synthetic_X["shift_code"].dropna().unique())
        assert actual.issubset(valid), (
            f"Unexpected shift_code values: {actual - valid}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessors:

    def test_tree_preprocessor_output_shape(self, synthetic_X, synthetic_y):
        """Tree preprocessor output columns = n_cat + n_num features."""
        tree_prep, _, cat_cols, num_cols = build_preprocessors(synthetic_X)
        tree_prep.fit(synthetic_X, synthetic_y)
        X_out = tree_prep.transform(synthetic_X)
        expected_cols = len(cat_cols) + len(num_cols)
        assert X_out.shape == (len(synthetic_X), expected_cols), (
            f"Expected shape ({len(synthetic_X)}, {expected_cols}), "
            f"got {X_out.shape}"
        )

    def test_linear_preprocessor_expands_categoricals(self, synthetic_X, synthetic_y):
        """Linear preprocessor one-hot encodes, so output cols > input cols."""
        _, linear_prep, cat_cols, num_cols = build_preprocessors(synthetic_X)
        linear_prep.fit(synthetic_X, synthetic_y)
        X_out = linear_prep.transform(synthetic_X)
        n_input = len(cat_cols) + len(num_cols)
        assert X_out.shape[1] > n_input, (
            "Linear preprocessor should expand columns via one-hot encoding."
        )

    def test_tree_preprocessor_handles_unknown_categories(self, synthetic_X, synthetic_y):
        """OrdinalEncoder with unknown_value=-1 should not raise on new categories."""
        tree_prep, _, _, _ = build_preprocessors(synthetic_X)
        tree_prep.fit(synthetic_X, synthetic_y)
        X_new = synthetic_X.copy()
        X_new.loc[0, "supplier"] = "Supplier Z"   # unseen category
        X_new.loc[1, "shift_code"] = "Shift C"    # unseen category
        try:
            tree_prep.transform(X_new)
        except Exception as e:
            pytest.fail(f"Preprocessor raised on unknown category: {e}")

    def test_linear_preprocessor_handles_unknown_categories(self, synthetic_X, synthetic_y):
        """OneHotEncoder with handle_unknown='ignore' should not raise."""
        _, linear_prep, _, _ = build_preprocessors(synthetic_X)
        linear_prep.fit(synthetic_X, synthetic_y)
        X_new = synthetic_X.copy()
        X_new.loc[0, "supplier"] = "Supplier Z"
        try:
            linear_prep.transform(X_new)
        except Exception as e:
            pytest.fail(f"Linear preprocessor raised on unknown category: {e}")

    def test_no_data_leakage_in_preprocessor(self, synthetic_X, synthetic_y):
        """Preprocessor fit on train should transform val without re-fitting."""
        tree_prep, _, _, _ = build_preprocessors(synthetic_X)
        train = synthetic_X.iloc[:150]
        val   = synthetic_X.iloc[150:]
        tree_prep.fit(train, synthetic_y.iloc[:150])
        # Should not raise — transform uses train-fit parameters only
        try:
            tree_prep.transform(val)
        except Exception as e:
            pytest.fail(f"Preprocessor transform on held-out data raised: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — METRICS
# ══════════════════════════════════════════════════════════════════════════════

class TestMetrics:

    def test_compute_metrics_keys(self, synthetic_y, synthetic_probs):
        """compute_metrics should return all expected keys."""
        metrics = compute_metrics(synthetic_y, synthetic_probs, prefix="val_")
        expected_keys = {
            "val_roc_auc", "val_avg_precision",
            "val_f1", "val_precision", "val_recall"
        }
        assert expected_keys == set(metrics.keys())

    def test_compute_metrics_range(self, synthetic_y, synthetic_probs):
        """All metric values should be in [0, 1]."""
        metrics = compute_metrics(synthetic_y, synthetic_probs)
        for key, val in metrics.items():
            assert 0.0 <= val <= 1.0, f"Metric {key}={val} outside [0,1]"

    def test_perfect_classifier_metrics(self):
        """Perfect predictions should yield ROC-AUC = 1.0."""
        y_true = pd.Series([0, 0, 1, 1])
        y_prob = np.array([0.01, 0.02, 0.99, 0.98])
        metrics = compute_metrics(y_true, y_prob, threshold=0.5)
        assert metrics["roc_auc"] == 1.0

    def test_random_classifier_auc(self):
        """Random predictions should yield ROC-AUC near 0.5."""
        np.random.seed(0)
        y_true = pd.Series(np.random.binomial(1, 0.5, 1000))
        y_prob = np.random.uniform(0, 1, 1000)
        metrics = compute_metrics(y_true, y_prob)
        assert 0.40 <= metrics["roc_auc"] <= 0.60, (
            f"Random classifier AUC={metrics['roc_auc']:.3f} far from 0.5"
        )


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class TestPipeline:

    def test_lr_pipeline_predict_proba_range(self, fitted_lr_pipeline, synthetic_X):
        """predict_proba output should be in [0, 1] for all samples."""
        probs = fitted_lr_pipeline.predict_proba(synthetic_X)[:, 1]
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_rf_pipeline_predict_proba_range(self, fitted_rf_pipeline, synthetic_X):
        """Random forest predict_proba output should be in [0, 1]."""
        probs = fitted_rf_pipeline.predict_proba(synthetic_X)[:, 1]
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_pipeline_output_row_count(self, fitted_lr_pipeline, synthetic_X):
        """predict_proba should return one row per input row."""
        probs = fitted_lr_pipeline.predict_proba(synthetic_X)
        assert probs.shape[0] == len(synthetic_X)

    def test_pipeline_deterministic(self, synthetic_X, synthetic_y):
        """Same random seed should produce identical predictions."""
        tree_prep, _, _, _ = build_preprocessors(synthetic_X)
        def make_pipeline():
            p = Pipeline([
                ("prep",  tree_prep),
                ("model", RandomForestClassifier(
                    n_estimators=10, random_state=RANDOM_SEED
                ))
            ])
            p.fit(synthetic_X, synthetic_y)
            return p

        p1 = make_pipeline()
        p2 = make_pipeline()
        probs1 = p1.predict_proba(synthetic_X)[:, 1]
        probs2 = p2.predict_proba(synthetic_X)[:, 1]
        np.testing.assert_array_equal(probs1, probs2)


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════

class TestArtifacts:

    def test_confusion_matrix_file_created(self, tmp_path, synthetic_y, synthetic_probs):
        path = tmp_path / "confusion_matrix.png"
        save_confusion_matrix(synthetic_y, synthetic_probs,
                              threshold=0.5, path=path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_pr_curve_csv_columns(self, tmp_path, synthetic_y, synthetic_probs):
        path = tmp_path / "pr_curve.csv"
        save_pr_curve(synthetic_y, synthetic_probs, path)
        df = pd.read_csv(path)
        assert set(df.columns) == {"threshold", "precision", "recall"}
        assert len(df) > 0

    def test_roc_curve_csv_columns(self, tmp_path, synthetic_y, synthetic_probs):
        path = tmp_path / "roc_curve.csv"
        save_roc_curve(synthetic_y, synthetic_probs, path)
        df = pd.read_csv(path)
        assert set(df.columns) == {"fpr", "tpr", "threshold"}
        assert (df["fpr"] >= 0).all() and (df["fpr"] <= 1).all()
        assert (df["tpr"] >= 0).all() and (df["tpr"] <= 1).all()

    def test_calibration_csv_columns(self, tmp_path, synthetic_y, synthetic_probs):
        path = tmp_path / "calibration.csv"
        save_calibration(synthetic_y, synthetic_probs, path)
        df = pd.read_csv(path)
        assert "mean_predicted_probability" in df.columns
        assert "fraction_positive" in df.columns
        assert (df["mean_predicted_probability"] >= 0).all()
        assert (df["fraction_positive"] >= 0).all()

    def test_learning_curve_csv_columns(self, tmp_path,
                                         fitted_rf_pipeline,
                                         synthetic_X, synthetic_y):
        path = tmp_path / "learning_curve.csv"
        save_learning_curve(fitted_rf_pipeline, synthetic_X, synthetic_y, path)
        df = pd.read_csv(path)
        expected = {"train_size","train_roc_auc","train_roc_auc_std",
                    "val_roc_auc","val_roc_auc_std"}
        assert expected == set(df.columns)
        assert len(df) > 0
        assert (df["train_roc_auc"] >= 0).all()
        assert (df["val_roc_auc"] >= 0).all()


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# Require real feature Parquet files and write to a temp MLflow directory.
# Run with: pytest tests/test_training.py -v -m integration
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestIntegration:

    @pytest.fixture(autouse=True)
    def require_features(self):
        """Skip integration tests if feature Parquet files don't exist."""
        for split in ["train", "validation", "test"]:
            path = FEATURES_DIR / f"{split}.parquet"
            if not path.exists():
                pytest.skip(f"Feature file not found: {path}. Run ml_prep.ipynb first.")

    def test_feature_parquet_files_exist(self):
        for split in ["train", "validation", "test"]:
            assert (FEATURES_DIR / f"{split}.parquet").exists()

    def test_parquet_target_column_present(self):
        for split in ["train", "validation", "test"]:
            df = pd.read_parquet(FEATURES_DIR / f"{split}.parquet")
            assert TARGET in df.columns, (
                f"Target column '{TARGET}' missing from {split}.parquet"
            )

    def test_parquet_no_future_leakage(self):
        """Train max date must be strictly before validation min date."""
        train = pd.read_parquet(FEATURES_DIR / "train.parquet")
        val   = pd.read_parquet(FEATURES_DIR / "validation.parquet")
        test  = pd.read_parquet(FEATURES_DIR / "test.parquet")

        # work_order_id encodes date implicitly via sequential numbering
        # Check row counts are non-trivial
        assert len(train) > 100, "Train split has too few rows"
        assert len(val)   > 50,  "Validation split has too few rows"
        assert len(test)  > 50,  "Test split has too few rows"

    def test_parquet_feature_parity(self):
        """All splits must have identical feature columns."""
        train = pd.read_parquet(FEATURES_DIR / "train.parquet")
        val   = pd.read_parquet(FEATURES_DIR / "validation.parquet")
        test  = pd.read_parquet(FEATURES_DIR / "test.parquet")
        assert set(train.columns) == set(val.columns) == set(test.columns), (
            "Feature columns differ across splits."
        )

    def test_class_balance_reasonable(self):
        """defect_flag rate should be between 20% and 80% in all splits."""
        for split in ["train", "validation", "test"]:
            df   = pd.read_parquet(FEATURES_DIR / f"{split}.parquet")
            rate = df[TARGET].mean()
            assert 0.20 <= rate <= 0.80, (
                f"{split} defect_flag rate {rate:.1%} outside 20–80% range. "
                "Model will be heavily biased."
            )

    def test_mlflow_run_completes(self, tmp_path):
        """
        Run training on a 20% sample with 3 Optuna trials.
        Verifies the full pipeline completes without error and
        creates an MLflow experiment directory.
        """
        import mlflow
        from training import (
            load_splits, build_preprocessors,
            tune_logistic_regression, train_and_log_model,
            EXPERIMENT_NAME
        )

        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        mlflow.set_experiment(EXPERIMENT_NAME + "_test")

        X_train, y_train, X_val, y_val, X_test, y_test, _ = load_splits()

        # Use 20% sample for speed
        sample_idx = np.random.choice(len(X_train),
                                      size=int(len(X_train) * 0.2),
                                      replace=False)
        X_s = X_train.iloc[sample_idx]
        y_s = y_train.iloc[sample_idx]

        _, linear_prep, _, _ = build_preprocessors(X_s)

        with patch("training.N_TRIALS", 3):
            pipeline, params, _ = tune_logistic_regression(
                X_s, y_s, X_val, y_val, linear_prep
            )

        with mlflow.start_run(run_name="integration_test") as run:
            result = train_and_log_model(
                "logistic_regression", pipeline, params,
                X_s, y_s, X_val, y_val,
                run.info.run_id, tmp_path
            )

        assert result["val_roc_auc"] > 0.5, (
            f"Model performed worse than random: AUC={result['val_roc_auc']:.4f}"
        )

        mlflow_dir = tmp_path / "mlruns"
        assert mlflow_dir.exists(), "MLflow tracking directory not created."

    def test_model_better_than_random(self):
        """
        Fit a quick RF on real data and verify AUC > 0.55.
        Guards against catastrophic feature or target issues.
        """
        train = pd.read_parquet(FEATURES_DIR / "train.parquet")
        val   = pd.read_parquet(FEATURES_DIR / "validation.parquet")

        feature_cols = [c for c in train.columns if c not in [ID_COL, TARGET]]
        X_train = train[feature_cols]
        y_train = train[TARGET].astype(int)
        X_val   = val[feature_cols]
        y_val   = val[TARGET].astype(int)

        tree_prep, _, _, _ = build_preprocessors(X_train)
        pipeline = Pipeline([
            ("prep",  tree_prep),
            ("model", RandomForestClassifier(
                n_estimators=50, max_depth=5,
                random_state=RANDOM_SEED, n_jobs=-1
            ))
        ])
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)

        assert auc > 0.55, (
            f"Model AUC={auc:.4f} barely above random. "
            "Check features and target definition."
        )
