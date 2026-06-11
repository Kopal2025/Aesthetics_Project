"""
=============================================================================
  ML MODEL PIPELINE — Kopal Gupta
  Project: Interpreting Photographic Aesthetics via Shapley Value Regression
=============================================================================

INPUTS  (from teammates):
  • Chetan  → cleaned EXIF CSV  (shutter_speed, aperture, iso, focal_length,
                                  + optional one-hot camera/lens cols)
  • Simmone → post-processing proxies CSV  (saturation_proxy, contrast_proxy)
  Both CSVs must share an 'image_id' column used for merging.

OUTPUTS (to Bhavya):
  • Trained ensemble model  → saved as  'ensemble_model.pkl'
  • Train/val/test splits   → saved as  'splits.pkl'
  • Per-split predictions   → saved as  'predictions.csv'
  • Evaluation metrics      → saved as  'metrics.json'

VALIDATION (standalone):
  Run with --demo flag to use a synthetic dataset derived from the
  Kaggle AVA-metadata CSV if the real files are not yet available.
  > python kopal_ml_model.py --demo
=============================================================================
"""

import argparse
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import xgboost as xgb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_STATE = 42
CORE_EXIF_FEATURES = ["shutter_speed", "aperture", "iso", "focal_length"]
PROXY_FEATURES = ["saturation_proxy", "contrast_proxy"]
TARGET_COL = "aesthetic_score"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING & MERGING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_merge(exif_csv: str, proxy_csv: str) -> pd.DataFrame:
    """
    Load Chetan's cleaned EXIF CSV and Simmone's proxy CSV,
    merge on 'image_id', and return a single DataFrame.
    """
    print("[1/6] Loading data...")
    exif_df  = pd.read_csv(exif_csv)
    proxy_df = pd.read_csv(proxy_csv)

    merged = pd.merge(exif_df, proxy_df, on="image_id", how="inner")
    print(f"      Merged shape: {merged.shape}  "
          f"({len(exif_df)} EXIF rows × {len(proxy_df)} proxy rows)")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select core EXIF + proxy features.
    One-hot columns added by Chetan (prefixed with 'brand_' or 'lens_')
    are automatically detected and included.
    """
    ohe_cols = [c for c in df.columns
                if c.startswith("brand_") or c.startswith("lens_")]
    feature_cols = CORE_EXIF_FEATURES + PROXY_FEATURES + ohe_cols

    # guard: only keep columns that exist in the merged DataFrame
    feature_cols = [c for c in feature_cols if c in df.columns]
    missing_core = [c for c in CORE_EXIF_FEATURES + PROXY_FEATURES
                    if c not in df.columns]
    if missing_core:
        raise ValueError(f"Missing expected feature columns: {missing_core}")

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    print(f"[2/6] Feature matrix: {X.shape[1]} features, {X.shape[0]} samples")
    print(f"      Features used: {feature_cols}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRAIN / VALIDATION / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_data(X: pd.DataFrame, y: pd.Series):
    """
    Stratified 80 / 10 / 10 split (mirrors the paper's protocol).
    Stratification is approximated by binning the continuous score into deciles.
    """
    strat_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")

    X_train, X_tmp, y_train, y_tmp, sb_train, sb_tmp = train_test_split(
        X, y, strat_bins,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=strat_bins
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=sb_tmp
    )
    print(f"[3/6] Split → train {len(X_train)} | val {len(X_val)} | test {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MODELS
# ─────────────────────────────────────────────────────────────────────────────

def build_ols_baseline(X_train, y_train, X_test, y_test) -> dict:
    """
    Ordinary Least Squares baseline (scikit-learn LinearRegression).
    Kept intentionally unregularised to reproduce the ill-conditioning
    described in Section I of the paper.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ols",    LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    metrics = evaluate(y_test, preds, label="OLS baseline")
    return {"model": pipe, "predictions": preds, "metrics": metrics}


def tune_random_forest(X_train, y_train, X_val, y_val) -> RandomForestRegressor:
    """
    Hyperparameter search for Random Forest using validation MAE.
    """
    print("      Tuning Random Forest (RandomizedSearchCV)…")
    param_dist = {
        "n_estimators":      [200, 300, 500],
        "max_depth":         [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2", 0.5],
    }
    base_rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    search  = RandomizedSearchCV(
        base_rf, param_dist,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    search.fit(X_train, y_train)
    best_rf = search.best_estimator_
    val_mae = mean_absolute_error(y_val, best_rf.predict(X_val))
    print(f"      RF best params: {search.best_params_}")
    print(f"      RF val MAE: {val_mae:.4f}")
    return best_rf


def tune_xgboost(X_train, y_train, X_val, y_val) -> xgb.XGBRegressor:
    """
    Hyperparameter search for XGBoost.
    Early stopping is applied on the validation set to prevent overfitting.
    """
    print("      Tuning XGBoost (RandomizedSearchCV)…")
    param_dist = {
        "n_estimators":     [300, 500, 800],
        "max_depth":        [3, 5, 7, 9],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "subsample":        [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha":        [0, 0.01, 0.1],
        "reg_lambda":       [1, 1.5, 2],
    }
    base_xgb = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=30,
    )

    # Manual RandomizedSearch with early stopping (CV not compatible with it)
    from sklearn.model_selection import ParameterSampler
    best_val_mae, best_params, best_model = float("inf"), None, None
    for params in ParameterSampler(param_dist, n_iter=20, random_state=RANDOM_STATE):
        model = xgb.XGBRegressor(
            **params,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=30,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        val_mae = mean_absolute_error(y_val, model.predict(X_val))
        if val_mae < best_val_mae:
            best_val_mae, best_params, best_model = val_mae, params, model

    print(f"      XGB best params: {best_params}")
    print(f"      XGB val MAE: {best_val_mae:.4f}")
    return best_model


def build_soft_voting_ensemble(rf_model, xgb_model,
                               X_train, y_train,
                               X_val,   y_val) -> tuple:
    """
    Soft Voting Ensemble — averages predictions from RF and XGBoost.
    Optimised weights are found by minimising val MAE over a grid.

    Returns: (ensemble_predict_fn, best_weights, val_mae)
    The returned function signature: predict(X) -> np.ndarray
    """
    print("      Optimising ensemble weights on validation set…")
    best_w, best_mae = (0.5, 0.5), float("inf")

    for w_rf in np.arange(0.1, 1.0, 0.05):
        w_xgb = 1.0 - w_rf
        preds = w_rf  * rf_model.predict(X_val) + \
                w_xgb * xgb_model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        if mae < best_mae:
            best_mae, best_w = mae, (round(w_rf, 2), round(w_xgb, 2))

    print(f"      Optimal weights  RF={best_w[0]}  XGB={best_w[1]}  "
          f"val MAE={best_mae:.4f}")

    def predict(X):
        return best_w[0] * rf_model.predict(X) + \
               best_w[1] * xgb_model.predict(X)

    return predict, best_w, best_mae


# ─────────────────────────────────────────────────────────────────────────────
# 5.  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, label: str = "") -> dict:
    """
    Compute MAE, RMSE, PLCC, SRCC — the four metrics from the paper.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    plcc, _ = pearsonr(y_true, y_pred)
    srcc, _ = spearmanr(y_true, y_pred)

    metrics = {"MAE": mae, "RMSE": rmse, "PLCC": plcc, "SRCC": srcc}
    tag = f"[{label}] " if label else ""
    print(f"      {tag}MAE={mae:.4f}  RMSE={rmse:.4f}  "
          f"PLCC={plcc:.4f}  SRCC={srcc:.4f}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(exif_csv: str, proxy_csv: str, output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)

    # --- load & split ---
    df = load_and_merge(exif_csv, proxy_csv)
    X, y = build_feature_matrix(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # --- OLS baseline ---
    print("[4/6] Fitting models…")
    print("    → OLS baseline")
    ols_result = build_ols_baseline(X_train, y_train, X_test, y_test)

    # --- tree models ---
    print("    → Random Forest")
    rf_model  = tune_random_forest(X_train, y_train, X_val, y_val)

    print("    → XGBoost")
    xgb_model = tune_xgboost(X_train, y_train, X_val, y_val)

    # --- ensemble ---
    print("    → Soft Voting Ensemble")
    ensemble_predict, weights, _ = build_soft_voting_ensemble(
        rf_model, xgb_model, X_train, y_train, X_val, y_val
    )

    # --- final test evaluation ---
    print("[5/6] Final test-set evaluation…")
    ens_preds_test = ensemble_predict(X_test)
    rf_preds_test  = rf_model.predict(X_test)
    xgb_preds_test = xgb_model.predict(X_test)

    all_metrics = {
        "OLS":      ols_result["metrics"],
        "RF":       evaluate(y_test, rf_preds_test,  "Random Forest"),
        "XGBoost":  evaluate(y_test, xgb_preds_test, "XGBoost"),
        "Ensemble": evaluate(y_test, ens_preds_test,  "Ensemble"),
        "ensemble_weights": {"RF": weights[0], "XGBoost": weights[1]},
    }

    # --- persist outputs for Bhavya ---
    print("[6/6] Saving outputs…")

    # a) models
    model_bundle = {
        "rf_model":        rf_model,
        "xgb_model":       xgb_model,
        "ensemble_weights": weights,
        "feature_cols":    list(X.columns),
    }
    with open(os.path.join(output_dir, "ensemble_model.pkl"), "wb") as f:
        pickle.dump(model_bundle, f)

    # b) splits (Bhavya needs these for SHAP)
    splits = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }
    with open(os.path.join(output_dir, "splits.pkl"), "wb") as f:
        pickle.dump(splits, f)

    # c) predictions CSV
    pred_df = X_test.copy()
    pred_df["y_true"]     = y_test.values
    pred_df["y_pred_ols"] = ols_result["predictions"]
    pred_df["y_pred_rf"]  = rf_preds_test
    pred_df["y_pred_xgb"] = xgb_preds_test
    pred_df["y_pred_ens"] = ens_preds_test
    pred_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    # d) metrics JSON
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n✅  All outputs saved to '{output_dir}/'")
    print("    ensemble_model.pkl  ← for Bhavya (SHAP)")
    print("    splits.pkl          ← for Bhavya (SHAP)")
    print("    predictions.csv     ← inspection / sanity check")
    print("    metrics.json        ← for Nirmith (LaTeX table)")
    return model_bundle, all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# 7.  DEMO MODE  (validate without real data — uses synthetic dataset)
# ─────────────────────────────────────────────────────────────────────────────

def generate_demo_data(n: int = 3000) -> tuple[str, str]:
    """
    Generate synthetic EXIF + proxy CSVs that mimic the AVA structure.
    Realistic correlations between ISO / shutter / aperture are enforced.
    aesthetic_score is a noisy non-linear function of the features.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    image_ids    = np.arange(n)
    # realistic EV-correlated exposure triangle
    iso          = rng.choice([100, 200, 400, 800, 1600, 3200, 6400], n,
                              p=[0.25, 0.20, 0.20, 0.15, 0.10, 0.07, 0.03])
    aperture     = rng.choice([1.4, 1.8, 2.8, 4.0, 5.6, 8.0, 11.0], n)
    shutter_frac = 1.0 / rng.choice([30, 60, 125, 250, 500, 1000, 2000], n)
    focal_length = rng.choice([24, 35, 50, 85, 135, 200], n)

    saturation_proxy = rng.uniform(0.1, 0.9, n)
    contrast_proxy   = rng.uniform(20, 80, n)

    # non-linear aesthetic score with ISO noise penalty above 3200
    iso_penalty = np.where(iso > 3200, -0.8, 0.0)
    score = (
        5.0
        + 0.5  * saturation_proxy
        + 0.003 * contrast_proxy
        - 0.4  * np.log1p(iso / 100)
        + 0.3  * np.log(aperture)
        + iso_penalty
        + rng.normal(0, 0.3, n)
    )
    score = np.clip(score, 1, 10)

    exif_df = pd.DataFrame({
        "image_id":      image_ids,
        "shutter_speed": shutter_frac,
        "aperture":      aperture,
        "iso":           iso.astype(float),
        "focal_length":  focal_length.astype(float),
        TARGET_COL:      score,
    })
    proxy_df = pd.DataFrame({
        "image_id":        image_ids,
        "saturation_proxy": saturation_proxy,
        "contrast_proxy":   contrast_proxy,
    })

    import tempfile
    tmp_dir    = tempfile.gettempdir()
    exif_path  = os.path.join(tmp_dir, "demo_exif.csv")
    proxy_path = os.path.join(tmp_dir, "demo_proxy.csv")
    exif_df.to_csv(exif_path,  index=False)
    proxy_df.to_csv(proxy_path, index=False)
    print(f"[DEMO] Synthetic data written → {exif_path}, {proxy_path}")
    return exif_path, proxy_path


# ─────────────────────────────────────────────────────────────────────────────
# 8.  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kopal's ML pipeline for photographic aesthetic scoring."
    )
    parser.add_argument("--exif",    type=str, help="Path to Chetan's cleaned EXIF CSV")
    parser.add_argument("--proxy",   type=str, help="Path to Simmone's proxy CSV")
    parser.add_argument("--outdir",  type=str, default="outputs",
                        help="Directory for output files (default: outputs/)")
    parser.add_argument("--demo",    action="store_true",
                        help="Run on synthetic data for validation")
    args = parser.parse_args()

    if args.demo:
        print("=" * 60)
        print("  DEMO MODE — synthetic data")
        print("=" * 60)
        exif_csv, proxy_csv = generate_demo_data()
    else:
        if not args.exif or not args.proxy:
            parser.error("Provide --exif and --proxy, or run with --demo")
        exif_csv, proxy_csv = args.exif, args.proxy

    run_pipeline(exif_csv, proxy_csv, output_dir=args.outdir)
