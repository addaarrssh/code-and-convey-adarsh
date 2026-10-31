#!/usr/bin/env python3
"""
baseline_model.py - Official Starter Kit Baseline for Code & Convey Challenge
ACC Club | National Institute of Technology (NIT) Jamshedpur

Competition: Code & Convey - The Machine Learning & Presentation Challenge
Task: Predict campus placement salary offers ('target_offered_ctc' in ₹ LPA)

Demonstrates:
  1. Vectorized data loading of train.csv and test.csv (100k rows each).
  2. Exploratory Data Analysis (EDA) summary of target & missing values.
  3. Preprocessing numerical, categorical, and Missing Not At Random (MNAR) features.
  4. Continuous XOR specialization feature engineering (|z_DSA - z_Core|).
  5. Text feature extraction from unstructured interview feedback notes (TF-IDF).
  6. Fast Gradient Boosting training with 5-fold Out-Of-Fold (OOF) cross-validation.
  7. Verification and export of valid submission.csv matching sample_submission.csv format.
"""

import sys
import os
import time
import argparse
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import HistGradientBoostingRegressor

# Graceful algorithm fallback: LightGBM -> HistGradientBoostingRegressor
try:
    import lightgbm as lgb
    USE_LIGHTGBM = True
except ImportError:
    lgb = None
    USE_LIGHTGBM = False


def find_data_files(data_dir: Optional[str] = None) -> Tuple[str, str]:
    """Locate train.csv and test.csv across standard candidate paths."""
    candidates = [
        data_dir,
        ".",
        "participant_download",
        "../participant_download",
        os.path.dirname(__file__),
        os.path.join(os.path.dirname(__file__), "participant_download"),
    ]

    for c in candidates:
        if c:
            train_p = os.path.join(c, "train.csv")
            test_p = os.path.join(c, "test.csv")
            if os.path.exists(train_p) and os.path.exists(test_p):
                return os.path.abspath(train_p), os.path.abspath(test_p)

    raise FileNotFoundError(
        f"Could not locate both 'train.csv' and 'test.csv'. Searched directories: {[c for c in candidates if c]}"
    )


def run_baseline_pipeline(
    data_dir: Optional[str] = None,
    output_path: str = "submission.csv",
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Execute complete end-to-end baseline training and inference pipeline."""
    start_total = time.time()
    print("=" * 78)
    print("       ACC CLUB NIT JAMSHEDPUR | CODE & CONVEY BASELINE STARTER KIT       ")
    print("=" * 78)

    train_path, test_path = find_data_files(data_dir)

    # ---------------------------------------------------------
    # 1. Ingest Data
    # ---------------------------------------------------------
    print("\n[1/6] Ingesting datasets...")
    print(f"      • Train source : {train_path}")
    print(f"      • Test source  : {test_path}")
    t0 = time.time()
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"      Train Shape: {train_df.shape} | Test Shape: {test_df.shape} (Loaded in {time.time()-t0:.2f}s)")

    target_col = "target_offered_ctc"
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in train.csv!")

    y = train_df[target_col].to_numpy(dtype=np.float64)
    candidate_ids_test = test_df["candidate_id"].values

    # ---------------------------------------------------------
    # 2. Exploratory Data Analysis & Target Distribution
    # ---------------------------------------------------------
    print("\n[2/6] Target Variable & Missing Value Inspection:")
    print(
        f"      CTC (₹ LPA) - Mean: {np.mean(y):.2f}, Median: {np.median(y):.2f}, "
        f"Min: {np.min(y):.2f}, Max: {np.max(y):.2f}, Std: {np.std(y):.2f}"
    )

    missing_12th_train = int(train_df["twelfth_percentage"].isna().sum())
    missing_12th_test = int(test_df["twelfth_percentage"].isna().sum())
    print(
        f"      MNAR Missing 12th Marks: "
        f"Train={missing_12th_train:,} ({missing_12th_train/len(train_df):.1%}), "
        f"Test={missing_12th_test:,} ({missing_12th_test/len(test_df):.1%})"
    )

    # ---------------------------------------------------------
    # 3. Preprocessing & Feature Engineering
    # ---------------------------------------------------------
    print("\n[3/6] Preprocessing and Feature Engineering...")
    t0 = time.time()

    # Combine train (without target) and test for uniform feature transformation
    df_all = pd.concat([train_df.drop(columns=[target_col]), test_df], axis=0, ignore_index=True)

    # Trap 1 / Feature A: Missing Not At Random (MNAR) Indicator & Imputation
    # Lower CGPA candidates systematically omit 12th marks; the missingness itself conveys signal!
    df_all["twelfth_is_missing"] = df_all["twelfth_percentage"].isna().astype(np.float32)
    median_12th = float(df_all["twelfth_percentage"].median())
    df_all["twelfth_percentage_imputed"] = df_all["twelfth_percentage"].fillna(median_12th)

    # Trap 2 / Feature B: Continuous XOR Boundary (|z_DSA - z_Core|)
    # Deep specialization in either DSA or Core domain commands a premium over unfocused generalists
    dsa_mean, dsa_std = float(df_all["dsa_problems_solved"].mean()), float(df_all["dsa_problems_solved"].std())
    core_mean, core_std = float(df_all["core_domain_score"].mean()), float(df_all["core_domain_score"].std())
    z_dsa = (df_all["dsa_problems_solved"] - dsa_mean) / max(dsa_std, 1e-6)
    z_core = (df_all["core_domain_score"] - core_mean) / max(core_std, 1e-6)
    df_all["specialization_xor"] = np.abs(z_dsa - z_core)

    # Feature C: Academic Composite & Ratios
    df_all["academic_composite"] = (df_all["cgpa"] / 10.0) * (df_all["tenth_percentage"] / 100.0)

    # Trap 3 / Feature D: Simpson's Paradox & Project Interaction
    proj_col = "projects_count" if "projects_count" in df_all.columns else "project_count"
    df_all["projects_per_cgpa"] = df_all[proj_col] / np.maximum(df_all["cgpa"], 1.0)
    df_all["projects_x_core"] = df_all[proj_col] * (df_all["core_domain_score"] / 100.0)

    # Feature E: Categorical Encodings (One-Hot for branch, gender)
    branch_dummies = pd.get_dummies(df_all["branch"], prefix="br", drop_first=True, dtype=np.float32)
    gender_dummies = pd.get_dummies(df_all["gender"], prefix="gen", drop_first=True, dtype=np.float32)

    # Trap 4 / Feature F: Unstructured Interview Notes Vectorization (TF-IDF)
    # Extracts semantic tokens from qualitative interview remarks (capturing praise, skepticism, keywords)
    print("      Extracting TF-IDF text features from interview notes...")
    tfidf = TfidfVectorizer(max_features=25, stop_words="english")
    text_sparse = tfidf.fit_transform(df_all["interview_feedback_notes"].fillna(""))
    text_dense = text_sparse.toarray().astype(np.float32)
    text_feat_df = pd.DataFrame(
        text_dense, columns=[f"tfidf_{w}" for w in tfidf.get_feature_names_out()]
    )

    # Trap 5 & 6 Caution Notes:
    # - 'alumni_referral_rating' is a poison feature: correlated in train (Tech Boom), pure noise in test!
    # - 'decoy_entropy_token' is random Gaussian noise with zero causal signal.
    # To demonstrate robust modeling, we exclude decoy_entropy_token and include internship experience.
    numeric_cols = [
        "cgpa",
        "tenth_percentage",
        "twelfth_percentage_imputed",
        "twelfth_is_missing",
        "dsa_problems_solved",
        "core_domain_score",
        "specialization_xor",
        "academic_composite",
        proj_col,
        "projects_per_cgpa",
        "projects_x_core",
        "internship_months",
    ]

    # Assemble complete design matrix
    X_all = pd.concat([df_all[numeric_cols], branch_dummies, gender_dummies, text_feat_df], axis=1)

    n_train = len(train_df)
    X_train = X_all.iloc[:n_train].to_numpy(dtype=np.float32)
    X_test = X_all.iloc[n_train:].to_numpy(dtype=np.float32)

    print(
        f"      Engineered {X_train.shape[1]} features across {n_train:,} training samples in {time.time()-t0:.2f}s."
    )

    # ---------------------------------------------------------
    # 4. Out-Of-Fold Cross-Validation & Model Training
    # ---------------------------------------------------------
    engine_name = "LightGBM Regressor" if USE_LIGHTGBM else "HistGradientBoostingRegressor"
    print(f"\n[4/6] Training {engine_name} with {n_splits}-Fold Cross Validation...")
    t0 = time.time()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(n_train, dtype=np.float64)
    test_preds = np.zeros(len(test_df), dtype=np.float64)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, y_tr = X_train[train_idx], y[train_idx]
        X_va = X_train[val_idx]

        if USE_LIGHTGBM:
            model = lgb.LGBMRegressor(
                n_estimators=120,
                learning_rate=0.08,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state + fold,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(X_tr, y_tr)
        else:
            model = HistGradientBoostingRegressor(
                max_iter=100,
                learning_rate=0.08,
                max_leaf_nodes=31,
                random_state=random_state + fold,
            )
            model.fit(X_tr, y_tr)

        val_pred = model.predict(X_va)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test) / float(n_splits)

    cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    cv_mae = mean_absolute_error(y, oof_preds)
    cv_r2 = r2_score(y, oof_preds)

    print(f"      Training complete in {time.time()-t0:.2f}s.")
    print("      " + "-" * 56)
    print(f"      LOCAL {n_splits}-FOLD CROSS-VALIDATION RESULTS:")
    print(f"        • Out-Of-Fold RMSE : {cv_rmse:.4f} ₹ LPA")
    print(f"        • Out-Of-Fold MAE  : {cv_mae:.4f} ₹ LPA")
    print(f"        • Out-Of-Fold R²   : {cv_r2:.4f}")
    print("      " + "-" * 56)

    # ---------------------------------------------------------
    # 5. Submission Export & Post-processing
    # ---------------------------------------------------------
    print(f"\n[5/6] Formatting submission file -> '{output_path}'...")
    # Clip predictions to realistic placement package bounds [3.0, 75.0] LPA
    clipped_preds = np.round(np.clip(test_preds, 3.0, 75.0), 2)

    sub_df = pd.DataFrame({
        "candidate_id": candidate_ids_test,
        "target_offered_ctc": clipped_preds,
    })

    # Ensure parent directories exist
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    sub_df.to_csv(output_path, index=False)

    # ---------------------------------------------------------
    # 6. Verification Sanity Checks
    # ---------------------------------------------------------
    print("\n[6/6] Verifying submission integrity:")
    print(f"      • File Exists     : {os.path.exists(output_path)}")
    print(f"      • Row Count       : {len(sub_df):,} (Expected: {len(test_df):,})")
    print(f"      • Nulls / NaNs    : {sub_df['target_offered_ctc'].isna().sum()} (Must be 0)")
    print(
        f"      • CTC Prediction Range: [{sub_df['target_offered_ctc'].min():.2f}, "
        f"{sub_df['target_offered_ctc'].max():.2f}] ₹ LPA"
    )
    print("=" * 78)
    print(f"SUCCESS: Baseline submission generated in {time.time()-start_total:.2f}s!")
    print("=" * 78)

    return sub_df


def main():
    parser = argparse.ArgumentParser(description="Code & Convey Baseline ML Starter Kit")
    parser.add_argument(
        "--data-dir",
        "-d",
        default=None,
        help="Directory containing train.csv and test.csv (default: auto-detected)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="submission.csv",
        help="Output submission CSV path (default: submission.csv)",
    )
    parser.add_argument(
        "--folds",
        "-k",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    try:
        run_baseline_pipeline(
            data_dir=args.data_dir,
            output_path=args.output,
            n_splits=args.folds,
            random_state=args.seed,
        )
        sys.exit(0)
    except Exception as e:
        sys.stderr.write(f"\n[BASELINE ERROR]: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
