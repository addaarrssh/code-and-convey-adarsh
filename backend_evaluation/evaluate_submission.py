#!/usr/bin/env python3
"""
evaluate_submission.py - Official Scoring Engine for Code & Convey ML Challenge
ACC Club | National Institute of Technology (NIT) Jamshedpur

Evaluates participant submissions against server_ground_truth.csv.
Computes RMSE, MAE, and R² Score across Public (50,000), Private (50,000), and Overall (100,000) splits.

Dual interface:
  - CLI: python3 evaluate_submission.py --submission <path> --ground-truth <path> [--json] [--output <path>]
  - Python API: score_submission(submission, ground_truth=None)
"""

import sys
import os
import argparse
import json
import datetime
from typing import Union, Tuple, Optional
import numpy as np
import pandas as pd


class SubmissionValidationError(Exception):
    """Raised when submission fails format, schema, row count, or data integrity validation."""
    pass


def find_ground_truth(gt_path: Optional[str] = None) -> Optional[str]:
    """Resolve ground truth path from argument or default locations."""
    if gt_path and os.path.exists(gt_path):
        return gt_path

    candidates = [
        gt_path,
        "server_ground_truth.csv",
        "backend_evaluation/server_ground_truth.csv",
        "../backend_evaluation/server_ground_truth.csv",
        os.path.join(os.path.dirname(__file__), "server_ground_truth.csv"),
        os.path.join(os.path.dirname(__file__), "backend_evaluation", "server_ground_truth.csv"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


def compute_split_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, and R² Score with high numerical stability."""
    diff = y_true - y_pred
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum(diff ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2_score": round(r2, 4),
        "r2": round(r2, 4),
    }


def validate_and_align_data(
    sub_df: pd.DataFrame, gt_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Execute 7-step multi-tier validation pipeline on participant submission.
    Returns: (y_true, y_pred, public_mask, private_mask, validation_metadata)
    """
    meta = {
        "is_valid": False,
        "total_rows_evaluated": 0,
        "public_rows": 0,
        "private_rows": 0,
        "candidate_id_match": False,
        "order_aligned": True,
        "has_nan": False,
        "has_inf": False,
        "negative_predictions_count": 0,
        "extreme_predictions_count": 0,
    }

    # Step 1: Input Readability / Non-empty validation
    if sub_df is None or sub_df.empty:
        raise SubmissionValidationError("Submission DataFrame is empty or null.")
    if gt_df is None or gt_df.empty:
        raise SubmissionValidationError("Ground truth DataFrame is empty or null.")

    # Step 2: Schema & Column Presence Check
    sub_cols = [str(c).strip() for c in sub_df.columns]
    sub_df = sub_df.copy()
    sub_df.columns = sub_cols

    gt_cols = [str(c).strip() for c in gt_df.columns]
    gt_df = gt_df.copy()
    gt_df.columns = gt_cols

    req_sub_cols = ["candidate_id", "target_offered_ctc"]
    for col in req_sub_cols:
        if col not in sub_df.columns:
            raise SubmissionValidationError(
                f"Missing required column in submission: '{col}'. Found columns: {list(sub_df.columns)}"
            )

    req_gt_cols = ["candidate_id", "target_offered_ctc", "Usage"]
    for col in req_gt_cols:
        if col not in gt_df.columns:
            raise SubmissionValidationError(
                f"Ground truth is missing required column '{col}'. Found columns: {list(gt_df.columns)}"
            )

    # Step 3: Exact Row Count Check
    expected_rows = len(gt_df)
    actual_rows = len(sub_df)
    if actual_rows != expected_rows:
        raise SubmissionValidationError(
            f"Row count mismatch: Submission contains {actual_rows:,} rows; expected exactly {expected_rows:,} rows."
        )
    meta["total_rows_evaluated"] = actual_rows

    # Step 4: Candidate ID Set Integrity Check (Bijection, no duplicates, no missing, no extras)
    sub_ids = sub_df["candidate_id"].astype(str).str.strip()
    gt_ids = gt_df["candidate_id"].astype(str).str.strip()

    set_sub = set(sub_ids)
    set_gt = set(gt_ids)

    if len(set_sub) != len(sub_ids):
        dupes = sub_ids[sub_ids.duplicated()].unique()
        diff_count = len(sub_ids) - len(set_sub)
        raise SubmissionValidationError(
            f"Submission contains {diff_count} duplicate candidate IDs! Example: {list(dupes[:5])}"
        )

    missing_ids = set_gt - set_sub
    if missing_ids:
        raise SubmissionValidationError(
            f"Submission is missing {len(missing_ids)} expected candidate IDs. Examples: {list(missing_ids)[:5]}"
        )

    extra_ids = set_sub - set_gt
    if extra_ids:
        raise SubmissionValidationError(
            f"Submission contains {len(extra_ids)} unrecognized candidate IDs. Examples: {list(extra_ids)[:5]}"
        )
    meta["candidate_id_match"] = True

    # Step 5: ID Order Alignment Check & Auto-Reindexing
    if not (sub_ids.values == gt_ids.values).all():
        meta["order_aligned"] = False
        sub_aligned = sub_df.copy()
        sub_aligned["candidate_id"] = sub_ids
        sub_aligned = sub_aligned.set_index("candidate_id").reindex(gt_ids.values).reset_index()
    else:
        meta["order_aligned"] = True
        sub_aligned = sub_df

    # Step 6: Data Quality & Type Checking (NaN / Inf Rejection)
    pred_series = pd.to_numeric(sub_aligned["target_offered_ctc"], errors="coerce")
    nan_count = int(pred_series.isna().sum())
    if nan_count > 0:
        meta["has_nan"] = True
        bad_idx = np.where(pred_series.isna().to_numpy())[0][:5]
        bad_cands = gt_ids.iloc[bad_idx].tolist()
        raise SubmissionValidationError(
            f"Submission contains {nan_count} non-numeric, null, or NaN predictions. Candidate IDs: {bad_cands}"
        )

    inf_count = int(np.isinf(pred_series.to_numpy()).sum())
    if inf_count > 0:
        meta["has_inf"] = True
        raise SubmissionValidationError(
            f"Submission contains {inf_count} infinite (inf / -inf) prediction values."
        )

    # Step 7: Domain Sanity & Range Inspection (Warning flags)
    neg_count = int((pred_series < 0).sum())
    meta["negative_predictions_count"] = neg_count
    if neg_count > 0:
        sys.stderr.write(
            f"WARNING: Submission contains {neg_count} negative CTC predictions. This will severely penalize RMSE.\n"
        )

    extreme_count = int((pred_series > 200).sum())
    meta["extreme_predictions_count"] = extreme_count
    if extreme_count > 0:
        sys.stderr.write(
            f"WARNING: Submission contains {extreme_count} predictions exceeding 200 LPA.\n"
        )

    y_pred = pred_series.to_numpy(dtype=np.float64)
    y_true = pd.to_numeric(gt_df["target_offered_ctc"], errors="raise").to_numpy(dtype=np.float64)

    usage_arr = gt_df["Usage"].astype(str).str.strip().values
    public_mask = (usage_arr == "Public")
    private_mask = (usage_arr == "Private")

    meta["public_rows"] = int(public_mask.sum())
    meta["private_rows"] = int(private_mask.sum())
    meta["is_valid"] = True

    return y_true, y_pred, public_mask, private_mask, meta


def score_submission_dataframes(
    sub_df: pd.DataFrame, gt_df: pd.DataFrame, submission_name: str = "DataFrame"
) -> dict:
    """Core scoring function operating directly on Pandas DataFrames."""
    y_true, y_pred, pub_mask, priv_mask, meta = validate_and_align_data(sub_df, gt_df)

    pub_metrics = compute_split_metrics(y_true[pub_mask], y_pred[pub_mask])
    priv_metrics = compute_split_metrics(y_true[priv_mask], y_pred[priv_mask])
    overall_metrics = compute_split_metrics(y_true, y_pred)

    return {
        "status": "success",
        "error_message": None,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "submission_file": submission_name,
        "validation": meta,
        "public": pub_metrics,
        "private": priv_metrics,
        "overall": overall_metrics,
        "metrics": {
            "public": pub_metrics,
            "private": priv_metrics,
            "overall": overall_metrics,
        },
        "leaderboard": {
            "public_rmse": pub_metrics["rmse"],
            "public_r2": pub_metrics["r2_score"],
            "private_rmse": priv_metrics["rmse"],
            "private_r2": priv_metrics["r2_score"],
            "primary_metric": "rmse",
        },
    }


def evaluate_submission(submission_path: str, ground_truth_path: Optional[str] = None) -> dict:
    """Evaluate submission file on disk against ground truth file."""
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found: {submission_path}")

    resolved_gt = find_ground_truth(ground_truth_path)
    if not resolved_gt or not os.path.exists(resolved_gt):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path or 'default candidates'}")

    sub_df = pd.read_csv(submission_path)
    gt_df = pd.read_csv(resolved_gt)

    return score_submission_dataframes(
        sub_df, gt_df, submission_name=os.path.basename(submission_path)
    )


def score_submission(
    submission: Union[str, pd.DataFrame],
    ground_truth: Optional[Union[str, pd.DataFrame]] = None,
) -> dict:
    """
    Universal Python API for scoring a participant submission.
    Accepts either file paths (str) or Pandas DataFrames for submission and ground_truth.
    """
    if isinstance(submission, str):
        sub_path = submission
        if not os.path.exists(sub_path):
            raise FileNotFoundError(f"Submission file not found: {sub_path}")
        sub_df = pd.read_csv(sub_path)
        sub_name = os.path.basename(sub_path)
    elif isinstance(submission, pd.DataFrame):
        sub_df = submission
        sub_name = "DataFrame"
    else:
        raise TypeError(f"Expected submission as filepath or DataFrame, got {type(submission)}")

    if ground_truth is None or isinstance(ground_truth, str):
        resolved_gt = find_ground_truth(ground_truth)
        if not resolved_gt or not os.path.exists(resolved_gt):
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth or 'default candidates'}")
        gt_df = pd.read_csv(resolved_gt)
    elif isinstance(ground_truth, pd.DataFrame):
        gt_df = ground_truth
    else:
        raise TypeError(f"Expected ground_truth as filepath or DataFrame, got {type(ground_truth)}")

    return score_submission_dataframes(sub_df, gt_df, submission_name=sub_name)


def format_terminal_dashboard(results: dict, sub_path: str, gt_path: str) -> str:
    """Render high-polish ASCII dashboard for terminal viewing."""
    m = results["metrics"]
    v = results["validation"]

    lines = [
        "=" * 88,
        "             ACC CLUB NIT JAMSHEDPUR | CODE & CONVEY EVALUATION ENGINE                  ",
        "=" * 88,
        f" Submission File   : {sub_path}",
        f" Ground Truth File : {gt_path}",
        f" Evaluation Time   : {results['timestamp']}",
        f" Validation Status : PASSED ({v['total_rows_evaluated']:,} / {v['total_rows_evaluated']:,} rows valid)",
        f" Order Alignment   : {'Exact Match' if v['order_aligned'] else 'Re-aligned to Canonical Order'}",
        "-" * 88,
        f" {'SPLIT':<17} {'ROWS':<11} {'RMSE (₹ LPA)':<17} {'MAE (₹ LPA)':<17} {'R² SCORE':<12}",
        "-" * 88,
        (
            f" {'Public (50%)':<17} {v['public_rows']:<11,} "
            f"{m['public']['rmse']:<17.4f} {m['public']['mae']:<17.4f} {m['public']['r2_score']:<12.4f}"
        ),
        (
            f" {'Private (50%)':<17} {v['private_rows']:<11,} "
            f"{m['private']['rmse']:<17.4f} {m['private']['mae']:<17.4f} {m['private']['r2_score']:<12.4f}"
        ),
        (
            f" {'Overall (100%)':<17} {v['total_rows_evaluated']:<11,} "
            f"{m['overall']['rmse']:<17.4f} {m['overall']['mae']:<17.4f} {m['overall']['r2_score']:<12.4f}"
        ),
        "=" * 88,
        (
            f" >>> PUBLIC LEADERBOARD SCORE : RMSE = {m['public']['rmse']:.4f}  |  "
            f"R² = {m['public']['r2_score']:.4f} <<<"
        ),
        (
            f" >>> PRIVATE HOLDOUT SCORE    : RMSE = {m['private']['rmse']:.4f}  |  "
            f"R² = {m['private']['r2_score']:.4f} [CONFIDENTIAL] <<<"
        ),
        "=" * 88,
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Code & Convey Submission Evaluator")
    parser.add_argument("submission", nargs="?", help="Path to participant submission CSV")
    parser.add_argument("ground_truth", nargs="?", default=None, help="Path to server_ground_truth.csv")
    parser.add_argument("--submission", "-s", dest="sub_opt", help="Path to participant submission CSV")
    parser.add_argument("--ground-truth", "-g", dest="gt_opt", help="Path to server_ground_truth.csv")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--output", "-o", help="Optional path to write JSON output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress human-readable dashboard")

    args = parser.parse_args()

    sub_path = args.sub_opt or args.submission
    gt_path = args.gt_opt or args.ground_truth

    if not sub_path:
        parser.print_help()
        sys.exit(2)

    resolved_gt = find_ground_truth(gt_path)
    if not resolved_gt or not os.path.exists(resolved_gt):
        err_res = {
            "status": "error",
            "error_type": "FileNotFoundError",
            "error_message": f"Server ground truth file not found: {gt_path or 'default locations'}",
        }
        if args.json:
            print(json.dumps(err_res, indent=2))
        else:
            print(f"ERROR: {err_res['error_message']}", file=sys.stderr)
        sys.exit(2)

    try:
        results = evaluate_submission(sub_path, resolved_gt)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

        if args.json:
            print(json.dumps(results, indent=2))
        elif not args.quiet:
            print(format_terminal_dashboard(results, sub_path, resolved_gt))

        sys.exit(0)

    except SubmissionValidationError as sve:
        err_res = {
            "status": "validation_error",
            "error_type": "SubmissionValidationError",
            "error_message": str(sve),
        }
        if args.output:
            with open(args.output, "w") as f:
                json.dump(err_res, f, indent=2)
        if args.json:
            print(json.dumps(err_res, indent=2))
        else:
            print(f"\n[VALIDATION FAILED]: {sve}\n", file=sys.stderr)
        sys.exit(1)

    except FileNotFoundError as fnfe:
        err_res = {
            "status": "system_error",
            "error_type": "FileNotFoundError",
            "error_message": str(fnfe),
        }
        if args.output:
            with open(args.output, "w") as f:
                json.dump(err_res, f, indent=2)
        if args.json:
            print(json.dumps(err_res, indent=2))
        else:
            print(f"\n[FILE NOT FOUND]: {fnfe}\n", file=sys.stderr)
        sys.exit(2)

    except Exception as e:
        err_res = {
            "status": "system_error",
            "error_type": type(e).__name__,
            "error_message": str(e),
        }
        if args.output:
            with open(args.output, "w") as f:
                json.dump(err_res, f, indent=2)
        if args.json:
            print(json.dumps(err_res, indent=2))
        else:
            print(f"\n[EXECUTION ERROR]: {e}\n", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
