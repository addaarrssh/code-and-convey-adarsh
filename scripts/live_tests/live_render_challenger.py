#!/usr/bin/env python3
"""
Empirical Challenger Test Suite for Live Render Deployment
Target: https://code-and-convey-adarsh.onrender.com
"""

import sys
import io
import time
import json
import hashlib
import requests
import pandas as pd
import numpy as np

BASE_URL = "https://code-and-convey-adarsh.onrender.com"

results = []

def record(name, passed, details):
    status = "PASS" if bool(passed) else "FAIL"
    print(f"[{status}] {name}: {details}")
    results.append({"test": name, "passed": bool(passed), "details": str(details)})

print(f"--- STARTING EMPIRICAL CHALLENGER TESTS AGAINST {BASE_URL} ---")

# 1. Base Connectivity & Security Headers
try:
    r = requests.get(BASE_URL, timeout=30)
    server_header = r.headers.get("x-render-origin-server", "")
    content_type = r.headers.get("content-type", "")
    passed = (r.status_code == 200 and "Werkzeug" in server_header and "text/html" in content_type)
    record(
        "Base_Connectivity_And_Headers",
        passed,
        f"Status: {r.status_code}, Origin Server: {server_header}, Content-Type: {content_type}"
    )
except Exception as e:
    record("Base_Connectivity_And_Headers", False, str(e))

# 2. Dataset Download Integrity
sample_sub_bytes = None
try:
    # 2.1 Download sample_submission.csv
    r = requests.get(f"{BASE_URL}/download/sample_submission.csv", timeout=30)
    passed_status = (r.status_code == 200)
    sample_sub_bytes = r.content
    df_sample = pd.read_csv(io.BytesIO(sample_sub_bytes))
    
    expected_rows = 100000
    actual_rows = len(df_sample)
    cols = list(df_sample.columns)
    first_id = str(df_sample.iloc[0]["candidate_id"])
    last_id = str(df_sample.iloc[-1]["candidate_id"])
    
    intact = (
        actual_rows == expected_rows and
        cols == ["candidate_id", "target_offered_ctc"] and
        first_id == "NITJ_CC_200001" and
        last_id == "NITJ_CC_300000" and
        df_sample["target_offered_ctc"].notnull().all()
    )
    record(
        "Dataset_Integrity_Sample_Submission",
        passed_status and intact,
        f"Rows: {actual_rows}, Cols: {cols}, First: {first_id}, Last: {last_id}, Bytes: {len(sample_sub_bytes)}"
    )
except Exception as e:
    record("Dataset_Integrity_Sample_Submission", False, str(e))

try:
    # 2.2 Download baseline_model.py
    r = requests.get(f"{BASE_URL}/download/baseline_model.py", timeout=30)
    passed = (r.status_code == 200 and len(r.content) == 12942 and "import" in r.text)
    record("Dataset_Integrity_Baseline_Model", passed, f"Status: {r.status_code}, Length: {len(r.content)}")
except Exception as e:
    record("Dataset_Integrity_Baseline_Model", False, str(e))

try:
    # 2.3 Download train.csv (streaming read of first chunk to verify availability and headers)
    r = requests.get(f"{BASE_URL}/download/train.csv", stream=True, timeout=30)
    chunk = next(r.iter_content(chunk_size=1024))
    r.close()
    passed = (r.status_code == 200 and b"candidate_id,branch,cgpa" in chunk and b"NITJ_CC_100001" in chunk)
    record("Dataset_Integrity_Train_CSV", passed, f"Status: {r.status_code}, Chunk: {chunk[:80].decode('utf-8', errors='ignore')}")
except Exception as e:
    record("Dataset_Integrity_Train_CSV", False, str(e))

try:
    # 2.4 Download test.csv (streaming read of first chunk to verify availability and headers)
    r = requests.get(f"{BASE_URL}/download/test.csv", stream=True, timeout=30)
    chunk = next(r.iter_content(chunk_size=1024))
    r.close()
    passed = (r.status_code == 200 and b"candidate_id,branch,cgpa" in chunk and b"NITJ_CC_200001" in chunk)
    record("Dataset_Integrity_Test_CSV", passed, f"Status: {r.status_code}, Chunk: {chunk[:80].decode('utf-8', errors='ignore')}")
except Exception as e:
    record("Dataset_Integrity_Test_CSV", False, str(e))

try:
    # 2.5 Security test: unauthorized download of server_ground_truth.csv
    r = requests.get(f"{BASE_URL}/download/server_ground_truth.csv", timeout=30)
    passed = (r.status_code == 403)
    record("Security_Ground_Truth_Exclusion", passed, f"Status: {r.status_code} (Expected 403 Forbidden)")
except Exception as e:
    record("Security_Ground_Truth_Exclusion", False, str(e))

# 3. Invalid Submission Tests
try:
    # 3.1 Missing form fields
    r = requests.post(f"{BASE_URL}/api/submit", data={"team_name": ""}, timeout=30)
    passed = (r.status_code == 400)
    record("Validation_Empty_Fields", passed, f"Status: {r.status_code}, Response: {r.text[:100]}")
except Exception as e:
    record("Validation_Empty_Fields", False, str(e))

try:
    # 3.2 Non-csv file
    files = {"submission_file": ("test.txt", io.BytesIO(b"dummy,text\n1,2"), "text/plain")}
    data = {"team_name": "TestTeam_BadExt", "team_email": "test@test.com"}
    r = requests.post(f"{BASE_URL}/api/submit", data=data, files=files, timeout=30)
    passed = (r.status_code == 400 and "csv" in r.text.lower())
    record("Validation_Non_CSV_Extension", passed, f"Status: {r.status_code}, Response: {r.text[:100]}")
except Exception as e:
    record("Validation_Non_CSV_Extension", False, str(e))

try:
    # 3.3 CSV missing required column
    bad_csv = b"candidate_id,wrong_col\nNITJ_CC_200001,10.5\n"
    files = {"submission_file": ("sub.csv", io.BytesIO(bad_csv), "text/csv")}
    data = {"team_name": "TestTeam_MissingCol", "team_email": "test@test.com"}
    r = requests.post(f"{BASE_URL}/api/submit", data=data, files=files, timeout=30)
    passed = (r.status_code == 400 and "header rejected" in r.text.lower())
    record("Validation_Missing_Columns", passed, f"Status: {r.status_code}, Response: {r.text[:100]}")
except Exception as e:
    record("Validation_Missing_Columns", False, str(e))

try:
    # 3.4 CSV with NaN / non-numeric
    bad_csv = b"candidate_id,target_offered_ctc\nNITJ_CC_200001,NaN\nNITJ_CC_200002,NOT_A_NUMBER\n"
    files = {"submission_file": ("sub.csv", io.BytesIO(bad_csv), "text/csv")}
    data = {"team_name": "TestTeam_NaN", "team_email": "test@test.com"}
    r = requests.post(f"{BASE_URL}/api/submit", data=data, files=files, timeout=30)
    passed = (r.status_code == 400 and "nan" in r.text.lower())
    record("Validation_NaN_Inf", passed, f"Status: {r.status_code}, Response: {r.text[:100]}")
except Exception as e:
    record("Validation_NaN_Inf", False, str(e))

try:
    # 3.5 CSV with wrong row count (only 1 row)
    bad_csv = b"candidate_id,target_offered_ctc\nNITJ_CC_200001,12.5\n"
    files = {"submission_file": ("sub.csv", io.BytesIO(bad_csv), "text/csv")}
    data = {"team_name": "TestTeam_ShortRows", "team_email": "test@test.com"}
    r = requests.post(f"{BASE_URL}/api/submit", data=data, files=files, timeout=30)
    passed = (r.status_code == 400 and "mismatch" in r.text.lower())
    record("Validation_Row_Count_Mismatch", passed, f"Status: {r.status_code}, Response: {r.text[:100]}")
except Exception as e:
    record("Validation_Row_Count_Mismatch", False, str(e))

# 4. Valid Submission, Public RMSE Calculation & Leaderboard Verification
team_a = f"Challenger_LiveA_{int(time.time())}"
team_b = f"Challenger_LiveB_{int(time.time())}"

score_a = None
score_b = None

try:
    # 4.1 Valid submission for Team Alpha (Sample submission baseline)
    files = {"submission_file": ("sample_submission.csv", io.BytesIO(sample_sub_bytes), "text/csv")}
    data = {"team_name": team_a, "team_email": "live_a@nitjsr.ac.in"}
    r = requests.post(f"{BASE_URL}/api/submit", data=data, files=files, timeout=60)
    res_a = r.json()
    passed = (
        r.status_code == 200 and
        res_a.get("success") is True and
        "public_rmse" in res_a and
        "public_r2" in res_a and
        "private_rmse" not in res_a and # Private metrics must be withheld!
        res_a.get("submissions_today") == 1
    )
    score_a = res_a.get("public_rmse")
    r2_a = res_a.get("public_r2")
    record(
        "Valid_Submission_Team_Alpha",
        passed,
        f"Status: {r.status_code}, RMSE: {score_a}, R2: {r2_a}, SubmissionsToday: {res_a.get('submissions_today')}"
    )
except Exception as e:
    record("Valid_Submission_Team_Alpha", False, str(e))

try:
    # 4.2 Valid submission for Team Beta (Shift predictions by -1.0 to get different score)
    df_beta = pd.read_csv(io.BytesIO(sample_sub_bytes))
    df_beta["target_offered_ctc"] = 11.50
    beta_buf = io.BytesIO()
    df_beta.to_csv(beta_buf, index=False)
    beta_bytes = beta_buf.getvalue()
    
    files = {"submission_file": ("beta_submission.csv", io.BytesIO(beta_bytes), "text/csv")}
    data = {"team_name": team_b, "team_email": "live_b@nitjsr.ac.in"}
    r = requests.post(f"{BASE_URL}/api/submit", data=data, files=files, timeout=60)
    res_b = r.json()
    score_b = res_b.get("public_rmse")
    r2_b = res_b.get("public_r2")
    passed = (
        r.status_code == 200 and
        res_b.get("success") is True and
        score_b != score_a and
        "private_rmse" not in res_b
    )
    record(
        "Valid_Submission_Team_Beta",
        passed,
        f"Status: {r.status_code}, RMSE: {score_b}, R2: {r2_b}, SubmissionsToday: {res_b.get('submissions_today')}"
    )
except Exception as e:
    record("Valid_Submission_Team_Beta", False, str(e))

# 4.3 Verify Leaderboard Content and Sorting
try:
    r = requests.get(f"{BASE_URL}/api/leaderboard", timeout=30)
    lb_data = r.json()
    lb = lb_data.get("leaderboard", [])
    
    teams_in_lb = {entry["team_name"]: entry for entry in lb}
    passed_presence = bool(team_a in teams_in_lb and team_b in teams_in_lb)
    
    email_a_masked = teams_in_lb.get(team_a, {}).get("team_email", "")
    passed_mask = bool("***@" in email_a_masked)
    
    r2_vals = [entry["best_public_r2"] for entry in lb]
    is_sorted_desc = bool(all(r2_vals[i] >= r2_vals[i+1] for i in range(len(r2_vals)-1)))
    
    ranks = [entry["rank"] for entry in lb]
    expected_ranks = list(range(1, len(lb) + 1))
    ranks_correct = bool(ranks == expected_ranks)
    
    overall_lb_passed = passed_presence and passed_mask and is_sorted_desc and ranks_correct
    record(
        "Leaderboard_Sorting_And_Integrity",
        overall_lb_passed,
        f"TotalTeams: {len(lb)}, AlphaPresent: {team_a in teams_in_lb}, BetaPresent: {team_b in teams_in_lb}, Mask: {email_a_masked}, SortedR2: {is_sorted_desc}, RanksCorrect: {ranks_correct}"
    )
except Exception as e:
    record("Leaderboard_Sorting_And_Integrity", False, str(e))

# 5. Rate Limiting / Daily Limit Enforcement (5 attempts max per team per day)
team_rate = f"RateLimit_Bot2_{int(time.time())}"
try:
    successes = 0
    statuses = []
    
    for i in range(1, 7):
        files = {"submission_file": ("sample_sub.csv", io.BytesIO(sample_sub_bytes), "text/csv")}
        data = {"team_name": team_rate, "team_email": "bot2@nitjsr.ac.in"}
        r = requests.post(f"{BASE_URL}/api/submit", data=data, files=files, timeout=60)
        statuses.append((i, r.status_code))
        if r.status_code == 200:
            successes += 1
            
    first_5_ok = bool(all(s[1] == 200 for s in statuses[:5]))
    sixth_429 = bool(statuses[5][1] == 429)
    rate_passed = first_5_ok and sixth_429
    record(
        "Rate_Limit_Enforcement_5_Per_Day",
        rate_passed,
        f"Statuses: {statuses}, First5OK: {first_5_ok}, 6thIs429: {sixth_429}"
    )
except Exception as e:
    record("Rate_Limit_Enforcement_5_Per_Day", False, str(e))

# 6. Admin Security Endpoints
try:
    r = requests.post(f"{BASE_URL}/api/admin/toggle-private", json={"pin": "WRONG_PIN"}, timeout=30)
    passed_toggle = bool(r.status_code == 403)
    
    r2 = requests.get(f"{BASE_URL}/api/admin/export-archive?pin=WRONG_PIN", timeout=30)
    passed_export = bool(r2.status_code == 403)
    
    record("Admin_Security_Endpoints", passed_toggle and passed_export, f"ToggleWrongPin: {r.status_code}, ExportWrongPin: {r2.status_code}")
except Exception as e:
    record("Admin_Security_Endpoints", False, str(e))

# Summary
print("\n" + "="*50)
total_tests = len(results)
passed_tests = sum(1 for r in results if r["passed"])
failed_tests = total_tests - passed_tests
print(f"TEST SUMMARY: {passed_tests}/{total_tests} PASSED ({failed_tests} FAILED)")
print("="*50)

results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, "test_results.json"), "w") as f:
    json.dump({"total": total_tests, "passed": passed_tests, "failed": failed_tests, "details": results}, f, indent=2)

if failed_tests > 0:
    sys.exit(1)
else:
    sys.exit(0)
