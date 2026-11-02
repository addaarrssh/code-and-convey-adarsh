import os
import io
import time
import concurrent.futures
import requests
import pytest

BASE_URL = os.environ.get("BASE_URL", "https://code-and-convey-adarsh.onrender.com")
ADMIN_PIN = os.environ.get("ADMIN_PIN", "MANNU")

# =========================================================================
# 1. MALFORMED CSV SUBMISSION TESTS
# =========================================================================

def test_submit_missing_file():
    """Verify POST /api/submit without file payload returns 400 Bad Request."""
    data = {"team_name": "TestTeam", "team_email": "test@nitjsr.ac.in"}
    resp = requests.post(f"{BASE_URL}/api/submit", data=data, timeout=15)
    assert resp.status_code == 400
    res = resp.json()
    assert res.get("success") is False
    assert "No submission CSV payload detected" in res.get("error", "")

def test_submit_non_csv_extension():
    """Verify non-CSV files (e.g. .txt, .zip) are rejected with 400."""
    data = {"team_name": "TestTeam", "team_email": "test@nitjsr.ac.in"}
    files = {"submission_file": ("exploit.txt", io.BytesIO(b"dummy payload"), "text/plain")}
    resp = requests.post(f"{BASE_URL}/api/submit", data=data, files=files, timeout=15)
    assert resp.status_code == 400
    res = resp.json()
    assert res.get("success") is False
    assert "Payload format rejected" in res.get("error", "")

def test_submit_missing_team_or_email():
    """Verify missing team name or email returns 400."""
    csv_bytes = io.BytesIO(b"candidate_id,target_offered_ctc\n")
    files = {"submission_file": ("test.csv", csv_bytes, "text/csv")}
    resp1 = requests.post(f"{BASE_URL}/api/submit", data={"team_name": "", "team_email": "test@nitjsr.ac.in"}, files=files, timeout=15)
    assert resp1.status_code == 400
    assert "Team Identifier and Student Email are required" in resp1.json().get("error", "")

    csv_bytes.seek(0)
    files = {"submission_file": ("test.csv", csv_bytes, "text/csv")}
    resp2 = requests.post(f"{BASE_URL}/api/submit", data={"team_name": "TestTeam", "team_email": ""}, files=files, timeout=15)
    assert resp2.status_code == 400
    assert "Team Identifier and Student Email are required" in resp2.json().get("error", "")

def test_submit_corrupted_headers():
    """Verify CSV with wrong/corrupted column headers is rejected with 400."""
    csv_data = "student_id,salary\nNITJ_CC_200001,15.5\n"
    files = {"submission_file": ("sub.csv", io.BytesIO(csv_data.encode()), "text/csv")}
    data = {"team_name": "HeaderTestTeam", "team_email": "headertest@nitjsr.ac.in"}
    resp = requests.post(f"{BASE_URL}/api/submit", data=data, files=files, timeout=15)
    assert resp.status_code == 400
    res = resp.json()
    assert res.get("success") is False
    assert "CSV header rejected" in res.get("error", "")

def test_submit_missing_target_column():
    """Verify CSV missing required target_offered_ctc column is rejected with 400."""
    csv_data = "candidate_id\nNITJ_CC_200001\n"
    files = {"submission_file": ("sub.csv", io.BytesIO(csv_data.encode()), "text/csv")}
    data = {"team_name": "ColTestTeam", "team_email": "coltest@nitjsr.ac.in"}
    resp = requests.post(f"{BASE_URL}/api/submit", data=data, files=files, timeout=15)
    assert resp.status_code == 400
    res = resp.json()
    assert res.get("success") is False
    assert "CSV header rejected" in res.get("error", "")

def test_submit_nan_and_corrupted_values():
    """Verify CSV with NaN, null, or string values in target column is rejected with 400."""
    csv_data = "candidate_id,target_offered_ctc\nNITJ_CC_200001,NaN\nNITJ_CC_200002,NULL\nNITJ_CC_200003,abc\n"
    files = {"submission_file": ("sub.csv", io.BytesIO(csv_data.encode()), "text/csv")}
    data = {"team_name": "NaNTestTeam", "team_email": "nantest@nitjsr.ac.in"}
    resp = requests.post(f"{BASE_URL}/api/submit", data=data, files=files, timeout=15)
    assert resp.status_code == 400
    res = resp.json()
    assert res.get("success") is False
    assert "NaN, null, or non-numeric values detected" in res.get("error", "")

def test_submit_wrong_row_count():
    """Verify CSV with insufficient rows (e.g. 5 instead of 100,000) is rejected with 400."""
    csv_data = "candidate_id,target_offered_ctc\n" + "\n".join([f"NITJ_CC_{200001+i},12.5" for i in range(5)])
    files = {"submission_file": ("sub.csv", io.BytesIO(csv_data.encode()), "text/csv")}
    data = {"team_name": "RowCountTestTeam", "team_email": "rowcount@nitjsr.ac.in"}
    resp = requests.post(f"{BASE_URL}/api/submit", data=data, files=files, timeout=15)
    assert resp.status_code == 400
    res = resp.json()
    assert res.get("success") is False
    assert "Row count / ID mismatch" in res.get("error", "")


# =========================================================================
# 2. SQL INJECTION / XSS & BOUNDARY TESTS IN TEAM NAMES
# =========================================================================

def test_boundary_excessive_lengths():
    """Verify team_name > 60 chars or email > 100 chars are rejected with 400."""
    csv_bytes = io.BytesIO(b"dummy")
    files = {"submission_file": ("sub.csv", csv_bytes, "text/csv")}
    
    # Team name > 60 chars
    long_team = "A" * 61
    resp = requests.post(f"{BASE_URL}/api/submit", data={"team_name": long_team, "team_email": "test@nitjsr.ac.in"}, files=files, timeout=15)
    assert resp.status_code == 400
    assert "Input length exceeds allowed boundaries" in resp.json().get("error", "")

    # Team email > 100 chars
    csv_bytes.seek(0)
    files = {"submission_file": ("sub.csv", csv_bytes, "text/csv")}
    long_email = ("b" * 90) + "@nitjsr.ac.in"
    resp = requests.post(f"{BASE_URL}/api/submit", data={"team_name": "ValidTeam", "team_email": long_email}, files=files, timeout=15)
    assert resp.status_code == 400
    assert "Input length exceeds allowed boundaries" in resp.json().get("error", "")

def test_sqli_payloads_in_team_name():
    """Verify SQL injection strings in team_name do not cause server crash / 500 error."""
    sqli_payloads = [
        "' OR '1'='1",
        "'; DROP TABLE submissions; --",
        "' UNION SELECT 1,2,3,4,5,6,7,8,9,10,11,12 --",
        "admin'--",
    ]
    for payload in sqli_payloads:
        # Send with malformed CSV so it rejects gracefully at CSV check or WAF/SQL check without crashing
        csv_data = "candidate_id,target_offered_ctc\nNITJ_CC_200001,12.5\n"
        files = {"submission_file": ("sub.csv", io.BytesIO(csv_data.encode()), "text/csv")}
        data = {"team_name": payload, "team_email": "sqli@nitjsr.ac.in"}
        resp = requests.post(f"{BASE_URL}/api/submit", data=data, files=files, timeout=15)
        # Should be blocked by WAF (403) or rejected by app validation (400), NEVER 500
        assert resp.status_code in [400, 403], f"Payload '{payload}' returned unexpected code {resp.status_code}: {resp.text}"

    # Verify leaderboard endpoint remains operational and database is uncorrupted
    lb_resp = requests.get(f"{BASE_URL}/api/leaderboard", timeout=15)
    assert lb_resp.status_code == 200
    assert "leaderboard" in lb_resp.json()

def test_xss_payload_in_team_name():
    """Verify XSS strings in team_name are handled without server crash."""
    xss_payload = "<script>alert('xss')</script>"
    csv_data = "candidate_id,target_offered_ctc\nNITJ_CC_200001,12.5\n"
    files = {"submission_file": ("sub.csv", io.BytesIO(csv_data.encode()), "text/csv")}
    data = {"team_name": xss_payload, "team_email": "xss@nitjsr.ac.in"}
    resp = requests.post(f"{BASE_URL}/api/submit", data=data, files=files, timeout=15)
    assert resp.status_code in [400, 403], f"XSS payload caused unexpected code {resp.status_code}: {resp.text}"

def test_file_download_whitelist_security():
    """Verify path traversal and unauthorized files cannot be downloaded."""
    forbidden_targets = [
        "../web_app/app.py",
        "../../etc/passwd",
        "competition_vault.db",
        "server_ground_truth.csv",
        "../backend_evaluation/server_ground_truth.csv",
        "Dockerfile",
        "render.yaml"
    ]
    for target in forbidden_targets:
        resp = requests.get(f"{BASE_URL}/download/{target}", timeout=15)
        assert resp.status_code in [403, 404], f"Forbidden file '{target}' was not blocked! Status: {resp.status_code}"


# =========================================================================
# 3. ADMIN REVEAL & EXPORT AUTHENTICATION SECURITY
# =========================================================================

def test_admin_toggle_invalid_pins():
    """Verify invalid admin PINs are rejected with 403 Forbidden on POST /api/admin/toggle-private."""
    invalid_pins = [
        "",
        "wrong_pin",
        "1234",
        "admin",
        "' OR '1'='1",
        "acc2026",  # wrong case
        "ACC2025"
    ]
    for pin in invalid_pins:
        resp = requests.post(f"{BASE_URL}/api/admin/toggle-private", json={"pin": pin}, timeout=15)
        assert resp.status_code == 403, f"PIN '{pin}' was not rejected! Status: {resp.status_code}"
        res = resp.json()
        assert res.get("success") is False
        assert "Security clearance denied: Invalid Admin PIN" in res.get("error", "")

def test_admin_toggle_missing_body():
    """Verify missing JSON body or null pin returns 403 or 415."""
    resp1 = requests.post(f"{BASE_URL}/api/admin/toggle-private", json={}, timeout=15)
    assert resp1.status_code == 403
    resp2 = requests.post(f"{BASE_URL}/api/admin/toggle-private", data="", timeout=15)
    assert resp2.status_code in [403, 415]

def test_admin_export_archive_unauthorized():
    """Verify GET /api/admin/export-archive rejects unauthorized requests with 403."""
    # No pin
    resp = requests.get(f"{BASE_URL}/api/admin/export-archive", timeout=15)
    assert resp.status_code == 403
    # Wrong pin
    resp = requests.get(f"{BASE_URL}/api/admin/export-archive?pin=invalid", timeout=15)
    assert resp.status_code == 403

def test_admin_toggle_valid_pin_lifecycle():
    """Verify valid PIN ACC2026 toggles private scores on and back off cleanly."""
    # Check initial state
    init_lb = requests.get(f"{BASE_URL}/api/leaderboard", timeout=15).json()
    orig_state = init_lb.get("is_private_revealed", False)

    # Toggle state
    resp1 = requests.post(f"{BASE_URL}/api/admin/toggle-private", json={"pin": ADMIN_PIN}, timeout=15)
    assert resp1.status_code == 200
    res1 = resp1.json()
    assert res1.get("success") is True
    assert res1.get("is_private_revealed") == (not orig_state)

    # Verify leaderboard reflects the toggled state
    toggled_lb = requests.get(f"{BASE_URL}/api/leaderboard", timeout=15).json()
    assert toggled_lb.get("is_private_revealed") == (not orig_state)

    # Toggle back to original state
    resp2 = requests.post(f"{BASE_URL}/api/admin/toggle-private", json={"pin": ADMIN_PIN}, timeout=15)
    assert resp2.status_code == 200
    res2 = resp2.json()
    assert res2.get("success") is True
    assert res2.get("is_private_revealed") == orig_state

    # Verify leaderboard is back to original
    final_lb = requests.get(f"{BASE_URL}/api/leaderboard", timeout=15).json()
    assert final_lb.get("is_private_revealed") == orig_state


# =========================================================================
# 4. SERVER STABILITY UNDER CONCURRENT REQUESTS
# =========================================================================

def test_concurrent_request_stability():
    """Verify server stability under 100 concurrent requests across multiple endpoints."""
    endpoints = [
        "/",
        "/api/leaderboard",
        "/download/sample_submission.csv"
    ]
    num_threads = 20
    requests_per_thread = 5
    total_requests = num_threads * requests_per_thread

    def worker_task(worker_id):
        results = []
        for i in range(requests_per_thread):
            target = endpoints[(worker_id + i) % len(endpoints)]
            start_time = time.time()
            try:
                # Use HEAD or GET
                r = requests.get(f"{BASE_URL}{target}", timeout=10)
                dur = time.time() - start_time
                results.append((r.status_code, dur))
            except Exception as e:
                results.append((0, str(e)))
        return results

    start_all = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker_task, wid) for wid in range(num_threads)]
        all_results = []
        for f in concurrent.futures.as_completed(futures):
            all_results.extend(f.result())
    total_elapsed = time.time() - start_all

    status_codes = [r[0] for r in all_results]
    durations = [r[1] for r in all_results if isinstance(r[1], (int, float))]

    success_count = sum(1 for c in status_codes if c == 200)
    assert success_count == total_requests, f"Concurrent failure! Only {success_count}/{total_requests} succeeded. Codes: {status_codes}"

    p50 = sorted(durations)[len(durations) // 2]
    p95 = sorted(durations)[int(len(durations) * 0.95)]
    print(f"\nConcurrency Results: {total_requests} requests in {total_elapsed:.2f}s | p50={p50:.3f}s, p95={p95:.3f}s")
