import sys
import os
import io
import pytest
import sqlite3

# Ensure web_app is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "web_app"))

from app import app, get_db

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

# =========================================================================
# 1. P0: CSV HEADER DISCREPANCY VERIFICATION
# =========================================================================

def test_p0_csv_headers_in_html():
    """Verify that index.html specifies candidate_id,target_offered_ctc and has no trace of ID,Placed_CTC_LPA."""
    html_path = os.path.join(BASE_DIR, "web_app", "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    assert "candidate_id,target_offered_ctc" in html
    assert "candidate_id, target_offered_ctc" in html
    assert "ID,Placed_CTC_LPA" not in html
    assert "ID, Placed_CTC_LPA" not in html


# =========================================================================
# 2. P1: LEADERBOARD SCORE DECOUPLING VERIFICATION
# =========================================================================

def test_p1_leaderboard_no_score_decoupling(client):
    """
    Verify that in /api/leaderboard:
    - best_public_rmse and best_public_mae match the exact submission with the highest public_r2.
    - They are NOT decoupled (e.g. MIN(rmse) from one sub and MAX(r2) from another sub).
    """
    test_team = "DecouplingAuditTeam"
    test_email = "decouple@nitjsr.ac.in"

    with get_db() as conn:
        conn.execute("DELETE FROM submissions WHERE team_name = ?", (test_team,))
        # Submission 1: Higher public_r2 (0.85), but worse (higher) RMSE (2.5)
        conn.execute("""
            INSERT INTO submissions (
                team_name, team_email, submission_filename, notebook_filename,
                public_r2, public_rmse, public_mae,
                private_r2, private_rmse, private_mae,
                timestamp, date_str
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_team, test_email, "sub1.csv", None,
            0.8500, 2.5000, 1.8000,
            0.7000, 3.0000, 2.2000,
            "2026-11-01 10:00:00 UTC", "2026-11-01"
        ))
        # Submission 2: Lower public_r2 (0.75), but better (lower) RMSE (1.2)
        conn.execute("""
            INSERT INTO submissions (
                team_name, team_email, submission_filename, notebook_filename,
                public_r2, public_rmse, public_mae,
                private_r2, private_rmse, private_mae,
                timestamp, date_str
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_team, test_email, "sub2.csv", None,
            0.7500, 1.2000, 0.9000,
            0.9000, 0.8000, 0.5000,
            "2026-11-01 11:00:00 UTC", "2026-11-01"
        ))
        conn.commit()

    # Public Leaderboard Check
    resp = client.get("/api/leaderboard")
    assert resp.status_code == 200
    data = resp.get_json()
    team_entry = next((item for item in data["leaderboard"] if item["team_name"] == test_team), None)
    assert team_entry is not None

    # Crucial assertion: best_public_r2 MUST be 0.8500 AND best_public_rmse MUST be 2.5000
    # If decoupled, best_public_rmse would incorrectly be 1.2000!
    assert team_entry["best_public_r2"] == 0.8500
    assert team_entry["best_public_rmse"] == 2.5000
    assert team_entry["best_public_mae"] == 1.8000
    assert team_entry["total_submissions"] == 2

    # Clean up
    with get_db() as conn:
        conn.execute("DELETE FROM submissions WHERE team_name = ?", (test_team,))
        conn.commit()


# =========================================================================
# 3. FEATURE/AUTH: TEAM LEADER ROLL NUMBER VERIFICATION
# =========================================================================

def test_auth_registered_team_success(client):
    """Verify that a registered team leader with valid roll number can authenticate."""
    data = {
        "team_name": "GradientPioneers",
        "team_email": "pioneers@nitjsr.ac.in",
        "leader_roll_no": "2024ugpi059"
    }
    # Send without CSV file to test that auth check succeeds and moves to file payload check (400)
    resp = client.post("/api/submit", data=data)
    assert resp.status_code == 400
    assert "No submission CSV payload detected" in resp.get_json().get("error", "")

def test_auth_case_and_whitespace_insensitivity(client):
    """Verify normalization with .strip().lower() for team_name, email, and roll number."""
    data = {
        "team_name": "  gradientpioneers  ",
        "team_email": " PIONEERS@NITJSR.AC.IN  ",
        "leader_roll_no": " 2024UGPI059 "
    }
    resp = client.post("/api/submit", data=data)
    # Auth must succeed and fail only at missing CSV payload
    assert resp.status_code == 400
    assert "No submission CSV payload detected" in resp.get_json().get("error", "")

def test_auth_wrong_roll_number(client):
    """Verify that wrong leader roll number returns 401."""
    data = {
        "team_name": "GradientPioneers",
        "team_email": "pioneers@nitjsr.ac.in",
        "leader_roll_no": "2024ugcs999"  # Incorrect roll number
    }
    resp = client.post("/api/submit", data=data)
    assert resp.status_code == 401
    res = resp.get_json()
    assert res.get("success") is False
    assert "Verification failed: Team Name, Leader Email, or Leader Roll Number does not match registered records." in res.get("error", "")

def test_auth_missing_roll_number(client):
    """Verify that missing leader roll number returns 401."""
    data = {
        "team_name": "GradientPioneers",
        "team_email": "pioneers@nitjsr.ac.in",
        "leader_roll_no": ""
    }
    resp = client.post("/api/submit", data=data)
    assert resp.status_code == 401
    assert "Verification failed: Team Name, Leader Email, or Leader Roll Number does not match registered records." in resp.get_json().get("error", "")

def test_auth_unregistered_team(client):
    """Verify that an unregistered team returns 401."""
    data = {
        "team_name": "UnregisteredRogueTeam",
        "team_email": "rogue@nitjsr.ac.in",
        "leader_roll_no": "2024ugpi059"
    }
    resp = client.post("/api/submit", data=data)
    assert resp.status_code == 401
    assert "Verification failed: Team Name, Leader Email, or Leader Roll Number does not match registered records." in resp.get_json().get("error", "")

def test_auth_inactive_team(client):
    """Verify that an inactive team (is_active = 0) returns 401."""
    inactive_team = "InactiveBannedTeam"
    with get_db() as conn:
        conn.execute("DELETE FROM registered_teams WHERE team_name = ?", (inactive_team,))
        conn.execute("""
            INSERT INTO registered_teams (team_name, leader_name, leader_email, leader_roll_no, is_active)
            VALUES (?, ?, ?, ?, 0)
        """, (inactive_team, "Banned Leader", "banned@nitjsr.ac.in", "2024ugxx001"))
        conn.commit()

    data = {
        "team_name": inactive_team,
        "team_email": "banned@nitjsr.ac.in",
        "leader_roll_no": "2024ugxx001"
    }
    resp = client.post("/api/submit", data=data)
    assert resp.status_code == 401
    assert "Verification failed" in resp.get_json().get("error", "")

    with get_db() as conn:
        conn.execute("DELETE FROM registered_teams WHERE team_name = ?", (inactive_team,))
        conn.commit()


# =========================================================================
# 4. P2: ACCESSIBILITY & MOBILE NAVIGATION VERIFICATION
# =========================================================================

def test_p2_accessibility_and_mobile_nav():
    """Verify all form inputs have labels or aria-labels, modals have aria-label=Close, and mobile nav has min-h-[44px]."""
    html_path = os.path.join(BASE_DIR, "web_app", "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Mobile nav touch targets
    assert 'min-h-[44px]' in html
    assert 'min-w-[44px]' in html
    assert 'id="mobile-nav-leaderboard"' in html
    assert 'id="mobile-nav-submit"' in html
    assert 'id="mobile-nav-downloads"' in html
    assert 'id="mobile-nav-protocol"' in html

    # Form labels & associations
    assert 'for="field-team-name"' in html
    assert 'for="field-team-email"' in html
    assert 'for="field-leader-roll-no"' in html
    assert 'for="field-file-csv"' in html
    assert 'for="field-file-nb"' in html
    assert 'for="input-admin-pin"' in html
    assert 'aria-label="Search teams by name"' in html
    assert 'aria-label="Organizer Secret PIN"' in html

    # Modal close buttons
    assert 'onclick="closeReceiptModal()" aria-label="Close"' in html
    assert 'onclick="closeAdminModal()" aria-label="Close"' in html


# =========================================================================
# 5. P3: STARS CANVAS REDUCED MOTION & VISIBILITY PAUSE VERIFICATION
# =========================================================================

def test_p3_canvas_animation_pause_and_reduced_motion():
    """Verify canvas animation pauses on document.hidden or prefers-reduced-motion."""
    html_path = os.path.join(BASE_DIR, "web_app", "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    assert "prefers-reduced-motion: reduce" in html
    assert "document.hidden || prefersReducedMotion" in html
    assert "cancelAnimationFrame" in html
    assert "visibilitychange" in html


# =========================================================================
# 6. END-TO-END VALID SUBMISSION & LEADERBOARD INTEGRATION
# =========================================================================

def test_end_to_end_valid_submission_flow(client):
    """Verify that a registered team can submit a valid 100,000-row CSV and appear on leaderboard."""
    sample_sub_path = os.path.join(BASE_DIR, "participant_download", "sample_submission.csv")
    if not os.path.exists(sample_sub_path):
        sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")

    assert os.path.exists(sample_sub_path), "sample_submission.csv must exist for testing"

    team_name = "GradientPioneers"
    team_email = "pioneers@nitjsr.ac.in"
    leader_roll = "2024ugpi059"

    with get_db() as conn:
        conn.execute("DELETE FROM submissions WHERE LOWER(team_name) = LOWER(?)", (team_name,))
        conn.commit()

    with open(sample_sub_path, "rb") as f:
        file_bytes = f.read()

    data = {
        "team_name": team_name,
        "team_email": team_email,
        "leader_roll_no": leader_roll,
        "submission_file": (io.BytesIO(file_bytes), "sample_submission.csv", "text/csv")
    }

    resp = client.post("/api/submit", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200, f"Submission failed with status {resp.status_code}: {resp.text}"
    result = resp.get_json()
    assert result.get("success") is True
    assert result.get("team_name") == team_name
    assert "public_r2" in result
    assert "public_rmse" in result
    assert "public_mae" in result

    # Check that leaderboard returns this team's exact scores
    lb_resp = client.get("/api/leaderboard")
    assert lb_resp.status_code == 200
    lb_data = lb_resp.get_json()
    pioneers_entry = next((item for item in lb_data["leaderboard"] if item["team_name"] == team_name), None)
    assert pioneers_entry is not None
    assert pioneers_entry["best_public_r2"] == result["public_r2"]
    assert pioneers_entry["best_public_rmse"] == result["public_rmse"]


# =========================================================================
# 7. ADVANCED AUDIT: CANONICAL NORMALIZATION & ADMIN TEAM REGISTRATION
# =========================================================================

def test_canonical_team_name_and_casing_deduplication(client):
    """
    Verify that submissions with varied casing:
    1. Are mapped to canonical team_name from registered_teams.
    2. Are grouped into a single team row on the leaderboard rather than duplicate rows.
    """
    test_team = "CaseTestTeam"
    with get_db() as conn:
        conn.execute("DELETE FROM submissions WHERE LOWER(team_name) = LOWER(?)", (test_team,))
        conn.execute("DELETE FROM registered_teams WHERE LOWER(team_name) = LOWER(?)", (test_team,))
        conn.execute("""
            INSERT INTO registered_teams (team_name, leader_name, leader_email, leader_roll_no, is_active)
            VALUES (?, ?, ?, ?, 1)
        """, (test_team, "Case Leader", "case@nitjsr.ac.in", "2024ugcs888"))
        # Insert two submissions directly with different casing and valid private metrics
        conn.execute("""
            INSERT INTO submissions (
                team_name, team_email, submission_filename,
                public_r2, public_rmse, public_mae,
                private_r2, private_rmse, private_mae,
                timestamp, date_str
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (test_team.upper(), "case@nitjsr.ac.in", "s1.csv", 0.60, 2.0, 1.5, 0.55, 2.2, 1.7, "2026-11-01 10:00:00 UTC", "2026-11-01"))
        conn.execute("""
            INSERT INTO submissions (
                team_name, team_email, submission_filename,
                public_r2, public_rmse, public_mae,
                private_r2, private_rmse, private_mae,
                timestamp, date_str
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (test_team.lower(), "case@nitjsr.ac.in", "s2.csv", 0.88, 1.1, 0.8, 0.85, 1.2, 0.9, "2026-11-01 11:00:00 UTC", "2026-11-01"))
        conn.commit()

    resp = client.get("/api/leaderboard")
    assert resp.status_code == 200
    lb = resp.get_json()["leaderboard"]
    matching_entries = [item for item in lb if item["team_name"].lower() == test_team.lower()]
    # Must be unified into exactly ONE row
    assert len(matching_entries) == 1, f"Expected 1 grouped entry, got {len(matching_entries)}"
    assert matching_entries[0]["total_submissions"] == 2
    assert matching_entries[0]["best_public_r2"] == 0.88

    # Clean up
    with get_db() as conn:
        conn.execute("DELETE FROM submissions WHERE LOWER(team_name) = LOWER(?)", (test_team,))
        conn.execute("DELETE FROM registered_teams WHERE LOWER(team_name) = LOWER(?)", (test_team,))
        conn.commit()


def test_extreme_negative_r2_sorting(client):
    """Verify that models with extreme negative R2 (e.g. -1500) are ranked correctly below better models."""
    team_a = "ExtremeModelTeamA"
    team_b = "ExtremeModelTeamB"
    with get_db() as conn:
        conn.execute("DELETE FROM submissions WHERE team_name IN (?, ?)", (team_a, team_b))
        conn.execute("""
            INSERT INTO submissions (
                team_name, team_email, submission_filename,
                public_r2, public_rmse, public_mae,
                private_r2, private_rmse, private_mae,
                timestamp, date_str
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (team_a, "ea@nitjsr.ac.in", "ea.csv", -1500.0, 50.0, 35.0, -1500.0, 50.0, 35.0, "2026-11-01 10:00:00 UTC", "2026-11-01"))
        conn.execute("""
            INSERT INTO submissions (
                team_name, team_email, submission_filename,
                public_r2, public_rmse, public_mae,
                private_r2, private_rmse, private_mae,
                timestamp, date_str
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (team_b, "eb@nitjsr.ac.in", "eb.csv", -2.0, 10.0, 8.0, -2.0, 10.0, 8.0, "2026-11-01 10:00:00 UTC", "2026-11-01"))
        conn.commit()

    resp = client.get("/api/leaderboard")
    assert resp.status_code == 200
    lb = resp.get_json()["leaderboard"]
    entry_a = next(item for item in lb if item["team_name"] == team_a)
    entry_b = next(item for item in lb if item["team_name"] == team_b)

    # Team B (R2 = -2.0) must rank HIGHER (lower rank number) than Team A (R2 = -1500.0)
    assert entry_b["rank"] < entry_a["rank"]

    # Clean up
    with get_db() as conn:
        conn.execute("DELETE FROM submissions WHERE team_name IN (?, ?)", (team_a, team_b))
        conn.commit()


def test_admin_register_team_api(client):
    """Verify POST /api/admin/register-team security and functional registration."""
    new_team = "AdminRegisteredSquad"
    new_email = "squad@nitjsr.ac.in"
    new_roll = "2024ugmm077"

    # Wrong PIN
    bad_resp = client.post("/api/admin/register-team", json={
        "pin": "WRONGPIN",
        "team_name": new_team,
        "leader_email": new_email,
        "leader_roll_no": new_roll
    })
    assert bad_resp.status_code == 403

    # Valid PIN
    good_resp = client.post("/api/admin/register-team", json={
        "pin": "MANNU",
        "team_name": new_team,
        "leader_name": "Squad Leader",
        "leader_email": new_email,
        "leader_roll_no": new_roll
    })
    assert good_resp.status_code == 200
    assert good_resp.get_json()["success"] is True

    # Now verify that this newly registered team can authenticate on /api/submit
    submit_check = client.post("/api/submit", data={
        "team_name": new_team,
        "team_email": new_email,
        "leader_roll_no": new_roll
    })
    # Should authenticate and fail only at missing CSV (400), not auth rejection (401)
    assert submit_check.status_code == 400
    assert "No submission CSV payload detected" in submit_check.get_json()["error"]

    # Clean up
    with get_db() as conn:
        conn.execute("DELETE FROM registered_teams WHERE team_name = ?", (new_team,))
        conn.commit()


def test_google_form_csv_import_script():
    """Verify that import_teams_csv.py can parse a Google Form CSV export and register teams."""
    import tempfile
    import subprocess

    csv_content = """Timestamp,Email Address,Team Name,Team Leader Name,Leader Roll Number
2026/11/02 10:00:00 AM,scriptleader@nitjsr.ac.in,ScriptImportedTeam,Script Leader,2024ugce011
"""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        temp_csv = f.name

    try:
        script_path = os.path.join(BASE_DIR, "scripts", "import_teams_csv.py")
        db_path = os.path.join(BASE_DIR, "web_app", "competition_vault.db")
        result = subprocess.run([sys.executable, script_path, temp_csv, "--db", db_path], capture_output=True, text=True)
        assert result.returncode == 0
        assert "registered" in result.stdout

        # Verify team is in database
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT leader_roll_no FROM registered_teams WHERE team_name = 'ScriptImportedTeam'")
            row = cur.fetchone()
            assert row is not None
            assert row["leader_roll_no"] == "2024ugce011"
            conn.execute("DELETE FROM registered_teams WHERE team_name = 'ScriptImportedTeam'")
            conn.commit()
    finally:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)


