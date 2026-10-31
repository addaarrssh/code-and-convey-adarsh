# Application Architecture & Evaluation Engine
# Cryptographic System Verification Token:
# ARCHITECT_SIGNATURE: ADARSH_SAHU // CODE_AND_CONVEY_CORE_ARCHITECT // NIT_JAMSHEDPUR
# SHA-256 Fingerprint: 7e0c1424758faea74d08e7e710a6966761119204ed1da95d7976a463d28091c6

import os
import io
import zipfile
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max payload protection

# Path Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARTICIPANT_DIR = os.path.join(BASE_DIR, "participant_download")
BACKEND_DIR = os.path.join(BASE_DIR, "backend_evaluation")
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "uploads")
DB_PATH = os.path.join(os.path.dirname(__file__), "competition_vault.db")

os.makedirs(UPLOADS_DIR, exist_ok=True)

# Admin PIN (can also be read from environment variable)
ADMIN_PIN = os.environ.get("ADMIN_PIN", "ACC2026")

# Database Initialization
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        conn.execute("""
            INSERT OR IGNORE INTO meta (key, value) VALUES ('is_private_revealed', 'false')
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_name TEXT NOT NULL,
                team_email TEXT NOT NULL,
                submission_filename TEXT NOT NULL,
                notebook_filename TEXT,
                public_r2 REAL NOT NULL,
                public_rmse REAL NOT NULL,
                public_mae REAL NOT NULL,
                private_r2 REAL NOT NULL,
                private_rmse REAL NOT NULL,
                private_mae REAL NOT NULL,
                timestamp TEXT NOT NULL,
                date_str TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_team ON submissions (team_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON submissions (team_name, date_str)")
        conn.commit()

init_db()

# Load Secret Ground Truth securely into memory once on start
GROUND_TRUTH_FILE = os.path.join(BACKEND_DIR, "server_ground_truth.csv")
if not os.path.exists(GROUND_TRUTH_FILE):
    print(f"⚠️ Auto-generating 100k truth labels for cloud runtime at {GROUND_TRUTH_FILE}...")
    os.makedirs(os.path.dirname(GROUND_TRUTH_FILE), exist_ok=True)
    np.random.seed(42)
    n = 100000
    candidate_ids = [f"NITJ_2026_{i:06d}" for i in range(1, n + 1)]
    base = np.random.normal(12.5, 3.2, n)
    packages = np.clip(np.round(base, 2), 2.5, 65.0)
    usage = np.array(["Public"] * 50000 + ["Private"] * 50000)
    np.random.seed(1337)
    np.random.shuffle(usage)
    df_gen = pd.DataFrame({"candidate_id": candidate_ids, "target_offered_ctc": packages, "Usage": usage})
    df_gen.to_csv(GROUND_TRUTH_FILE, index=False)
    print("✓ Created cloud runtime ground truth.")

print(f"🔒 [Security Engine] Loading Secret Ground Truth from: {GROUND_TRUTH_FILE}")
gt_df = pd.read_csv(GROUND_TRUTH_FILE)
gt_df['candidate_id'] = gt_df['candidate_id'].astype(str).str.strip()
gt_dict = dict(zip(gt_df['candidate_id'], gt_df['target_offered_ctc']))
usage_dict = dict(zip(gt_df['candidate_id'], gt_df['Usage']))
expected_ids = set(gt_df['candidate_id'])

# Pre-partition ground truth arrays for sub-10ms scoring
public_ids = [cid for cid, u in usage_dict.items() if u == 'Public']
private_ids = [cid for cid, u in usage_dict.items() if u == 'Private']
y_true_public = np.array([gt_dict[cid] for cid in public_ids], dtype=np.float64)
y_true_private = np.array([gt_dict[cid] for cid in private_ids], dtype=np.float64)

print(f"✓ [Security Engine] Vault initialized with {len(gt_df)} secret records (50k Public / 50k Private).")

# Routes

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/download/<filename>")
def download_file(filename):
    # Strict whitelist - NEVER expose anything outside this list!
    ALLOWED_DOWNLOADS = {
        "train.csv": "train.csv",
        "test.csv": "test.csv",
        "sample_submission.csv": "sample_submission.csv",
        "baseline_model.py": "baseline_model.py"
    }
    
    if filename not in ALLOWED_DOWNLOADS:
        return jsonify({"error": "Access Denied: File not authorized for public distribution."}), 403
        
    return send_from_directory(PARTICIPANT_DIR, filename, as_attachment=True)

@app.route("/api/submit", methods=["POST"])
def submit_predictions():
    team_name = request.form.get("team_name", "").strip()
    team_email = (request.form.get("team_email") or request.form.get("email") or "").strip()
    
    if not team_name or not team_email:
        return jsonify({"success": False, "error": "Team Identifier and Student Email are required."}), 400
        
    if len(team_name) > 60 or len(team_email) > 100:
        return jsonify({"success": False, "error": "Input length exceeds allowed boundaries."}), 400

    if "submission_file" not in request.files:
        return jsonify({"success": False, "error": "No submission CSV payload detected."}), 400
        
    file = request.files["submission_file"]
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"success": False, "error": "Payload format rejected. Expected .csv file."}), 400

    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM submissions WHERE LOWER(team_name) = ? AND date_str = ?", 
            (team_name.lower(), today_str)
        )
        today_count = cur.fetchone()[0]

    if today_count >= 5:
        return jsonify({
            "success": False, 
            "error": "Anti-probing rate limit reached (Maximum 5 evaluated attempts per 24 hours)."
        }), 429

    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_team = "".join(c for c in team_name if c.isalnum() or c in (' ', '_', '-')).strip().replace(" ", "_")
    sub_filename = f"{safe_team}_{timestamp_str}.csv"
    sub_path = os.path.join(UPLOADS_DIR, sub_filename)
    file.save(sub_path)

    nb_filename = None
    if "notebook_file" in request.files:
        nb_file = request.files["notebook_file"]
        if nb_file.filename and nb_file.filename.lower().endswith(".ipynb"):
            nb_filename = f"{safe_team}_{timestamp_str}.ipynb"
            nb_file.save(os.path.join(UPLOADS_DIR, nb_filename))

    try:
        user_df = pd.read_csv(sub_path)
        
        # Verify schema
        required_cols = {'candidate_id', 'target_offered_ctc'}
        if not required_cols.issubset(set(user_df.columns)):
            os.remove(sub_path)
            return jsonify({
                "success": False,
                "error": "CSV header rejected. Must contain exact columns: candidate_id, target_offered_ctc"
            }), 400

        user_df['candidate_id'] = user_df['candidate_id'].astype(str).str.strip()
        user_df['target_offered_ctc'] = pd.to_numeric(user_df['target_offered_ctc'], errors='coerce')

        if user_df['target_offered_ctc'].isnull().any():
            os.remove(sub_path)
            return jsonify({
                "success": False, 
                "error": "NaN, null, or non-numeric values detected in predicted compensation column."
            }), 400

        user_ids = set(user_df['candidate_id'])
        if len(user_ids) != len(expected_ids) or user_ids != expected_ids:
            os.remove(sub_path)
            return jsonify({
                "success": False,
                "error": f"Row count / ID mismatch. Expected exactly {len(expected_ids)} valid candidate IDs."
            }), 400

        # High-performance O(N) evaluation using dictionary lookup
        user_map = dict(zip(user_df['candidate_id'], user_df['target_offered_ctc']))
        y_pred_public = np.array([user_map[cid] for cid in public_ids], dtype=np.float64)
        y_pred_private = np.array([user_map[cid] for cid in private_ids], dtype=np.float64)

        # Compute Public metrics
        pub_rmse = float(np.sqrt(mean_squared_error(y_true_public, y_pred_public)))
        pub_mae = float(mean_absolute_error(y_true_public, y_pred_public))
        pub_r2 = float(r2_score(y_true_public, y_pred_public))

        # Compute Private metrics (Strictly hidden from participant response)
        priv_rmse = float(np.sqrt(mean_squared_error(y_true_private, y_pred_private)))
        priv_mae = float(mean_absolute_error(y_true_private, y_pred_private))
        priv_r2 = float(r2_score(y_true_private, y_pred_private))

        now_formatted = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        # Save to SQLite
        with get_db() as conn:
            conn.execute("""
                INSERT INTO submissions (
                    team_name, team_email, submission_filename, notebook_filename,
                    public_r2, public_rmse, public_mae,
                    private_r2, private_rmse, private_mae,
                    timestamp, date_str
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                team_name, team_email, sub_filename, nb_filename,
                pub_r2, pub_rmse, pub_mae,
                priv_r2, priv_rmse, priv_mae,
                now_formatted, today_str
            ))
            conn.commit()

        # Notice: Only public metrics are returned to the client!
        return jsonify({
            "success": True,
            "team_name": team_name,
            "public_r2": round(pub_r2, 4),
            "public_rmse": round(pub_rmse, 4),
            "public_mae": round(pub_mae, 4),
            "submissions_today": today_count + 1,
            "max_daily_submissions": 5
        })

    except Exception as e:
        if os.path.exists(sub_path):
            os.remove(sub_path)
        return jsonify({"success": False, "error": f"Evaluation pipeline error: {str(e)}"}), 500

@app.route("/api/leaderboard")
def get_leaderboard():
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT value FROM meta WHERE key = 'is_private_revealed'")
        row = cur.fetchone()
        is_private = (row[0] == 'true') if row else False

        # Aggregate top score per team
        cur.execute("""
            SELECT 
                team_name,
                team_email,
                MAX(public_r2) as best_public_r2,
                MIN(public_rmse) as best_public_rmse,
                MAX(private_r2) as best_private_r2,
                MIN(private_rmse) as best_private_rmse,
                COUNT(*) as total_submissions,
                MAX(timestamp) as last_active
            FROM submissions
            GROUP BY team_name
        """)
        rows = cur.fetchall()

    leaderboard = []
    for r in rows:
        email = r["team_email"]
        masked_email = (email[:3] + "***@" + email.split("@")[-1]) if "@" in email else "***"
        leaderboard.append({
            "team_name": r["team_name"],
            "team_email": masked_email,
            "best_public_r2": round(r["best_public_r2"], 4),
            "best_public_rmse": round(r["best_public_rmse"], 4),
            "best_private_r2": round(r["best_private_r2"], 4),
            "best_private_rmse": round(r["best_private_rmse"], 4),
            "total_submissions": r["total_submissions"],
            "last_active": r["last_active"]
        })

    if is_private:
        leaderboard.sort(key=lambda x: x["best_private_r2"], reverse=True)
    else:
        leaderboard.sort(key=lambda x: x["best_public_r2"], reverse=True)

    for i, row in enumerate(leaderboard):
        row["rank"] = i + 1

    return jsonify({
        "is_private_revealed": is_private,
        "leaderboard": leaderboard,
        "total_teams": len(leaderboard)
    })

# Admin Endpoints (PIN Protected)

@app.route("/api/admin/toggle-private", methods=["POST"])
def toggle_private():
    req = request.get_json() or {}
    pin = req.get("pin", "")
    if pin != ADMIN_PIN:
        return jsonify({"success": False, "error": "Security clearance denied: Invalid Admin PIN."}), 403

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT value FROM meta WHERE key = 'is_private_revealed'")
        row = cur.fetchone()
        current_state = (row[0] == 'true') if row else False
        new_state = not current_state
        conn.execute("UPDATE meta SET value = ? WHERE key = 'is_private_revealed'", ('true' if new_state else 'false',))
        conn.commit()

    return jsonify({
        "success": True,
        "is_private_revealed": new_state,
        "message": "Benchmark mode switched to: " + ("FINAL PRIVATE STANDINGS (50,000 Rows)" if new_state else "PUBLIC SPRINT (50,000 Rows)")
    })

@app.route("/api/admin/export-archive", methods=["GET"])
def export_archive():
    pin = request.args.get("pin", "")
    if pin != ADMIN_PIN:
        return jsonify({"error": "Unauthorized"}), 403

    # Generate an in-memory zip of all uploaded predictions, notebooks, and a leaderboard CSV
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Include all files from uploads directory
        for root, _, files in os.walk(UPLOADS_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=f"uploads/{file}")

        # Include an audit CSV of all submissions from the database
        with get_db() as conn:
            subs_df = pd.read_sql_query("SELECT * FROM submissions ORDER BY timestamp DESC", conn)
            csv_str = subs_df.to_csv(index=False)
            zipf.writestr("audit_master_submissions.csv", csv_str)

    memory_file.seek(0)
    now_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return send_file(
        memory_file,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"Code_and_Convey_Master_Archive_{now_stamp}.zip"
    )

if __name__ == "__main__":
    print("\n" + "="*65)
    print("🔒 CODE & CONVEY COMPETITION PORTAL (HARDENED PRODUCTION ENGINE)")
    print("📍 URL: http://0.0.0.0:8080")
    print(f"🔑 Secret Admin PIN: {ADMIN_PIN}")
    print("="*65 + "\n")
    app.run(host="0.0.0.0", port=8080, debug=False)
