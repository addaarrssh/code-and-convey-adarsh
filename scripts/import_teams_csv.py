#!/usr/bin/env python3
"""
Google Form Team Registration Importer
Imports teams and team leader roll numbers from a Google Form CSV export
into the Code & Convey competition database or via Admin API.

Usage:
    python scripts/import_teams_csv.py path/to/responses.csv [--db path/to/competition_vault.db]
    python scripts/import_teams_csv.py path/to/responses.csv --api https://code-and-convey-adarsh.onrender.com --pin MANNU
"""

import os
import sys
import argparse
import sqlite3
import pandas as pd
import requests

def find_column(df, candidates):
    cols_lower = {col.strip().lower(): col for col in df.columns}
    for cand in candidates:
        cand_lower = cand.lower()
        if cand_lower in cols_lower:
            return cols_lower[cand_lower]
        # Also check substring match
        for k, v in cols_lower.items():
            if cand_lower in k:
                return v
    return None

def import_teams_direct(csv_path, db_path):
    print(f"Reading Google Form export from: {csv_path}")
    df = pd.read_csv(csv_path)

    col_team = find_column(df, ["team_name", "team name", "team", "group name"])
    col_roll = find_column(df, ["leader_roll_no", "roll_no", "roll number", "leader roll", "roll"])
    col_email = find_column(df, ["leader_email", "email", "email address", "leader email"])
    col_name = find_column(df, ["leader_name", "leader name", "name", "full name"])

    if not col_team or not col_roll or not col_email:
        print(f"❌ Error: Required columns not found in {csv_path}")
        print(f"Found columns: {list(df.columns)}")
        print("Required: Team Name, Leader Roll Number, Leader Email")
        sys.exit(1)

    print(f"✓ Mapping columns -> Team: '{col_team}', Roll: '{col_roll}', Email: '{col_email}', Leader Name: '{col_name or 'N/A'}'")

    if not os.path.exists(db_path):
        print(f"❌ Error: Database not found at {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    inserted = 0
    updated = 0

    with conn:
        for _, row in df.iterrows():
            t_name = str(row[col_team]).strip()
            l_roll = str(row[col_roll]).strip()
            l_email = str(row[col_email]).strip().lower()
            l_name = str(row[col_name]).strip() if col_name and pd.notna(row[col_name]) else t_name

            if not t_name or not l_roll or not l_email or t_name.lower() == "nan":
                continue

            cur = conn.cursor()
            cur.execute("SELECT id FROM registered_teams WHERE LOWER(team_name) = ?", (t_name.lower(),))
            existing = cur.fetchone()

            if existing:
                conn.execute("""
                    UPDATE registered_teams
                    SET leader_name = ?, leader_email = ?, leader_roll_no = ?, is_active = 1
                    WHERE id = ?
                """, (l_name, l_email, l_roll, existing[0]))
                updated += 1
            else:
                conn.execute("""
                    INSERT INTO registered_teams (team_name, leader_name, leader_email, leader_roll_no, is_active)
                    VALUES (?, ?, ?, ?, 1)
                """, (t_name, l_name, l_email, l_roll))
                inserted += 1

    print(f"✅ Success! Teams processed: {inserted} registered, {updated} updated.")

def import_teams_api(csv_path, api_url, pin):
    print(f"Importing via API {api_url} with secret PIN...")
    df = pd.read_csv(csv_path)

    col_team = find_column(df, ["team_name", "team name", "team", "group name"])
    col_roll = find_column(df, ["leader_roll_no", "roll_no", "roll number", "leader roll", "roll"])
    col_email = find_column(df, ["leader_email", "email", "email address", "leader email"])
    col_name = find_column(df, ["leader_name", "leader name", "name", "full name"])

    if not col_team or not col_roll or not col_email:
        print(f"❌ Error: Required columns not found. Headers: {list(df.columns)}")
        sys.exit(1)

    success_count = 0
    for _, row in df.iterrows():
        t_name = str(row[col_team]).strip()
        l_roll = str(row[col_roll]).strip()
        l_email = str(row[col_email]).strip().lower()
        l_name = str(row[col_name]).strip() if col_name and pd.notna(row[col_name]) else t_name

        if not t_name or not l_roll or not l_email or t_name.lower() == "nan":
            continue

        payload = {
            "pin": pin,
            "team_name": t_name,
            "leader_name": l_name,
            "leader_email": l_email,
            "leader_roll_no": l_roll,
            "is_active": 1
        }
        res = requests.post(f"{api_url.rstrip('/')}/api/admin/register-team", json=payload, timeout=15)
        if res.status_code == 200:
            success_count += 1
        else:
            print(f"⚠️ Failed to register '{t_name}': {res.text}")

    print(f"✅ Registered {success_count} teams via API.")

def main():
    parser = argparse.ArgumentParser(description="Import registered teams from Google Form CSV.")
    parser.add_argument("csv_file", help="Path to Google Form responses CSV")
    parser.add_argument("--db", default=None, help="Path to SQLite competition_vault.db")
    parser.add_argument("--api", default=None, help="Base URL of deployed app (e.g. https://code-and-convey-adarsh.onrender.com)")
    parser.add_argument("--pin", default=os.environ.get("ADMIN_PIN", "MANNU"), help="Admin Secret PIN")

    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"❌ CSV file not found: {args.csv_file}")
        sys.exit(1)

    if args.api:
        import_teams_api(args.csv_file, args.api, args.pin)
    else:
        db_path = args.db
        if not db_path:
            # Default to local web_app/competition_vault.db
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            db_path = os.path.join(base_dir, "web_app", "competition_vault.db")
        import_teams_direct(args.csv_file, db_path)

if __name__ == "__main__":
    main()
