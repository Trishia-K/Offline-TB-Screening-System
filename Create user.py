# backup way to add a staff account from the terminal
# normally you add staff on the Staff page inside the app
# run app.py at least once first (it creates the database), then:
#   python create_user.py
#
# roles:
#   screener - can run screenings and register patients
#   records  - can also open past patient results

import getpass
import os
import sqlite3

from werkzeug.security import generate_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tb_screening.db")

if not os.path.exists(DB_PATH):
    print("No database yet. Run python app.py once first, then try again.")
    raise SystemExit

name = input("Full name: ").strip()
email = input("Email: ").strip().lower()
role = input("Role (screener / records): ").strip().lower()
password = getpass.getpass("Password (typing is hidden): ")

if role not in ("screener", "records"):
    print("Role has to be 'screener' or 'records'.")
    raise SystemExit
if len(password) < 8:
    print("Use at least 8 characters for the password.")
    raise SystemExit

conn = sqlite3.connect(DB_PATH)
try:
    conn.execute(
        "INSERT INTO users (email, name, password_hash, role) VALUES (?, ?, ?, ?)",
        (email, name, generate_password_hash(password), role),
    )
    conn.commit()
    print(f"Added {name} ({email}) as {role}.")
except sqlite3.IntegrityError:
    print("There's already an account with that email.")
finally:
    conn.close()