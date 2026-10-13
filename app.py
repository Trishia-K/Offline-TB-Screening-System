# TB screening backend
# run with:  python app.py   then open http://localhost:5000

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # must be set BEFORE tensorflow gets imported

import base64
import json
import secrets
import sqlite3
import time
from datetime import datetime, date, timedelta
from functools import wraps

import cv2
import numpy as np
import tensorflow as tf  # noqa: F401  (imported so the env var above takes effect)
import tf_keras as keras
from flask import Flask, request, jsonify, send_from_directory, send_file, session, g
from werkzeug.security import generate_password_hash, check_password_hash


# paths and numbers we might want to tweak

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "mobilenetv2_tb.h5")

# everything we save lives in data/ (database + images)
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "tb_screening.db")
XRAY_DIR = os.path.join(DATA_DIR, "xrays")
HEATMAP_DIR = os.path.join(DATA_DIR, "heatmaps")
for folder in (DATA_DIR, XRAY_DIR, HEATMAP_DIR):
    os.makedirs(folder, exist_ok=True)

IMG_SIZE = 224
THRESHOLD = 0.5        # combined score at or above this = TB Positive
IMAGE_WEIGHT = 0.7     # 70% from the x-ray model
SYMPTOM_WEIGHT = 0.3   # 30% from the symptom answers

RECORDS_TIMEOUT_MIN = 15   # records lock again after this many minutes of no activity

# WHO-based symptom weights, same as the Symptom_scoring notebook (they add up to 1.0)
SYMPTOM_WEIGHTS = {
    "cough_weeks":  0.25,
    "weight_loss":  0.20,
    "night_sweats": 0.15,
    "fever":        0.15,
    "hiv_positive": 0.15,
    "tb_contact":   0.10,
}

# the html sends different names, so map them to the notebook names
FORM_KEYS = {
    "cough":       "cough_weeks",
    "weightLoss":  "weight_loss",
    "nightSweats": "night_sweats",
    "fever":       "fever",
    "hiv":         "hiv_positive",
    "contact":     "tb_contact",
}

# vitals are saved with each screening but the model doesn't use them.
# all optional. (column name, label, lowest, highest, whole number?)
VITALS = [
    ("weight_kg",     "Weight",            1,   300, False),
    ("height_cm",     "Height",            30,  250, False),
    ("bp_systolic",   "Systolic BP",       50,  260, True),
    ("bp_diastolic",  "Diastolic BP",      30,  160, True),
    ("temperature_c", "Temperature",       30,  45,  False),
    ("pulse",         "Pulse",             20,  250, True),
    ("spo2",          "Oxygen saturation", 50,  100, True),
]
VITAL_COLS = [v[0] for v in VITALS]


def load_secret_key():
    # random key saved to a file, so logins survive restarting the server
    path = os.path.join(DATA_DIR, "secret.key")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(os.urandom(32))
    with open(path, "rb") as f:
        return f.read()


app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.secret_key = load_secret_key()


# sqlite database, one file in data/

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT UNIQUE NOT NULL,
    name          TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('screener', 'records')),
    active        INTEGER NOT NULL DEFAULT 1,
    must_change_password INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT
);

CREATE TABLE IF NOT EXISTS patients (
    patient_id    TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    age           INTEGER,
    sex           TEXT,
    phone         TEXT,
    address       TEXT,
    registered_at TEXT NOT NULL,
    registered_by TEXT,
    next_checkup  TEXT
);

CREATE TABLE IF NOT EXISTS screenings (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id       TEXT NOT NULL REFERENCES patients(patient_id),
    screened_at      TEXT NOT NULL,
    screened_by      TEXT,
    age_at_screening INTEGER,
    result           TEXT NOT NULL,
    confidence       TEXT,
    image_score      REAL,
    symptom_score    REAL,
    combined_score   REAL,
    symptoms         TEXT,
    weight_kg        REAL,
    height_cm        REAL,
    bp_systolic      INTEGER,
    bp_diastolic     INTEGER,
    temperature_c    REAL,
    pulse            INTEGER,
    spo2             INTEGER,
    xray_file        TEXT,
    heatmap_file     TEXT
);
"""


def get_db():
    # one connection per request
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(error):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)

    # older databases don't have the newer columns yet, so add them
    cols = [row[1] for row in conn.execute("PRAGMA table_info(users)")]
    if "active" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN active INTEGER NOT NULL DEFAULT 1")
    if "must_change_password" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0")
    if "created_at" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN created_at TEXT")

    patient_cols = [row[1] for row in conn.execute("PRAGMA table_info(patients)")]
    if "address" not in patient_cols:
        conn.execute("ALTER TABLE patients ADD COLUMN address TEXT")

    screening_cols = [row[1] for row in conn.execute("PRAGMA table_info(screenings)")]
    for col, _label, _low, _high, whole in VITALS:
        if col not in screening_cols:
            conn.execute(f"ALTER TABLE screenings ADD COLUMN {col} {'INTEGER' if whole else 'REAL'}")

    # no accounts yet = the page shows a "set up the first account" screen
    if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        print("no accounts yet, open the app in the browser to create the first one")

    # warn if the old demo accounts are still switched on
    demo = conn.execute(
        "SELECT email FROM users WHERE email IN ('screener@tb.local', 'records@tb.local') AND active = 1"
    ).fetchall()
    if demo:
        print("heads up: demo accounts are still active, turn them off on the Staff page:",
              ", ".join(r[0] for r in demo))

    conn.commit()
    conn.close()


init_db()


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def next_patient_id(db):
    # ids look like TB-2026-0001, counting up each year
    prefix = f"TB-{datetime.now().year}-"
    row = db.execute(
        "SELECT patient_id FROM patients WHERE patient_id LIKE ? ORDER BY patient_id DESC LIMIT 1",
        (prefix + "%",),
    ).fetchone()
    number = int(row["patient_id"][-4:]) + 1 if row else 1
    return f"{prefix}{number:04d}"


# decorators that block routes until someone signs in

def still_active():
    # if a supervisor turns someone's account off they get signed out on their next click
    row = get_db().execute("SELECT active FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    return bool(row and row["active"])


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session or not still_active():
            session.clear()
            return jsonify(error="Please sign in."), 401
        # someone on a temporary password can't do anything until they pick their own
        if session.get("must_change_password"):
            return jsonify(error="password_change_required"), 403
        return f(*args, **kwargs)
    return wrapper


def records_unlocked():
    return session.get("records_until", 0) > time.time()


def records_required(f):
    # supervisor accounts only, and they have to enter their password to open records
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session or not still_active():
            session.clear()
            return jsonify(error="Please sign in."), 401
        if session.get("must_change_password"):
            return jsonify(error="password_change_required"), 403
        if session.get("role") != "records":
            return jsonify(error="Only supervisor accounts can open this."), 403
        if not records_unlocked():
            session.pop("records_until", None)
            return jsonify(error="records_locked"), 403
        # still active, push the timeout forward
        session["records_until"] = time.time() + RECORDS_TIMEOUT_MIN * 60
        return f(*args, **kwargs)
    return wrapper


def check_password(email, password):
    user = get_db().execute(
        "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
    ).fetchone()
    if user and user["active"] and check_password_hash(user["password_hash"], password):
        return user
    return None


def password_problem(password):
    # simple rules, long enough and not just numbers
    if len(password) < 8:
        return "Use at least 8 characters."
    if password.isdigit():
        return "Don't use only numbers."
    return None


def make_temp_password():
    # easy to read out loud: no 0/O or 1/l mix-ups
    letters = "abcdefghjkmnpqrstuvwxyz23456789"
    return "".join(secrets.choice(letters) for _ in range(4)) + "-" + \
           "".join(secrets.choice(letters) for _ in range(4))


# load the mobilenet model once when the server starts

def build_model():
    # same Sequential setup as the training notebook, then load weights into it
    # weights=None so it doesn't try to download imagenet weights (we're offline)
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights=None
    )
    base.trainable = False

    model = keras.Sequential([
        base,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.build((None, IMG_SIZE, IMG_SIZE, 3))
    return model


print("loading model...")
model = build_model()
model.load_weights(MODEL_PATH)

base_model = model.layers[0]                              # mobilenetv2 part, outputs 7x7x1280
dense_weights = model.layers[-1].get_weights()[0][:, 0]   # (1280,) for CAM

# run one fake prediction so the first real request isn't slow
model.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32"), verbose=0)
print("model ready")


# reading, resizing and the CAM heatmap

def read_image(file_storage):
    # turn the uploaded file into an opencv image (BGR)
    data = np.frombuffer(file_storage.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)  # returns None if it's not a valid image


def preprocess(img_bgr):
    # same as training: RGB, 224x224, scaled to 0-1
    # nearest resize because that's what ImageDataGenerator / load_img use by default
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    arr = resized.astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)  # add batch dimension


def predict_with_cam(batch, original_bgr):
    # get the feature maps from mobilenet, then push them through the rest of the model
    # so we get the prediction and the CAM from one pass
    features = base_model(batch, training=False)
    out = features
    for layer in model.layers[1:]:
        out = layer(out, training=False)
    prob = float(out.numpy()[0][0])

    # CAM = feature maps weighted by the dense layer weights (same as the CAM notebook)
    fmap = features.numpy()[0]            # (7, 7, 1280)
    cam = np.dot(fmap, dense_weights)     # (7, 7)
    cam = cam - cam.min()                 # min-max scale so it still shows when activations are low
    if cam.max() > 0:
        cam = cam / cam.max()

    # blow it up and lay it over the x-ray
    size = 448
    xray = cv2.resize(original_bgr, (size, size))
    heat = cv2.resize(cam, (size, size))
    heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(xray, 0.6, heat, 0.4, 0)

    ok, buf = cv2.imencode(".png", overlay)
    return prob, buf.tobytes()


# symptom score and confidence label

def symptom_risk_score(symptoms):
    # same as the notebook: symptoms is a dict of 1s and 0s, result is 0-1
    return round(sum(SYMPTOM_WEIGHTS[k] * symptoms.get(k, 0) for k in SYMPTOM_WEIGHTS), 4)


def confidence_label(combined):
    # how far from the threshold the score landed
    gap = abs(combined - THRESHOLD)
    if gap >= 0.3:
        return "High"
    if gap >= 0.15:
        return "Moderate"
    return "Low"


def read_vitals(form):
    # returns (values, error). blank = not measured, saved as None
    values = {}
    for key, label, low, high, whole in VITALS:
        raw = form.get(key, "").strip()
        if raw == "":
            values[key] = None
            continue
        try:
            number = float(raw)
        except ValueError:
            return None, f"{label} must be a number."
        if not low <= number <= high:
            return None, f"{label} should be between {low} and {high}."
        values[key] = int(round(number)) if whole else round(number, 1)

    systolic, diastolic = values["bp_systolic"], values["bp_diastolic"]
    if (systolic is None) != (diastolic is None):
        return None, "Enter both blood pressure numbers or leave both blank."
    if systolic is not None and diastolic >= systolic:
        return None, "Diastolic pressure should be lower than systolic."
    return values, None


# the html page

@app.route("/")
def index():
    # serves the frontend so you can just open localhost:5000
    return send_from_directory(app.static_folder, "index.html")


# staff sign in / out

def user_json():
    return dict(
        name=session["name"], email=session["email"], role=session["role"],
        records_unlocked=records_unlocked(),
        must_change_password=bool(session.get("must_change_password")),
    )


def start_session(user):
    session.clear()
    session["user_id"] = user["id"]
    session["name"] = user["name"]
    session["email"] = user["email"]
    session["role"] = user["role"]
    session["must_change_password"] = bool(user["must_change_password"])


@app.route("/api/setup")
def setup_status():
    # the page asks this first, empty database = show the setup screen
    count = get_db().execute("SELECT COUNT(*) FROM users").fetchone()[0]
    return jsonify(needs_setup=count == 0)


@app.route("/api/setup", methods=["POST"])
def setup_first_account():
    db = get_db()
    if db.execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0:
        return jsonify(error="The system is already set up. Sign in instead."), 400

    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    if not name or "@" not in email:
        return jsonify(error="Enter your full name and email."), 400
    problem = password_problem(password)
    if problem:
        return jsonify(error=problem), 400

    # first person is a supervisor so they can add everyone else
    db.execute(
        "INSERT INTO users (email, name, password_hash, role, created_at) VALUES (?, ?, ?, 'records', ?)",
        (email, name, generate_password_hash(password), now_iso()),
    )
    db.commit()
    user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    start_session(user)
    return jsonify(user_json())


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    user = check_password(data.get("email", ""), data.get("password", ""))
    if not user:
        return jsonify(error="Wrong email or password."), 401
    start_session(user)
    return jsonify(user_json())


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify(ok=True)


@app.route("/api/me")
def me():
    # lets the page stay signed in after a refresh
    # no login_required here so someone mid password change still gets their info
    if "user_id" not in session:
        return jsonify(error="Please sign in."), 401
    return jsonify(user_json())


@app.route("/api/account/password", methods=["POST"])
def change_password():
    # used for the first sign-in and for "change my password" later
    if "user_id" not in session:
        return jsonify(error="Please sign in."), 401

    data = request.get_json(silent=True) or {}
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    if not user or not check_password_hash(user["password_hash"], data.get("current_password", "")):
        return jsonify(error="Your current password is wrong."), 400

    new = data.get("new_password", "")
    problem = password_problem(new)
    if problem:
        return jsonify(error=problem), 400
    if check_password_hash(user["password_hash"], new):
        return jsonify(error="Pick a password different from the current one."), 400

    db.execute(
        "UPDATE users SET password_hash = ?, must_change_password = 0 WHERE id = ?",
        (generate_password_hash(new), user["id"]),
    )
    db.commit()
    session["must_change_password"] = False
    return jsonify(user_json())


# looking up a returning patient by their id

@app.route("/api/patients/<patient_id>")
@login_required
def find_patient(patient_id):
    # only basic details here, past results need the records sign-in
    p = get_db().execute(
        "SELECT patient_id, name, age, sex, registered_at FROM patients WHERE patient_id = ?",
        (patient_id.strip().upper(),),
    ).fetchone()
    if not p:
        return jsonify(error="No patient found with that ID."), 404
    return jsonify(dict(p))


# the main screening route

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    form = request.form
    patient_type = form.get("patient_type", "new")

    # check the image
    if "xray" not in request.files or request.files["xray"].filename == "":
        return jsonify(error="No X-ray image was sent."), 400

    img = read_image(request.files["xray"])
    if img is None:
        return jsonify(error="Could not read the image. Use a JPG or PNG."), 400

    # check age
    try:
        age = int(form.get("age", ""))
        if not 0 <= age <= 120:
            raise ValueError
    except ValueError:
        return jsonify(error="Enter a valid age."), 400

    # check patient details
    db = get_db()
    if patient_type == "returning":
        patient_id = form.get("patient_id", "").strip().upper()
        patient = db.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
        if not patient:
            return jsonify(error="No patient found with that ID."), 400
    else:
        name = form.get("name", "").strip()
        sex = form.get("sex", "")
        if not name:
            return jsonify(error="Patient name is required."), 400
        if sex not in ("Female", "Male"):
            return jsonify(error="Select the patient's sex."), 400

    # vitals (optional)
    vitals, vitals_error = read_vitals(form)
    if vitals_error:
        return jsonify(error=vitals_error), 400

    # check all 6 answers are there and are yes/no, turn them into 1/0
    symptoms = {}
    for form_key, notebook_key in FORM_KEYS.items():
        value = form.get(form_key, "").strip().lower()
        if value not in ("yes", "no"):
            return jsonify(error=f"Missing or invalid answer for '{form_key}'."), 400
        symptoms[notebook_key] = 1 if value == "yes" else 0

    # run the model + scoring
    batch = preprocess(img)
    image_prob, heatmap_png = predict_with_cam(batch, img)
    symptom_score = symptom_risk_score(symptoms)

    combined = round(IMAGE_WEIGHT * image_prob + SYMPTOM_WEIGHT * symptom_score, 4)
    positive = combined >= THRESHOLD
    result = "TB Positive" if positive else "TB Negative"
    confidence = confidence_label(combined)
    screened_at = now_iso()

    # save the patient (if new) and the screening
    if patient_type == "returning":
        db.execute("UPDATE patients SET age = ? WHERE patient_id = ?", (age, patient_id))
        name, sex = patient["name"], patient["sex"]
    else:
        patient_id = next_patient_id(db)
        db.execute(
            "INSERT INTO patients (patient_id, name, age, sex, phone, address, registered_at, registered_by) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (patient_id, name, age, sex, form.get("phone", "").strip(),
             form.get("address", "").strip()[:200], screened_at, session["name"]),
        )

    cur = db.execute(
        "INSERT INTO screenings (patient_id, screened_at, screened_by, age_at_screening, result, "
        "confidence, image_score, symptom_score, combined_score, symptoms, "
        + ", ".join(VITAL_COLS) + ") "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, " + ", ".join("?" * len(VITAL_COLS)) + ")",
        (patient_id, screened_at, session["name"], age, result, confidence,
         round(image_prob, 4), symptom_score, combined, json.dumps(symptoms))
        + tuple(vitals[c] for c in VITAL_COLS),
    )
    screening_id = cur.lastrowid

    # images are saved as files, the database just keeps the file names
    xray_file = f"{screening_id}.png"
    heatmap_file = f"{screening_id}.png"
    cv2.imwrite(os.path.join(XRAY_DIR, xray_file), img)
    with open(os.path.join(HEATMAP_DIR, heatmap_file), "wb") as f:
        f.write(heatmap_png)
    db.execute(
        "UPDATE screenings SET xray_file = ?, heatmap_file = ? WHERE id = ?",
        (xray_file, heatmap_file, screening_id),
    )
    db.commit()

    return jsonify(
        result=result,
        confidence=confidence,
        heatmap="data:image/png;base64," + base64.b64encode(heatmap_png).decode("utf-8"),
        image_score=round(image_prob, 4),
        symptom_score=symptom_score,
        combined_score=combined,
        screening_id=screening_id,
        screened_at=screened_at,
        patient_id=patient_id,
        patient_name=name,
        age=age,
        sex=sex,
        vitals=vitals,
        is_new=patient_type != "returning",
    )


# today's screenings, any staff can see these but only for the current day
# after midnight they're only reachable through patient records

def today_str():
    return date.today().isoformat()


@app.route("/api/today")
@login_required
def todays_screenings():
    rows = get_db().execute(
        """
        SELECT s.id, s.patient_id, s.screened_at, s.screened_by, s.age_at_screening,
               s.result, s.confidence, s.image_score, s.symptom_score, s.combined_score,
               s.symptoms, s.weight_kg, s.height_cm, s.bp_systolic, s.bp_diastolic,
               s.temperature_c, s.pulse, s.spo2,
               p.name, p.sex, p.registered_at
        FROM screenings s JOIN patients p ON p.patient_id = s.patient_id
        WHERE substr(s.screened_at, 1, 10) = ?
        ORDER BY s.screened_at DESC, s.id DESC
        """,
        (today_str(),),
    ).fetchall()

    screenings = []
    for r in rows:
        item = dict(r)
        item["symptoms"] = json.loads(item["symptoms"] or "{}")
        # a new patient's registration time is the same as their first screening
        item["is_new"] = item.pop("registered_at") == item["screened_at"]
        item["xray_url"] = f"/api/today/{r['id']}/xray"
        item["heatmap_url"] = f"/api/today/{r['id']}/heatmap"
        screenings.append(item)
    return jsonify(screenings=screenings, date=today_str())


@app.route("/api/today/<int:screening_id>/<kind>")
@login_required
def todays_image(screening_id, kind):
    if kind not in ("xray", "heatmap"):
        return jsonify(error="Not found."), 404
    row = get_db().execute(
        "SELECT screened_at, xray_file, heatmap_file FROM screenings WHERE id = ?", (screening_id,)
    ).fetchone()
    # only today's images, older ones need the records sign-in
    if not row or row["screened_at"][:10] != today_str():
        return jsonify(error="Not found."), 404

    folder = XRAY_DIR if kind == "xray" else HEATMAP_DIR
    filename = row["xray_file"] if kind == "xray" else row["heatmap_file"]
    path = os.path.join(folder, filename or "")
    if not filename or not os.path.exists(path):
        return jsonify(error="Image missing."), 404
    return send_file(path, mimetype="image/png")


# patient records, supervisors only, and they enter their password to open them

@app.route("/api/records/unlock", methods=["POST"])
@login_required
def unlock_records():
    if session.get("role") != "records":
        return jsonify(error="Only supervisor accounts can open this."), 403
    data = request.get_json(silent=True) or {}
    # they're already signed in, so check the password against their own account
    user = check_password(session["email"], data.get("password", ""))
    if not user:
        return jsonify(error="Wrong password."), 400
    session["records_until"] = time.time() + RECORDS_TIMEOUT_MIN * 60
    session["records_user"] = user["name"]
    return jsonify(ok=True, records_user=user["name"])


@app.route("/api/records/lock", methods=["POST"])
@login_required
def lock_records():
    session.pop("records_until", None)
    session.pop("records_user", None)
    return jsonify(ok=True)


# staff accounts, managed by supervisors behind the records sign-in

def staff_json(row):
    return dict(
        id=row["id"], name=row["name"], email=row["email"], role=row["role"],
        active=bool(row["active"]), must_change_password=bool(row["must_change_password"]),
        created_at=row["created_at"],
    )


@app.route("/api/staff")
@records_required
def list_staff():
    rows = get_db().execute(
        "SELECT * FROM users ORDER BY active DESC, name COLLATE NOCASE"
    ).fetchall()
    return jsonify(staff=[staff_json(r) for r in rows], me=session["user_id"])


@app.route("/api/staff", methods=["POST"])
@records_required
def add_staff():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    role = data.get("role", "screener")
    if not name:
        return jsonify(error="Enter the health worker's full name."), 400
    if "@" not in email or "." not in email.split("@")[-1]:
        return jsonify(error="Enter a valid email address."), 400
    if role not in ("screener", "records"):
        return jsonify(error="Pick a role."), 400

    temp = make_temp_password()
    db = get_db()
    try:
        db.execute(
            "INSERT INTO users (email, name, password_hash, role, must_change_password, created_at) "
            "VALUES (?, ?, ?, ?, 1, ?)",
            (email, name, generate_password_hash(temp), role, now_iso()),
        )
    except sqlite3.IntegrityError:
        return jsonify(error="There's already an account with that email."), 400
    db.commit()
    # the temp password is only ever shown this once
    return jsonify(ok=True, name=name, email=email, temp_password=temp)


@app.route("/api/staff/<int:user_id>/reset", methods=["POST"])
@records_required
def reset_staff_password(user_id):
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        return jsonify(error="Account not found."), 404

    temp = make_temp_password()
    db.execute(
        "UPDATE users SET password_hash = ?, must_change_password = 1 WHERE id = ?",
        (generate_password_hash(temp), user_id),
    )
    db.commit()
    return jsonify(ok=True, name=user["name"], email=user["email"], temp_password=temp)


@app.route("/api/staff/<int:user_id>/active", methods=["PUT"])
@records_required
def set_staff_active(user_id):
    active = bool((request.get_json(silent=True) or {}).get("active"))
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        return jsonify(error="Account not found."), 404
    if not active and user_id == session["user_id"]:
        return jsonify(error="You can't turn off the account you're signed in with."), 400

    # always keep at least one supervisor who can get in
    if not active and user["role"] == "records":
        others = db.execute(
            "SELECT COUNT(*) FROM users WHERE role = 'records' AND active = 1 AND id != ?", (user_id,)
        ).fetchone()[0]
        if others == 0:
            return jsonify(error="This is the last active supervisor account, add another one first."), 400

    db.execute("UPDATE users SET active = ? WHERE id = ?", (1 if active else 0, user_id))
    db.commit()
    return jsonify(ok=True)


@app.route("/api/records/patients")
@records_required
def list_patients():
    db = get_db()
    show = request.args.get("filter", "all")
    search = request.args.get("q", "").strip()
    soon = (date.today() + timedelta(days=7)).isoformat()

    # each patient + their latest screening + how many times they've been screened / positive
    sql = """
        SELECT p.patient_id, p.name, p.age, p.sex, p.next_checkup,
               s.result AS last_result, s.screened_at AS last_screened,
               (SELECT COUNT(*) FROM screenings WHERE patient_id = p.patient_id) AS screening_count,
               (SELECT COUNT(*) FROM screenings WHERE patient_id = p.patient_id
                    AND result = 'TB Positive') AS positive_count
        FROM patients p
        LEFT JOIN screenings s ON s.id = (
            SELECT id FROM screenings WHERE patient_id = p.patient_id
            ORDER BY screened_at DESC, id DESC LIMIT 1)
    """
    rows = [dict(r) for r in db.execute(sql).fetchall()]

    # counts for the filter buttons
    counts = dict(
        all=len(rows),
        positive=sum(1 for r in rows if r["positive_count"] > 0),
        due=sum(1 for r in rows if r["next_checkup"] and r["next_checkup"] <= soon),
    )

    if show == "positive":
        rows = [r for r in rows if r["positive_count"] > 0]
    elif show == "due":
        rows = [r for r in rows if r["next_checkup"] and r["next_checkup"] <= soon]

    if search:
        s = search.lower()
        rows = [r for r in rows if s in r["name"].lower() or s in r["patient_id"].lower()]

    # check-ups sorted by soonest, everything else by most recent screening
    if show == "due":
        rows.sort(key=lambda r: r["next_checkup"])
    else:
        rows.sort(key=lambda r: r["last_screened"] or "", reverse=True)

    return jsonify(patients=rows, counts=counts)


@app.route("/api/records/patients/<patient_id>")
@records_required
def patient_detail(patient_id):
    db = get_db()
    p = db.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
    if not p:
        return jsonify(error="Patient not found."), 404

    screenings = []
    for s in db.execute(
        "SELECT * FROM screenings WHERE patient_id = ? ORDER BY screened_at DESC, id DESC",
        (patient_id,),
    ).fetchall():
        item = dict(s)
        item["symptoms"] = json.loads(item["symptoms"] or "{}")
        item["xray_url"] = f"/api/records/screenings/{s['id']}/xray"
        item["heatmap_url"] = f"/api/records/screenings/{s['id']}/heatmap"
        screenings.append(item)

    return jsonify(patient=dict(p), screenings=screenings)


@app.route("/api/records/patients/<patient_id>/checkup", methods=["PUT"])
@records_required
def set_checkup(patient_id):
    data = request.get_json(silent=True) or {}
    value = data.get("next_checkup") or None
    if value:
        try:
            date.fromisoformat(value)
        except ValueError:
            return jsonify(error="Invalid date."), 400

    db = get_db()
    cur = db.execute("UPDATE patients SET next_checkup = ? WHERE patient_id = ?", (value, patient_id))
    if cur.rowcount == 0:
        return jsonify(error="Patient not found."), 404
    db.commit()
    return jsonify(ok=True, next_checkup=value)


@app.route("/api/records/screenings/<int:screening_id>/<kind>")
@records_required
def screening_image(screening_id, kind):
    if kind not in ("xray", "heatmap"):
        return jsonify(error="Not found."), 404
    row = get_db().execute(
        "SELECT xray_file, heatmap_file FROM screenings WHERE id = ?", (screening_id,)
    ).fetchone()
    if not row:
        return jsonify(error="Not found."), 404

    folder = XRAY_DIR if kind == "xray" else HEATMAP_DIR
    filename = row["xray_file"] if kind == "xray" else row["heatmap_file"]
    path = os.path.join(folder, filename or "")
    if not filename or not os.path.exists(path):
        return jsonify(error="Image missing."), 404
    return send_file(path, mimetype="image/png")


if __name__ == "__main__":
    # debug=False on purpose, debug mode reloads and would load the model twice
    app.run(host="127.0.0.1", port=5000, debug=False)