

import os
import base64
import json
import secrets
import sqlite3
import time
from datetime import datetime, date, timedelta
from functools import wraps
from threading import Lock

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import (
    Flask, request, jsonify, send_from_directory,
    send_file, session, g
)
from werkzeug.security import (
    generate_password_hash, check_password_hash
)

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    Interpreter = tf.lite.Interpreter


# Configuration


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_NAME = "mobilenetv2_full_fp16.tflite"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

GRADCAM_MODEL_NAME = "mobilenetv2_full_seed42.keras"
GRADCAM_MODEL_PATH = os.path.join(
    MODEL_DIR, GRADCAM_MODEL_NAME
)

IMG_SIZE = 224
THRESHOLD = 0.3904514014720917


DECISION_RULE = "image_only"
RECORDS_TIMEOUT_MIN = 15

for path in [MODEL_PATH, GRADCAM_MODEL_PATH]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required model file not found: {path}"
        )

DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "tb_screening.db")
XRAY_DIR = os.path.join(DATA_DIR, "xrays")
HEATMAP_DIR = os.path.join(DATA_DIR, "heatmaps")

for folder in (DATA_DIR, XRAY_DIR, HEATMAP_DIR):
    os.makedirs(folder, exist_ok=True)

SYMPTOM_WEIGHTS = {
    "cough_weeks": 0.25,
    "weight_loss": 0.20,
    "night_sweats": 0.15,
    "fever": 0.15,
    "hiv_positive": 0.15,
    "tb_contact": 0.10,
}

FORM_KEYS = {
    "cough": "cough_weeks",
    "weightLoss": "weight_loss",
    "nightSweats": "night_sweats",
    "fever": "fever",
    "hiv": "hiv_positive",
    "contact": "tb_contact",
}

VITALS = [
    ("weight_kg", "Weight", 1, 300, False),
    ("height_cm", "Height", 30, 250, False),
    ("bp_systolic", "Systolic BP", 50, 260, True),
    ("bp_diastolic", "Diastolic BP", 30, 160, True),
    ("temperature_c", "Temperature", 30, 45, False),
    ("pulse", "Pulse", 20, 250, True),
    ("spo2", "Oxygen saturation", 50, 100, True),
]

VITAL_COLS = [item[0] for item in VITALS]


def load_secret_key():
    path = os.path.join(DATA_DIR, "secret.key")

    if not os.path.exists(path):
        with open(path, "wb") as file:
            file.write(os.urandom(32))

    with open(path, "rb") as file:
        return file.read()


app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.secret_key = load_secret_key()

# Database


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('screener', 'records')),
    active INTEGER NOT NULL DEFAULT 1,
    must_change_password INTEGER NOT NULL DEFAULT 0,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER,
    sex TEXT,
    phone TEXT,
    address TEXT,
    registered_at TEXT NOT NULL,
    registered_by TEXT,
    next_checkup TEXT
);

CREATE TABLE IF NOT EXISTS screenings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id),
    screened_at TEXT NOT NULL,
    screened_by TEXT,
    age_at_screening INTEGER,
    result TEXT NOT NULL,
    confidence TEXT,
    image_score REAL,
    symptom_score REAL,
    combined_score REAL,
    symptoms TEXT,
    weight_kg REAL,
    height_cm REAL,
    bp_systolic INTEGER,
    bp_diastolic INTEGER,
    temperature_c REAL,
    pulse INTEGER,
    spo2 INTEGER,
    xray_file TEXT,
    heatmap_file TEXT
);
"""


def get_db():
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

    user_cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(users)")
    }

    for name, definition in [
        ("active", "INTEGER NOT NULL DEFAULT 1"),
        ("must_change_password", "INTEGER NOT NULL DEFAULT 0"),
        ("created_at", "TEXT"),
    ]:
        if name not in user_cols:
            conn.execute(
                f"ALTER TABLE users ADD COLUMN {name} {definition}"
            )

    patient_cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(patients)")
    }

    if "address" not in patient_cols:
        conn.execute(
            "ALTER TABLE patients ADD COLUMN address TEXT"
        )

    screening_cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(screenings)")
    }

    for name, definition in [
        (
            "decision_rule",
            "TEXT NOT NULL DEFAULT 'legacy_combined'"
        ),
        ("model_name", "TEXT"),
        ("model_threshold", "REAL"),
    ]:
        if name not in screening_cols:
            conn.execute(
                f"ALTER TABLE screenings ADD COLUMN {name} {definition}"
            )

    for name, label, low, high, whole in VITALS:
        if name not in screening_cols:
            data_type = "INTEGER" if whole else "REAL"
            conn.execute(
                f"ALTER TABLE screenings ADD COLUMN {name} {data_type}"
            )

    count = conn.execute(
        "SELECT COUNT(*) FROM users"
    ).fetchone()[0]

    if count == 0:
        print("No accounts yet. Create the first account in the browser.")

    demo = conn.execute(
        """
        SELECT email FROM users
        WHERE email IN ('screener@tb.local', 'records@tb.local')
          AND active = 1
        """
    ).fetchall()

    if demo:
        print(
            "WARNING: Demo accounts are active:",
            ", ".join(row[0] for row in demo)
        )

    conn.commit()
    conn.close()


init_db()


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def next_patient_id(db):
    prefix = f"TB-{datetime.now().year}-"

    row = db.execute(
        """
        SELECT patient_id FROM patients
        WHERE patient_id LIKE ?
        ORDER BY patient_id DESC LIMIT 1
        """,
        (prefix + "%",)
    ).fetchone()

    number = int(row["patient_id"][-4:]) + 1 if row else 1
    return f"{prefix}{number:04d}"

# Authentication


def still_active():
    row = get_db().execute(
        "SELECT active FROM users WHERE id = ?",
        (session["user_id"],)
    ).fetchone()

    return bool(row and row["active"])


def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session or not still_active():
            session.clear()
            return jsonify(error="Please sign in."), 401

        if session.get("must_change_password"):
            return jsonify(error="password_change_required"), 403

        return func(*args, **kwargs)

    return wrapper


def records_unlocked():
    return session.get("records_until", 0) > time.time()


def records_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session or not still_active():
            session.clear()
            return jsonify(error="Please sign in."), 401

        if session.get("must_change_password"):
            return jsonify(error="password_change_required"), 403

        if session.get("role") != "records":
            return jsonify(
                error="Only supervisor accounts can open this."
            ), 403

        if not records_unlocked():
            session.pop("records_until", None)
            return jsonify(error="records_locked"), 403

        session["records_until"] = (
            time.time() + RECORDS_TIMEOUT_MIN * 60
        )
        return func(*args, **kwargs)

    return wrapper


def check_password(email, password):
    user = get_db().execute(
        "SELECT * FROM users WHERE email = ?",
        (email.strip().lower(),)
    ).fetchone()

    if (
        user
        and user["active"]
        and check_password_hash(user["password_hash"], password)
    ):
        return user

    return None


def password_problem(password):
    if len(password) < 8:
        return "Use at least 8 characters."
    if password.isdigit():
        return "Don't use only numbers."
    return None


def make_temp_password():
    letters = "abcdefghjkmnpqrstuvwxyz23456789"
    return (
        "".join(secrets.choice(letters) for _ in range(4))
        + "-"
        + "".join(secrets.choice(letters) for _ in range(4))
    )


# FP16 classifier


print(f"Loading FP16 screening model: {MODEL_PATH}")

interpreter = Interpreter(
    model_path=MODEL_PATH,
    num_threads=2
)
interpreter.allocate_tensors()

input_info = interpreter.get_input_details()[0]
output_info = interpreter.get_output_details()[0]

if input_info["shape"].tolist() != [1, 224, 224, 3]:
    raise RuntimeError(
        f"Unexpected model input shape: {input_info['shape']}"
    )

if input_info["dtype"] != np.float32:
    raise RuntimeError("Expected float32 model input.")

if output_info["dtype"] != np.float32:
    raise RuntimeError("Expected float32 model output.")

model_lock = Lock()

print("FP16 classifier ready.")
print("Model:", MODEL_NAME)
print("Threshold:", THRESHOLD)


def read_image(file_storage):
    data = np.frombuffer(file_storage.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def preprocess(img_bgr):
    """
    Match the full-X-ray TBX11K preprocessing:
    grayscale -> bicubic 224x224 -> float32 / 255
    -> repeat grayscale across 3 channels.

    The classifier contains its own subsequent rescaling.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    gray = np.asarray(
        Image.fromarray(rgb).convert("L").resize(
            (IMG_SIZE, IMG_SIZE),
            resample=Image.Resampling.BICUBIC
        ),
        dtype=np.uint8
    )

    normalized = gray.astype(np.float32) / 255.0

    return np.repeat(
        normalized[..., None],
        3,
        axis=-1
    )[None, ...]


def predict_tflite(batch):
    if batch.shape != (1, IMG_SIZE, IMG_SIZE, 3):
        raise ValueError(f"Incorrect input shape: {batch.shape}")

    with model_lock:
        interpreter.set_tensor(
            input_info["index"],
            batch.astype(np.float32)
        )
        interpreter.invoke()
        probability = float(
            interpreter.get_tensor(
                output_info["index"]
            ).reshape(-1)[0]
        )

    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise RuntimeError("Invalid classifier output.")

    return probability


# ------------------------------------------------------------
# Grad-CAM using original Keras model
# ------------------------------------------------------------

print(f"Loading Grad-CAM model: {GRADCAM_MODEL_PATH}")

gradcam_model = tf.keras.models.load_model(
    GRADCAM_MODEL_PATH,
    compile=False
)

BACKBONE_NAME = "mobilenetv2_1.00_224"
backbone = gradcam_model.get_layer(BACKBONE_NAME)
target_layer = backbone.get_layer("out_relu")

if backbone.layers[-1].name != "out_relu":
    raise RuntimeError(
        "Backbone does not end at out_relu. "
        "Grad-CAM requires a different feature extraction path."
    )

# The saved model is a sequential outer graph whose nested
# MobileNetV2 produces the final feature maps.
outer_layers = gradcam_model.layers
backbone_position = next(
    i for i, layer in enumerate(outer_layers)
    if layer.name == BACKBONE_NAME
)

if not any(
    isinstance(layer, tf.keras.layers.GlobalAveragePooling2D)
    for layer in outer_layers[backbone_position + 1:]
):
    raise RuntimeError("The expected GAP classification head is missing.")

gradcam_lock = Lock()


def forward_for_gradcam(input_tensor, tape):
    """
    Execute the saved model's layers in order, capturing
    the nested MobileNetV2 output with the gradient tape.
    """
    x = input_tensor
    features = None

    for layer in outer_layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        if layer.name == BACKBONE_NAME:
            x = layer(x, training=False)
            features = x
            tape.watch(features)

        elif isinstance(
            layer,
            (
                tf.keras.layers.Dropout,
                tf.keras.layers.RandomRotation,
                tf.keras.layers.RandomZoom
            )
        ):
            x = layer(x, training=False)

        else:
            x = layer(x)

    if features is None:
        raise RuntimeError("Grad-CAM feature maps were not found.")

    return features, x


def generate_gradcam(batch, original_bgr):
    """
    Grad-CAM of the original FP32 Keras model.

    The FP16 LiteRT model determines the screening score.
    Grad-CAM is an exploratory explanation of the saved
    FP32 model, not verified TB lesion localisation.
    """
    input_tensor = tf.convert_to_tensor(
        batch,
        dtype=tf.float32
    )

    with gradcam_lock:
        with tf.GradientTape() as tape:
            feature_maps, predictions = forward_for_gradcam(
                input_tensor, tape
            )

            probability = tf.clip_by_value(
                predictions[:, 0],
                1e-6,
                1.0 - 1e-6
            )

            # Use the positive-class logit to avoid saturation in the sigmoid probability gradient.
            target_score = tf.math.log(
                probability / (1.0 - probability)
            )

        gradients = tape.gradient(
            target_score,
            feature_maps
        )

        # Check our manual forward pass against the original
        # saved model to detect unexpected architecture issues.
        direct_prediction = gradcam_model(
            input_tensor,
            training=False
        )

    if gradients is None:
        raise RuntimeError("Grad-CAM gradients are unavailable.")

    keras_probability = float(predictions.numpy().reshape(-1)[0])
    direct_probability = float(
        direct_prediction.numpy().reshape(-1)[0]
    )

    if abs(keras_probability - direct_probability) > 1e-4:
        raise RuntimeError(
            "Grad-CAM forward pass does not reproduce "
            "the saved Keras model prediction."
        )

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    cam = tf.reduce_sum(
        feature_maps * weights[:, None, None, :],
        axis=-1
    )
    cam = tf.nn.relu(cam)
    cam = cam[0].numpy()

    if not np.all(np.isfinite(cam)):
        raise RuntimeError("Grad-CAM produced invalid values.")

    maximum = float(np.max(cam))

    if maximum > 1e-8:
        cam = cam / maximum
    else:
        cam = np.zeros_like(cam)

    height, width = original_bgr.shape[:2]

    cam = cv2.resize(
        cam,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )
    cam = np.clip(cam, 0.0, 1.0)
    cam = cv2.GaussianBlur(
        cam, (0, 0), sigmaX=5, sigmaY=5
    )

    if np.max(cam) > 1e-8:
        cam /= np.max(cam)

    heatmap = cv2.applyColorMap(
        np.uint8(cam * 255),
        cv2.COLORMAP_JET
    )

    alpha = (0.40 * cam)[..., None]

    overlay = (
        original_bgr.astype(np.float32) * (1.0 - alpha)
        + heatmap.astype(np.float32) * alpha
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    ok, encoded = cv2.imencode(".png", overlay)

    if not ok:
        raise RuntimeError("Failed to encode Grad-CAM image.")

    return encoded.tobytes(), keras_probability


print("Grad-CAM backbone:", backbone.name)
print("Grad-CAM target layer:", target_layer.name)
print("Grad-CAM model ready.")


# Symptoms and vitals


def symptom_risk_score(symptoms):
    return round(
        sum(
            SYMPTOM_WEIGHTS[key] * symptoms.get(key, 0)
            for key in SYMPTOM_WEIGHTS
        ),
        4
    )


def read_vitals(form):
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

        if not np.isfinite(number) or not low <= number <= high:
            return None, f"{label} should be between {low} and {high}."

        values[key] = (
            int(round(number))
            if whole else round(number, 1)
        )

    systolic = values["bp_systolic"]
    diastolic = values["bp_diastolic"]

    if (systolic is None) != (diastolic is None):
        return None, (
            "Enter both blood pressure numbers or leave both blank."
        )

    if systolic is not None and diastolic >= systolic:
        return None, (
            "Diastolic pressure should be lower than systolic."
        )

    return values, None


# ------------------------------------------------------------
# Frontend
# ------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(
        app.static_folder,
        "index.html"
    )


# Staff login and setup


def user_json():
    return dict(
        name=session["name"],
        email=session["email"],
        role=session["role"],
        records_unlocked=records_unlocked(),
        must_change_password=bool(
            session.get("must_change_password")
        )
    )


def start_session(user):
    session.clear()
    session["user_id"] = user["id"]
    session["name"] = user["name"]
    session["email"] = user["email"]
    session["role"] = user["role"]
    session["must_change_password"] = bool(
        user["must_change_password"]
    )


@app.route("/api/setup")
def setup_status():
    count = get_db().execute(
        "SELECT COUNT(*) FROM users"
    ).fetchone()[0]
    return jsonify(needs_setup=count == 0)


@app.route("/api/setup", methods=["POST"])
def setup_first_account():
    db = get_db()

    if db.execute(
        "SELECT COUNT(*) FROM users"
    ).fetchone()[0] > 0:
        return jsonify(
            error="The system is already set up. Sign in instead."
        ), 400

    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not name or "@" not in email:
        return jsonify(
            error="Enter your full name and email."
        ), 400

    problem = password_problem(password)
    if problem:
        return jsonify(error=problem), 400

    db.execute(
        """
        INSERT INTO users
        (email, name, password_hash, role, created_at)
        VALUES (?, ?, ?, 'records', ?)
        """,
        (
            email,
            name,
            generate_password_hash(password),
            now_iso()
        )
    )
    db.commit()

    user = db.execute(
        "SELECT * FROM users WHERE email = ?",
        (email,)
    ).fetchone()

    start_session(user)
    return jsonify(user_json())


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}

    user = check_password(
        data.get("email", ""),
        data.get("password", "")
    )

    if not user:
        return jsonify(
            error="Wrong email or password."
        ), 401

    start_session(user)
    return jsonify(user_json())


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify(ok=True)


@app.route("/api/me")
def me():
    if "user_id" not in session or not still_active():
        session.clear()
        return jsonify(error="Please sign in."), 401

    return jsonify(user_json())


@app.route("/api/account/password", methods=["POST"])
def change_password():
    if "user_id" not in session:
        return jsonify(error="Please sign in."), 401

    data = request.get_json(silent=True) or {}
    db = get_db()

    user = db.execute(
        "SELECT * FROM users WHERE id = ?",
        (session["user_id"],)
    ).fetchone()

    if not user or not check_password_hash(
        user["password_hash"],
        data.get("current_password", "")
    ):
        return jsonify(
            error="Your current password is wrong."
        ), 400

    new = data.get("new_password", "")
    problem = password_problem(new)

    if problem:
        return jsonify(error=problem), 400

    if check_password_hash(user["password_hash"], new):
        return jsonify(
            error="Pick a password different from the current one."
        ), 400

    db.execute(
        """
        UPDATE users
        SET password_hash = ?, must_change_password = 0
        WHERE id = ?
        """,
        (generate_password_hash(new), user["id"])
    )
    db.commit()

    session["must_change_password"] = False
    return jsonify(user_json())


# Patient lookup


@app.route("/api/patients/<patient_id>")
@login_required
def find_patient(patient_id):
    patient = get_db().execute(
        """
        SELECT patient_id, name, age, sex, registered_at
        FROM patients
        WHERE patient_id = ?
        """,
        (patient_id.strip().upper(),)
    ).fetchone()

    if not patient:
        return jsonify(
            error="No patient found with that ID."
        ), 404

    return jsonify(dict(patient))



# Screening


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    form = request.form
    patient_type = form.get("patient_type", "new")

    if (
        "xray" not in request.files
        or request.files["xray"].filename == ""
    ):
        return jsonify(
            error="No X-ray image was sent."
        ), 400

    img = read_image(request.files["xray"])

    if img is None:
        return jsonify(
            error="Could not read the image. Use a JPG or PNG."
        ), 400

    try:
        age = int(form.get("age", ""))
        if not 0 <= age <= 120:
            raise ValueError
    except ValueError:
        return jsonify(
            error="Enter a valid age."
        ), 400

    db = get_db()

    if patient_type == "returning":
        patient_id = (
            form.get("patient_id", "").strip().upper()
        )
        patient = db.execute(
            "SELECT * FROM patients WHERE patient_id = ?",
            (patient_id,)
        ).fetchone()

        if not patient:
            return jsonify(
                error="No patient found with that ID."
            ), 400

    else:
        name = form.get("name", "").strip()
        sex = form.get("sex", "")

        if not name:
            return jsonify(
                error="Patient name is required."
            ), 400

        if sex not in ("Female", "Male"):
            return jsonify(
                error="Select the patient's sex."
            ), 400

    vitals, vitals_error = read_vitals(form)

    if vitals_error:
        return jsonify(error=vitals_error), 400

    symptoms = {}

    for form_key, notebook_key in FORM_KEYS.items():
        value = form.get(form_key, "").strip().lower()

        if value not in ("yes", "no"):
            return jsonify(
                error=f"Missing or invalid answer for '{form_key}'."
            ), 400

        symptoms[notebook_key] = (
            1 if value == "yes" else 0
        )

    # FP16 prediction
    batch = preprocess(img)
    image_prob = predict_tflite(batch)

    # FP32 Keras Grad-CAM explanation
    heatmap_png, keras_prob = generate_gradcam(
        batch, img
    )

    score_difference = abs(image_prob - keras_prob)
    decision_agreement = (
        (image_prob >= THRESHOLD)
        == (keras_prob >= THRESHOLD)
    )

    if score_difference > 0.05:
        print(
            "WARNING: FP16/Keras score difference:",
            round(score_difference, 6)
        )

    if not decision_agreement:
        print(
            "WARNING: FP16 and Keras disagree "
            "at the decision threshold."
        )

    # Symptoms are descriptive and do not affect prediction.
    symptom_score = symptom_risk_score(symptoms)
    positive = image_prob >= THRESHOLD

    # Stored values kept for compatibility with existing records.
    # The frontend displays TB Suspected / TB Not Indicated.
    result = "TB Positive" if positive else "TB Negative"

    confidence = "Not assessed"
    combined = round(image_prob, 4)
    screened_at = now_iso()

    if patient_type == "returning":
        db.execute(
            "UPDATE patients SET age = ? WHERE patient_id = ?",
            (age, patient_id)
        )
        name = patient["name"]
        sex = patient["sex"]

    else:
        patient_id = next_patient_id(db)

        db.execute(
            """
            INSERT INTO patients
            (patient_id, name, age, sex, phone, address,
             registered_at, registered_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                name,
                age,
                sex,
                form.get("phone", "").strip(),
                form.get("address", "").strip()[:200],
                screened_at,
                session["name"]
            )
        )

    sql = (
        "INSERT INTO screenings "
        "(patient_id, screened_at, screened_by, "
        "age_at_screening, result, confidence, "
        "image_score, symptom_score, combined_score, "
        "symptoms, decision_rule, model_name, model_threshold, "
        + ", ".join(VITAL_COLS)
        + ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        + ", ".join("?" for _ in VITAL_COLS)
        + ")"
    )

    parameters = (
        patient_id,
        screened_at,
        session["name"],
        age,
        result,
        confidence,
        round(image_prob, 4),
        symptom_score,
        combined,
        json.dumps(symptoms),
        DECISION_RULE,
        MODEL_NAME,
        THRESHOLD
    ) + tuple(vitals[col] for col in VITAL_COLS)

    cur = db.execute(sql, parameters)
    screening_id = cur.lastrowid

    xray_file = f"{screening_id}.png"
    heatmap_file = f"{screening_id}.png"

    xray_path = os.path.join(XRAY_DIR, xray_file)
    heatmap_path = os.path.join(HEATMAP_DIR, heatmap_file)

    if not cv2.imwrite(xray_path, img):
        db.rollback()
        return jsonify(
            error="Failed to save X-ray."
        ), 500

    try:
        with open(heatmap_path, "wb") as file:
            file.write(heatmap_png)
    except OSError:
        db.rollback()
        try:
            os.remove(xray_path)
        except OSError:
            pass
        return jsonify(
            error="Failed to save Grad-CAM image."
        ), 500

    db.execute(
        """
        UPDATE screenings
        SET xray_file = ?, heatmap_file = ?
        WHERE id = ?
        """,
        (xray_file, heatmap_file, screening_id)
    )
    db.commit()

    return jsonify(
        result=result,
        confidence=confidence,
        heatmap=(
            "data:image/png;base64,"
            + base64.b64encode(heatmap_png).decode("utf-8")
        ),
        image_score=round(image_prob, 4),
        symptom_score=symptom_score,
        combined_score=combined,
        model_threshold=THRESHOLD,
        decision_rule=DECISION_RULE,
        model_name=MODEL_NAME,
        screening_id=screening_id,
        screened_at=screened_at,
        patient_id=patient_id,
        patient_name=name,
        age=age,
        sex=sex,
        vitals=vitals,
        is_new=patient_type != "returning",
        gradcam_model=GRADCAM_MODEL_NAME,
        gradcam_model_score=round(keras_prob, 6),
        gradcam_score_difference=round(score_difference, 6),
        gradcam_decision_agreement=bool(decision_agreement),
        research_only=True,
        warning="Result should not be used for clinical diagnosis."
    )
# Today's screenings


def today_str():
    return date.today().isoformat()


@app.route("/api/today")
@login_required
def todays_screenings():
    rows = get_db().execute(
        """
        SELECT s.id, s.patient_id, s.screened_at, s.screened_by,
               s.age_at_screening, s.result, s.confidence,
               s.image_score, s.symptom_score, s.combined_score,
               s.decision_rule, s.model_name, s.model_threshold,
               s.symptoms, s.weight_kg, s.height_cm,
               s.bp_systolic, s.bp_diastolic, s.temperature_c,
               s.pulse, s.spo2,
               p.name, p.sex, p.registered_at
        FROM screenings s
        JOIN patients p ON p.patient_id = s.patient_id
        WHERE substr(s.screened_at, 1, 10) = ?
        ORDER BY s.screened_at DESC, s.id DESC
        """,
        (today_str(),)
    ).fetchall()

    screenings = []

    for row in rows:
        item = dict(row)
        item["symptoms"] = json.loads(
            item["symptoms"] or "{}"
        )
        item["is_new"] = (
            item.pop("registered_at") == item["screened_at"]
        )
        item["xray_url"] = f"/api/today/{row['id']}/xray"
        item["heatmap_url"] = (
            f"/api/today/{row['id']}/heatmap"
        )
        screenings.append(item)

    return jsonify(
        screenings=screenings,
        date=today_str()
    )


@app.route("/api/today/<int:screening_id>/<kind>")
@login_required
def todays_image(screening_id, kind):
    if kind not in ("xray", "heatmap"):
        return jsonify(error="Not found."), 404

    row = get_db().execute(
        """
        SELECT screened_at, xray_file, heatmap_file
        FROM screenings WHERE id = ?
        """,
        (screening_id,)
    ).fetchone()

    if not row or row["screened_at"][:10] != today_str():
        return jsonify(error="Not found."), 404

    folder = XRAY_DIR if kind == "xray" else HEATMAP_DIR
    filename = (
        row["xray_file"]
        if kind == "xray"
        else row["heatmap_file"]
    )

    if not filename:
        return jsonify(error="Image missing."), 404

    path = os.path.join(folder, filename)

    if not os.path.isfile(path):
        return jsonify(error="Image missing."), 404

    return send_file(path, mimetype="image/png")


# ------------------------------------------------------------
# Supervisor records access
# ------------------------------------------------------------

@app.route("/api/records/unlock", methods=["POST"])
@login_required
def unlock_records():
    if session.get("role") != "records":
        return jsonify(
            error="Only supervisor accounts can open this."
        ), 403

    data = request.get_json(silent=True) or {}
    user = check_password(
        session["email"],
        data.get("password", "")
    )

    if not user:
        return jsonify(error="Wrong password."), 400

    session["records_until"] = (
        time.time() + RECORDS_TIMEOUT_MIN * 60
    )
    session["records_user"] = user["name"]

    return jsonify(
        ok=True,
        records_user=user["name"]
    )


@app.route("/api/records/lock", methods=["POST"])
@login_required
def lock_records():
    session.pop("records_until", None)
    session.pop("records_user", None)
    return jsonify(ok=True)


# Staff accounts


def staff_json(row):
    return dict(
        id=row["id"],
        name=row["name"],
        email=row["email"],
        role=row["role"],
        active=bool(row["active"]),
        must_change_password=bool(
            row["must_change_password"]
        ),
        created_at=row["created_at"]
    )


@app.route("/api/staff")
@records_required
def list_staff():
    rows = get_db().execute(
        """
        SELECT * FROM users
        ORDER BY active DESC, name COLLATE NOCASE
        """
    ).fetchall()

    return jsonify(
        staff=[staff_json(row) for row in rows],
        me=session["user_id"]
    )


@app.route("/api/staff", methods=["POST"])
@records_required
def add_staff():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    role = data.get("role", "screener")

    if not name:
        return jsonify(
            error="Enter the health worker's full name."
        ), 400

    if "@" not in email or "." not in email.split("@")[-1]:
        return jsonify(
            error="Enter a valid email address."
        ), 400

    if role not in ("screener", "records"):
        return jsonify(error="Pick a role."), 400

    temp = make_temp_password()
    db = get_db()

    try:
        db.execute(
            """
            INSERT INTO users
            (email, name, password_hash, role,
             must_change_password, created_at)
            VALUES (?, ?, ?, ?, 1, ?)
            """,
            (
                email,
                name,
                generate_password_hash(temp),
                role,
                now_iso()
            )
        )
    except sqlite3.IntegrityError:
        return jsonify(
            error="There's already an account with that email."
        ), 400

    db.commit()

    return jsonify(
        ok=True,
        name=name,
        email=email,
        temp_password=temp
    )


@app.route("/api/staff/<int:user_id>/reset", methods=["POST"])
@records_required
def reset_staff_password(user_id):
    db = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE id = ?",
        (user_id,)
    ).fetchone()

    if not user:
        return jsonify(error="Account not found."), 404

    temp = make_temp_password()

    db.execute(
        """
        UPDATE users
        SET password_hash = ?, must_change_password = 1
        WHERE id = ?
        """,
        (generate_password_hash(temp), user_id)
    )
    db.commit()

    return jsonify(
        ok=True,
        name=user["name"],
        email=user["email"],
        temp_password=temp
    )


@app.route("/api/staff/<int:user_id>/active", methods=["PUT"])
@records_required
def set_staff_active(user_id):
    active = bool(
        (request.get_json(silent=True) or {}).get("active")
    )

    db = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE id = ?",
        (user_id,)
    ).fetchone()

    if not user:
        return jsonify(error="Account not found."), 404

    if not active and user_id == session["user_id"]:
        return jsonify(
            error="You can't turn off the account you're signed in with."
        ), 400

    if not active and user["role"] == "records":
        others = db.execute(
            """
            SELECT COUNT(*) FROM users
            WHERE role = 'records'
              AND active = 1
              AND id != ?
            """,
            (user_id,)
        ).fetchone()[0]

        if others == 0:
            return jsonify(
                error=(
                    "This is the last active supervisor account. "
                    "Add another one first."
                )
            ), 400

    db.execute(
        "UPDATE users SET active = ? WHERE id = ?",
        (1 if active else 0, user_id)
    )
    db.commit()

    return jsonify(ok=True)


# Patient records

@app.route("/api/records/patients")
@records_required
def list_patients():
    db = get_db()
    show = request.args.get("filter", "all")
    search = request.args.get("q", "").strip()

    soon = (
        date.today() + timedelta(days=7)
    ).isoformat()

    sql = """
        SELECT p.patient_id, p.name, p.age, p.sex, p.next_checkup,
               s.result AS last_result,
               s.screened_at AS last_screened,
               (
                   SELECT COUNT(*)
                   FROM screenings
                   WHERE patient_id = p.patient_id
               ) AS screening_count,
               (
                   SELECT COUNT(*)
                   FROM screenings
                   WHERE patient_id = p.patient_id
                     AND result = 'TB Positive'
               ) AS positive_count
        FROM patients p
        LEFT JOIN screenings s ON s.id = (
            SELECT id FROM screenings
            WHERE patient_id = p.patient_id
            ORDER BY screened_at DESC, id DESC LIMIT 1
        )
    """

    rows = [dict(row) for row in db.execute(sql).fetchall()]

    counts = dict(
        all=len(rows),
        positive=sum(
            1 for row in rows if row["positive_count"] > 0
        ),
        due=sum(
            1 for row in rows
            if row["next_checkup"] and row["next_checkup"] <= soon
        )
    )

    if show == "positive":
        rows = [
            row for row in rows
            if row["positive_count"] > 0
        ]
    elif show == "due":
        rows = [
            row for row in rows
            if row["next_checkup"] and row["next_checkup"] <= soon
        ]

    if search:
        term = search.lower()
        rows = [
            row for row in rows
            if term in row["name"].lower()
            or term in row["patient_id"].lower()
        ]

    if show == "due":
        rows.sort(key=lambda row: row["next_checkup"])
    else:
        rows.sort(
            key=lambda row: row["last_screened"] or "",
            reverse=True
        )

    return jsonify(patients=rows, counts=counts)


@app.route("/api/records/patients/<patient_id>")
@records_required
def patient_detail(patient_id):
    db = get_db()
    patient = db.execute(
        "SELECT * FROM patients WHERE patient_id = ?",
        (patient_id,)
    ).fetchone()

    if not patient:
        return jsonify(error="Patient not found."), 404

    screenings = []

    for screening in db.execute(
        """
        SELECT * FROM screenings
        WHERE patient_id = ?
        ORDER BY screened_at DESC, id DESC
        """,
        (patient_id,)
    ).fetchall():
        item = dict(screening)
        item["symptoms"] = json.loads(
            item["symptoms"] or "{}"
        )
        item["xray_url"] = (
            f"/api/records/screenings/{screening['id']}/xray"
        )
        item["heatmap_url"] = (
            f"/api/records/screenings/{screening['id']}/heatmap"
            if screening["heatmap_file"] else None
        )
        screenings.append(item)

    return jsonify(
        patient=dict(patient),
        screenings=screenings
    )


@app.route(
    "/api/records/patients/<patient_id>/checkup",
    methods=["PUT"]
)
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
    cur = db.execute(
        "UPDATE patients SET next_checkup = ? WHERE patient_id = ?",
        (value, patient_id)
    )

    if cur.rowcount == 0:
        return jsonify(error="Patient not found."), 404

    db.commit()

    return jsonify(
        ok=True,
        next_checkup=value
    )


@app.route(
    "/api/records/screenings/<int:screening_id>/<kind>"
)
@records_required
def screening_image(screening_id, kind):
    if kind not in ("xray", "heatmap"):
        return jsonify(error="Not found."), 404

    row = get_db().execute(
        """
        SELECT xray_file, heatmap_file
        FROM screenings
        WHERE id = ?
        """,
        (screening_id,)
    ).fetchone()

    if not row:
        return jsonify(error="Not found."), 404

    folder = XRAY_DIR if kind == "xray" else HEATMAP_DIR
    filename = (
        row["xray_file"]
        if kind == "xray" else row["heatmap_file"]
    )

    if not filename:
        return jsonify(error="Image missing."), 404

    path = os.path.join(folder, filename)

    if not os.path.isfile(path):
        return jsonify(error="Image missing."), 404

    return send_file(path, mimetype="image/png")


# Start Flask

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
