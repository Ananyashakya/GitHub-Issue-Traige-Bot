import io, json, os, time, uuid, hashlib, secrets
from collections import defaultdict
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd
import torch
from flask import (Flask, jsonify, render_template, request,
                   send_file, session)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── App setup ───────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# ── Config ──────────────────────────────────────────────────────
MODEL_DIR = "roberta_model"
MAX_LEN   = 384
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Demo users  (replace with DB in production) ─────────────────
def _hash(pw): return hashlib.sha256(pw.encode()).hexdigest()

USERS = {
    "admin":     {"password": _hash("admin123"),  "role": "admin",  "name": "Admin User"},
    "analyst":   {"password": _hash("analyst123"),"role": "analyst","name": "ML Analyst"},
    "viewer":    {"password": _hash("viewer123"),  "role": "viewer", "name": "Viewer"},
}

# ── Load model ──────────────────────────────────────────────────
print(f"[startup] Loading model from '{MODEL_DIR}' on {DEVICE} …")
t0 = time.time()

with open(os.path.join(MODEL_DIR, "label_config.json")) as f:
    label_cfg = json.load(f)

ID2LABEL = label_cfg["id2label"]          # {"0":"bug", …}
LABEL2ID = label_cfg["labels"]            # {"bug":0, …}
LABELS   = list(LABEL2ID.keys())
N_LABELS = len(LABELS)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, num_labels=N_LABELS, ignore_mismatched_sizes=True
).to(DEVICE)
model.eval()
print(f"[startup] Ready in {time.time()-t0:.1f}s  labels={LABELS}  device={DEVICE}")

# Colour palette — auto-assigned for any label set
BASE_COLORS = [
    "#E05A5A","#4F8EFF","#3DD68C","#F5C842","#9B7EF5",
    "#F57C42","#42C8F5","#F542A7","#82C45A","#F5A742",
]
LABEL_COLORS = {lbl: BASE_COLORS[i % len(BASE_COLORS)] for i, lbl in enumerate(LABELS)}

# Load training history from results.json if available
TRAINING_HISTORY = []
TEST_REPORT_RAW  = ""
try:
    with open(os.path.join(MODEL_DIR, "results.json")) as f:
        res = json.load(f)
    TRAINING_HISTORY = res.get("history", [])
    TEST_REPORT_RAW  = res.get("test_report", "")
    MODEL_ACC = res.get("test_acc", 0)
    MODEL_F1  = res.get("test_f1",  0)
except Exception:
    MODEL_ACC = 0
    MODEL_F1  = 0

# ── Session analytics store ──────────────────────────────────────
session_log = []     # global across users (per-server memory)
batch_store = {}     # batch_id → bytes

# ── Auth helpers ─────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            return jsonify({"error": "Authentication required."}), 401
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            return jsonify({"error": "Authentication required."}), 401
        user = USERS.get(session["username"], {})
        if user.get("role") != "admin":
            return jsonify({"error": "Admin access required."}), 403
        return f(*args, **kwargs)
    return decorated

# ── Inference ────────────────────────────────────────────────────
def _predict_one(title: str, body: str) -> dict:
    """
    Exact replication of training tokenisation:
      tokenizer(title, body, max_length=MAX_LEN, padding='max_length',
                truncation=True, return_tensors='pt')
    Works for any label set — reads from label_config.json dynamically.
    """
    title = (title or "").strip()
    body  = (body  or "").strip()

    enc = tokenizer(
        title,
        body if body else None,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE),
        ).logits
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()


            

    pred_id    = int(np.argmax(probs))
    pred_label = ID2LABEL[str(pred_id)]
    confidence = float(probs[pred_id])
    return {
        "label":       pred_label,
        "confidence":  round(confidence * 100, 1),
        "confidences": {ID2LABEL[str(i)]: round(float(p) * 100, 1)
                        for i, p in enumerate(probs)},
        "color":       LABEL_COLORS.get(pred_label, "#888"),
    }

# ── Routes — Auth ────────────────────────────────────────────────
@app.route("/api/login", methods=["POST"])
def login():
    d = request.get_json(force=True)
    username = d.get("username", "").strip().lower()
    password = d.get("password", "")
    user = USERS.get(username)
    if not user or user["password"] != _hash(password):
        return jsonify({"error": "Invalid credentials."}), 401
    session["username"] = username
    return jsonify({"username": username, "name": user["name"], "role": user["role"]})

@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})

@app.route("/api/me")
def me():
    if "username" not in session:
        return jsonify({"authenticated": False})
    u = USERS[session["username"]]
    return jsonify({"authenticated": True, "username": session["username"],
                    "name": u["name"], "role": u["role"]})

# ── Routes — Predict ─────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
@login_required
def predict_single():
    print("API CALLED")
    d     = request.get_json(force=True)
    title = d.get("title", "").strip()
    body  = d.get("body",  "").strip()
    if not title and not body:
        return jsonify({"error": "Provide at least a title or body."}), 400
    try:
        result = _predict_one(title, body)
        session_log.append({
            "ts":         datetime.now().isoformat(),
            "label":      result["label"],
            "confidence": result["confidence"],
            "source":     "single",
            "user":       session.get("username", "?"),
        })
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/batch", methods=["POST"])
@login_required
def predict_batch():
    if "file" not in request.files:
        return jsonify({"error": "No file attached."}), 400
    f    = request.files["file"]
    name = f.filename.lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(f)
        elif name.endswith(".csv"):
            df = pd.read_csv(f)
        else:
            return jsonify({"error": "Only .xlsx, .xls, or .csv files accepted."}), 400
    except Exception as e:
        return jsonify({"error": f"Could not read file: {e}"}), 400

    df.columns = [c.strip().lower() for c in df.columns]
    if "title" not in df.columns:
        return jsonify({"error": "File must contain a 'title' column."}), 400
    if "body" not in df.columns:
        df["body"] = ""

    # Drop phantom empty rows Excel sometimes appends
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"].replace("nan", "").str.len() > 0].reset_index(drop=True)
    df["body"] = df["body"].fillna("").astype(str)

    if len(df) == 0:
        return jsonify({"error": "File has no valid rows (title column is empty)."}), 400
    if len(df) > 1000:
        return jsonify({"error": "Maximum 1,000 rows per batch."}), 400

    print(f"[batch] {len(df)} valid rows after stripping empty titles")

    results = []
    for _, row in df.iterrows():
        r = _predict_one(str(row["title"]), str(row.get("body", "")))
        results.append(r)
        session_log.append({
            "ts": datetime.now().isoformat(), "label": r["label"],
            "confidence": r["confidence"], "source": "batch",
            "user": session.get("username", "?"),
        })

    # Build output Excel
    out = df.copy()
    out["predicted_label"] = [r["label"]      for r in results]
    out["confidence_%"]    = [r["confidence"] for r in results]
    for lbl in LABELS:
        out[f"score_{lbl}"] = [r["confidences"].get(lbl, 0) for r in results]

    bid = str(uuid.uuid4())[:8]
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            out.to_excel(writer, index=False, sheet_name="Predictions")
            wb = writer.book; ws = writer.sheets["Predictions"]
            ci = list(out.columns).index("predicted_label")
            for ri, lbl in enumerate(out["predicted_label"], 1):
                hx  = LABEL_COLORS.get(lbl, "#888888").lstrip("#")
                fmt = wb.add_format({"bg_color": f"#{hx}", "font_color": "#000000", "bold": True})
                ws.write(ri, ci, lbl, fmt)
        buf.seek(0)
        batch_store[bid] = buf.getvalue()
        print(f"[batch] Excel saved id={bid}")
    except Exception as excel_err:
        print(f"[batch] Excel failed ({excel_err}), saving CSV")
        buf2 = io.BytesIO()
        buf2.write(out.to_csv(index=False).encode("utf-8"))
        buf2.seek(0)
        batch_store[bid] = buf2.getvalue()

    lc = defaultdict(int)
    for r in results: lc[r["label"]] += 1
    avg_conf = round(sum(r["confidence"] for r in results) / len(results), 1)

    return jsonify({
        "total": len(results), "results": results,
        "titles": df["title"].tolist(),
        "label_counts": dict(lc), "avg_confidence": avg_conf,
        "download_id": bid, "labels": LABELS,
        "label_colors": LABEL_COLORS,
    })

@app.route("/api/download/<bid>")
@login_required
def download(bid):
    if bid not in batch_store:
        return "File not found or expired.", 404
    data = batch_store[bid]
    is_excel = data[:4] == b"PK\x03\x04"
    if is_excel:
        return send_file(io.BytesIO(data),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True, download_name=f"triage_{bid}.xlsx")
    return send_file(io.BytesIO(data),
        mimetype="text/csv",
        as_attachment=True, download_name=f"triage_{bid}.csv")

@app.route("/api/analytics")
@login_required
def analytics():
    if not session_log:
        return jsonify({"empty": True})
    lc = defaultdict(int); cb = defaultdict(list)
    for e in session_log:
        lc[e["label"]] += 1
        cb[e["label"]].append(e["confidence"])
    return jsonify({
        "total":             len(session_log),
        "label_counts":      dict(lc),
        "avg_conf_by_label": {l: round(sum(v)/len(v), 1) for l, v in cb.items()},
        "timeline":          session_log[-120:],
        "single_count":      sum(1 for e in session_log if e["source"] == "single"),
        "batch_count":       sum(1 for e in session_log if e["source"] == "batch"),
        "label_colors":      LABEL_COLORS,
    })

@app.route("/api/health")
def health():
    return jsonify({
        "status": "operational", "model": "roberta-base",
        "labels": LABELS, "n_labels": N_LABELS,
        "device": str(DEVICE),
        "test_accuracy": f"{MODEL_ACC*100:.2f}%",
        "macro_f1":      f"{MODEL_F1*100:.2f}%",
        "uptime_s":      round(time.time() - t0),
    })

@app.route("/api/model-info")
@login_required
def model_info():
    return jsonify({
        "labels":           LABELS,
        "label_colors":     LABEL_COLORS,
        "training_history": TRAINING_HISTORY,
        "test_acc":         MODEL_ACC,
        "test_f1":          MODEL_F1,
        "n_labels":         N_LABELS,
        "test_report":      TEST_REPORT_RAW,
    })

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def spa(path):
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)