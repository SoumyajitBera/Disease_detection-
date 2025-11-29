# new_api.py  (or disease_api.py)
#
# AgroAI – Rice/Wheat Disease Detection + State-wise Cure
# - CNN model inference (Rice + Wheat)
# - State-wise cure lookup via cure_db.py
# - HTML UI:
#     * Upload image
#     * Live camera capture (webcam / phone)
# - JSON APIs:
#     * /predict
#     * /predict_with_cure

import json
import base64
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cure_db import cure_db  # uses india_statewise_rice_wheat_focused_diseases.csv

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

DISEASE_DIR = BASE_DIR / "Disease_Detection"

# ---------------- RICE ----------------
RICE_MODEL_PATH = DISEASE_DIR / "outputs_rice" / "rice_cnn_best.keras"
RICE_CLASS_JSON = DISEASE_DIR / "outputs_rice" / "class_indices.json"
RICE_FALLBACK_CLASSES = ["Bacterialblight", "Brownspot", "Leafsmut"]

# ---------------- WHEAT ----------------
WHEAT_MODEL_PATH = DISEASE_DIR / "outputs_wheat" / "wheat_cnn_best.keras"
WHEAT_CLASS_JSON = DISEASE_DIR / "outputs_wheat" / "class_indices.json"
WHEAT_FALLBACK_CLASSES = [
    "BlackPoint",
    "FusariumFootRot",
    "HealthyLeaf",
    "LeafBlight",
    "WheatBlast",
]

IMG_SIZE = (224, 224)

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
)
log = logging.getLogger("disease-api")

# Turn this to True temporarily if you want to inspect live camera frames
DEBUG_SAVE = False

# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(
    title="AgroAI – Disease + Cure (Rice/Wheat) with Live Camera",
    version="3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# MODEL + LABEL LOADING
# ============================================================
def load_labels(json_path: Path, fallback: List[str]) -> List[str]:
    """Load label list from class_indices.json or fallback list."""
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            labels = [None] * len(data)
            for k, v in data.items():
                labels[v] = k
            return labels
        except Exception as e:
            log.warning("Could not parse class map %s: %s", json_path, e)
    log.warning("Using fallback labels for %s", json_path)
    return fallback


def load_model(model_path: Path, tag: str):
    """Load a keras model and log."""
    if not model_path.exists():
        raise FileNotFoundError(f"[{tag}] Model not found: {model_path}")
    model = keras.models.load_model(model_path)
    log.info("[%s] Model loaded from %s", tag, model_path)
    return model


RICE_CLASS_NAMES = load_labels(RICE_CLASS_JSON, RICE_FALLBACK_CLASSES)
WHEAT_CLASS_NAMES = load_labels(WHEAT_CLASS_JSON, WHEAT_FALLBACK_CLASSES)

RICE_MODEL = load_model(RICE_MODEL_PATH, "rice")
WHEAT_MODEL = load_model(WHEAT_MODEL_PATH, "wheat")

MODELS = {"rice": RICE_MODEL, "wheat": WHEAT_MODEL}
CLASS_MAP = {"rice": RICE_CLASS_NAMES, "wheat": WHEAT_CLASS_NAMES}


def get_model_and_labels(crop: str):
    crop_key = crop.lower().strip()
    if crop_key not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported crop '{crop}'. Use one of: {list(MODELS.keys())}",
        )
    return MODELS[crop_key], CLASS_MAP[crop_key]


# ============================================================
# UTILS
# ============================================================
def preprocess(img_bytes: bytes):
    """
    Decode image, center-crop to square, then resize to IMG_SIZE.
    This makes live camera input closer to training distribution.
    """
    # Decode
    img = tf.io.decode_image(img_bytes, channels=3)
    img.set_shape([None, None, 3])  # ensure rank known

    # Center-crop to square (important for webcam frames)
    shape = tf.shape(img)
    h = shape[0]
    w = shape[1]
    side = tf.minimum(h, w)

    offset_h = (h - side) // 2
    offset_w = (w - side) // 2

    img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, side, side)

    # Normalize & resize exactly as training
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, IMG_SIZE)

    # Add batch dimension
    return tf.expand_dims(img, 0)


def gradcam_heatmap(model, img_tensor, class_index: int):
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
    if not last_conv:
        raise RuntimeError("No Conv2D layer found for Grad-CAM")

    grad_model = keras.Model(
        [model.inputs],
        [model.get_layer(last_conv).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(weights[:, None, None, :] * conv_out, axis=-1)[0].numpy()

    cam = np.maximum(cam, 0)
    cam /= np.max(cam) + 1e-8
    return cam


def overlay_heatmap(img_bytes: bytes, heatmap: np.ndarray):
    import cv2

    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    _, buffer = cv2.imencode(".jpg", overlay)
    return base64.b64encode(buffer).decode("utf-8")


# ============================================================
# Pydantic SCHEMA
# ============================================================
class PredictResponse(BaseModel):
    crop: str
    label: str
    confidence: float
    probs: Dict[str, float]
    gradcam_b64: Optional[str] = None


class CureInfo(BaseModel):
    crop: str
    state: str
    disease: str
    cultural_management: str
    chemical_management: str
    dose: str
    season: str
    notes: str


class PredictWithCureResponse(BaseModel):
    crop: str
    label: str
    confidence: float
    probs: Dict[str, float]
    state: str
    cure: Optional[CureInfo] = None
    gradcam_b64: Optional[str] = None


# ============================================================
# HTML UI – ROOT (upload + live camera)
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    HTML UI:
    - Upload image OR use live camera.
    - User selects crop + state.
    - Frontend calls /predict_with_cure and renders disease + cure.
    """
    state_options = """
        <option value="">--Choose State--</option>
        <option value="Punjab">Punjab</option>
        <option value="Haryana">Haryana</option>
        <option value="Uttar Pradesh">Uttar Pradesh</option>
        <option value="Bihar">Bihar</option>
        <option value="Rajasthan">Rajasthan</option>
        <option value="Madhya Pradesh">Madhya Pradesh</option>
        <option value="Gujarat">Gujarat</option>
        <option value="Maharashtra">Maharashtra</option>
        <option value="West Bengal">West Bengal</option>
        <option value="Odisha">Odisha</option>
        <option value="Assam">Assam</option>
        <option value="Jharkhand">Jharkhand</option>
        <option value="Chhattisgarh">Chhattisgarh</option>
        <option value="Tamil Nadu">Tamil Nadu</option>
        <option value="Karnataka">Karnataka</option>
        <option value="Andhra Pradesh">Andhra Pradesh</option>
        <option value="Telangana">Telangana</option>
        <option value="Kerala">Kerala</option>
        <option value="Himachal Pradesh">Himachal Pradesh</option>
        <option value="Uttarakhand">Uttarakhand</option>
        <option value="Jammu & Kashmir">Jammu & Kashmir</option>
    """

    html = """
    <html>
    <head>
        <title>AgroAI – Live Disease & Cure</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <style>
            :root {
                --green-dark: #0b4f32;
                --green: #198754;
                --green-soft: #dff6e6;
                --gold: #f2b705;
                --bg: #0f172a;
                --card-bg: #0b2532;
                --border-soft: #1f2937;
                --text-main: #f9fafb;
                --text-muted: #9ca3af;
            }

            * {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: radial-gradient(circle at top, #14532d 0, #020617 45%, #020617 100%);
                color: var(--text-main);
                -webkit-font-smoothing: antialiased;
            }

            .page {
                max-width: 1080px;
                margin: 0 auto;
                padding: 20px 16px 40px 16px;
            }

            .top-bar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 20px;
            }

            .brand {
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .brand-logo {
                width: 40px;
                height: 40px;
                border-radius: 12px;
                background: conic-gradient(from 160deg, #22c55e, #facc15, #22c55e);
                display: flex;
                align-items: center;
                justify-content: center;
                color: #022c22;
                font-size: 22px;
                font-weight: 800;
            }

            .brand-text-title {
                font-weight: 700;
                letter-spacing: 0.04em;
                font-size: 18px;
                text-transform: uppercase;
            }

            .brand-text-sub {
                font-size: 11px;
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 0.12em;
            }

            .badge-env {
                padding: 4px 10px;
                border-radius: 999px;
                background: rgba(34, 197, 94, 0.1);
                border: 1px solid rgba(34, 197, 94, 0.4);
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #bbf7d0;
            }

            .hero {
                display: grid;
                grid-template-columns: minmax(0, 2.2fr) minmax(0, 1.5fr);
                gap: 24px;
                align-items: stretch;
                margin-bottom: 24px;
            }

            @media (max-width: 900px) {
                .hero {
                    grid-template-columns: minmax(0, 1fr);
                }
            }

            .hero-main {
                background: radial-gradient(circle at top left, rgba(34,197,94,0.16), transparent 55%),
                            radial-gradient(circle at bottom right, rgba(234,179,8,0.12), transparent 55%),
                            linear-gradient(135deg, #020617, #020617);
                border-radius: 24px;
                padding: 22px 22px 18px 22px;
                border: 1px solid rgba(148,163,184,0.25);
                box-shadow: 0 18px 40px rgba(15,23,42,0.75);
            }

            .hero-tagline {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 4px 10px;
                border-radius: 999px;
                background: rgba(15,23,42,0.7);
                border: 1px solid rgba(148,163,184,0.5);
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: #e5e7eb;
            }

            .hero-tagline span.dot {
                width: 7px;
                height: 7px;
                border-radius: 999px;
                background: #22c55e;
                box-shadow: 0 0 0 5px rgba(34,197,94,0.4);
            }

            .hero-title {
                margin-top: 14px;
                font-size: 28px;
                font-weight: 700;
                line-height: 1.25;
            }

            .hero-title span-em {
                color: #bbf7d0;
            }

            .hero-subtitle {
                margin-top: 10px;
                font-size: 14px;
                color: var(--text-muted);
            }

            .hero-pills {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-top: 14px;
            }

            .pill {
                padding: 6px 10px;
                border-radius: 999px;
                border: 1px solid rgba(148,163,184,0.5);
                font-size: 11px;
                display: inline-flex;
                align-items: center;
                gap: 6px;
                background: rgba(15,23,42,0.8);
            }

            .pill.green {
                border-color: rgba(34,197,94,0.6);
                background: radial-gradient(circle at top left, rgba(34,197,94,0.16), rgba(15,23,42,0.9));
            }

            .hero-stats {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 10px;
                margin-top: 18px;
            }

            @media (max-width: 600px) {
                .hero-stats {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }

            .stat-card {
                padding: 10px 10px 9px 10px;
                border-radius: 14px;
                background: rgba(15,23,42,0.95);
                border: 1px solid rgba(148,163,184,0.25);
                font-size: 11px;
                color: var(--text-muted);
            }

            .stat-value {
                font-size: 16px;
                font-weight: 600;
                color: #e5e7eb;
            }

            .hero-note {
                margin-top: 12px;
                font-size: 11px;
                color: #9ca3af;
            }

            .hero-secondary {
                border-radius: 24px;
                padding: 16px 16px 14px 16px;
                background: linear-gradient(145deg, #052e16, #0b1120);
                border: 1px solid rgba(34,197,94,0.5);
                box-shadow: 0 16px 40px rgba(0,0,0,0.6);
            }

            .mini-heading {
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.16em;
                color: #bbf7d0;
            }

            .hero-secondary-title {
                margin-top: 6px;
                font-size: 15px;
                font-weight: 600;
            }

            .hero-secondary-sub {
                margin-top: 8px;
                font-size: 12px;
                color: #d1fae5;
            }

            .hero-secondary-list {
                margin-top: 10px;
                padding-left: 16px;
                font-size: 11px;
                color: #a7f3d0;
            }

            .mode-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 16px;
                margin-top: 10px;
            }

            @media (max-width: 900px) {
                .mode-grid {
                    grid-template-columns: minmax(0, 1fr);
                }
            }

            .block {
                border-radius: 18px;
                padding: 16px 14px 14px 14px;
                background: linear-gradient(160deg, rgba(15,23,42,0.96), rgba(15,23,42,0.9));
                border: 1px solid rgba(55,65,81,0.7);
                box-shadow: 0 16px 35px rgba(15,23,42,0.75);
            }

            .block-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 6px;
            }

            .block-title {
                font-size: 15px;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 6px;
            }

            .block-title span.icon {
                font-size: 17px;
            }

            .block-subtitle {
                font-size: 11px;
                color: var(--text-muted);
                margin-bottom: 10px;
            }

            label {
                font-size: 11px;
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }

            select, input[type="file"] {
                margin-top: 4px;
                padding: 7px 8px;
                border-radius: 10px;
                border: 1px solid rgba(75,85,99,0.9);
                background: rgba(15,23,42,0.9);
                color: var(--text-main);
                font-size: 12px;
                width: 100%;
            }

            input[type="file"] {
                padding: 8px;
                font-size: 12px;
            }

            .field-row {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 10px;
                margin-bottom: 10px;
            }

            @media (max-width: 600px) {
                .field-row {
                    grid-template-columns: minmax(0, 1fr);
                }
            }

            .checkbox-row {
                margin-top: 8px;
                display: flex;
                align-items: center;
                gap: 6px;
                font-size: 12px;
                color: var(--text-muted);
            }

            .checkbox-row input[type="checkbox"] {
                width: 14px;
                height: 14px;
            }

            button {
                padding: 8px 14px;
                margin-top: 10px;
                border-radius: 999px;
                border: none;
                background: linear-gradient(135deg, #22c55e, #22c55e);
                color: #022c22;
                font-weight: 600;
                font-size: 13px;
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                gap: 6px;
                box-shadow: 0 10px 25px rgba(22,163,74,0.55);
                transition: transform 0.08s ease, box-shadow 0.08s ease, filter 0.08s ease;
            }

            button:hover {
                transform: translateY(-1px);
                filter: brightness(1.05);
                box-shadow: 0 14px 30px rgba(22,163,74,0.65);
            }

            button:active {
                transform: translateY(0);
                box-shadow: 0 8px 18px rgba(22,163,74,0.55);
            }

            button.secondary {
                background: transparent;
                border: 1px solid rgba(148,163,184,0.7);
                color: #e5e7eb;
                box-shadow: none;
            }

            button.secondary:hover {
                background: rgba(15,23,42,0.8);
                box-shadow: 0 8px 20px rgba(15,23,42,0.8);
            }

            video {
                max-width: 100%;
                border-radius: 14px;
                border: 1px solid rgba(55,65,81,0.9);
                margin-top: 10px;
                background: #020617;
            }

            #results {
                margin-top: 22px;
            }

            #results.block {
                background: radial-gradient(circle at top left, rgba(34,197,94,0.12), rgba(15,23,42,0.96));
                border: 1px solid rgba(52,211,153,0.7);
            }

            #results h2 {
                margin-top: 0;
            }

            .result-header {
                display: flex;
                flex-wrap: wrap;
                align-items: baseline;
                gap: 8px;
                margin-bottom: 6px;
            }

            .result-pill {
                padding: 3px 10px;
                border-radius: 999px;
                background: rgba(15,23,42,0.9);
                border: 1px solid rgba(16,185,129,0.8);
                font-size: 11px;
            }

            .result-meta {
                font-size: 12px;
                color: var(--text-muted);
                margin-bottom: 8px;
            }

            .result-section-title {
                margin-top: 10px;
                font-size: 13px;
                font-weight: 600;
            }

            .result-grid {
                display: grid;
                grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
                gap: 18px;
                margin-top: 6px;
            }

            @media (max-width: 900px) {
                .result-grid {
                    grid-template-columns: minmax(0, 1fr);
                }
            }

            .cure-text p {
                font-size: 12px;
                margin: 4px 0;
            }

            .cure-text b {
                color: #d1fae5;
            }

            .gradcam-wrapper img {
                max-width: 100%;
                border-radius: 14px;
                border: 1px solid rgba(148,163,184,0.7);
                background: #020617;
            }

            .error-text {
                color: #fecaca;
                font-size: 13px;
            }

            small.hint {
                display: block;
                margin-top: 4px;
                font-size: 11px;
                color: var(--text-muted);
            }
        </style>
    </head>
    <body>
        <div class="page">
            <header class="top-bar">
                <div class="brand">
                    <div class="brand-logo">🌾</div>
                    <div>
                        <div class="brand-text-title">AgroAI</div>
                        <div class="brand-text-sub">Leaf disease & state-wise cure</div>
                    </div>
                </div>
                <div class="badge-env">Field Pilot · India</div>
            </header>

            <section class="hero">
                <div class="hero-main">
                    <div class="hero-tagline">
                        <span class="dot"></span>
                        <span>Real-time crop health assistant</span>
                    </div>
                    <h1 class="hero-title">
                        Scan a <span-em>rice</span-em> or <span-em>wheat</span-em> leaf.
                        Get instant disease <span-em>diagnosis & cure</span-em>
                        for your state.
                    </h1>
                    <p class="hero-subtitle">
                        Farmers and agronomists can upload a leaf photo or use live camera to detect disease
                        and view state-specific management practices – cultural, chemical and doses.
                    </p>

                    <div class="hero-pills">
                        <div class="pill green">🌱 Rice · Wheat</div>
                        <div class="pill">🧪 CNN model · Grad-CAM explainability</div>
                        <div class="pill">🧭 State-wise cure mapped to India</div>
                    </div>

                    <div class="hero-stats">
                        <div class="stat-card">
                            <div class="stat-label">Supported crops</div>
                            <div class="stat-value">2</div>
                            <div class="stat-foot">Rice, Wheat</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Disease classes</div>
                            <div class="stat-value">Multiple</div>
                            <div class="stat-foot">Model-based detection</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Cure lookup</div>
                            <div class="stat-value">State-wise</div>
                            <div class="stat-foot">CSV-backed knowledge</div>
                        </div>
                    </div>

                    <p class="hero-note">
                        Tip: For best results, capture a clear close-up of the infected portion of the leaf in daylight.
                    </p>
                </div>

                <aside class="hero-secondary">
                    <div class="mini-heading">how it works</div>
                    <div class="hero-secondary-title">From farm camera to cure card</div>
                    <p class="hero-secondary-sub">
                        1. Choose crop & state<br/>
                        2. Upload or capture a leaf photo<br/>
                        3. AgroAI predicts disease & fetches the cure for your state
                    </p>
                    <ul class="hero-secondary-list">
                        <li>No manual typing of disease names</li>
                        <li>Visual explanation with Grad-CAM (optional)</li>
                        <li>Configurable CSV for agronomy teams</li>
                    </ul>
                </aside>
            </section>

            <section class="mode-grid">
                <!-- Upload Image Block -->
                <div class="block">
                    <div class="block-header">
                        <div class="block-title">
                            <span class="icon">📤</span>
                            <span>Upload leaf image</span>
                        </div>
                    </div>
                    <div class="block-subtitle">
                        For farmers sending photos via WhatsApp, extension agents, or any stored leaf images.
                    </div>

                    <form id="uploadForm">
                        <div class="field-row">
                            <div>
                                <label for="uploadCrop">Crop</label>
                                <select id="uploadCrop" required>
                                    <option value="rice">Rice</option>
                                    <option value="wheat">Wheat</option>
                                </select>
                            </div>
                            <div>
                                <label for="uploadState">State</label>
                                <select id="uploadState" required>
                                    __STATE_OPTIONS__
                                </select>
                            </div>
                        </div>

                        <label for="uploadFile">Leaf photo</label>
                        <input type="file" id="uploadFile" accept="image/*" capture="environment" required />
                        <small class="hint">
                          Use a sharp close-up of a single leaf, avoiding background clutter.
                        </small>

                        <div class="checkbox-row">
                            <input type="checkbox" id="uploadGradcam" />
                            <span>Show Grad-CAM (how the model “sees” the disease)</span>
                        </div>

                        <button type="submit">
                            🔍 Predict disease & show cure
                        </button>
                    </form>
                </div>

                <!-- Live Camera Block -->
                <div class="block">
                    <div class="block-header">
                        <div class="block-title">
                            <span class="icon">📷</span>
                            <span>Live camera scan</span>
                        </div>
                    </div>
                    <div class="block-subtitle">
                        Use directly on a smartphone or laptop webcam standing in the field.
                    </div>

                    <div class="field-row">
                        <div>
                            <label for="liveCrop">Crop</label>
                            <select id="liveCrop" required>
                                <option value="rice">Rice</option>
                                <option value="wheat">Wheat</option>
                            </select>
                        </div>
                        <div>
                            <label for="liveState">State</label>
                            <select id="liveState" required>
                                __STATE_OPTIONS__
                            </select>
                        </div>
                    </div>

                    <div class="checkbox-row">
                        <input type="checkbox" id="liveGradcam" />
                        <span>Show Grad-CAM overlay</span>
                    </div>

                    <button id="startCameraBtn" type="button" class="secondary">
                        ▶ Start camera
                    </button>

                    <video id="video" autoplay playsinline style="display:none;"></video>
                    <canvas id="canvas" style="display:none;"></canvas>

                    <button id="captureBtn" type="button" style="display:none;margin-left:0;margin-top:10px;">
                        📸 Capture & predict
                    </button>

                    <small class="hint">
                      Hold a single infected leaf so it fills most of the frame, keep your hand steady,
                      then tap “Capture & predict”.
                    </small>
                </div>
            </section>

            <section id="results" class="block" style="display:none;"></section>
        </div>

        <script>
            const resultsDiv = document.getElementById("results");

            function renderResult(data) {
                if (!data) {
                    resultsDiv.style.display = "block";
                    resultsDiv.innerHTML = "<p class='error-text'>No response.</p>";
                    return;
                }

                let cureHtml = "";
                if (data.cure) {
                    cureHtml += "<div class='result-section-title'>Cure recommendation – "
                             + data.crop.toUpperCase() + " · " + data.label + " · " + data.state + "</div>";
                    cureHtml += "<div class='cure-text'>";
                    cureHtml += "<p><b>Cultural management:</b> " + data.cure.cultural_management + "</p>";
                    cureHtml += "<p><b>Chemical management:</b> " + data.cure.chemical_management + "</p>";
                    cureHtml += "<p><b>Dose:</b> " + data.cure.dose + "</p>";
                    cureHtml += "<p><b>Season:</b> " + data.cure.season + "</p>";
                    cureHtml += "<p><b>Notes:</b> " + data.cure.notes + "</p>";
                    cureHtml += "</div>";
                } else {
                    cureHtml += "<p class='error-text'>No cure record found for this crop/disease/state combination.</p>";
                }

                let gradcamHtml = "";
                if (data.gradcam_b64) {
                    gradcamHtml += "<div class='result-section-title'>Grad-CAM view</div>";
                    gradcamHtml += "<div class='gradcam-wrapper'><img src='data:image/jpeg;base64,"
                                + data.gradcam_b64 + "' /></div>";
                }

                let html = "";
                html += "<div class='result-header'>";
                html += "<h2>Diagnosis report</h2>";
                html += "<span class='result-pill'>Confidence: " + (data.confidence * 100).toFixed(2) + "%</span>";
                html += "</div>";
                html += "<div class='result-meta'>";
                html += "Crop: <b>" + data.crop + "</b> · Disease: <b>" + data.label + "</b> · State: <b>" + data.state + "</b>";
                html += "</div>";

                html += "<div class='result-grid'>";
                html += "<div>" + cureHtml + "</div>";
                html += "<div>" + gradcamHtml + "</div>";
                html += "</div>";

                resultsDiv.style.display = "block";
                resultsDiv.innerHTML = html;
            }

            async function callPredictWithCure(file, crop, state, withGradcam) {
                const formData = new FormData();
                formData.append("file", file, "leaf.jpg");
                const url = "/predict_with_cure?crop="
                            + encodeURIComponent(crop)
                            + "&state=" + encodeURIComponent(state)
                            + "&with_gradcam=" + (withGradcam ? "true" : "false");

                const resp = await fetch(url, {
                    method: "POST",
                    body: formData
                });
                if (!resp.ok) {
                    const txt = await resp.text();
                    throw new Error("Request failed: " + resp.status + " " + txt);
                }
                return resp.json();
            }

            // Upload form
            document.getElementById("uploadForm").addEventListener("submit", async (e) => {
                e.preventDefault();
                const fileInput = document.getElementById("uploadFile");
                if (!fileInput.files.length) {
                    alert("Please select an image.");
                    return;
                }
                const crop = document.getElementById("uploadCrop").value;
                const state = document.getElementById("uploadState").value;
                if (!state) {
                    alert("Please select state.");
                    return;
                }
                const withGradcam = document.getElementById("uploadGradcam").checked;

                try {
                    const data = await callPredictWithCure(fileInput.files[0], crop, state, withGradcam);
                    renderResult(data);
                } catch (err) {
                    console.error(err);
                    resultsDiv.style.display = "block";
                    resultsDiv.innerHTML = "<p class='error-text'>Error: " + err.message + "</p>";
                }
            });

            // Live camera
            const startCameraBtn = document.getElementById("startCameraBtn");
            const captureBtn = document.getElementById("captureBtn");
            const video = document.getElementById("video");
            const canvas = document.getElementById("canvas");
            let stream = null;

            startCameraBtn.addEventListener("click", async () => {
                try {
                    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
                    video.srcObject = stream;
                    video.style.display = "block";
                    captureBtn.style.display = "inline-flex";
                } catch (err) {
                    alert("Could not access camera: " + err.message);
                }
            });

            captureBtn.addEventListener("click", async () => {
                if (!stream) {
                    alert("Camera not started.");
                    return;
                }
                const crop = document.getElementById("liveCrop").value;
                const state = document.getElementById("liveState").value;
                if (!state) {
                    alert("Please select state.");
                    return;
                }
                const withGradcam = document.getElementById("liveGradcam").checked;

                const w = video.videoWidth || 640;
                const h = video.videoHeight || 480;
                canvas.width = w;
                canvas.height = h;
                const ctx = canvas.getContext("2d");

                // small delay so autofocus / exposure can settle
                await new Promise((resolve) => setTimeout(resolve, 200));

                ctx.drawImage(video, 0, 0, w, h);

                canvas.toBlob(async (blob) => {
                    if (!blob) {
                        alert("Failed to capture image.");
                        return;
                    }
                    try {
                        const data = await callPredictWithCure(blob, crop, state, withGradcam);
                        renderResult(data);
                    } catch (err) {
                        console.error(err);
                        resultsDiv.style.display = "block";
                        resultsDiv.innerHTML = "<p class='error-text'>Error: " + err.message + "</p>";
                    }
                }, "image/jpeg", 0.9);
            });
        </script>
    </body>
    </html>
    """

    # inject state options into both dropdowns
    html = html.replace("__STATE_OPTIONS__", state_options)
    return HTMLResponse(content=html)


# ============================================================
# JSON APIs – programmatic access
# ============================================================

@app.post("/predict", response_model=PredictResponse)
async def predict_api(
    file: UploadFile = File(...),
    crop: str = Query(..., description="Crop type: 'rice' or 'wheat'"),
    with_gradcam: bool = Query(False),
):
    """Disease prediction only (no cure)."""
    model, class_names = get_model_and_labels(crop)

    data = await file.read()
    x = preprocess(data)
    preds = model.predict(x, verbose=0)[0]

    idx = int(np.argmax(preds))
    label = class_names[idx]
    conf = float(preds[idx])
    probs = {class_names[i]: float(p) for i, p in enumerate(preds)}

    gradcam_b64 = None
    if with_gradcam:
        try:
            cam = gradcam_heatmap(model, x, idx)
            gradcam_b64 = overlay_heatmap(data, cam)
        except Exception as e:
            log.warning("Grad-CAM failed: %s", e)

    return PredictResponse(
        crop=crop.lower(),
        label=label,
        confidence=round(conf, 4),
        probs=probs,
        gradcam_b64=gradcam_b64,
    )


@app.post("/predict_with_cure", response_model=PredictWithCureResponse)
async def predict_with_cure_api(
    file: UploadFile = File(...),
    crop: str = Query(..., description="Crop type: 'rice' or 'wheat'"),
    state: str = Query(..., description="Indian state, e.g. 'Punjab'"),
    with_gradcam: bool = Query(False),
):
    """
    Main API:
      - takes image + crop + state
      - predicts disease
      - looks up state-wise cure from cure_db
      - returns both prediction + cure
    """
    model, class_names = get_model_and_labels(crop)

    data = await file.read()

    # Optional: debug save to inspect live frames vs uploads
    if DEBUG_SAVE:
        debug_path = BASE_DIR / "debug_live_frame.jpg"
        with open(debug_path, "wb") as f:
            f.write(data)
        log.info("Saved debug frame to %s", debug_path)

    x = preprocess(data)
    preds = model.predict(x, verbose=0)[0]

    idx = int(np.argmax(preds))
    label = class_names[idx]
    conf = float(preds[idx])
    probs = {class_names[i]: float(p) for i, p in enumerate(preds)}

    rec = cure_db.get_cure(crop=crop, disease=label, state=state)
    cure_payload: Optional[CureInfo] = None
    if rec:
        cure_payload = CureInfo(**rec.to_dict())
    else:
        log.warning(
            "No cure record found for crop=%s, disease=%s, state=%s",
            crop,
            label,
            state,
        )

    gradcam_b64 = None
    if with_gradcam:
        try:
            cam = gradcam_heatmap(model, x, idx)
            gradcam_b64 = overlay_heatmap(data, cam)
        except Exception as e:
            log.warning("Grad-CAM failed in /predict_with_cure: %s", e)

    return PredictWithCureResponse(
        crop=crop.lower(),
        label=label,
        confidence=round(conf, 4),
        probs=probs,
        state=state,
        cure=cure_payload,
        gradcam_b64=gradcam_b64,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("new_api:app", host="0.0.0.0", port=8001, reload=True)
