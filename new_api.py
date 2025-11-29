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
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
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
    img = tf.io.decode_image(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, IMG_SIZE)
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
        <style>
            body {
                font-family: sans-serif;
                max-width: 900px;
                margin: 20px auto;
                text-align: center;
            }
            .block {
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 16px;
                margin-bottom: 20px;
            }
            #results {
                margin-top: 20px;
                text-align: left;
            }
            video {
                max-width: 100%;
                border: 1px solid #ccc;
                border-radius: 8px;
            }
            button {
                padding: 6px 16px;
                margin-top: 8px;
            }
        </style>
    </head>
    <body>
        <h1>🌾 AgroAI – Disease Detection + Cure</h1>
        <p>Select crop and state, then either upload an image or use live camera. The system will detect the disease and show the cure.</p>

        <div class="block">
            <h2>1️⃣ Upload Image</h2>
            <form id="uploadForm">
                <label>Crop:</label>
                <select id="uploadCrop" required>
                    <option value="rice">Rice</option>
                    <option value="wheat">Wheat</option>
                </select>
                &nbsp;&nbsp;
                <label>State:</label>
                <select id="uploadState" required>
                    __STATE_OPTIONS__
                </select>
                <br><br>
                <input type="file" id="uploadFile" accept="image/*" capture="environment" required />
                <br><br>
                <label><input type="checkbox" id="uploadGradcam" /> With Grad-CAM</label>
                <br><br>
                <button type="submit">Predict Disease & Cure</button>
            </form>
        </div>

        <div class="block">
            <h2>2️⃣ Live Camera</h2>
            <p>Use this on a laptop with webcam or a phone browser.</p>
            <label>Crop:</label>
            <select id="liveCrop" required>
                <option value="rice">Rice</option>
                <option value="wheat">Wheat</option>
            </select>
            &nbsp;&nbsp;
            <label>State:</label>
            <select id="liveState" required>
                __STATE_OPTIONS__
            </select>
            <br><br>
            <button id="startCameraBtn" type="button">Start Camera</button>
            <br><br>
            <video id="video" autoplay playsinline style="display:none;"></video>
            <canvas id="canvas" style="display:none;"></canvas>
            <br>
            <label><input type="checkbox" id="liveGradcam" /> With Grad-CAM</label>
            <br>
            <button id="captureBtn" type="button" style="display:none;">Capture & Predict</button>
        </div>

        <div id="results" class="block" style="display:none;"></div>

        <script>
            const resultsDiv = document.getElementById("results");

            function renderResult(data) {
                if (!data) {
                    resultsDiv.style.display = "block";
                    resultsDiv.innerHTML = "<p style='color:red;'>No response.</p>";
                    return;
                }

                let cureHtml = "";
                if (data.cure) {
                    cureHtml += "<h3>Cure Recommendation – " + data.crop.toUpperCase() +
                                " / " + data.label + " / " + data.state + "</h3>";
                    cureHtml += "<p><b>Cultural management:</b> " + data.cure.cultural_management + "</p>";
                    cureHtml += "<p><b>Chemical management:</b> " + data.cure.chemical_management + "</p>";
                    cureHtml += "<p><b>Dose:</b> " + data.cure.dose + "</p>";
                    cureHtml += "<p><b>Season:</b> " + data.cure.season + "</p>";
                    cureHtml += "<p><b>Notes:</b> " + data.cure.notes + "</p>";
                } else {
                    cureHtml += "<p style='color:red;'>No cure record found for this crop/disease/state combination.</p>";
                }

                let gradcamHtml = "";
                if (data.gradcam_b64) {
                    gradcamHtml += "<h4>Grad-CAM</h4>";
                    gradcamHtml += "<img src='data:image/jpeg;base64," + data.gradcam_b64 +
                                   "' style='max-width:100%;border:1px solid #ccc;border-radius:6px;'/>";
                }

                let html = "";
                html += "<h2>Prediction</h2>";
                html += "<p><b>Crop:</b> " + data.crop + "</p>";
                html += "<p><b>Disease:</b> " + data.label + "</p>";
                html += "<p><b>Confidence:</b> " + (data.confidence * 100).toFixed(2) + "%</p>";
                html += cureHtml;
                html += gradcamHtml;

                resultsDiv.style.display = "block";
                resultsDiv.innerHTML = html;
            }

            async function callPredictWithCure(file, crop, state, withGradcam) {
                const formData = new FormData();
                formData.append("file", file, "leaf.jpg");
                const url = "/predict_with_cure?crop=" +
                            encodeURIComponent(crop) +
                            "&state=" + encodeURIComponent(state) +
                            "&with_gradcam=" + (withGradcam ? "true" : "false");

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
                    resultsDiv.innerHTML = "<p style='color:red;'>Error: " + err.message + "</p>";
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
                    captureBtn.style.display = "inline-block";
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
                        resultsDiv.innerHTML = "<p style='color:red;'>Error: " + err.message + "</p>";
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
