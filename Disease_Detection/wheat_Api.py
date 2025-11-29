import os
import io
import json
import base64
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -------------------------------------
# CONFIG
# -------------------------------------
# -------------------------------------
# CONFIG
# -------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "outputs_wheat" / "wheat_cnn_best.keras"
CLASS_MAP_JSON = BASE_DIR / "outputs_wheat" / "class_indices.json"
IMG_SIZE = (224, 224)
FALLBACK_CLASSES = ["BlackPoint", "FusariumFootRot", "HealthyLeaf", "LeafBlight", "WheatBlast"]


# -------------------------------------
# LOGGING
# -------------------------------------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')
log = logging.getLogger("wheat-api")

# -------------------------------------
# APP SETUP
# -------------------------------------
app = FastAPI(title="Wheat Disease Classifier", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -------------------------------------
# MODEL LOADING
# -------------------------------------
def load_labels(json_path: Path) -> List[str]:
    if json_path.exists():
        try:
            data = json.load(open(json_path))
            labels = [None] * len(data)
            for k, v in data.items():
                labels[v] = k
            return labels
        except Exception as e:
            log.warning("Could not parse class map: %s", e)
    return FALLBACK_CLASSES

CLASS_NAMES = load_labels(CLASS_MAP_JSON)

def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = keras.models.load_model(model_path)
    log.info("Model loaded: %s", model_path)
    return model

MODEL = load_model(MODEL_PATH)

# -------------------------------------
# UTILS
# -------------------------------------
def preprocess(img_bytes: bytes):
    img = tf.io.decode_image(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, IMG_SIZE)
    return tf.expand_dims(img, 0)

def gradcam_heatmap(model, img_tensor, class_index):
    # find last conv
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
    if not last_conv:
        raise RuntimeError("No Conv2D layer found for Grad-CAM")
    grad_model = keras.Model([model.inputs], [model.get_layer(last_conv).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        loss = preds[:, class_index]
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(tf.multiply(weights[:, None, None, :], conv_out), axis=-1)[0].numpy()
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

# -------------------------------------
# SCHEMA
# -------------------------------------
class PredictResponse(BaseModel):
    label: str
    confidence: float
    probs: Dict[str, float]
    gradcam_b64: Optional[str] = None

# -------------------------------------
# ROUTES
# -------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Wheat Disease Classifier 🌾</title>
        </head>
        <body style="font-family:sans-serif;text-align:center;">
            <h1>🌾 Wheat Disease Classifier</h1>
            <p>Upload a wheat leaf image to get the predicted disease.</p>
            <form action="/upload" enctype="multipart/form-data" method="post">
                <input type="file" name="file" accept="image/*" required><br><br>
                <label><input type="checkbox" name="with_gradcam"> Show Grad-CAM</label><br><br>
                <input type="submit" value="Predict" style="padding:6px 16px;">
            </form>
        </body>
    </html>
    """

@app.post("/upload", response_class=HTMLResponse)
async def upload_image(file: UploadFile = File(...), with_gradcam: bool = Form(False)):
    try:
        data = await file.read()
        x = preprocess(data)
        preds = MODEL.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        label = CLASS_NAMES[idx]
        conf = float(preds[idx])
        html = f"<h2>Prediction: <b>{label}</b> ({conf*100:.2f}%)</h2>"
        if with_gradcam:
            try:
                cam = gradcam_heatmap(MODEL, x, idx)
                b64 = overlay_heatmap(data, cam)
                html += f'<img src="data:image/jpeg;base64,{b64}" style="max-width:90%;border:1px solid #ccc;margin-top:10px;">'
            except Exception as e:
                html += f"<p style='color:red;'>Grad-CAM failed: {e}</p>"
        return f"<html><body style='text-align:center;'>{html}<br><a href='/'>← Try another</a></body></html>"
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict", response_model=PredictResponse)
async def predict_api(file: UploadFile = File(...), with_gradcam: bool = Query(False)):
    data = await file.read()
    x = preprocess(data)
    preds = MODEL.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    label = CLASS_NAMES[idx]
    conf = float(preds[idx])
    result = {
        "label": label,
        "confidence": round(conf, 4),
        "probs": {CLASS_NAMES[i]: float(p) for i, p in enumerate(preds)},
    }
    if with_gradcam:
        try:
            cam = gradcam_heatmap(MODEL, x, idx)
            result["gradcam_b64"] = overlay_heatmap(data, cam)
        except Exception as e:
            log.warning("Grad-CAM failed: %s", e)
    return result