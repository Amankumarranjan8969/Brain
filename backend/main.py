"""FastAPI server for Brain Tumor Detection."""
import io
import os
import base64
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from tensorflow.keras.applications.densenet import preprocess_input

from backend.model_loader import load_model_once
from backend.gradcam import compute_gradcam, overlay_heatmap

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "backend", "models")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# ---------- App ----------
app = FastAPI(title="Brain Tumor Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend statically
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.on_event("startup")
def _startup():
    """Load the model exactly once when the server boots."""
    load_model_once(MODEL_DIR)


@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an MRI image, returns:
      - predicted class
      - confidence
      - full probability distribution
      - Grad-CAM overlay as base64 PNG
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    model, class_names, config = load_model_once(MODEL_DIR)
    img_size = config["img_size"]

    # --- Read image ---
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # Resize for model
    original_rgb = np.array(pil_img)                       # for overlay
    resized = cv2.resize(original_rgb, (img_size, img_size))
    model_input = preprocess_input(resized.astype(np.float32))
    batch = np.expand_dims(model_input, axis=0)

    # --- Predict ---
    probs = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    # --- Grad-CAM ---
    try:
        heatmap, _ = compute_gradcam(batch, class_idx=pred_idx)
        # Resize original to model size for overlay consistency
        overlay = overlay_heatmap(resized, heatmap)
        # Encode to base64
        _, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        gradcam_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception as e:
        print(f"[gradcam] failed: {e}")
        gradcam_b64 = None

    # --- Original image as base64 (so we can display side-by-side) ---
    _, buf_orig = cv2.imencode(".png", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    original_b64 = base64.b64encode(buf_orig.tobytes()).decode("utf-8")

    return {
        "predicted_class": pred_class,
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            class_names[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(class_names))
        },
        "gradcam_image": f"data:image/png;base64,{gradcam_b64}" if gradcam_b64 else None,
        "original_image": f"data:image/png;base64,{original_b64}",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
