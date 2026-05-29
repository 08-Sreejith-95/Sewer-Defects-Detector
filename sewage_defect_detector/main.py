# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import onnxruntime as ort
import numpy as np
from preprocess import preprocess_image_bytes
from src.config.config import load_config
from src.utils.arg_parser import parse_args

args = parse_args()
cfg = load_config(args.config)

# ── Sewer-ML class names (17 defect classes + normal) ──────────────
CLASS_NAMES = cfg.class_names  # fallback if config not loaded
THRESHOLD = cfg.onnx_inference.threshold  # fallback if config not loaded

# ── Load ONNX model ONCE at startup (not on every request) for latency optimization ─────────
session = ort.InferenceSession(
    cfg.onnx_inference.onnx_model_path,
    providers=["CPUExecutionProvider"]
)
input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ── FastAPI app ─────────────────────────────────────────────────────
app = FastAPI(
    title="Sewer Defect Detector",
    description="Multi-label defect classification on sewer inspection images",
    version="1.0.0"
)

@app.get("/health")
def health():
    """Simple health check — confirms the server is running."""
    return {"status": "ok", "model": "sewer_convnext_tiny_int8"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a sewer inspection image.
    Returns detected defect classes with confidence scores.
    """
    # 1. Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected image, got {file.content_type}"
        )

    # 2. Read raw bytes from the upload
    image_bytes = await file.read()

    # 3. Preprocess → NCHW float32 array
    input_tensor = preprocess_image_bytes(image_bytes, cfg.dataset.img_size, cfg.dataset.mean, cfg.dataset.std)

    # 4. ONNX inference
    raw_output = session.run(
        [output_name],
        {input_name: input_tensor}
    )[0]  # shape: (1, 17)

    # 5. Post-process: sigmoid → probabilities
    probabilities = 1 / (1 + np.exp(-raw_output[0]))  # sigmoid

    # 6. Build response
    all_scores = {
        CLASS_NAMES[i]: round(float(probabilities[i]), 4)
        for i in range(len(CLASS_NAMES))
    }
    detected = [
        {"class": CLASS_NAMES[i], "confidence": round(float(probabilities[i]), 4)}
        for i in range(len(CLASS_NAMES))
        if probabilities[i] >= THRESHOLD
    ]
    is_normal = len(detected) == 0 or (
        len(detected) == 1 and detected[0]["class"] == "OK"
    )

    return JSONResponse({
        "filename": file.filename,
        "detected_defects": detected,
        "is_normal": is_normal,
        "all_scores": all_scores,
        "threshold": THRESHOLD
    })