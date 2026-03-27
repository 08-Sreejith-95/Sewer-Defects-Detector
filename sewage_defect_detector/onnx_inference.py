"""
ONNX inference module — CPU deployment, no PyTorch required at runtime.
Replicates the exact preprocessing from infer.py.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image


# -----------------!!!Must match config exactly!!!-----------------
IMG_SIZE   = 224 #change for modified head model, for default head model it is 224 --- IGNORE ---
MEAN       = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD        = np.array([0.229, 0.224, 0.225], dtype=np.float32)
THRESHOLD  = 0.4
CLASS_NAMES = [
    "VA","RB","OB","PF","DE","FS","IS","RO","IN",
    "AF","BE","FO","GR","PH","PB","OS","OP","OK","ND"
]


def preprocess(image_path: str) -> np.ndarray:
    """PIL load --> resize --> float32 --> normalize ---> NCHW."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0          # [0,1]
    arr = (arr - MEAN) / STD                                # normalize
    arr = arr.transpose(2, 0, 1)[np.newaxis]                # HWC ---> NCHW
    return arr


def predict_single(
    session: ort.InferenceSession,
    image_path: str,
    threshold: float = THRESHOLD,
) -> dict:
    input_tensor = preprocess(image_path)
    t0 = time.perf_counter()
    logits = session.run(None, {"image": input_tensor})[0][0]        # (19,)
    latency_ms = (time.perf_counter() - t0) * 1000

    probs  = 1.0 / (1.0 + np.exp(-logits))                 # sigmoid
    labels = [c for c, p in zip(CLASS_NAMES, probs) if p >= threshold]
    return {
        "image":        image_path,
        "labels":       labels if labels else ["OK"],
        "probabilities": dict(zip(CLASS_NAMES, probs.tolist())),
        "latency_ms":   round(latency_ms, 2),
    }


def run_batch(
    model_path: str,
    image_dir:  str,
    threshold:  float = THRESHOLD,
    output_csv: str   = "predictions.csv",
):
    sess = ort.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )
    images = sorted(
        list(Path(image_dir).glob("*.jpg")) +
        list(Path(image_dir).glob("*.png"))
    )
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    results = [predict_single(sess, str(p), threshold) for p in images]

    df = pd.DataFrame([{
        "Filename":   r["image"],
        "Defects":    " ".join(r["labels"]),
        "Latency_ms": r["latency_ms"],
    } for r in results])
    df.to_csv(output_csv, index=False)

    print(f"Saved {len(df)} predictions → {output_csv}")
    print(f"Avg latency : {df.Latency_ms.mean():.1f} ms/image")
    print(f"Throughput  : {1000 / df.Latency_ms.mean():.1f} images/sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ONNX inference for sewer defect detection"
    )
    parser.add_argument("--model",     required=True, help="Path to .onnx model")
    parser.add_argument("--image_dir", required=True, help="Directory of test images")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--output",    default="predictions.csv")
    args = parser.parse_args()
    run_batch(args.model, args.image_dir, args.threshold, args.output)