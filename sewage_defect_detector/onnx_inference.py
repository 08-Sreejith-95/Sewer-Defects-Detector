"""
ONNX inference module — CPU deployment, no PyTorch required at runtime.
Replicates the exact preprocessing from infer.py.
"""
import argparse
import time
from pathlib import Path
from src.config.config import load_config
from src.utils.arg_parser import parse_args

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image
from preprocess import preprocess_image_path



cfg = load_config('configs/configs.yaml')

def predict_single(
    session: ort.InferenceSession,
    image_path: str,
    img_size: int = cfg.dataset.img_size,
    threshold: float = cfg.onnx_inference.threshold,
    class_names: list = cfg.dataset.class_names,
    mean: float = cfg.dataset.mean,
    std: float = cfg.dataset.std
) -> dict:
    input_tensor = preprocess_image_path(image_path, img_size, mean, std)
    t0 = time.perf_counter()
    logits = session.run(None, {"image": input_tensor})[0][0]        # (19,)
    latency_ms = (time.perf_counter() - t0) * 1000

    probs  = 1.0 / (1.0 + np.exp(-logits))                 # sigmoid
    labels = [c for c, p in zip(class_names, probs) if p >= threshold]
    return {
        "image":        image_path,
        "labels":       labels if labels else ["OK"],
        "probabilities": dict(zip(class_names, probs.tolist())),
        "latency_ms":   round(latency_ms, 2),
    }


def run_batch(
    model_path: str,
    image_dir: str,
    threshold: float,
    class_names,
    img_size: int,
    mean,
    std,
    output_csv: str,
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

    results = [predict_single(sess, str(p), img_size, threshold, class_names, mean, std) for p in images]

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
    args = parse_args()
    cfg = load_config(args.config)  
    run_batch(cfg.onnx_inference.onnx_model_path, cfg.onnx_inference.test_image_dir, 
              cfg.onnx_inference.threshold, cfg.dataset.class_names, cfg.dataset.img_size, 
              cfg.dataset.mean, cfg.dataset.std, cfg.onnx_inference.output_csv)