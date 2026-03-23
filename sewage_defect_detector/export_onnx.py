import argparse
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
import numpy as np
from omegaconf import OmegaConf
from src.model.transformer_models import build_vit_model


def export(checkpoint_path: str, config_path: str, output_path: str):
    cfg = OmegaConf.load(config_path)

    # --- Build model and load checkpoint ---
    model = build_vit_model(cfg)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, cfg.dataset.img_size, cfg.dataset.img_size)

    # --- Export FP32 ---
    fp32_path = output_path.replace(".onnx", "_fp32.onnx")
    torch.onnx.export(
        model,
        dummy,
        fp32_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    onnx.checker.check_model(fp32_path)
    print(f"FP32 exported --> {fp32_path}")

    # --- INT8 dynamic quantization ---
    int8_path = output_path.replace(".onnx", "_int8.onnx")
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
    print(f"INT8 exported --> {int8_path}")

    # --- Sanity check: compare FP32 vs INT8 outputs ---
    sess_fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    input = {"image": dummy.numpy()}
    out_fp32 = sess_fp32.run(None, input)[0]
    out_int8 = sess_int8.run(None, input)[0]
    max_diff = float(np.abs(out_fp32 - out_int8).max())
    print(f"Max logit diff FP32 vs INT8: {max_diff:.4f}  (should be < 0.5)")

    # --- checking Latency benchmark ---
    import time
    runs = 50
    t0 = time.perf_counter()
    for _ in range(runs):
        sess_int8.run(None, input)
    avg_ms = (time.perf_counter() - t0) / runs * 1000
    print(f"INT8 avg latency (CPU, batch=1): {avg_ms:.1f} ms/image")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to EMA .pt checkpoint")
    parser.add_argument("--config",     required=True, help="Path to config YAML")
    parser.add_argument("--output",     default="model.onnx")
    args = parser.parse_args()
    export(args.checkpoint, args.config, args.output)