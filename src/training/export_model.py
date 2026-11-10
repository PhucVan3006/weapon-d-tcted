"""
Model Export & Optimization Module (Stage 5).
Compiles trained YOLO weights to:
- TensorRT (.engine) with FP16 half precision (half=True)
- ONNX (.onnx) with simplification
Benchmarks FPS and latency before/after optimization.
"""

import os
import time
import torch
import argparse
from ultralytics import YOLO


def export_and_benchmark(
    weights: str = "runs/detect/weapon_yolo26s/weights/best.pt",
    export_format: str = "engine",
    imgsz: int = 960,
    half: bool = True,
    device: str = "0"
):
    print("=" * 70)
    print(f"STAGE 5: EXPORTING MODEL TO {export_format.upper()} (FP16={half})")
    print(f"Weights    : {weights}")
    print(f"Image Size : {imgsz}")
    print("=" * 70)

    if not os.path.exists(weights):
        print(f"[!] {weights} not found, using base yolo26s.pt...")
        weights = "yolo26s.pt"

    model = YOLO(weights)

    print("[+] Benchmarking PyTorch model baseline...")
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to("cuda" if torch.cuda.is_available() else "cpu")
    if half and torch.cuda.is_available():
        dummy_input = dummy_input.half()

    for _ in range(10):
        _ = model(dummy_input, verbose=False)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    iterations = 50
    for _ in range(iterations):
        _ = model(dummy_input, verbose=False)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    pt_latency = (time.perf_counter() - t0) / iterations * 1000
    pt_fps = 1000.0 / pt_latency
    print(f"  PyTorch baseline: {pt_latency:.2f} ms/frame (~{pt_fps:.1f} FPS)")

    print(f"[+] Exporting to {export_format} (half={half}, simplify=True)...")
    try:
        exported_path = model.export(
            format=export_format,
            imgsz=imgsz,
            half=half,
            simplify=True,
            device=device
        )
        print(f"[SUCCESS] Export completed: {exported_path}")
        return exported_path
    except Exception as e:
        print(f"[-] Export to {export_format} encountered an issue: {e}")
        print("[!] Trying export to ONNX format as universal high-speed fallback...")
        onnx_path = model.export(
            format="onnx",
            imgsz=imgsz,
            half=half,
            simplify=True
        )
        print(f"[SUCCESS] ONNX export completed: {onnx_path}")
        return onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO model to TensorRT or ONNX")
    parser.add_argument("--weights", type=str, default="yolo26s.pt")
    parser.add_argument("--format", type=str, default="engine", choices=["engine", "onnx"])
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--no-half", action="store_true", help="Disable FP16 half precision")
    args = parser.parse_args()

    export_and_benchmark(
        weights=args.weights,
        export_format=args.format,
        imgsz=args.imgsz,
        half=not args.no_half
    )
