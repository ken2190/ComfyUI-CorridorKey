#!/usr/bin/env python3
"""Pre-build ONNX export and TensorRT engine cache for CorridorKey.

Run this once per inference_size + batch_size combination to avoid the
5-15 minute first-run penalty during actual workflow execution.

Usage:
    python scripts/build_trt_engine.py --img-size 2048 --max-batch 4
    python scripts/build_trt_engine.py --img-size 2048 --max-batch 4 --gpu 0 --gpu 1
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add parent dir to path so we can import corridor_key
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
LOGGER = logging.getLogger("build_trt_engine")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-build TensorRT engine for CorridorKey")
    parser.add_argument("--img-size", type=int, default=2048, choices=[768, 1024, 1536, 2048])
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--gpu", type=int, action="append", default=None,
                        help="GPU device IDs to build for (repeat for multiple). Default: all GPUs.")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 (use FP32)")
    args = parser.parse_args()

    import torch
    from corridor_key.engine import resolve_checkpoint_path, MODELS_DIR
    from corridor_key.model_transformer import GreenFormer
    from corridor_key.onnx_trt_backend import export_onnx, OnnxTrtSession

    # Determine GPUs
    if args.gpu is not None:
        gpu_ids = args.gpu
    elif torch.cuda.is_available():
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = []
        LOGGER.warning("No CUDA GPUs found. Will export ONNX only.")

    fp16 = not args.no_fp16

    LOGGER.info("Config: img_size=%d, max_batch=%d, fp16=%s, gpus=%s",
                args.img_size, args.max_batch, fp16, gpu_ids)

    # Load model on CPU for ONNX export
    checkpoint_path = resolve_checkpoint_path()
    LOGGER.info("Loading model from %s...", checkpoint_path)

    model = GreenFormer(
        encoder_name="hiera_base_plus_224.mae_in1k_ft_in1k",
        img_size=args.img_size,
        use_refiner=True,
    )
    model.eval()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_checkpoint(state_dict)

    # Step 1: Export ONNX
    onnx_path = export_onnx(
        model=model,
        models_dir=MODELS_DIR,
        img_size=args.img_size,
        max_batch=args.max_batch,
        device=torch.device("cpu"),
    )
    LOGGER.info("ONNX model: %s", onnx_path)

    # Step 2: Build TRT engines for each GPU
    for gpu_id in gpu_ids:
        LOGGER.info("Building TRT engine for GPU %d (this may take 5-15 minutes)...", gpu_id)
        t0 = time.monotonic()
        session = OnnxTrtSession(
            onnx_path=onnx_path,
            device_id=gpu_id,
            img_size=args.img_size,
            max_batch=args.max_batch,
            fp16=fp16,
        )
        elapsed = time.monotonic() - t0
        LOGGER.info("GPU %d: provider=%s, built in %.1fs", gpu_id, session.active_provider, elapsed)

    LOGGER.info("Done. TRT engines are cached and will be reused automatically.")


if __name__ == "__main__":
    main()
