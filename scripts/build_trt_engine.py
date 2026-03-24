#!/usr/bin/env python3
"""Export ONNX model and (optionally) pre-build TensorRT engine cache.

Run this once on the host (or inside the container with GPU access).
The .pth model is loaded from --models-dir, ONNX is exported there too.
TRT engine cache goes to --trt-cache-dir (defaults to --models-dir).

Usage:
    # Export ONNX only (no GPU needed)
    python scripts/build_trt_engine.py \
        --models-dir /app/custom_nodes/ComfyUI-CorridorKey/models

    # Export ONNX + pre-build TRT engines into persistent mount
    docker exec comfyui python /app/custom_nodes/ComfyUI-CorridorKey/scripts/build_trt_engine.py \
        --models-dir /app/custom_nodes/ComfyUI-CorridorKey/models \
        --trt-cache-dir /trt-cache \
        --img-size 2048 --max-batch 2 \
        --gpu 0 --gpu 1

The TRT engine cache is GPU-architecture-specific (e.g. SM 70 for V100).
It persists across container rebuilds as long as the same GPU hardware is used.
Set CORRIDORKEY_TRT_CACHE_DIR in docker-compose to point to the persistent mount.
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
    parser = argparse.ArgumentParser(description="Export ONNX and pre-build TensorRT engine for CorridorKey")
    parser.add_argument("--models-dir", type=str, default=None,
                        help="Directory containing the .pth model file (ONNX will be written here too)")
    parser.add_argument("--trt-cache-dir", type=str, default=None,
                        help="Directory for TRT engine cache (default: same as --models-dir). "
                             "Use a persistent mount like /trt-cache to survive container rebuilds.")
    # Keep --output-dir as hidden alias for backwards compatibility
    parser.add_argument("--output-dir", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--img-size", type=int, default=2048, choices=[768, 1024, 1536, 2048])
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--gpu", type=int, action="append", default=None,
                        help="GPU device IDs to build TRT engines for (repeat for multiple). "
                             "Omit to export ONNX only (no GPU needed).")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 (use FP32)")
    args = parser.parse_args()

    # Handle --output-dir as alias for --models-dir (backwards compat)
    if args.models_dir is None and args.output_dir is not None:
        args.models_dir = args.output_dir
    if args.models_dir is None:
        parser.error("--models-dir is required")

    import torch
    from corridor_key.model_transformer import GreenFormer
    from corridor_key.onnx_trt_backend import export_onnx, OnnxTrtSession

    models_dir = Path(args.models_dir)
    trt_cache_dir = Path(args.trt_cache_dir) if args.trt_cache_dir else models_dir
    fp16 = not args.no_fp16
    gpu_ids = args.gpu or []

    # Find checkpoint in models_dir
    pth_files = sorted(models_dir.glob("*.pth"))
    if not pth_files:
        LOGGER.error("No .pth model found in %s", models_dir)
        sys.exit(1)
    checkpoint_path = pth_files[0]

    LOGGER.info("Config: img_size=%d, max_batch=%d, fp16=%s, gpus=%s",
                args.img_size, args.max_batch, fp16, gpu_ids)
    LOGGER.info("Models dir: %s", models_dir)
    LOGGER.info("TRT cache dir: %s", trt_cache_dir)
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

    # Step 1: Export ONNX to models_dir (alongside .pth)
    onnx_path = export_onnx(
        model=model,
        output_dir=models_dir,
        img_size=args.img_size,
        max_batch=args.max_batch,
        device=torch.device("cpu"),
    )
    LOGGER.info("ONNX model: %s", onnx_path)

    # Step 2: Build TRT engines for each GPU (optional)
    trt_cache_dir.mkdir(parents=True, exist_ok=True)
    for gpu_id in gpu_ids:
        LOGGER.info("Building TRT engine for GPU %d (this may take 5-15 minutes)...", gpu_id)
        t0 = time.monotonic()
        session = OnnxTrtSession(
            onnx_path=onnx_path,
            trt_cache_dir=trt_cache_dir,
            device_id=gpu_id,
            img_size=args.img_size,
            max_batch=args.max_batch,
            fp16=fp16,
        )
        elapsed = time.monotonic() - t0
        LOGGER.info("GPU %d: provider=%s, built in %.1fs", gpu_id, session.active_provider, elapsed)

    if not gpu_ids:
        LOGGER.info("ONNX export complete. TRT engine will be built on first use inside the container.")
    else:
        LOGGER.info("Done. ONNX + TRT engines ready in %s", trt_cache_dir)


if __name__ == "__main__":
    main()