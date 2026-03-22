#!/usr/bin/env python3
"""Export ONNX model and (optionally) pre-build TensorRT engine cache.

Run this once on the host. The exported .onnx file is placed alongside the
.pth model file so it gets mounted into the container read-only, same as
the checkpoint. The TRT engine cache is GPU-specific and rebuilt inside the
container on first use.

Usage (on host, outside container):
    # Export ONNX to the corridorkey models dir
    python scripts/build_trt_engine.py --output-dir /home/ubuntu/DATA/ComfyUI/models/corridorkey

    # Export + pre-build TRT engines (run inside container or with GPU access)
    python scripts/build_trt_engine.py --output-dir /path/to/models/corridorkey --gpu 0 --gpu 1
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
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write ONNX file (same dir as .pth model)")
    parser.add_argument("--img-size", type=int, default=2048, choices=[768, 1024, 1536, 2048])
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--gpu", type=int, action="append", default=None,
                        help="GPU device IDs to build TRT engines for (repeat for multiple). "
                             "Omit to export ONNX only (no GPU needed).")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 (use FP32)")
    args = parser.parse_args()

    import torch
    from corridor_key.model_transformer import GreenFormer
    from corridor_key.onnx_trt_backend import export_onnx, OnnxTrtSession

    output_dir = Path(args.output_dir)
    fp16 = not args.no_fp16
    gpu_ids = args.gpu or []

    # Find checkpoint in output_dir (same dir as where we write ONNX)
    pth_files = sorted(output_dir.glob("*.pth"))
    if not pth_files:
        LOGGER.error("No .pth model found in %s", output_dir)
        sys.exit(1)
    checkpoint_path = pth_files[0]

    LOGGER.info("Config: img_size=%d, max_batch=%d, fp16=%s, gpus=%s",
                args.img_size, args.max_batch, fp16, gpu_ids)
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

    # Step 1: Export ONNX to output_dir (alongside .pth)
    onnx_path = export_onnx(
        model=model,
        output_dir=output_dir,
        img_size=args.img_size,
        max_batch=args.max_batch,
        device=torch.device("cpu"),
    )
    LOGGER.info("ONNX model: %s", onnx_path)

    # Step 2: Build TRT engines for each GPU (optional)
    for gpu_id in gpu_ids:
        LOGGER.info("Building TRT engine for GPU %d (this may take 5-15 minutes)...", gpu_id)
        t0 = time.monotonic()
        session = OnnxTrtSession(
            onnx_path=onnx_path,
            trt_cache_dir=output_dir,
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
        LOGGER.info("Done. ONNX + TRT engines ready.")


if __name__ == "__main__":
    main()