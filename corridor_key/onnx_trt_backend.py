"""ONNX export and TensorRT/CUDA inference backend for CorridorKey.

Fallback chain: TensorRT EP -> CUDA EP -> None (caller uses PyTorch).

The ONNX model is exported once on the host and placed alongside the .pth
file (e.g. /home/ubuntu/DATA/ComfyUI/models/corridorkey/). It is then
mounted read-only into the container just like the .pth, and symlinked
into the custom node's models/ dir by entrypoint.sh.

The TRT engine cache (GPU-specific, rebuilt per container) goes into the
custom node's writable models/ dir.
"""
from __future__ import annotations

import contextlib
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ONNX-compatible wrapper: flattens dict output to tuple for ONNX export
# ---------------------------------------------------------------------------

class _GreenFormerONNXWrapper(nn.Module):
    """Thin wrapper that converts GreenFormer's dict output to a tuple
    (alpha, fg) which ONNX can export."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x)
        return out["alpha"], out["fg"]


# ---------------------------------------------------------------------------
# SDPA decomposition for ONNX export compatibility
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _sdpa_decomposition_ctx():
    """Temporarily replace F.scaled_dot_product_attention with an explicit
    matmul-softmax-matmul decomposition so torch.onnx.export can trace it.
    Restored on context exit."""
    import torch.nn.functional as _F

    _original_sdpa = _F.scaled_dot_product_attention

    def _decomposed_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                         is_causal=False, scale=None):
        scale = scale if scale is not None else (query.shape[-1] ** -0.5)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        if is_causal:
            L, S = query.shape[-2], key.shape[-2]
            causal_mask = torch.triu(
                torch.ones(L, S, dtype=torch.bool, device=query.device), diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, value)

    _F.scaled_dot_product_attention = _decomposed_sdpa
    try:
        yield
    finally:
        _F.scaled_dot_product_attention = _original_sdpa


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def _onnx_path_for(models_dir: Path, img_size: int) -> Path:
    return models_dir / f"CorridorKey_s{img_size}.onnx"


def export_onnx(
    model: nn.Module,
    output_dir: Path,
    img_size: int,
    max_batch: int = 4,
    device: torch.device | None = None,
) -> Path:
    """Export GreenFormer to ONNX. Writes to output_dir (must be writable).

    Meant to be run once on the host, then the .onnx file is placed alongside
    the .pth in the models directory and mounted into the container read-only.
    """
    onnx_path = _onnx_path_for(output_dir, img_size)
    if onnx_path.exists():
        LOGGER.info("ONNX model already exists: %s", onnx_path)
        return onnx_path

    LOGGER.info("Exporting GreenFormer to ONNX (size=%d, max_batch=%d, dir=%s)...",
                img_size, max_batch, output_dir)
    t0 = time.monotonic()

    wrapper = _GreenFormerONNXWrapper(model)
    wrapper.eval()

    # Use CPU for export to avoid GPU memory pressure
    export_device = device or torch.device("cpu")
    wrapper = wrapper.to(export_device)

    dummy_input = torch.randn(1, 4, img_size, img_size, device=export_device)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    with _sdpa_decomposition_ctx(), torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            str(onnx_path),
            opset_version=17,
            input_names=["input"],
            output_names=["alpha", "fg"],
            dynamic_axes={
                "input": {0: "batch"},
                "alpha": {0: "batch"},
                "fg": {0: "batch"},
            },
        )

    elapsed = time.monotonic() - t0
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    LOGGER.info("ONNX export done: %s (%.1f MB) in %.1fs", onnx_path.name, size_mb, elapsed)

    # Move model back to original device if we moved it
    if device is not None and device != export_device:
        wrapper.model.to(device)

    return onnx_path


def find_onnx_model(models_dir: Path, img_size: int) -> Path | None:
    """Look for a pre-exported ONNX model in the models dir (read-only is fine)."""
    onnx_path = _onnx_path_for(models_dir, img_size)
    if onnx_path.exists():
        return onnx_path
    # Also check via symlink (entrypoint symlinks host files into custom node models/)
    return None


# ---------------------------------------------------------------------------
# ORT inference session with TensorRT EP
# ---------------------------------------------------------------------------

_TRT_AVAILABLE: bool | None = None  # Module-level cache: None = not checked yet
_TRT_SM_MIN = 70  # Minimum SM version for TensorRT. SM 70 (Volta/V100) supported by TRT <=10.4. SM <70 (Pascal) not supported.


def _get_gpu_sm_version(device_id: int = 0) -> int:
    """Return the SM version (e.g. 70 for V100, 75 for T4, 80 for A100) for a CUDA device.
    Returns 0 if CUDA is not available."""
    try:
        if not torch.cuda.is_available():
            return 0
        major, minor = torch.cuda.get_device_capability(device_id)
        return major * 10 + minor
    except Exception:
        return 0


def _check_trt_available() -> bool:
    """Check if TensorRT EP is available in ONNX Runtime AND the GPU SM version is supported.
    Result is cached globally so we only probe once per process."""
    global _TRT_AVAILABLE
    if _TRT_AVAILABLE is not None:
        return _TRT_AVAILABLE
    try:
        import onnxruntime as ort
        has_ep = "TensorrtExecutionProvider" in ort.get_available_providers()
        if not has_ep:
            _TRT_AVAILABLE = False
            LOGGER.warning(
                "TensorRT EP not found in ORT providers. "
                "TRT backend disabled. Install tensorrt-cu12 or use backend='pytorch'."
            )
            return _TRT_AVAILABLE

        # Check GPU SM compatibility — recent TRT drops Volta (SM 70)
        sm = _get_gpu_sm_version(0)
        if 0 < sm < _TRT_SM_MIN:
            _TRT_AVAILABLE = False
            LOGGER.warning(
                "GPU SM %d (Pascal/older) is not supported by TensorRT "
                "(requires SM >= %d). TRT backend disabled, using PyTorch.",
                sm, _TRT_SM_MIN,
            )
            return _TRT_AVAILABLE

        _TRT_AVAILABLE = True
        LOGGER.info("TensorRT EP available via ONNX Runtime (GPU SM %d).", sm)
    except Exception:
        _TRT_AVAILABLE = False
        LOGGER.warning(
            "onnxruntime not available. TRT backend disabled."
        )
    return _TRT_AVAILABLE


class OnnxTrtSession:
    """ONNX Runtime inference session with TensorRT or CUDA execution provider."""

    def __init__(
        self,
        onnx_path: Path,
        trt_cache_dir: Path,
        device_id: int = 0,
        img_size: int = 2048,
        max_batch: int = 4,
        fp16: bool = True,
    ) -> None:
        import onnxruntime as ort

        self.device_id = device_id
        self.img_size = img_size
        self._fp16 = fp16
        self.active_provider: str = "unknown"

        # TRT engine cache goes to the writable custom node models dir
        cache_dir = trt_cache_dir / f"trt_cache_s{img_size}_b{max_batch}_gpu{device_id}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        shape_str = f"input:{max_batch}x4x{img_size}x{img_size}"

        providers = []

        # Try TensorRT EP first (only if libnvinfer is actually available)
        trt_available = _check_trt_available()

        if trt_available:
            trt_opts = {
                "device_id": device_id,
                "trt_fp16_enable": fp16,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache_dir),
                "trt_max_workspace_size": str(8 * 1024 ** 3),  # 8 GB
                "trt_builder_optimization_level": "5",
                "trt_profile_min_shapes": f"input:1x4x{img_size}x{img_size}",
                "trt_profile_max_shapes": shape_str,
                "trt_profile_opt_shapes": shape_str,
            }
            providers.append(("TensorrtExecutionProvider", trt_opts))

            # CUDA EP as fallback only when TRT is present (avoids 16GB Softmax OOM)
            cuda_opts = {
                "device_id": device_id,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
            }
            providers.append(("CUDAExecutionProvider", cuda_opts))
        else:
            # TRT not available — don't use ORT CUDA EP (it allocates ~16GB for
            # Softmax at 2048, worse than PyTorch). Raise so caller falls back to PyTorch.
            raise RuntimeError(
                "TensorRT runtime (libnvinfer.so) not found. "
                "ORT CUDA EP is not used as fallback because it consumes more VRAM than PyTorch. "
                "Install TensorRT or use backend='pytorch'."
            )

        LOGGER.info(
            "Creating ORT session (device=%d, size=%d, batch<=%d, fp16=%s)...",
            device_id, img_size, max_batch, fp16,
        )
        t0 = time.monotonic()

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=providers,
        )

        elapsed = time.monotonic() - t0
        active = self._session.get_providers()
        if "TensorrtExecutionProvider" in active:
            self.active_provider = "TensorRT"
        elif "CUDAExecutionProvider" in active:
            self.active_provider = "CUDA"
        else:
            self.active_provider = active[0] if active else "CPU"

        LOGGER.info(
            "ORT session ready: provider=%s, device=%d (%.1fs)",
            self.active_provider, device_id, elapsed,
        )

    def __call__(
        self, input_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference. Input: GPU tensor [N,4,S,S]. Returns (alpha, fg) as GPU tensors."""
        import onnxruntime as ort

        # Use IO binding to keep data on GPU (avoids CPU roundtrip)
        io_binding = self._session.io_binding()

        # Ensure contiguous float32 on the correct GPU
        inp = input_tensor.contiguous().float()

        # Bind input from PyTorch GPU tensor via data_ptr
        io_binding.bind_input(
            name="input",
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float32,
            shape=tuple(inp.shape),
            buffer_ptr=inp.data_ptr(),
        )

        # Pre-allocate output tensors on GPU
        batch = inp.shape[0]
        alpha_out = torch.empty(
            (batch, 1, self.img_size, self.img_size),
            dtype=torch.float32,
            device=f"cuda:{self.device_id}",
        )
        fg_out = torch.empty(
            (batch, 3, self.img_size, self.img_size),
            dtype=torch.float32,
            device=f"cuda:{self.device_id}",
        )

        io_binding.bind_output(
            name="alpha",
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float32,
            shape=tuple(alpha_out.shape),
            buffer_ptr=alpha_out.data_ptr(),
        )
        io_binding.bind_output(
            name="fg",
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float32,
            shape=tuple(fg_out.shape),
            buffer_ptr=fg_out.data_ptr(),
        )

        self._session.run_with_iobinding(io_binding)

        return alpha_out, fg_out


# ---------------------------------------------------------------------------
# Session cache and convenience constructor
# ---------------------------------------------------------------------------

_ORT_SESSION_CACHE: dict[tuple[str, int, int, int], OnnxTrtSession | None] = {}
_ORT_FAILED_KEYS: set[tuple[str, int, int, int]] = set()  # Negative cache: don't retry failed sessions


def get_ort_session(
    model: nn.Module,
    models_dir: Path,
    device_id: int,
    img_size: int,
    max_batch: int = 4,
    fp16: bool = True,
) -> OnnxTrtSession | None:
    """Get or create a cached ORT TensorRT session.

    Looks for a pre-exported ONNX model in models_dir (read-only, alongside .pth).
    If not found, returns None — run scripts/build_trt_engine.py on the host first.
    TRT engine cache is written to models_dir (must be writable for trt_cache/).
    """
    # Early exit if TRT is known to be unavailable (avoids repeated log spam)
    if not _check_trt_available():
        return None

    cache_key = (str(models_dir), device_id, img_size, max_batch)

    # Negative cache: don't retry sessions that already failed (e.g. SM incompatibility)
    if cache_key in _ORT_FAILED_KEYS:
        return None

    cached = _ORT_SESSION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        onnx_path = find_onnx_model(models_dir, img_size)
        if onnx_path is None:
            LOGGER.warning(
                "ONNX model not found: %s. "
                "Run scripts/build_trt_engine.py on the host to export it, "
                "then place it in the models directory alongside the .pth file.",
                _onnx_path_for(models_dir, img_size).name,
            )
            _ORT_FAILED_KEYS.add(cache_key)
            return None

        LOGGER.info("Found ONNX model: %s", onnx_path)

        # TRT engine cache: writable dir inside the custom node's models/
        session = OnnxTrtSession(
            onnx_path=onnx_path,
            trt_cache_dir=models_dir,
            device_id=device_id,
            img_size=img_size,
            max_batch=max_batch,
            fp16=fp16,
        )

        _ORT_SESSION_CACHE[cache_key] = session
        return session

    except Exception:
        LOGGER.warning("Failed to create ORT/TRT session", exc_info=True)
        # Cache the failure so we don't retry on every meta-batch
        _ORT_FAILED_KEYS.add(cache_key)
        # Also mark TRT as globally unavailable if this looks like an SM compatibility issue
        global _TRT_AVAILABLE
        _TRT_AVAILABLE = False
        LOGGER.warning(
            "TRT marked as unavailable for this process to prevent repeated retries. "
            "Using PyTorch backend for all subsequent batches."
        )
        return None


def free_ort_sessions() -> int:
    """Destroy all cached ORT sessions and free their VRAM.
    Note: does NOT clear _ORT_FAILED_KEYS — failed sessions stay failed for the process lifetime."""
    count = len(_ORT_SESSION_CACHE)
    for key, session in list(_ORT_SESSION_CACHE.items()):
        if session is not None and hasattr(session, '_session'):
            del session._session
    _ORT_SESSION_CACHE.clear()
    LOGGER.info("Freed %d ORT session(s).", count)
    return count