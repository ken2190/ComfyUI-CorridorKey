from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import color_utils as cu
from .model_transformer import GreenFormer

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_CHECKPOINT_NAME = "CorridorKey.pth"
CHECKPOINT_DOWNLOAD_URL = "https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth"
_ENGINE_CACHE: dict[tuple, "CorridorKeyEngine"] = {}


def _import_cv2():
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for CorridorKey. Install requirements.txt in the ComfyUI Python environment."
        ) from exc
    return cv2


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _prefer_channels_last() -> bool:
    return _parse_bool_env("CORRIDORKEY_PREFER_CHANNELS_LAST", True)


def _enable_tf32() -> bool:
    return _parse_bool_env("CORRIDORKEY_ENABLE_TF32", True)


def _use_torch_compile() -> bool:
    # Default off: torch.compile with reduce-overhead takes 20-60min to JIT on V100
    # and provides minimal speedup on Volta GPUs. Set CORRIDORKEY_TORCH_COMPILE=true
    # to enable on Ampere+ (A100/H100) where it can help.
    return _parse_bool_env("CORRIDORKEY_TORCH_COMPILE", False)


def _configure_torch_for_inference(device_type: str) -> None:
    if device_type != "cuda":
        return
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = _enable_tf32()
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = _enable_tf32()


def resolve_checkpoint_path() -> Path:
    explicit = MODELS_DIR / DEFAULT_CHECKPOINT_NAME
    if explicit.is_file():
        return explicit

    if not MODELS_DIR.exists():
        raise FileNotFoundError(
            f"CorridorKey model directory is missing: {MODELS_DIR}. "
            "Create it and place CorridorKey.pth inside. "
            f"Download URL: {CHECKPOINT_DOWNLOAD_URL}"
        )

    candidates = sorted(MODELS_DIR.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"No CorridorKey model file found in {MODELS_DIR}. "
            "Download the model checkpoint and place it there. "
            f"Download URL: {CHECKPOINT_DOWNLOAD_URL}"
        )
    return candidates[0]


def get_available_gpu_count() -> int:
    """Return number of CUDA GPUs available, 0 if none."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


class CorridorKeyEngine:
    def __init__(
        self,
        checkpoint_path: Path,
        device: str | None = None,
        img_size: int = 2048,
        use_refiner: bool = True,
        backend: str = "auto",
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.img_size = img_size
        self.checkpoint_path = Path(checkpoint_path)
        self.use_refiner = use_refiner
        self.backend = backend
        self.channels_last = self.device.type == "cuda" and _prefer_channels_last()

        # Numpy mean/std for legacy process_frame
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # GPU-resident mean/std tensors for process_frame_tensor (shape: [1, 3, 1, 1])
        self.mean_t = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        self.std_t = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1).to(self.device)

        _configure_torch_for_inference(self.device.type)
        # PyTorch model — lazily loaded only when needed (TRT path skips it entirely)
        self.model: GreenFormer | None = None
        self._compiled = False

        # ORT/TRT session — lazily initialized on first inference
        self._ort_session = None
        self._ort_init_attempted = False

    def _ensure_model_loaded(self) -> GreenFormer:
        """Load PyTorch model on demand. Skipped entirely when TRT handles inference."""
        if self.model is not None:
            return self.model
        self.model = self._load_model()
        return self.model

    def _load_model(self) -> GreenFormer:
        model = GreenFormer(
            encoder_name="hiera_base_plus_224.mae_in1k_ft_in1k",
            img_size=self.img_size,
            use_refiner=self.use_refiner,
        )
        model = model.to(self.device)
        if self.channels_last:
            model = model.to(memory_format=torch.channels_last)
        model.eval()

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"CorridorKey model file not found: {self.checkpoint_path}. "
                "Place CorridorKey.pth in the local models directory. "
                f"Download URL: {CHECKPOINT_DOWNLOAD_URL}"
            )

        checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_checkpoint(state_dict)

        # torch.compile for optimized inference (first call will be slower due to JIT)
        if self.device.type == "cuda" and _use_torch_compile():
            try:
                model = torch.compile(model, mode="reduce-overhead")
                self._compiled = True
                LOGGER.info("CorridorKey model compiled with torch.compile (device=%s)", self.device)
            except Exception as e:
                LOGGER.warning("torch.compile failed, falling back to eager mode: %s", e)

        return model

    def _autocast_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _get_ort_session(self, max_batch: int = 4):
        """Lazily initialize ORT/TRT session. Returns session or None.

        When backend='tensorrt' but TRT is unavailable (e.g. GPU SM too old),
        logs a warning and falls back to PyTorch instead of crashing.
        """
        if self._ort_init_attempted:
            return self._ort_session
        self._ort_init_attempted = True

        if self.backend == "pytorch" or self.device.type != "cuda":
            return None

        try:
            from .onnx_trt_backend import _check_trt_available, get_ort_session

            # Check TRT availability before attempting session creation
            if not _check_trt_available():
                if self.backend == "tensorrt":
                    LOGGER.warning(
                        "backend='tensorrt' requested but TensorRT is not available for this GPU. "
                        "Falling back to PyTorch. Set backend='auto' or 'pytorch' to suppress this warning."
                    )
                return None

            device_id = self.device.index if self.device.index is not None else 0
            self._ort_session = get_ort_session(
                model=None,  # model only needed for ONNX export; pre-exported via build script
                models_dir=MODELS_DIR,
                device_id=device_id,
                img_size=self.img_size,
                max_batch=max_batch,
            )
            if self._ort_session is not None:
                LOGGER.info(
                    "TRT backend active: provider=%s, device=%s",
                    self._ort_session.active_provider, self.device,
                )
            elif self.backend == "tensorrt":
                LOGGER.warning(
                    "backend='tensorrt' requested but TRT session creation failed. "
                    "Falling back to PyTorch."
                )
        except Exception as e:
            LOGGER.warning("ORT/TRT backend init failed, using PyTorch: %s", e)

        return self._ort_session

    @torch.no_grad()
    def _run_model_batch(
        self,
        images: list[np.ndarray],
        masks: list[np.ndarray],
        input_is_linear: bool = False,
        refiner_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run model inference on a batch of frames. Returns (alpha, fg) tensors
        at original resolution, still on GPU. Shape: [N, C, H, W].
        """
        n = len(images)
        height, width = images[0].shape[:2]

        # Build batch tensor on GPU
        img_list = []
        mask_list = []
        for i in range(n):
            img = images[i].astype(np.float32) if images[i].dtype != np.float32 else images[i]
            if img.max() > 1.01 and img.dtype == np.float32:
                img = img / 255.0
            msk = masks[i].astype(np.float32) if masks[i].dtype != np.float32 else masks[i]
            if msk.ndim == 2:
                msk = msk[:, :, np.newaxis]
            img_list.append(torch.from_numpy(img).permute(2, 0, 1))  # [3,H,W]
            mask_list.append(torch.from_numpy(msk).permute(2, 0, 1))  # [1,H,W]

        img_batch = torch.stack(img_list).to(self.device)   # [N,3,H,W]
        mask_batch = torch.stack(mask_list).to(self.device)  # [N,1,H,W]
        del img_list, mask_list

        # Resize to inference size on GPU
        img_resized = F.interpolate(img_batch, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        mask_resized = F.interpolate(mask_batch, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        del img_batch, mask_batch

        if input_is_linear:
            img_resized = cu.linear_to_srgb(img_resized)

        # Normalize
        img_norm = (img_resized - self.mean_t) / self.std_t
        input_tensor = torch.cat([img_norm, mask_resized], dim=1)  # [N,4,S,S]
        del img_resized, mask_resized, img_norm

        # Try ORT/TRT backend (skip when refiner_scale != 1.0 — hook can't work with ORT)
        ort_session = self._get_ort_session(max_batch=n) if refiner_scale == 1.0 else None

        if ort_session is not None:
            # ORT/TRT path: zero-copy GPU inference via IO binding
            alpha_raw, fg_raw = ort_session(input_tensor)
        else:
            # PyTorch path (fallback or refiner_scale != 1.0)
            # Lazy-load model only when actually needed
            model = self._ensure_model_loaded()

            if self.channels_last:
                input_tensor = input_tensor.to(memory_format=torch.channels_last)

            hook_handle = None
            if refiner_scale != 1.0 and model.refiner is not None:
                def scale_hook(_module, _inputs, output):
                    return output * refiner_scale
                hook_handle = model.refiner.register_forward_hook(scale_hook)

            with self._autocast_context():
                output = model(input_tensor)

            if hook_handle is not None:
                hook_handle.remove()

            alpha_raw = output["alpha"]
            fg_raw = output["fg"]
            del output

        del input_tensor

        # Resize back to original resolution on GPU
        alpha = F.interpolate(alpha_raw, size=(height, width), mode="bilinear", align_corners=False)
        fg = F.interpolate(fg_raw, size=(height, width), mode="bilinear", align_corners=False)
        del alpha_raw, fg_raw

        return alpha, fg  # [N,1,H,W], [N,3,H,W]

    def postprocess_frame(
        self,
        alpha_chw: torch.Tensor,
        fg_chw: torch.Tensor,
        despill_strength: float = 1.0,
        auto_despeckle: bool = True,
        despeckle_size: int = 400,
        compute_qc: bool = False,
        compute_processed: bool = False,
    ) -> dict[str, np.ndarray]:
        """Post-process a single frame's model output (GPU tensors [C,H,W]) into numpy results.

        When compute_processed=False and compute_qc=False, skips despill/premultiply
        entirely and returns 1×1 pixel placeholders for 'processed' and 'comp' outputs.
        This saves ~6GB of CPU RAM for 120 frames at 1080p.
        """
        height = alpha_chw.shape[1]
        width = alpha_chw.shape[2]
        _PLACEHOLDER_RGB = np.zeros((1, 1, 3), dtype=np.float32)

        # Despeckle requires CPU/cv2
        if auto_despeckle:
            alpha_np = alpha_chw.permute(1, 2, 0).cpu().numpy()  # [H,W,1]
            processed_alpha_np = cu.clean_matte(alpha_np, area_threshold=despeckle_size, dilation=25, blur_size=5)
        else:
            processed_alpha_np = alpha_chw.permute(1, 2, 0).cpu().numpy()

        # fg and matte are always needed
        fg_out = fg_chw.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        matte_out = processed_alpha_np.astype(np.float32)
        if matte_out.ndim == 2:
            matte_out = matte_out[:, :, np.newaxis]

        result = {
            "fg": np.clip(fg_out, 0.0, 1.0),
            "matte": np.clip(matte_out, 0.0, 1.0),
        }

        # Only compute despill/premultiply/QC if actually needed
        if compute_processed or compute_qc:
            fg_hwc = fg_chw.permute(1, 2, 0)  # GPU [H,W,3]
            fg_despilled = cu.despill(fg_hwc, green_limit_mode="average", strength=despill_strength)
            fg_despilled_linear = cu.srgb_to_linear(fg_despilled)
            del fg_hwc, fg_despilled

            if compute_processed:
                alpha_t = torch.from_numpy(matte_out).to(self.device)
                fg_dl_t = fg_despilled_linear if isinstance(fg_despilled_linear, torch.Tensor) else torch.from_numpy(fg_despilled_linear).to(self.device)
                fg_premul = cu.premultiply(fg_dl_t, alpha_t)
                result["processed"] = np.clip(fg_premul.cpu().numpy().astype(np.float32), 0.0, 1.0)
                del alpha_t, fg_dl_t, fg_premul
            else:
                result["processed"] = _PLACEHOLDER_RGB

            if compute_qc:
                fg_dl_np = fg_despilled_linear.cpu().numpy().astype(np.float32) if isinstance(fg_despilled_linear, torch.Tensor) else fg_despilled_linear.astype(np.float32)
                checkerboard_srgb = cu.create_checkerboard(width, height, checker_size=128, color1=0.15, color2=0.55)
                checkerboard_linear = cu.srgb_to_linear(checkerboard_srgb)
                comp_linear = cu.composite_straight(fg_dl_np, checkerboard_linear, matte_out)
                comp_srgb = cu.linear_to_srgb(comp_linear)
                result["comp"] = np.clip(comp_srgb.astype(np.float32), 0.0, 1.0)
            else:
                result["comp"] = _PLACEHOLDER_RGB

            del fg_despilled_linear
        else:
            # Skip all expensive post-processing — saves significant RAM
            result["processed"] = _PLACEHOLDER_RGB
            result["comp"] = _PLACEHOLDER_RGB

        return result

    @torch.no_grad()
    def process_frame_tensor(
        self,
        image: np.ndarray,
        mask_linear: np.ndarray,
        refiner_scale: float = 1.0,
        input_is_linear: bool = False,
        despill_strength: float = 1.0,
        auto_despeckle: bool = True,
        despeckle_size: int = 400,
        compute_qc: bool = False,
        compute_processed: bool = False,
    ) -> dict[str, np.ndarray]:
        """GPU-resident single-frame processing. Kept for compatibility and single-GPU path."""
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        if mask_linear.dtype == np.uint8:
            mask_linear = mask_linear.astype(np.float32) / 255.0
        elif mask_linear.dtype == np.uint16:
            mask_linear = mask_linear.astype(np.float32) / 65535.0

        alpha, fg = self._run_model_batch([image], [mask_linear], input_is_linear, refiner_scale)
        result = self.postprocess_frame(
            alpha_chw=alpha[0], fg_chw=fg[0],
            despill_strength=despill_strength,
            auto_despeckle=auto_despeckle,
            despeckle_size=despeckle_size,
            compute_qc=compute_qc,
            compute_processed=compute_processed,
        )
        del alpha, fg
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    @torch.no_grad()
    def process_batch_tensor(
        self,
        images: list[np.ndarray],
        masks: list[np.ndarray],
        refiner_scale: float = 1.0,
        input_is_linear: bool = False,
        despill_strength: float = 1.0,
        auto_despeckle: bool = True,
        despeckle_size: int = 400,
        compute_qc: bool = False,
        compute_processed: bool = False,
    ) -> list[dict[str, np.ndarray]]:
        """Batched inference: run N frames through the model in one forward pass,
        then post-process each frame individually."""
        # Normalize inputs
        normed_imgs = []
        normed_masks = []
        for img, msk in zip(images, masks):
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            else:
                img = img.astype(np.float32)
            if msk.dtype == np.uint8:
                msk = msk.astype(np.float32) / 255.0
            elif msk.dtype == np.uint16:
                msk = msk.astype(np.float32) / 65535.0
            else:
                msk = msk.astype(np.float32)
            normed_imgs.append(img)
            normed_masks.append(msk)

        alpha_batch, fg_batch = self._run_model_batch(
            normed_imgs, normed_masks, input_is_linear, refiner_scale,
        )

        results = []
        for i in range(len(images)):
            result = self.postprocess_frame(
                alpha_chw=alpha_batch[i], fg_chw=fg_batch[i],
                despill_strength=despill_strength,
                auto_despeckle=auto_despeckle,
                despeckle_size=despeckle_size,
                compute_qc=compute_qc,
                compute_processed=compute_processed,
            )
            results.append(result)

        del alpha_batch, fg_batch
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return results

    @torch.no_grad()
    def process_frame(
        self,
        image: np.ndarray,
        mask_linear: np.ndarray,
        refiner_scale: float = 1.0,
        input_is_linear: bool = False,
        fg_is_straight: bool = True,
        despill_strength: float = 1.0,
        auto_despeckle: bool = True,
        despeckle_size: int = 400,
    ) -> dict[str, np.ndarray]:
        """Legacy numpy-based processing path. Kept for compatibility."""
        cv2 = _import_cv2()

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)

        if mask_linear.dtype == np.uint8:
            mask_linear = mask_linear.astype(np.float32) / 255.0
        elif mask_linear.dtype == np.uint16:
            mask_linear = mask_linear.astype(np.float32) / 65535.0
        else:
            mask_linear = mask_linear.astype(np.float32)

        height, width = image.shape[:2]

        if mask_linear.ndim == 2:
            mask_linear = mask_linear[:, :, np.newaxis]

        if input_is_linear:
            img_resized_linear = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            img_resized = cu.linear_to_srgb(img_resized_linear)
        else:
            img_resized = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        mask_resized = cv2.resize(mask_linear, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, np.newaxis]

        img_norm = (img_resized - self.mean) / self.std
        input_np = np.concatenate([img_norm, mask_resized], axis=-1)
        input_tensor = torch.from_numpy(input_np.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        if self.channels_last:
            input_tensor = input_tensor.to(memory_format=torch.channels_last)

        model = self._ensure_model_loaded()

        hook_handle = None
        if refiner_scale != 1.0 and model.refiner is not None:
            def scale_hook(_module, _inputs, output):
                return output * refiner_scale

            hook_handle = model.refiner.register_forward_hook(scale_hook)

        with self._autocast_context():
            output = model(input_tensor)

        if hook_handle is not None:
            hook_handle.remove()

        pred_alpha = output["alpha"]
        pred_fg = output["fg"]

        resized_alpha = pred_alpha[0].permute(1, 2, 0).cpu().numpy()
        resized_fg = pred_fg[0].permute(1, 2, 0).cpu().numpy()
        resized_alpha = cv2.resize(resized_alpha, (width, height), interpolation=cv2.INTER_LANCZOS4)
        resized_fg = cv2.resize(resized_fg, (width, height), interpolation=cv2.INTER_LANCZOS4)

        if resized_alpha.ndim == 2:
            resized_alpha = resized_alpha[:, :, np.newaxis]

        if auto_despeckle:
            processed_alpha = cu.clean_matte(resized_alpha, area_threshold=despeckle_size, dilation=25, blur_size=5)
        else:
            processed_alpha = resized_alpha

        fg_despilled = cu.despill(resized_fg, green_limit_mode="average", strength=despill_strength)
        fg_despilled_linear = cu.srgb_to_linear(fg_despilled)
        fg_premul_linear = cu.premultiply(fg_despilled_linear, processed_alpha)
        processed_rgba = np.concatenate([fg_premul_linear, processed_alpha], axis=-1)

        checkerboard_srgb = cu.create_checkerboard(width, height, checker_size=128, color1=0.15, color2=0.55)
        checkerboard_linear = cu.srgb_to_linear(checkerboard_srgb)
        if fg_is_straight:
            comp_linear = cu.composite_straight(fg_despilled_linear, checkerboard_linear, processed_alpha)
        else:
            comp_linear = cu.composite_premul(fg_despilled_linear, checkerboard_linear, processed_alpha)
        comp_srgb = cu.linear_to_srgb(comp_linear)

        return {
            "fg": np.clip(resized_fg.astype(np.float32), 0.0, 1.0),
            "raw_alpha": np.clip(resized_alpha.astype(np.float32), 0.0, 1.0),
            "matte": np.clip(processed_alpha.astype(np.float32), 0.0, 1.0),
            "processed": np.clip(fg_premul_linear.astype(np.float32), 0.0, 1.0),
            "processed_rgba": np.clip(processed_rgba.astype(np.float32), 0.0, 1.0),
            "comp": np.clip(comp_srgb.astype(np.float32), 0.0, 1.0),
        }


def get_cached_engine(
    device: str | None = None,
    img_size: int = 2048,
    use_refiner: bool = True,
    backend: str = "auto",
) -> CorridorKeyEngine:
    checkpoint_path = resolve_checkpoint_path()
    cache_key = (str(checkpoint_path), device or "", img_size, use_refiner, backend)
    cached = _ENGINE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    engine = CorridorKeyEngine(
        checkpoint_path=checkpoint_path,
        device=device,
        img_size=img_size,
        use_refiner=use_refiner,
        backend=backend,
    )
    _ENGINE_CACHE[cache_key] = engine
    return engine


def free_all_engines(keep_ort_sessions: bool = True) -> int:
    """Destroy all cached engines, release GPU VRAM, and clear CUDA cache.

    Args:
        keep_ort_sessions: If True (default), keep ORT/TRT sessions alive in
            the global cache so they can be reused without the 3-4s reload
            penalty per GPU. TRT sessions are lightweight (~50MB) compared
            to the PyTorch model (~2-4GB). Set False to free everything.

    Returns the number of engines freed.
    """
    count = len(_ENGINE_CACHE)
    for key, engine in list(_ENGINE_CACHE.items()):
        if not keep_ort_sessions:
            # Delete ORT session if present
            if hasattr(engine, '_ort_session') and engine._ort_session is not None:
                del engine._ort_session
                engine._ort_session = None
        # Delete model (PyTorch — the heavy part, ~2-4GB VRAM)
        if hasattr(engine, 'model') and engine.model is not None:
            del engine.model
            engine.model = None
        # Delete GPU tensors
        if hasattr(engine, 'mean_t'):
            del engine.mean_t
        if hasattr(engine, 'std_t'):
            del engine.std_t
    _ENGINE_CACHE.clear()

    if not keep_ort_sessions:
        # Also clear the separate ORT session cache in onnx_trt_backend
        try:
            from .onnx_trt_backend import free_ort_sessions
            free_ort_sessions()
        except Exception:
            pass

    import gc
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    LOGGER.info("Freed %d CorridorKey engine(s) and cleared CUDA cache (kept ORT sessions: %s).", count, keep_ort_sessions)
    return count


def get_multi_gpu_engines(
    img_size: int = 2048,
    num_gpus: int = 0,
    use_refiner: bool = True,
    backend: str = "auto",
) -> list[CorridorKeyEngine]:
    """Create one engine per GPU. num_gpus=0 means auto-detect all available GPUs."""
    available = get_available_gpu_count()
    if available == 0:
        return [get_cached_engine(device="cpu", img_size=img_size, use_refiner=use_refiner, backend=backend)]

    target = available if num_gpus <= 0 else min(num_gpus, available)
    engines = []
    for gpu_idx in range(target):
        device = f"cuda:{gpu_idx}"
        engines.append(get_cached_engine(device=device, img_size=img_size, use_refiner=use_refiner, backend=backend))
    return engines
