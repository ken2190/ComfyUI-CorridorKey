from __future__ import annotations

from typing import Callable

try:
    from .corridor_key import CorridorKeyProcessor, CorridorKeySettings, free_all_engines
    from .corridor_key.config import VALID_BACKENDS, VALID_INFERENCE_SIZES
except ImportError:
    from corridor_key import CorridorKeyProcessor, CorridorKeySettings, free_all_engines
    from corridor_key.config import VALID_BACKENDS, VALID_INFERENCE_SIZES


def _build_progress_reporter(unique_id: str | None) -> Callable[[str, int, int], None]:
    progress_bar = None
    prompt_server = None

    try:
        from comfy.utils import ProgressBar

        progress_bar = ProgressBar(1, node_id=unique_id)
    except Exception:
        progress_bar = None

    if unique_id:
        try:
            from server import PromptServer

            prompt_server = PromptServer.instance
        except Exception:
            prompt_server = None

    def report(message: str, completed: int, total: int) -> None:
        bounded_total = max(int(total), 1)
        bounded_completed = max(0, min(int(completed), bounded_total))

        if progress_bar is not None:
            progress_bar.update_absolute(bounded_completed, total=bounded_total)
        if prompt_server is not None and unique_id:
            prompt_server.send_progress_text(message, unique_id)

        print(f"[CorridorKey] {message}")

    return report


class CorridorKey:
    DESCRIPTION = (
        "Refines a coarse per-frame alpha hint into CorridorKey FG, Matte, Processed, and QC passes. "
        "The alpha hint should be rough, slightly eroded, and soft rather than tightly expanded."
    )
    OUTPUT_TOOLTIPS = (
        "Raw straight foreground color. The model predicts this in sRGB space; convert to linear before manual compositing with the matte.",
        "Raw linear alpha matte.",
        "Linear foreground premultiplied by the linear matte for quick preview and simple downstream export.",
        "QC preview composite over a checkerboard in sRGB.",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Input RGB image or image batch to refine.",
                    },
                ),
                "mask": (
                    "MASK",
                    {
                        "tooltip": (
                            "Coarse Alpha Hint for the current frame. Keep it rough, soft, and slightly eroded. "
                            "For batched images, provide one matching mask per frame."
                        ),
                    },
                ),
                "gamma_space": (
                    ["sRGB", "Linear"],
                    {
                        "tooltip": "Interpret the incoming image as sRGB or already-linear before inference.",
                    },
                ),
                "despill_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Green despill amount after inference. 0 disables despill, 1 is the standard default. "
                            "Higher values are stronger and can push heavy spill toward magenta or purple."
                        ),
                    },
                ),
                "refiner_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "Scales the learned refiner delta. 1.0 is the standard default behavior.",
                    },
                ),
                "auto_despeckle": (
                    ["On", "Off"],
                    {
                        "tooltip": "Enable connected-component cleanup on the predicted alpha matte.",
                    },
                ),
                "despeckle_size": (
                    "INT",
                    {
                        "default": 400,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Minimum island area in pixels to preserve when auto-despeckle is enabled.",
                    },
                ),
                "inference_size": (
                    list(VALID_INFERENCE_SIZES),
                    {
                        "default": 2048,
                        "tooltip": (
                            "Resolution for the transformer inference pass. "
                            "2048 = highest quality (default), 1024 = ~4x faster, 768 = ~7x faster. "
                            "Lower values reduce quality but drastically cut processing time and VRAM."
                        ),
                    },
                ),
                "compute_qc": (
                    ["On", "Off"],
                    {
                        "tooltip": (
                            "Generate the QC checkerboard composite output. "
                            "Turn Off to save time and memory when the QC preview is not needed."
                        ),
                    },
                ),
                "compute_processed": (
                    ["On", "Off"],
                    {
                        "tooltip": (
                            "Generate the premultiplied foreground output. "
                            "Turn Off to save ~3GB RAM per 120 frames at 1080p when not needed."
                        ),
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": (
                            "Number of frames per model forward pass. "
                            "Higher values use more VRAM but can be faster. "
                            "At 2048: batch=2 needs ~20GB VRAM per GPU. Start with 1 and increase if VRAM allows."
                        ),
                    },
                ),
                "num_gpus": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 16,
                        "step": 1,
                        "tooltip": (
                            "Number of GPUs to use for parallel frame processing. "
                            "0 = auto-detect all available GPUs. "
                            "Frames are distributed round-robin across GPUs for near-linear speedup."
                        ),
                    },
                ),
                "backend": (
                    list(VALID_BACKENDS),
                    {
                        "default": "auto",
                        "tooltip": (
                            "Inference backend. "
                            "'auto' tries TensorRT (fastest) then falls back to PyTorch. "
                            "'tensorrt' forces ONNX Runtime + TensorRT EP (FP16, 30-50%% faster on V100). "
                            "'pytorch' uses the original PyTorch path. "
                            "First TensorRT run exports ONNX and builds the engine (5-15 min, cached after)."
                        ),
                    },
                ),
                "unload_model": (
                    ["Off", "On"],
                    {
                        "default": "Off",
                        "tooltip": (
                            "Unload the CorridorKey model from GPU after processing completes. "
                            "Frees ~2-4GB VRAM so downstream nodes or subsequent runs don't OOM. "
                            "The model will be reloaded automatically on the next run (adds ~5s). "
                            "Turn On if you experience VRAM issues between runs."
                        ),
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("fg", "matte", "processed", "QC")
    FUNCTION = "run"
    CATEGORY = "CorridorKey"

    def __init__(self) -> None:
        self._processor = CorridorKeyProcessor()

    def run(
        self,
        image,
        mask,
        gamma_space: str,
        despill_strength: float,
        refiner_strength: float,
        auto_despeckle: str,
        despeckle_size: int,
        inference_size: int = 2048,
        compute_qc: str = "Off",
        compute_processed: str = "Off",
        batch_size: int = 1,
        num_gpus: int = 0,
        backend: str = "auto",
        unload_model: str = "Off",
        unique_id: str | None = None,
    ):
        settings = CorridorKeySettings(
            gamma_space=str(gamma_space),
            despill_strength=float(despill_strength),
            refiner_strength=float(refiner_strength),
            auto_despeckle=str(auto_despeckle),
            despeckle_size=int(despeckle_size),
            inference_size=int(inference_size),
            compute_qc=str(compute_qc),
            compute_processed=str(compute_processed),
            batch_size=int(batch_size),
            num_gpus=int(num_gpus),
            backend=str(backend),
        )
        progress_callback = _build_progress_reporter(unique_id)
        result = self._processor.refine(
            image=image,
            mask=mask,
            settings=settings,
            progress_callback=progress_callback,
        )

        if unload_model == "On":
            # Keep ORT/TRT sessions cached — they're lightweight (~50MB) and
            # take 3-4s per GPU to recreate. Only free PyTorch model (~2-4GB).
            freed = free_all_engines(keep_ort_sessions=True)
            print(f"[CorridorKey] Unloaded {freed} engine(s), VRAM freed (TRT sessions kept).")

        return result


class CorridorKey_FreeVRAM:
    """Utility node to force-free all CorridorKey models from GPU VRAM.

    Place this anywhere in your workflow (connect any IMAGE through it).
    When executed, it destroys all cached CorridorKey engines and clears
    CUDA memory. The image passes through unchanged.

    Use case: add after CorridorKey to reclaim VRAM for downstream nodes,
    or run standalone to recover from OOM without restarting ComfyUI.
    """

    DESCRIPTION = (
        "Force-free all CorridorKey models from GPU VRAM. "
        "Pass-through node: the input image is returned unchanged. "
        "Use after CorridorKey or standalone to recover from OOM."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "Pass-through: images are returned unchanged after freeing VRAM.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "CorridorKey"
    OUTPUT_NODE = True

    def run(self, images):
        # Full cleanup: free everything including ORT/TRT sessions
        freed = free_all_engines(keep_ort_sessions=False)
        print(f"[CorridorKey_FreeVRAM] Freed {freed} engine(s) + ORT sessions, CUDA cache cleared.")
        return (images,)