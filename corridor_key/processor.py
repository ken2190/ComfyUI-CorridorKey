from __future__ import annotations

import gc
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np
import torch

from .config import CorridorKeySettings
from .engine import get_available_gpu_count, get_cached_engine, get_multi_gpu_engines
from .tensor_ops import (
    batch_to_numpy,
    ensure_image_tensor,
    ensure_mask_batch,
    stack_mask_frames,
    stack_rgb_frames,
)

LOGGER = logging.getLogger(__name__)


def _process_mini_batch_on_engine(
    engine,
    images: list[np.ndarray],
    masks: list[np.ndarray],
    settings: CorridorKeySettings,
) -> list[dict[str, np.ndarray]]:
    """Run a mini-batch of frames through a single engine (one GPU).
    Uses batched model inference when batch_size > 1."""
    return engine.process_batch_tensor(
        images=images,
        masks=masks,
        refiner_scale=settings.refiner_strength,
        input_is_linear=settings.input_is_linear,
        despill_strength=settings.despill_strength,
        auto_despeckle=settings.despeckle_enabled,
        despeckle_size=settings.despeckle_size,
        compute_qc=settings.qc_enabled,
        compute_processed=settings.processed_enabled,
    )


class CorridorKeyProcessor:
    def __init__(self, device: str | None = None) -> None:
        self._device = device

    def refine(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        settings: CorridorKeySettings,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(settings, CorridorKeySettings):
            raise ValueError("settings must be a CorridorKeySettings instance.")

        image_batch = ensure_image_tensor(image)
        mask_batch = ensure_mask_batch(
            mask=mask,
            batch_size=image_batch.shape[0],
            height=image_batch.shape[1],
            width=image_batch.shape[2],
        )

        total_frames = int(image_batch.shape[0])
        chunk_size = settings.chunk_size
        batch_size = max(1, settings.batch_size)

        # Determine GPU count
        num_gpus_available = get_available_gpu_count()
        if self._device and self._device.startswith("cuda"):
            # Explicit device: single-GPU mode
            num_gpus = 1
        elif settings.num_gpus <= 0:
            num_gpus = max(1, num_gpus_available)
        else:
            num_gpus = min(settings.num_gpus, max(1, num_gpus_available))

        if progress_callback is not None:
            progress_callback(
                f"Loading CorridorKey (size={settings.inference_size}, "
                f"gpus={num_gpus}, batch={batch_size}, backend={settings.backend})...",
                0,
                total_frames,
            )

        # Create engines — one per GPU (or single engine for CPU/explicit device)
        if num_gpus <= 1:
            engines = [get_cached_engine(
                device=self._device,
                img_size=settings.inference_size,
                backend=settings.backend,
            )]
        else:
            engines = get_multi_gpu_engines(
                img_size=settings.inference_size,
                num_gpus=num_gpus,
                backend=settings.backend,
            )

        LOGGER.info(
            "CorridorKey engines ready: %d GPU(s), inference_size=%d, batch_size=%d, compiled=%s",
            len(engines),
            settings.inference_size,
            batch_size,
            engines[0]._compiled if engines else "N/A",
        )

        image_frames = batch_to_numpy(image_batch)
        mask_frames = batch_to_numpy(mask_batch)
        del image_batch, mask_batch

        # Collect all results
        all_fg: list[torch.Tensor] = []
        all_matte: list[torch.Tensor] = []
        all_processed: list[torch.Tensor] = []
        all_comp: list[torch.Tensor] = []

        # Process in chunks (memory management boundary)
        for chunk_start in range(0, total_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_frames)
            chunk_frame_indices = list(range(chunk_start, chunk_end))

            # Split chunk into mini-batches, distributed across GPUs
            mini_batches = []
            for mb_start in range(0, len(chunk_frame_indices), batch_size):
                mb_indices = chunk_frame_indices[mb_start:mb_start + batch_size]
                mini_batches.append(mb_indices)

            # Ordered results for this chunk
            chunk_results: list[tuple[int, list[dict[str, np.ndarray]]]] = [None] * len(mini_batches)

            if len(engines) == 1:
                # Single GPU: sequential mini-batches (no thread overhead)
                for mb_idx, mb_indices in enumerate(mini_batches):
                    if progress_callback is not None:
                        progress_callback(
                            f"Processing frames {mb_indices[0]+1}-{mb_indices[-1]+1}/{total_frames}...",
                            mb_indices[0],
                            total_frames,
                        )
                    mb_images = [image_frames[i] for i in mb_indices]
                    mb_masks = [mask_frames[i] for i in mb_indices]
                    results = _process_mini_batch_on_engine(engines[0], mb_images, mb_masks, settings)
                    chunk_results[mb_idx] = (mb_idx, results)
            else:
                # Multi-GPU: distribute mini-batches across GPUs via thread pool
                # CUDA releases GIL during kernel execution, so threads give true parallelism
                with ThreadPoolExecutor(max_workers=len(engines)) as executor:
                    futures = {}
                    for mb_idx, mb_indices in enumerate(mini_batches):
                        engine = engines[mb_idx % len(engines)]
                        mb_images = [image_frames[i] for i in mb_indices]
                        mb_masks = [mask_frames[i] for i in mb_indices]
                        future = executor.submit(
                            _process_mini_batch_on_engine, engine, mb_images, mb_masks, settings,
                        )
                        futures[future] = mb_idx

                    for future in as_completed(futures):
                        mb_idx = futures[future]
                        mb_indices = mini_batches[mb_idx]
                        results = future.result()
                        chunk_results[mb_idx] = (mb_idx, results)
                        if progress_callback is not None:
                            progress_callback(
                                f"Processed frames {mb_indices[0]+1}-{mb_indices[-1]+1}/{total_frames} "
                                f"(GPU {mb_idx % len(engines)})",
                                mb_indices[-1] + 1,
                                total_frames,
                            )

            # Flatten chunk results in order
            chunk_fg = []
            chunk_matte = []
            chunk_processed = []
            chunk_comp = []
            for _, frame_results in sorted(chunk_results, key=lambda x: x[0]):
                for r in frame_results:
                    chunk_fg.append(r["fg"])
                    chunk_matte.append(r["matte"])
                    chunk_processed.append(r["processed"])
                    chunk_comp.append(r["comp"])

            # Stack chunk to tensors
            all_fg.append(stack_rgb_frames(chunk_fg))
            all_matte.append(stack_mask_frames(chunk_matte))
            all_processed.append(stack_rgb_frames(chunk_processed))
            all_comp.append(stack_rgb_frames(chunk_comp))

            del chunk_fg, chunk_matte, chunk_processed, chunk_comp, chunk_results
            gc.collect()

            LOGGER.info(
                "CorridorKey chunk %d-%d/%d done.",
                chunk_start + 1, chunk_end, total_frames,
            )

        if progress_callback is not None:
            progress_callback("CorridorKey complete.", total_frames, total_frames)

        return (
            torch.cat(all_fg, dim=0) if len(all_fg) > 1 else all_fg[0],
            torch.cat(all_matte, dim=0) if len(all_matte) > 1 else all_matte[0],
            torch.cat(all_processed, dim=0) if len(all_processed) > 1 else all_processed[0],
            torch.cat(all_comp, dim=0) if len(all_comp) > 1 else all_comp[0],
        )
