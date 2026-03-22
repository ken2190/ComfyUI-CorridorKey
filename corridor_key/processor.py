from __future__ import annotations

import gc
import logging
from typing import Callable

import numpy as np
import torch

from .config import CorridorKeySettings
from .engine import get_cached_engine
from .tensor_ops import (
    batch_to_numpy,
    ensure_image_tensor,
    ensure_mask_batch,
    stack_mask_frames,
    stack_rgb_frames,
)

LOGGER = logging.getLogger(__name__)


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

        if progress_callback is not None:
            progress_callback(
                f"Loading CorridorKey model (inference_size={settings.inference_size})...",
                0,
                total_frames,
            )

        engine = get_cached_engine(
            device=self._device,
            img_size=settings.inference_size,
        )

        image_frames = batch_to_numpy(image_batch)
        mask_frames = batch_to_numpy(mask_batch)

        # Free the original batched tensors to reduce peak memory
        del image_batch, mask_batch

        # Process frames in chunks to cap peak memory
        all_fg: list[torch.Tensor] = []
        all_matte: list[torch.Tensor] = []
        all_processed: list[torch.Tensor] = []
        all_comp: list[torch.Tensor] = []

        for chunk_start in range(0, total_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_frames)
            chunk_fg: list[np.ndarray] = []
            chunk_matte: list[np.ndarray] = []
            chunk_processed: list[np.ndarray] = []
            chunk_comp: list[np.ndarray] = []

            for frame_index in range(chunk_start, chunk_end):
                global_idx = frame_index + 1
                if progress_callback is not None:
                    progress_callback(
                        f"Processing frame {global_idx}/{total_frames}...",
                        frame_index,
                        total_frames,
                    )

                result = engine.process_frame_tensor(
                    image=image_frames[frame_index],
                    mask_linear=mask_frames[frame_index],
                    refiner_scale=settings.refiner_strength,
                    input_is_linear=settings.input_is_linear,
                    despill_strength=settings.despill_strength,
                    auto_despeckle=settings.despeckle_enabled,
                    despeckle_size=settings.despeckle_size,
                    compute_qc=settings.qc_enabled,
                )
                chunk_fg.append(result["fg"])
                chunk_matte.append(result["matte"])
                chunk_processed.append(result["processed"])
                chunk_comp.append(result["comp"])
                LOGGER.debug("CorridorKey processed frame %s", global_idx)

            # Stack chunk into tensors immediately and free numpy lists
            all_fg.append(stack_rgb_frames(chunk_fg))
            all_matte.append(stack_mask_frames(chunk_matte))
            all_processed.append(stack_rgb_frames(chunk_processed))
            all_comp.append(stack_rgb_frames(chunk_comp))

            del chunk_fg, chunk_matte, chunk_processed, chunk_comp
            gc.collect()

            LOGGER.info(
                "CorridorKey chunk %d-%d/%d done, stacked to tensor.",
                chunk_start + 1,
                chunk_end,
                total_frames,
            )

        if progress_callback is not None:
            progress_callback("CorridorKey complete.", total_frames, total_frames)

        # Concatenate all chunk tensors
        return (
            torch.cat(all_fg, dim=0) if len(all_fg) > 1 else all_fg[0],
            torch.cat(all_matte, dim=0) if len(all_matte) > 1 else all_matte[0],
            torch.cat(all_processed, dim=0) if len(all_processed) > 1 else all_processed[0],
            torch.cat(all_comp, dim=0) if len(all_comp) > 1 else all_comp[0],
        )