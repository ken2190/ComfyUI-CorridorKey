from __future__ import annotations

from dataclasses import dataclass

VALID_INFERENCE_SIZES = (768, 1024, 1536, 2048)
VALID_BACKENDS = ("auto", "tensorrt", "pytorch")


@dataclass(frozen=True, slots=True)
class CorridorKeySettings:
    gamma_space: str = "sRGB"
    despill_strength: float = 1.0
    refiner_strength: float = 1.0
    auto_despeckle: str = "On"
    despeckle_size: int = 400
    inference_size: int = 2048
    compute_qc: str = "Off"
    compute_processed: str = "Off"
    chunk_size: int = 50
    batch_size: int = 1
    num_gpus: int = 0  # 0 = auto-detect
    backend: str = "auto"  # "auto", "tensorrt", or "pytorch"

    def __post_init__(self) -> None:
        if self.gamma_space not in {"sRGB", "Linear"}:
            raise ValueError("gamma_space must be 'sRGB' or 'Linear'.")
        if not 0.0 <= self.despill_strength <= 1.0:
            raise ValueError("despill_strength must be between 0.0 and 1.0.")
        if not 0.0 <= self.refiner_strength <= 4.0:
            raise ValueError("refiner_strength must be between 0.0 and 4.0.")
        if self.auto_despeckle not in {"Off", "On"}:
            raise ValueError("auto_despeckle must be 'Off' or 'On'.")
        if not 0 <= self.despeckle_size <= 4096:
            raise ValueError("despeckle_size must be between 0 and 4096.")
        if self.inference_size not in VALID_INFERENCE_SIZES:
            raise ValueError(f"inference_size must be one of {VALID_INFERENCE_SIZES}.")
        if self.compute_qc not in {"Off", "On"}:
            raise ValueError("compute_qc must be 'Off' or 'On'.")
        if self.compute_processed not in {"Off", "On"}:
            raise ValueError("compute_processed must be 'Off' or 'On'.")
        if self.backend not in VALID_BACKENDS:
            raise ValueError(f"backend must be one of {VALID_BACKENDS}.")

    @property
    def input_is_linear(self) -> bool:
        return self.gamma_space == "Linear"

    @property
    def despeckle_enabled(self) -> bool:
        return self.auto_despeckle == "On"

    @property
    def qc_enabled(self) -> bool:
        return self.compute_qc == "On"

    @property
    def processed_enabled(self) -> bool:
        return self.compute_processed == "On"