"""
@author: local
@title: ComfyUI-CorridorKey
@nickname: CorridorKey
@description: ComfyUI node for CorridorKey inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .corridor_key import schedule_upstream_check
    from .nodes import CorridorKey, CorridorKey_FreeVRAM
except ImportError:
    package_dir = Path(__file__).resolve().parent
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))
    from corridor_key import schedule_upstream_check
    from nodes import CorridorKey, CorridorKey_FreeVRAM

NODE_CLASS_MAPPINGS = {
    "CorridorKey": CorridorKey,
    "CorridorKey_FreeVRAM": CorridorKey_FreeVRAM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CorridorKey": "CorridorKey",
    "CorridorKey_FreeVRAM": "CorridorKey Free VRAM",
}

schedule_upstream_check()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
