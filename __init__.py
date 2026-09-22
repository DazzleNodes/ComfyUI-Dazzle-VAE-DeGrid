"""
ComfyUI Dazzle VAE DeGrid - DazzleNodes Custom Node
Removes the 2px/4px VAE grid ("hatching") from Qwen/Wan/Krea decodes via
coherent global estimation.

Part of the DazzleNodes collection - standalone ComfyUI custom nodes.
"""

import os
import sys

# =====================================================
# DUAL-LOADING DETECTION
# Prevents issues when this node is installed both as a
# standalone node AND inside DazzleNodes. Uses a sys-level
# sentinel (shared across module namespaces) to detect the
# second load.
# =====================================================
_SENTINEL = "_dazzle_vae_degrid_loaded"
_is_duplicate_load = hasattr(sys, _SENTINEL)

if _is_duplicate_load:
    _first_path = getattr(sys, _SENTINEL)
    _this_path = os.path.dirname(os.path.abspath(__file__))
    print("[DazzleVAEDeGrid] WARNING: Duplicate installation detected!")
    print(f"[DazzleVAEDeGrid]   Already loaded from: {_first_path}")
    print(f"[DazzleVAEDeGrid]   Skipping this copy:  {_this_path}")
    print("[DazzleVAEDeGrid]   Fix: Remove one installation (standalone symlink or DazzleNodes submodule).")
else:
    setattr(sys, _SENTINEL, os.path.dirname(os.path.abspath(__file__)))

from .degrid import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

try:
    from .ComfyUI_Dazzle_VAE_DeGrid._version import __version__
except Exception:
    __version__ = "0.0.0-unknown"

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
