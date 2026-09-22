"""Node registrations for ComfyUI-Dazzle-VAE-DeGrid."""

from .degrid_image import DazzleVAEDeGridImage
from .grid_diagnostic import DazzleGridDiagnostic

NODE_CLASS_MAPPINGS = {
    "DazzleVAEDeGridImage": DazzleVAEDeGridImage,
    "DazzleGridDiagnostic": DazzleGridDiagnostic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DazzleVAEDeGridImage": "Dazzle VAE DeGrid (Image)",
    "DazzleGridDiagnostic": "Dazzle Grid Diagnostic",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
