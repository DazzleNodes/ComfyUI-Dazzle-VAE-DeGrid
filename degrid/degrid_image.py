"""DazzleVAEDeGrid (Image) — post-decode VAE grid removal node."""

import torch

from .degrid_core import degrid


class DazzleVAEDeGridImage:
    """Remove the Qwen/Wan VAE 2px grid (hatching) from decoded images.

    Wire directly after VAE Decode — and before any re-encode, upscale,
    or sharpen. The filter targets the decode-native pixel grid; once an
    image has been resized the grid moves off the 2px period and this
    node will no longer find it.
    """

    CATEGORY = "DazzleNodes/DeGrid"
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["coherent", "notch"], {
                    "default": "coherent",
                    "tooltip": "coherent: detail-preserving global grid estimation "
                               "(recommended). notch: community 7-tap reference — "
                               "kills the grid but blurs fine detail.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Multiplier on the subtracted grid estimate.",
                }),
                "sigma": ("FLOAT", {
                    "default": 48.0, "min": 8.0, "max": 128.0, "step": 4.0,
                    "tooltip": "Coherent mode: smoothing radius (px) for the grid "
                               "amplitude estimate. LARGER is better on both grid "
                               "removal and detail preservation for Qwen/Wan; do "
                               "not lower expecting a stronger kill.",
                }),
                "harmonics": (["2px", "2px+4px"], {
                    "default": "2px",
                    "tooltip": "2px (default) targets the VAE grid. 2px+4px also "
                               "removes 4px-period components — ONLY use on "
                               "straight-from-decode images: webp/JPEG-compressed "
                               "sources carry globally-coherent 4px DCT block "
                               "harmonics that this path would 'remove', "
                               "imprinting visible texture on flat regions.",
                }),
                "min_amplitude": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Amplitude gate (8-bit levels; 0 disables). Regions "
                               "whose local grid amplitude is below this are left "
                               "untouched, so flat backgrounds cannot pick up "
                               "rounding dither when saved to 8-bit.",
                }),
            },
        }

    def apply(self, image, mode, strength, sigma, harmonics, min_amplitude=0.3):
        dev = image.device
        run_dev = dev
        if dev.type == "cpu" and torch.cuda.is_available():
            run_dev = torch.device("cuda")
        x = image.to(run_dev)
        out = degrid(x, mode=mode, strength=strength, sigma=sigma,
                     harmonics=harmonics, min_amplitude=min_amplitude)
        return (out.to(dev),)
