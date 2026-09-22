"""Sweep smoothing sigma (and pass count) for the coherent de-grid:
grid-kill on a real Qwen image vs detail-preservation on its
grid-free (Lanczos 1/3) downscale. Picks the default settings.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_CORE = Path(__file__).resolve().parent.parent.parent / "degrid" / "degrid_core.py"
spec = importlib.util.spec_from_file_location("degrid_core", _CORE)
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)

REAL = r"C:\code\ComfyUI_experiment\output\d3\2026-07-02_21-29-37 _qwen_1.webp"

im = Image.open(REAL).convert("RGB")
full = torch.from_numpy(np.asarray(im).astype("float32") / 255.0).unsqueeze(0)
small = im.resize((im.width // 3, im.height // 3), Image.LANCZOS)
ref = torch.from_numpy(np.asarray(small).astype("float32") / 255.0).unsqueeze(0)

BINS2 = ("nyq_x", "nyq_y", "nyq_xy")
BINS4 = ("p4_x", "p4_y", "p4_xy")


def max_ratio(m, bins):
    return max(m["ratios"][b][0] for b in bins)


before = dc.measure_grid(full)
print(f"before: 2px={max_ratio(before, BINS2):.1f} 4px={max_ratio(before, BINS4):.1f}")
print(f"{'sigma':>6} {'passes':>6} | {'2px':>7} {'4px':>7} | {'AC2 PSNR':>9}")

for sigma in (16.0, 8.0, 4.0, 2.0):
    for passes in (1, 2):
        out = full
        for _ in range(passes):
            out = dc.degrid(out, mode="coherent", sigma=sigma, harmonics="2px+4px")
        m = dc.measure_grid(out)
        # AC2: same settings applied to the grid-free reference
        clean_out = ref
        for _ in range(passes):
            clean_out = dc.degrid(clean_out, mode="coherent", sigma=sigma,
                                  harmonics="2px+4px")
        p = dc.psnr(clean_out, ref)
        print(f"{sigma:>6} {passes:>6} | {max_ratio(m, BINS2):>7.2f} "
              f"{max_ratio(m, BINS4):>7.2f} | {p:>9.2f}")
