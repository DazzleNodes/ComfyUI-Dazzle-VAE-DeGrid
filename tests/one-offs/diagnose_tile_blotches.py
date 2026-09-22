"""Diagnose large square structures in the 2400-tall tiled render.

1. Dimensions + degrid-delta check (our filter should be ~zero at
   tile scale — confirm it neither causes nor can fix this).
2. Tile-lattice detection: fold low-passed-gradient energy by candidate
   spacings; a real tiling spikes at one spacing/offset.
3. Blotch amplitude: per-cell mean deviation of low-passed luma in the
   flat background vs. its neighborhood.
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_CORE = Path(__file__).resolve().parent.parent.parent / "degrid" / "degrid_core.py"
spec = importlib.util.spec_from_file_location("dc", _CORE)
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)

RAW = r"C:\code\ComfyUI_experiment\output\d3\2026-09-22_05-33-40 _qwen_tile_guide_1.webp"
COH = r"C:\code\ComfyUI_experiment\output\degrid\coherent_00001_.png"


def load(p):
    a = np.asarray(Image.open(p).convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(a).unsqueeze(0)


raw = load(RAW)
print(f"raw {tuple(raw.shape)}")
try:
    coh = load(COH)
    print(f"coh {tuple(coh.shape)}")
    if coh.shape == raw.shape:
        d = (coh - raw)[0]
        dl = dc._to_bhwc(dc._blur2d(dc._to_bchw(d.unsqueeze(0).mean(-1, keepdim=True)), 16.0))[0]
        print(f"degrid delta, low-passed (sigma16): mean|d|={255*dl.abs().mean():.4f} "
              f"max={255*dl.abs().max():.3f} 8-bit levels "
              f"(should be ~0: we do not touch tile scale)")
    else:
        print("coh is a different render — skipping delta check")
except Exception as e:
    print(f"coh load failed: {e}")

x = dc._to_bchw(raw.float())
luma = (0.2126 * x[:, 0] + 0.7152 * x[:, 1] + 0.0722 * x[:, 2]).unsqueeze(1)
lp = dc._blur2d(luma, 4.0)
# gradient magnitude of the low-passed luma
gy = (lp[:, :, 1:, :] - lp[:, :, :-1, :]).abs().squeeze()[..., :-1]
gx = (lp[:, :, :, 1:] - lp[:, :, :, :-1]).abs().squeeze()[:-1, :]
h, w = gx.shape
print(f"\nlattice scan (fold gradient energy by spacing; ratio = peak bin vs median bin):")
for axis, g, n in (("x", gx.mean(0), w), ("y", gy.mean(1), h)):
    prof = g.numpy()
    for s in (256, 320, 384, 448, 512, 640, 768, 896, 1024):
        if n < 2 * s:
            continue
        m = n - (n % s)
        fold = prof[:m].reshape(-1, s).mean(0)
        ratio = fold.max() / np.median(fold)
        if ratio > 1.5:
            print(f"  {axis}: spacing {s:>4}: peak/median={ratio:.2f} at offset {int(fold.argmax())}")

# blotch amplitude in the flat background: cell-mean deviations at various scales
print("\nblotch amplitude (std of cell means of low-passed luma, background strip x<400):")
bg = dc._blur2d(luma, 8.0)[0, 0, :, :400]
for cell in (64, 128, 256):
    hh = bg.shape[0] - bg.shape[0] % cell
    ww = 400 - 400 % cell
    cells = bg[:hh, :ww].reshape(hh // cell, cell, ww // cell, cell).mean(dim=(1, 3))
    # deviation of each cell from the local 3x3 cell-neighborhood mean
    pad = torch.nn.functional.avg_pool2d(cells.unsqueeze(0).unsqueeze(0), 3, 1, 1).squeeze()
    dev = (cells - pad).abs()
    print(f"  cell {cell:>3}px: mean|dev|={255*dev.mean():.3f} max={255*dev.max():.3f} 8-bit levels")
