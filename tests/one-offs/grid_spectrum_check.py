"""Measure 2px-grid (Nyquist) artifact strength in decoded images.

For each image:
  1. FFT-based: magnitude of spectrum at (0, Nyq), (Nyq, 0), (Nyq, Nyq)
     vs. median magnitude in the surrounding high-frequency band.
     A coherent 2px grid shows as isolated sharp peaks at these bins.
  2. Spatial parity check: mean luma per 2x2 phase class after high-pass.
     A grid gives systematic per-phase offsets; clean images ~0.

Usage: python grid_spectrum_check.py <image1> [image2 ...]
"""
import sys
import numpy as np
from PIL import Image


def analyze(path):
    img = Image.open(path).convert("RGB")
    a = np.asarray(img).astype(np.float64) / 255.0
    # luma
    y = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    H, W = y.shape
    # crop to even dims
    H -= H % 2
    W -= W % 2
    y = y[:H, :W]

    # --- FFT measure ---
    win = np.outer(np.hanning(H), np.hanning(W))
    F = np.fft.fft2((y - y.mean()) * win)
    M = np.abs(F)
    ny, nx = H // 2, W // 2  # Nyquist bin indices

    def peak_vs_bg(r, c, rad=6):
        peak = M[r % H, c % W]
        r0, r1 = max(0, r - rad), min(H, r + rad + 1)
        c0, c1 = max(0, c - rad), min(W, c + rad + 1)
        patch = M[r0:r1, c0:c1].copy()
        pr, pc = r - r0, c - c0
        patch[max(0, pr - 1):pr + 2, max(0, pc - 1):pc + 2] = np.nan
        bg = np.nanmedian(patch)
        return peak / max(bg, 1e-12)

    ratios = {
        "(Nyq_x, 0)   horiz-alternating": peak_vs_bg(0, nx),
        "(0, Nyq_y)   vert-alternating ": peak_vs_bg(ny, 0),
        "(Nyq, Nyq)   checkerboard     ": peak_vs_bg(ny, nx),
        "(Nyq/2, 0)   4px horiz        ": peak_vs_bg(0, nx // 2),
        "(0, Nyq/2)   4px vert         ": peak_vs_bg(ny // 2, 0),
        "(Nyq/2,Nyq/2) 4px diag        ": peak_vs_bg(ny // 2, nx // 2),
    }

    # --- spatial parity measure ---
    # high-pass: subtract 5x5 box blur (cheap)
    from scipy.ndimage import uniform_filter
    hp = y - uniform_filter(y, size=5)
    phases = {
        "ee": hp[0::2, 0::2].mean(), "eo": hp[0::2, 1::2].mean(),
        "oe": hp[1::2, 0::2].mean(), "oo": hp[1::2, 1::2].mean(),
    }
    # grid amplitude estimates (in 8-bit levels)
    gx = 255 * abs((phases["ee"] + phases["oe"]) - (phases["eo"] + phases["oo"])) / 2
    gy = 255 * abs((phases["ee"] + phases["eo"]) - (phases["oe"] + phases["oo"])) / 2
    gd = 255 * abs((phases["ee"] + phases["oo"]) - (phases["eo"] + phases["oe"])) / 2

    print(f"\n=== {path} ({W}x{H}) ===")
    print("FFT peak/background ratios (>3 = clear coherent grid):")
    for k, v in ratios.items():
        flag = "  <-- GRID" if v > 3 else ""
        print(f"  {k}: {v:7.2f}{flag}")
    print(f"Spatial parity grid amplitude (8-bit levels): "
          f"x={gx:.3f}  y={gy:.3f}  diag={gd:.3f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            analyze(p)
        except Exception as e:
            print(f"ERROR {p}: {type(e).__name__}: {e}")
