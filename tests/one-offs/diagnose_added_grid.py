"""Diagnose the reported 'coherent ADDED hatching' on the 2026-09-22 test.

Questions:
  1. Is raw_00001_.png pixel-identical to the source webp (wiring check)?
  2. Where does the source's grid energy actually live — exactly at
     Nyquist/4px (straight decode) or off-grid (resized 'tile guide')?
  3. What did the filter change? FFT the (coherent - raw) difference,
     globally and in the top-150-row band vs the image center.
  4. Save 100x100 top-left crops at 6x for eyeballing.
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_CORE = Path(__file__).resolve().parent.parent.parent / "degrid" / "degrid_core.py"
spec = importlib.util.spec_from_file_location("degrid_core", _CORE)
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)

SRC = r"C:\code\ComfyUI_experiment\output\d3\2026-09-22_03-52-30 _qwen_tile_guide_1.webp"
RAW = r"C:\code\ComfyUI_experiment\output\degrid\raw_00001_.png"
COH = r"C:\code\ComfyUI_experiment\output\degrid\coherent_00001_.png"
OUT = Path(r"C:\code\ComfyUI-Dazzle-VAE-DeGrid\local\tests\output")


def load(p):
    a = np.asarray(Image.open(p).convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(a).unsqueeze(0)


src, raw, coh = load(SRC), load(RAW), load(COH)
print(f"src {tuple(src.shape)}  raw {tuple(raw.shape)}  coh {tuple(coh.shape)}")

# --- 1. wiring check ---
if src.shape == raw.shape:
    d = (src - raw).abs()
    print(f"src vs raw: max|diff|={d.max():.6f} mean={d.mean():.6f} "
          f"({'IDENTICAL' if d.max() < 1/510 else 'DIFFERENT'})")
else:
    print("src vs raw: DIFFERENT SHAPES — user loaded a different file!")

# --- 2. where does source grid energy live? scan the FFT rows/cols ---
def spectrum_profile(img, name):
    y = (0.2126 * img[0, :, :, 0] + 0.7152 * img[0, :, :, 1]
         + 0.0722 * img[0, :, :, 2])
    h, w = y.shape
    h -= h % 2; w -= w % 2
    y = y[:h, :w]
    win = torch.outer(torch.hann_window(h, periodic=False),
                      torch.hann_window(w, periodic=False))
    M = torch.abs(torch.fft.fft2((y - y.mean()) * win))
    # horizontal-frequency profile along v=0 (top row of spectrum),
    # from 0.15*Nyq up to Nyq; print top-5 peaks with their period
    prof_x = M[0, : w // 2 + 1].clone()
    prof_y = M[: h // 2 + 1, 0].clone()
    for label, prof, n in (("x", prof_x, w), ("y", prof_y, h)):
        lo = int(0.15 * (len(prof) - 1))
        p = prof.clone(); p[:lo] = 0
        top = torch.topk(p, 5)
        peaks = []
        med = torch.median(prof[lo:])
        for v, i in zip(top.values, top.indices):
            period = n / max(int(i), 1)
            peaks.append(f"{period:.2f}px(x{v / med:.0f})")
        print(f"  {name} {label}-axis peaks: " + ", ".join(peaks))


print("\n--- spectral peaks (period in px, strength vs median) ---")
spectrum_profile(raw, "raw")
spectrum_profile(coh, "coh")

m_raw = dc.measure_grid(raw)
m_coh = dc.measure_grid(coh)
print("\nexact-bin ratios raw:", {k: round(v[0], 1) for k, v in m_raw["ratios"].items()})
print("exact-bin ratios coh:", {k: round(v[0], 1) for k, v in m_coh["ratios"].items()})

# --- 3. what did the filter change? ---
diff = (coh - raw)[0]  # (H, W, 3)
dl = diff.abs().mean(-1)
h, w = dl.shape
print(f"\ncoherent-raw: mean|d|={dl.mean():.5f} max={dl.max():.4f} "
      f"(8-bit: mean {255 * dl.mean():.2f}, max {255 * dl.max():.1f})")
bands = {"top150": dl[:150], "mid": dl[h // 2 - 75: h // 2 + 75],
         "bottom150": dl[-150:], "left150": dl[:, :150], "right150": dl[:, -150:]}
for k, b in bands.items():
    print(f"  band {k}: mean|d|={255 * b.mean():.2f}/255 max={255 * b.max():.1f}/255")

spectrum_profile((coh - raw + 0.5).clamp(0, 1), "DIFF(+0.5)")

# --- 4. crops for eyeballing: top-left 100x100 at 6x ---
OUT.mkdir(exist_ok=True)
for name, img in (("src", src), ("raw", raw), ("coh", coh)):
    a = (img[0, :100, :100].clamp(0, 1).numpy() * 255).astype("uint8")
    Image.fromarray(a).resize((600, 600), Image.NEAREST).save(OUT / f"tl100_{name}_6x.png")
amp = ((diff[:100, :100] * 10 + 0.5).clamp(0, 1).numpy() * 255).astype("uint8")
Image.fromarray(amp).resize((600, 600), Image.NEAREST).save(OUT / "tl100_diff_x10_6x.png")
print(f"\ncrops saved to {OUT}")
