"""Core algorithm for VAE 2px/4px grid ("hatching") removal.

Pure PyTorch — no ComfyUI imports — so the algorithm is testable offline
and reusable/upstreamable. Operates on ComfyUI IMAGE convention tensors:
(B, H, W, C) float32 in ~[0, 1].

Background
----------
The Qwen-Image / Wan 2.1 VAE decoder upsamples with nearest-exact 2x +
3x3 conv (three stages). The polyphase imbalance of that structure stamps
a globally phase-coherent 2px grid (Nyquist frequency) plus weaker 4px
harmonics onto every decoded image.

Two removal modes:

- ``notch``: the community 7-tap separable Nyquist notch
  (B = [-1, 6, -15, 20, -15, 6, -1] / 64, frequency response sin^6(w/2)).
  Kills the grid but attenuates ~42% of 3px-period detail — visible blur
  on hair/texture. Kept as a reference baseline.

- ``coherent`` (default): exploits the grid's *global phase coherence*.
  For each grid component, demodulate with the component's carrier
  (e.g. (-1)^x), low-pass the demodulated field heavily (the coherent
  grid amplitude survives; locally-incoherent true detail averages to
  ~zero), re-modulate and subtract. Only the spatially-coherent part of
  the Nyquist/4px energy is removed, so fine detail is preserved.
"""

import math

import torch
import torch.nn.functional as F

__all__ = [
    "degrid",
    "measure_grid",
    "inject_grid",
    "psnr",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _to_bchw(img_bhwc: torch.Tensor) -> torch.Tensor:
    return img_bhwc.permute(0, 3, 1, 2)


def _to_bhwc(img_bchw: torch.Tensor) -> torch.Tensor:
    return img_bchw.permute(0, 2, 3, 1)


def _gaussian_kernel1d(sigma: float, dtype, device) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _pad_mode_for(size: int, pad: int) -> str:
    # reflect padding requires pad < size; fall back for tiny images
    return "reflect" if pad < size else "replicate"


def _blur2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on (B, C, H, W) with reflect padding."""
    if sigma <= 0:
        return x
    b, c, h, w = x.shape
    k = _gaussian_kernel1d(sigma, x.dtype, x.device)
    # clamp kernel so padding stays legal on small images
    max_r = max(1, min(h, w) - 1)
    r = (k.numel() - 1) // 2
    if r > max_r:
        k = k[r - max_r : r + max_r + 1]
        k = k / k.sum()
        r = max_r
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    x = F.pad(x, (r, r, 0, 0), mode=_pad_mode_for(w, r))
    x = F.conv2d(x, kx, groups=c)
    x = F.pad(x, (0, 0, r, r), mode=_pad_mode_for(h, r))
    x = F.conv2d(x, ky, groups=c)
    return x


def psnr(a: torch.Tensor, b: torch.Tensor, peak: float = 1.0) -> float:
    """PSNR in dB between two tensors of identical shape."""
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(peak * peak / mse)


# --------------------------------------------------------------------------
# carriers
#
# Each grid component is described by quadrature carriers that are exact
# at integer pixel coordinates (no float trig error):
#   2px:  (-1)^x            -> [1, -1] cycle
#   4px:  cos(pi x / 2)     -> [1, 0, -1, 0] cycle
#         sin(pi x / 2)     -> [0, 1, 0, -1] cycle
#
# A quadrature is (row_vec | None, col_vec | None, recon_factor). The
# factor is 1 / mean(carrier^2) over pixels: 1 for the +-1 carriers,
# 2 per axis for the quadrature pairs (mean cos^2 = 1/2), 4 for 2D pairs.
# --------------------------------------------------------------------------

def _alt(n: int, dtype, device) -> torch.Tensor:
    """(-1)^i vector of length n."""
    i = torch.arange(n, device=device)
    return (1.0 - 2.0 * (i % 2).to(dtype))


def _quarter(n: int, phase: int, dtype, device) -> torch.Tensor:
    """cos(pi i / 2) for phase=0, sin(pi i / 2) for phase=1 — exact at ints."""
    table = torch.tensor(
        [1.0, 0.0, -1.0, 0.0] if phase == 0 else [0.0, 1.0, 0.0, -1.0],
        dtype=dtype,
        device=device,
    )
    i = torch.arange(n, device=device)
    return table[i % 4]


def _component_quadratures(h, w, harmonics, dtype, device):
    """List of (name, [(row_vec|None, col_vec|None, factor), ...]) sets."""
    sx = _alt(w, dtype, device)
    sy = _alt(h, dtype, device)
    sets = [
        ("nyq_x", [(None, sx, 1.0)]),
        ("nyq_y", [(sy, None, 1.0)]),
        ("nyq_xy", [(sy, sx, 1.0)]),
    ]
    if harmonics == "2px+4px":
        c4x = _quarter(w, 0, dtype, device)
        s4x = _quarter(w, 1, dtype, device)
        c4y = _quarter(h, 0, dtype, device)
        s4y = _quarter(h, 1, dtype, device)
        sets += [
            ("p4_x", [(None, c4x, 2.0), (None, s4x, 2.0)]),
            ("p4_y", [(c4y, None, 2.0), (s4y, None, 2.0)]),
            ("p4_xy", [
                (c4y, c4x, 4.0),
                (c4y, s4x, 4.0),
                (s4y, c4x, 4.0),
                (s4y, s4x, 4.0),
            ]),
        ]
    return sets


def _apply_carrier(x: torch.Tensor, row, col) -> torch.Tensor:
    """Multiply (B, C, H, W) by a separable carrier field."""
    if row is not None:
        x = x * row.view(1, 1, -1, 1)
    if col is not None:
        x = x * col.view(1, 1, 1, -1)
    return x


# --------------------------------------------------------------------------
# notch mode (community reference: 7-tap alternating binomial)
# --------------------------------------------------------------------------

_NOTCH_B = [-1.0, 6.0, -15.0, 20.0, -15.0, 6.0, -1.0]


def _notch_conv(x: torch.Tensor, axis: int) -> torch.Tensor:
    """Apply the 7-tap B kernel along one axis of (B, C, H, W)."""
    b, c, h, w = x.shape
    k = torch.tensor(_NOTCH_B, dtype=x.dtype, device=x.device) / 64.0
    if axis == 0:  # along H (y)
        kern = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
        x = F.pad(x, (0, 0, 3, 3), mode=_pad_mode_for(h, 3))
        return F.conv2d(x, kern, groups=c)
    kern = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    x = F.pad(x, (3, 3, 0, 0), mode=_pad_mode_for(w, 3))
    return F.conv2d(x, kern, groups=c)


def _notch_estimate(x: torch.Tensor) -> torch.Tensor:
    """Grid estimate per the reference GLSL: Bx + By - Bxy."""
    bx = _notch_conv(x, 1)
    by = _notch_conv(x, 0)
    bxy = _notch_conv(bx, 0)  # separable: B along x then y
    return bx + by - bxy


# --------------------------------------------------------------------------
# coherent mode
# --------------------------------------------------------------------------

_GATE_SIGMA = 16.0        # smoothing radius (px) for the gate's local field
_GATE_SMOOTH_SIGMA = 36.0  # smoothing of the agreement ratio itself: the
                           # local field's detail leak is zero-mean, so
                           # smoothing BEFORE clamping removes the downward
                           # bias that clamp01 alone would introduce
                           # (clamp cuts only the >1 side of the noise).
                           # 16/36 chosen by 4-criteria sweep 2026-09-22:
                           # synth kill <3x, PSNR 56dB, dither flips 0.2%,
                           # real denim parity 2.05 -> 0.26


def _coherent_estimate(x: torch.Tensor, sigma: float, harmonics: str,
                       return_maps: bool = False, gate_tau: float = 0.0):
    """Coherent grid estimate on (B, C, H, W).

    gate_tau (image-scale units; 0 disables): local-agreement gate.
    Per component, the global estimate s (sigma-smoothed) is scaled by

        gate = clamp01( <g, s> / (|s|^2 + tau^2) ) ^ 2

    where g is the LOCAL demodulated field (smoothed at _GATE_SIGMA) and
    <g, s> sums over the component's quadratures. This keys on PHASE
    AGREEMENT between the local field and the global estimate, which is
    what separates real grid from look-alikes:
      - uniform real grid, amplitude >> tau: g ~ s, gate ~ 1 — full kill;
      - region whose s is mostly bleed from gridded neighbours (or webp
        compression noise): g is phase-incoherent with s, <g,s> ~ 0,
        gate ~ 0 — flat areas stay bit-identical, no save-time dither;
      - uniform but sub-quantum grid (< tau): gate ~ (a/tau)^4 ~ 0 —
        invisible grid is not worth a sub-quantum correction that 8-bit
        rounding would amplify into organized dither;
      - textured region with real grid: detail leak in g is zero-mean
        against s, gate ~ 1 — subtract as planned.
    (2026-09-22: a pure amplitude gate failed on a webp-compressed test
    image — webp noise passes an amplitude test but not a phase test.)

    Returns (estimate, maps) where maps[name] is the smoothed demodulated
    amplitude field (B, C, H, W) per component (only if return_maps).
    """
    b, c, h, w = x.shape
    est = torch.zeros_like(x)
    maps = {} if return_maps else None
    for name, quads in _component_quadratures(h, w, harmonics, x.dtype, x.device):
        comp = torch.zeros_like(x)
        amp_sq = None
        corr = None
        s_pow = None
        for row, col, factor in quads:
            d = _apply_carrier(x, row, col)
            s = _blur2d(d, sigma)
            comp = comp + factor * _apply_carrier(s, row, col)
            if return_maps:
                a = (factor * s) ** 2
                amp_sq = a if amp_sq is None else amp_sq + a
            if gate_tau > 0:
                g = _blur2d(d, _GATE_SIGMA)
                fs = factor * s
                c = (factor * g) * fs
                p = fs * fs
                corr = c if corr is None else corr + c
                s_pow = p if s_pow is None else s_pow + p
        if gate_tau > 0:
            # two independent gates:
            # 1. AGREEMENT: does the local field match the global estimate's
            #    phase? (eps is only a numerical floor, NOT the threshold —
            #    folding tau in here was measured to depress the ratio for
            #    weak-but-real components and under-kill them)
            eps = 0.25 * gate_tau
            r = corr / (s_pow + eps * eps)
            agree = _blur2d(r, _GATE_SMOOTH_SIGMA).clamp(0.0, 1.0)
            # 2. AMPLITUDE: is the estimated grid big enough to matter?
            #    quartic on the global amplitude — sub-quantum grid is not
            #    worth a correction that 8-bit rounding turns into dither.
            tau4 = (gate_tau * gate_tau) ** 2
            amp_gate = s_pow * s_pow / (s_pow * s_pow + tau4)
            comp = comp * (agree * agree * amp_gate)
        est = est + comp
        if return_maps:
            maps[name] = torch.sqrt(amp_sq)
    return est, maps


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def degrid(
    img_bhwc: torch.Tensor,
    mode: str = "coherent",
    strength: float = 1.0,
    sigma: float = 48.0,
    harmonics: str = "2px",
    min_amplitude: float = 0.3,
) -> torch.Tensor:
    """Remove the VAE grid artifact from a batch of images.

    Args:
        img_bhwc: (B, H, W, C) float tensor (ComfyUI IMAGE).
        mode: "coherent" (default; detail-preserving) or "notch"
            (community 7-tap reference — blurs fine detail).
        strength: 0..1+ multiplier on the subtracted grid estimate.
        sigma: coherent mode only — Gaussian radius (px) for the
            demodulated-field smoothing. Empirically larger is better on
            BOTH axes for the Qwen/Wan grid (it is near-globally uniform):
            sigma=48 kills the grid below background AND preserves detail
            at 50+ dB, while sigma<=16 both leaves residual grid and leaks
            more detail into the estimate. Don't lower it expecting a
            better grid kill.
        harmonics: "2px" (default) or "2px+4px". WARNING: only use
            "2px+4px" on straight-from-decode images. Lossy-compressed
            sources (webp/JPEG) carry globally phase-coherent 4px
            harmonics of their 8x8 DCT block grid; the 4px path cannot
            distinguish those from VAE artifacts (they are genuinely
            coherent) and will imprint visible texture on flat regions
            when the subtraction lands on lossy-smoothed content
            (measured 2026-09-22: 34% of flat-background pixels flipped
            a quantization step with 4px enabled on a webp source, 0%
            with 2px only).
        min_amplitude: coherent mode only — amplitude gate threshold in
            8-bit levels (0 disables). Regions whose LOCAL grid amplitude
            is below this are left untouched, so sub-quantum corrections
            on flat areas cannot round into visible dither at save time.
            Default 0.3: grid below a third of a quantization step is
            invisible everywhere and not worth touching.

    Returns:
        (B, H, W, C) tensor, same dtype/device. Not clamped — VAE decode
        output may legitimately exceed [0, 1]; clamp at save time.
    """
    if img_bhwc.dim() != 4:
        raise ValueError(f"expected (B,H,W,C) tensor, got shape {tuple(img_bhwc.shape)}")
    if mode not in ("coherent", "notch"):
        raise ValueError(f"unknown mode {mode!r}")
    x = _to_bchw(img_bhwc.float())
    tau = float(min_amplitude) / 255.0
    if mode == "notch":
        est = _notch_estimate(x)
        if harmonics == "2px+4px":
            # the 7-tap notch has no 4px suppression; add the coherent
            # 4px-only estimate on top
            est4, _ = _coherent_estimate(x, sigma, "2px+4px", gate_tau=tau)
            est2, _ = _coherent_estimate(x, sigma, "2px", gate_tau=tau)
            est = est + (est4 - est2)
    else:
        est, _ = _coherent_estimate(x, sigma, harmonics, gate_tau=tau)
    out = x - float(strength) * est
    return _to_bhwc(out).to(img_bhwc.dtype)


def measure_grid(
    img_bhwc: torch.Tensor,
    sigma: float = 48.0,
    return_maps: bool = False,
):
    """Measure grid-artifact strength.

    Returns a dict with, per image in the batch:
      - fft peak/background ratios at the six grid bins
        (nyq_x, nyq_y, nyq_xy, p4_x, p4_y, p4_xy); >~3 indicates a
        coherent grid, clean images sit near 1
      - parity amplitudes (mean 2x2-phase-class offsets of a high-passed
        luma, in 8-bit levels): parity_x, parity_y, parity_diag
      - optionally 'maps': smoothed demodulated amplitude fields
        (B, C, H, W) per component, usable as diagnostic heatmaps
    """
    if img_bhwc.dim() != 4:
        raise ValueError(f"expected (B,H,W,C) tensor, got shape {tuple(img_bhwc.shape)}")
    x = _to_bchw(img_bhwc.float())
    b, c, h, w = x.shape
    # crop to even dims so the Nyquist bin is exact
    h2, w2 = h - (h % 2), w - (w % 2)
    x = x[:, :, :h2, :w2]
    h, w = h2, w2

    if c >= 3:
        luma = (0.2126 * x[:, 0] + 0.7152 * x[:, 1] + 0.0722 * x[:, 2])
    else:
        luma = x[:, 0]

    win = torch.outer(
        torch.hann_window(h, periodic=False, dtype=x.dtype, device=x.device),
        torch.hann_window(w, periodic=False, dtype=x.dtype, device=x.device),
    )
    z = (luma - luma.mean(dim=(-2, -1), keepdim=True)) * win
    mag = torch.abs(torch.fft.fft2(z))  # (B, H, W)

    ny, nx = h // 2, w // 2
    bins = {
        "nyq_x": (0, nx),
        "nyq_y": (ny, 0),
        "nyq_xy": (ny, nx),
        "p4_x": (0, nx // 2),
        "p4_y": (ny // 2, 0),
        "p4_xy": (ny // 2, nx // 2),
    }

    def peak_vs_bg(m, r, c_, rad=6):
        peak = m[r, c_]
        r0, r1 = max(0, r - rad), min(h, r + rad + 1)
        c0, c1 = max(0, c_ - rad), min(w, c_ + rad + 1)
        patch = m[r0:r1, c0:c1].clone()
        pr, pc = r - r0, c_ - c0
        patch[max(0, pr - 1) : pr + 2, max(0, pc - 1) : pc + 2] = float("nan")
        bg = torch.nanmedian(patch.flatten())
        return (peak / torch.clamp(bg, min=1e-12)).item()

    ratios = {
        name: [peak_vs_bg(mag[i], r, c_) for i in range(b)]
        for name, (r, c_) in bins.items()
    }

    # spatial parity amplitudes on high-passed luma
    hp = luma - _blur2d(luma.unsqueeze(1), 1.5).squeeze(1)
    ee = hp[:, 0::2, 0::2].mean(dim=(-2, -1))
    eo = hp[:, 0::2, 1::2].mean(dim=(-2, -1))
    oe = hp[:, 1::2, 0::2].mean(dim=(-2, -1))
    oo = hp[:, 1::2, 1::2].mean(dim=(-2, -1))
    parity = {
        "parity_x": (255.0 * ((ee + oe) - (eo + oo)).abs() / 2).tolist(),
        "parity_y": (255.0 * ((ee + eo) - (oe + oo)).abs() / 2).tolist(),
        "parity_diag": (255.0 * ((ee + oo) - (eo + oe)).abs() / 2).tolist(),
    }

    out = {"ratios": ratios, **parity, "height": h, "width": w}
    if return_maps:
        _, maps = _coherent_estimate(x, sigma, "2px+4px", return_maps=True)
        out["maps"] = maps
    return out


def inject_grid(
    img_bhwc: torch.Tensor,
    amp_x: float = 0.01,
    amp_y: float = 0.02,
    amp_xy: float = 0.005,
    amp_field: torch.Tensor = None,
    phase_flip: bool = False,
) -> torch.Tensor:
    """Synthesize a VAE-like 2px grid onto a clean image (test fixture).

    amp_field: optional (H, W) multiplier for spatially-varying amplitude.
    phase_flip: shift the grid phase by one pixel (parity robustness tests).
    """
    b, h, w, c = img_bhwc.shape
    dtype, device = img_bhwc.dtype, img_bhwc.device
    sx = _alt(w, dtype, device)
    sy = _alt(h, dtype, device)
    if phase_flip:
        sx = -sx
        sy = -sy
    gx = sx.view(1, 1, w, 1).expand(1, h, w, 1)
    gy = sy.view(1, h, 1, 1).expand(1, h, w, 1)
    gxy = (sy.view(h, 1) * sx.view(1, w)).view(1, h, w, 1)
    grid = amp_x * gx + amp_y * gy + amp_xy * gxy
    if amp_field is not None:
        grid = grid * amp_field.view(1, h, w, 1).to(dtype)
    return img_bhwc + grid.to(device)
