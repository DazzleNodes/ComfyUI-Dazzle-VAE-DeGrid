# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-22

### Added

- Core de-grid algorithm (`degrid/degrid_core.py`): removes the 2px grid ("hatching") that the Qwen-Image / Wan 2.1 VAE decoder stamps on decoded images. Default `coherent` mode estimates the globally phase-coherent grid (demodulate, smooth at sigma=48, re-modulate, subtract) so fine detail is preserved; `notch` mode reproduces the community 7-tap filter as a reference baseline.
- Agreement + amplitude gating: subtraction is scaled by how well the local field matches the global estimate's phase, and by a minimum-amplitude threshold (default 0.3/255), so flat regions stay bit-identical after 8-bit saves instead of picking up rounding dither.
- `Dazzle VAE DeGrid (Image)` node: mode, strength, sigma, harmonics, and min_amplitude controls.
- `Dazzle Grid Diagnostic` node: passthrough that measures grid strength (FFT peak ratios, parity amplitudes), prints/returns metrics JSON, and renders a grid-amplitude heatmap.
- Example A/B workflow (`examples/degrid-ab-test.json`): raw / coherent / notch / GLSL-reference branches with diagnostics and rgthree slide comparers.
- Test suite (12 passing + machine-specific real-image fixtures): synthetic grid recovery incl. spatially-varying amplitude and parity flips, detail preservation vs the notch baseline, flat-background bit-stability (dither regression), and region-based real-render acceptance.

### Notes

- `harmonics` defaults to `2px`. The `2px+4px` option should only be used on straight-from-decode images: lossy webp/JPEG sources carry globally phase-coherent 4px DCT block harmonics that the filter would "remove", imprinting visible texture on flat regions.

[Unreleased]: https://github.com/DazzleNodes/ComfyUI-Dazzle-VAE-DeGrid/compare/v0.1.0a1...HEAD
[0.1.0]: https://github.com/DazzleNodes/ComfyUI-Dazzle-VAE-DeGrid/releases/tag/v0.1.0
