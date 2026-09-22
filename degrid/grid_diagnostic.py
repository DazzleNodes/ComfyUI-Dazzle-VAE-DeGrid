"""DazzleGridDiagnostic — measure VAE grid artifact strength.

Passthrough node that measures the 2px/4px coherent grid components of an
image, emits the metrics as JSON (and to the console), and renders the
demodulated amplitude fields as a heatmap image for visual inspection.
"""

import json

import torch

from .degrid_core import measure_grid


class DazzleGridDiagnostic:
    CATEGORY = "DazzleNodes/DeGrid"
    FUNCTION = "measure"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "heatmap", "metrics_json")
    OUTPUT_TOOLTIPS = (
        "Input image, unchanged (passthrough).",
        "Grid amplitude heatmap (sum of coherent components, normalized).",
        "Peak/background ratios and parity amplitudes as JSON.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "label": ("STRING", {
                    "default": "",
                    "tooltip": "Optional tag echoed into the console/JSON — "
                               "useful when placing several diagnostics along "
                               "a pipeline (e.g. 'stage2-pre-encode').",
                }),
            },
        }

    def measure(self, image, label=""):
        dev = image.device
        run_dev = dev
        if dev.type == "cpu" and torch.cuda.is_available():
            run_dev = torch.device("cuda")
        x = image.to(run_dev)

        m = measure_grid(x, return_maps=True)
        maps = m.pop("maps")

        # heatmap: total coherent amplitude across components, channel-mean
        total = None
        for comp in maps.values():
            a = comp.mean(dim=1)  # (B, H, W)
            total = a if total is None else total + a
        peak = torch.quantile(total.flatten(1), 0.999, dim=1).view(-1, 1, 1)
        heat = (total / torch.clamp(peak, min=1e-8)).clamp(0.0, 1.0)
        heatmap = heat.unsqueeze(-1).expand(-1, -1, -1, 3).to(dev)

        metrics = {
            "label": label,
            "ratios": {k: [round(v, 2) for v in vals]
                       for k, vals in m["ratios"].items()},
            "parity_x": [round(v, 4) for v in m["parity_x"]],
            "parity_y": [round(v, 4) for v in m["parity_y"]],
            "parity_diag": [round(v, 4) for v in m["parity_diag"]],
            "note": "ratios > ~3 indicate a coherent grid; clean images sit near 1",
        }
        text = json.dumps(metrics, indent=2)
        tag = f" [{label}]" if label else ""
        print(f"[DazzleGridDiagnostic]{tag} " + json.dumps(metrics["ratios"]))
        return (image, heatmap, text)
