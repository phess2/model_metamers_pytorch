"""Build training configs for bound-search trials."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config import load_config


def _default_projection(bound_method: str, projection: str | None) -> str | None:
    if projection is not None:
        return projection
    if bound_method == "modular_linf":
        return "modular_linf_cap"
    return "spectral_normalize"


def build_trial_config(
    *,
    template_config_path: str | Path,
    output_path: str | Path,
    trial_id: str,
    hparams: dict[str, Any],
) -> Path:
    """Create a training config JSON for one search trial."""
    template = load_config(template_config_path)
    config = copy.deepcopy(template)

    bound_method = hparams["bound_method"]
    projection = _default_projection(bound_method, hparams.get("projection"))
    w_max = float(hparams["w_max"])

    config["model_name"] = hparams.get("model_name", config.get("model_name", "lipsresnet"))
    config.setdefault("hparams", {})
    config["hparams"]["bound_method"] = bound_method
    config["hparams"]["projection"] = projection
    config["hparams"]["w_max"] = w_max

    if "optimizer_lr" in hparams:
        optim = config.setdefault("optim_settings", {})
        if optim.get("mult_optimizers", False):
            for key in ("linear_optimizer", "conv_optimizer"):
                if key in optim:
                    optim[key]["kwargs"]["lr"] = float(hparams["optimizer_lr"])
        elif "optimizer" in optim:
            optim["optimizer"]["kwargs"]["lr"] = float(hparams["optimizer_lr"])

    config["bound_search"] = {
        "trial_id": trial_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "template_config": str(template_config_path),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4)
        handle.write("\n")
    return output_path
