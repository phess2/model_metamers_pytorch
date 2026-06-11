"""Propose the next bound-search trial."""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.robustness_search.config_builder import build_trial_config
from src.robustness_search.state import SearchState, TrialRecord, load_state, save_state, trial_signature
from src.utils.config import load_config


def _iter_candidates(search_space: dict[str, Any]) -> Iterator[dict[str, Any]]:
    model_names = search_space["model_name"]
    bound_methods = search_space["bound_method"]
    w_max_values = search_space["w_max"]
    optimizer_lrs = search_space.get("optimizer_lr", [None])
    projection_map = search_space.get("projection", {})

    for model_name, bound_method, w_max, optimizer_lr in itertools.product(
        model_names, bound_methods, w_max_values, optimizer_lrs
    ):
        projections = projection_map.get(bound_method, [None])
        for projection in projections:
            hparams = {
                "model_name": model_name,
                "bound_method": bound_method,
                "projection": projection,
                "w_max": w_max,
            }
            if optimizer_lr is not None:
                hparams["optimizer_lr"] = optimizer_lr
            yield hparams


def propose_next_trial(
    search_space_path: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """
    Pick the next untried bound configuration and write a morning-ready training config.

    Returns a dict with trial metadata and shell commands.
    """
    search_space_path = Path(search_space_path)
    spec = load_config(search_space_path)

    if spec.get("goal_met") and not force:
        state_path = Path(spec["output"]["state_dir"]) / "state.json"
        state = load_state(state_path)
        if state.goal_met:
            return {
                "status": "goal_already_met",
                "best_trial_id": state.best_trial_id,
                "message": "Search goal already met. Pass force=True to continue exploring.",
            }

    state_dir = Path(spec["output"]["state_dir"])
    state_path = state_dir / "state.json"
    state = load_state(state_path)
    tried = state.tried_signatures()

    candidate = None
    for hparams in _iter_candidates(spec["search_space"]):
        if trial_signature(hparams) not in tried:
            candidate = hparams
            break

    if candidate is None:
        return {
            "status": "search_exhausted",
            "message": "All combinations in search_space have been proposed.",
            "trials_completed": len(state.trials),
        }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    trial_id = (
        f"{timestamp}_{candidate['bound_method']}_w{candidate['w_max']}_"
        f"p{candidate.get('projection', 'auto')}_lr{candidate.get('optimizer_lr', 'default')}"
    )
    trial_id = trial_id.replace(".", "p").replace(" ", "_")

    generated_dir = Path(spec["output"]["generated_config_dir"])
    config_path = generated_dir / f"{trial_id}.json"
    build_trial_config(
        template_config_path=spec["template_config"],
        output_path=config_path,
        trial_id=trial_id,
        hparams=candidate,
    )

    trial = TrialRecord(
        trial_id=trial_id,
        config_path=str(config_path),
        hparams=candidate,
        proposed_at=datetime.now(timezone.utc).isoformat(),
        status="ready_for_training",
    )
    state.trials.append(trial)
    save_state(state_path, state)

    train_cmd = (
        "source /workspace/miniconda3/etc/profile.d/conda.sh && "
        "conda activate metamMuon && "
        "cd /workspace && "
        "export PYTHONPATH=/workspace && "
        f"python scripts/train_vision.py --config {config_path} --gpus 1"
    )
    eval_cmd = (
        "source /workspace/miniconda3/etc/profile.d/conda.sh && "
        "conda activate metamMuon && "
        "cd /workspace && "
        "export PYTHONPATH=/workspace && "
        f"python scripts/nightly_bound_search.py evaluate --trial-id {trial_id}"
    )

    ready_payload = {
        "status": "ready_for_morning",
        "trial_id": trial_id,
        "config_path": str(config_path),
        "hparams": candidate,
        "train_command": train_cmd,
        "evaluate_after_training_command": eval_cmd,
        "proposed_at": trial.proposed_at,
    }

    ready_path = Path(spec["output"]["ready_for_morning_file"])
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ready_path, "w", encoding="utf-8") as handle:
        json.dump(ready_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return ready_payload
