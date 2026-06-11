"""Persistence for nightly bound-search trials."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TrialRecord:
    trial_id: str
    config_path: str
    hparams: dict[str, Any]
    proposed_at: str
    status: str = "proposed"
    trained_at: str | None = None
    evaluated_at: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    passed_goal: bool = False


@dataclass
class SearchState:
    version: int = 1
    goal_met: bool = False
    best_trial_id: str | None = None
    trials: list[TrialRecord] = field(default_factory=list)

    def tried_signatures(self) -> set[str]:
        return {trial_signature(t.hparams) for t in self.trials}

    def get_trial(self, trial_id: str) -> TrialRecord | None:
        for trial in self.trials:
            if trial.trial_id == trial_id:
                return trial
        return None


def trial_signature(hparams: dict[str, Any]) -> str:
    keys = ("model_name", "bound_method", "projection", "w_max", "optimizer_lr")
    parts = []
    for key in keys:
        value = hparams.get(key)
        parts.append(f"{key}={value}")
    return "|".join(parts)


def load_state(path: Path) -> SearchState:
    if not path.exists():
        return SearchState()
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    trials = [TrialRecord(**item) for item in raw.get("trials", [])]
    return SearchState(
        version=raw.get("version", 1),
        goal_met=raw.get("goal_met", False),
        best_trial_id=raw.get("best_trial_id"),
        trials=trials,
    )


def save_state(path: Path, state: SearchState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": state.version,
        "goal_met": state.goal_met,
        "best_trial_id": state.best_trial_id,
        "trials": [asdict(trial) for trial in state.trials],
        "updated_at": _utc_now(),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
