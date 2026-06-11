"""Top-level ``src`` package with lazy subpackage loading."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "analysis",
    "data",
    "models",
    "optimizers",
    "training",
    "utils",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
