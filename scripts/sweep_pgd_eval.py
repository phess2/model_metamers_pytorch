#!/usr/bin/env python
"""Run L2 PGD evaluation across a grid of epsilon values."""

from __future__ import annotations

import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = ArgumentParser(description="Sweep L2 PGD evaluation over epsilon values")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument(
        "--epsilons",
        type=str,
        nargs="+",
        required=True,
        help="Epsilon values, e.g. 0 0.5 1 2 3",
    )
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--step_size", type=float, default=0.1)
    parser.add_argument("--indices", type=str, default="0-399")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exp_dir", type=str, default="experiments")
    args = parser.parse_args()

    attack_script = REPO_ROOT / "scripts" / "attack_models.py"
    for epsilon in args.epsilons:
        cmd = [
            sys.executable,
            str(attack_script),
            "--config",
            args.config,
            "--ckpt_path",
            args.ckpt_path,
            "--epsilon",
            str(epsilon),
            "--num_steps",
            str(args.num_steps),
            "--step_size",
            str(args.step_size),
            "--indices",
            args.indices,
            "--device",
            args.device,
            "--exp_dir",
            args.exp_dir,
        ]
        print(f"\n=== PGD eval: epsilon={epsilon} ===")
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
