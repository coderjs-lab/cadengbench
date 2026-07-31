"""One-process boundary for a frozen-L3 CalculiX solve."""

from __future__ import annotations

import argparse
from pathlib import Path

from cadengbench_p_eval.physics.calculix import load_physics_card, run_probe_card


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--step", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ccx", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--render", action="store_true")
    arguments = parser.parse_args()
    card = load_physics_card(arguments.card)
    run_probe_card(
        card,
        step_path=arguments.step,
        output_dir=arguments.output,
        ccx_path=arguments.ccx,
        timeout_seconds=arguments.timeout_seconds,
        render=arguments.render,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
