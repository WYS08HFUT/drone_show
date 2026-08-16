"""Command-line entry point for centralized offline aerial-show planning."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import ShowConfig
from .export import export_plan, load_plan
from .planner import plan_show


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="2026-龙-马-精-神", help="'-' separates display formations")
    parser.add_argument("--drones", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("outputs/2026_dragon_horse_spirit"))
    parser.add_argument("--font", type=Path)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--reuse-plan", action="store_true", help="render/export an existing output without replanning")
    parser.add_argument("--show", action="store_true", help="open an interactive Matplotlib window")
    parser.add_argument("--playback-speed", type=float, default=8.0)
    parser.add_argument(
        "--view",
        choices=("3d", "front", "top"),
        default="3d",
        help="camera preset: perspective 3D, formation front, or top-down",
    )
    parser.add_argument(
        "--drone-model",
        choices=("quadcopter", "point"),
        default="quadcopter",
        help="render lightweight quadcopters or faster point markers",
    )
    args = parser.parse_args()

    if args.reuse_plan:
        print(f"Loading existing plan from {args.output}...", flush=True)
        plan = load_plan(args.output)
        paths = {
            "manifest": args.output / "manifest.json",
            "commands": args.output / "drone_commands",
        }
    else:
        config = ShowConfig(num_drones=args.drones)
        print(f"Planning {args.text!r} for {args.drones} drones...", flush=True)
        plan = plan_show(
            args.text,
            config=config,
            font_path=args.font,
            progress=lambda message: print(message, flush=True),
        )
        paths = export_plan(plan, args.output)
    summary = {key: value for key, value in plan.metrics.items() if key != "transitions"}
    summary["transitions"] = [
        {key: value for key, value in transition.items() if key != "target_permutation"}
        for transition in plan.metrics["transitions"]
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Manifest: {paths['manifest']}")
    print(f"Per-drone commands: {paths['commands']}")
    if not args.no_video or args.show:
        if not args.show:
            os.environ.setdefault("MPLBACKEND", "Agg")
        from .visualize import animate_plan

        video = args.output / "show.mp4" if not args.no_video else None
        animate_plan(
            plan,
            save_path=video,
            show=args.show,
            playback_speed=args.playback_speed,
            view=args.view,
            drone_model=args.drone_model,
        )
