"""One-command train-if-needed and visual replay entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .play import animate, rollout
from .train import DEFAULT_MODEL, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--wind-x", type=float, default=0.0)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.retrain or not args.model.exists():
        train(args.timesteps, args.num_envs, args.model)
    data = rollout(args.model, wind_x=args.wind_x)
    animate(data, save_path=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
