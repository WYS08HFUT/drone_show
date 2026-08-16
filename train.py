"""Train a PPO collective-thrust/body-rate hover policy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from .agilicious_params import parameter_summary
from .environment import make_env


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_MODEL = ARTIFACT_DIR / "ppo_kingfisher_hover.zip"


def train(
    timesteps: int = 1_500_000,
    num_envs: int = 32,
    model_path: Path = DEFAULT_MODEL,
    seed: int = 7,
    resume: bool = False,
) -> Path:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    env = make_env(num_envs=num_envs, seed=seed)
    checkpoint = CheckpointCallback(
        save_freq=max(50_000 // num_envs, 1),
        save_path=str(model_path.parent / "checkpoints"),
        name_prefix="ppo_kingfisher",
    )
    if resume and model_path.exists():
        model = PPO.load(str(model_path), env=env, device="cpu")
        model.verbose = 1
        reset_num_timesteps = False
        print(f"Continuing from {model_path}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=min(512, num_envs * 256),
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.002,
            policy_kwargs={"net_arch": [128, 128], "log_std_init": -1.5},
            verbose=1,
            seed=seed,
            device="cpu",
        )
        reset_num_timesteps = True
    print(f"Kingfisher mapping: {parameter_summary()}")
    model.learn(
        total_timesteps=timesteps,
        callback=checkpoint,
        progress_bar=False,
        reset_num_timesteps=reset_num_timesteps,
    )
    model.save(str(model_path.with_suffix("")))
    env.close()
    print(f"Saved PPO policy to {model_path}")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args.timesteps, args.num_envs, args.model, args.seed, args.resume)


if __name__ == "__main__":
    main()
