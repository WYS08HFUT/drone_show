"""Roll out a trained PPO policy and show/save a true-geometry 3D animation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation
from stable_baselines3 import PPO

from .agilicious_params import DX, DY, PROPELLER_RADIUS_M, parameter_summary
from .environment import make_env
from .train import DEFAULT_MODEL


def rollout(
    model_path: Path,
    duration: float = 6.0,
    sim_rate: int = 100,
    wind_x: float = 0.0,
) -> dict[str, np.ndarray]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Policy not found: {model_path}. Run train.py first.")
    start = np.array([0.55, -0.40, 0.45])
    env = make_env(
        num_envs=1,
        max_time=duration + 1.0,
        sim_rate=sim_rate,
        random_reset=False,
        position=start,
        wind=(wind_x, 0.0, 0.0),
    )
    model = PPO.load(str(model_path), env=env, device="cpu")
    obs = env.reset()
    positions, quaternions, actions, rewards = [], [], [], []
    for _ in range(int(duration * sim_rate)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        positions.append(obs[0, 0:3].copy())
        quaternions.append(obs[0, 6:10].copy())
        actions.append(action[0].copy())
        rewards.append(float(reward[0]))
        if bool(done[0]):
            break
    env.close()
    return {
        "position": np.asarray(positions),
        "quaternion": np.asarray(quaternions),
        "action": np.asarray(actions),
        "reward": np.asarray(rewards),
        "dt": np.array(1.0 / sim_rate),
        "wind_x": np.array(wind_x),
    }


def animate(
    data: dict[str, np.ndarray],
    save_path: Path | None = None,
    show: bool = True,
    fps: int = 30,
) -> None:
    positions = data["position"]
    quaternions = data["quaternion"]
    dt = float(data["dt"])
    stride = max(1, round(1.0 / (fps * dt)))
    frame_indices = np.arange(0, len(positions), stride)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), zlim=(-0.5, 1.6))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_box_aspect((2.5, 2.5, 2.1))
    ax.scatter([0], [0], [0], marker="*", s=160, c="limegreen", label="hover target")
    wind_x = float(data["wind_x"])
    if abs(wind_x) > 1e-9:
        ax.quiver(
            -0.9,
            0.85,
            1.35,
            0.45 * np.sign(wind_x),
            0.0,
            0.0,
            color="crimson",
            linewidth=2.0,
            label=f"wind {wind_x:.1f} m/s",
        )
    ax.legend(loc="upper right")

    arm_lines = [ax.plot([], [], [], color="#ff7a18", lw=3)[0] for _ in range(4)]
    rotor_lines = [ax.plot([], [], [], color="#2d7ff9", lw=2)[0] for _ in range(4)]
    trail, = ax.plot([], [], [], color="#222222", lw=1.5, alpha=0.75)
    time_text = ax.text2D(0.03, 0.95, "", transform=ax.transAxes)
    body_rotors = np.array(
        [[-DX, DY, 0.0], [DX, DY, 0.0], [-DX, -DY, 0.0], [DX, -DY, 0.0]]
    )
    circle_angle = np.linspace(0.0, 2.0 * np.pi, 32)

    def update(frame_number: int):
        idx = int(frame_indices[frame_number])
        p = positions[idx]
        rotation = Rotation.from_quat(quaternions[idx]).as_matrix()
        rotors_world = (rotation @ body_rotors.T).T + p
        for motor_idx, rotor_world in enumerate(rotors_world):
            arm = np.vstack((p, rotor_world))
            arm_lines[motor_idx].set_data_3d(arm[:, 0], arm[:, 1], arm[:, 2])
            circle_body = np.column_stack(
                (
                    body_rotors[motor_idx, 0] + PROPELLER_RADIUS_M * np.cos(circle_angle),
                    body_rotors[motor_idx, 1] + PROPELLER_RADIUS_M * np.sin(circle_angle),
                    np.zeros_like(circle_angle),
                )
            )
            circle_world = (rotation @ circle_body.T).T + p
            rotor_lines[motor_idx].set_data_3d(
                circle_world[:, 0], circle_world[:, 1], circle_world[:, 2]
            )
        trail.set_data_3d(positions[: idx + 1, 0], positions[: idx + 1, 1], positions[: idx + 1, 2])
        error = np.linalg.norm(p)
        time_text.set_text(
            f"t={idx * dt:4.2f} s   |position error|={error:4.2f} m   "
            f"wind x={float(data['wind_x']):.1f} m/s"
        )
        return (*arm_lines, *rotor_lines, trail, time_text)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )
    fig.suptitle("RotorPy — Agilicious Kingfisher PPO hover")
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        animation.save(str(save_path), writer="ffmpeg", fps=fps, dpi=130)
        print(f"Saved visualization to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--wind-x", type=float, default=0.0)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    print(f"Kingfisher mapping: {parameter_summary()}")
    data = rollout(args.model, duration=args.duration, wind_x=args.wind_x)
    print(
        f"Replay: {len(data['position'])} steps, final position error="
        f"{np.linalg.norm(data['position'][-1]):.3f} m, mean reward={np.mean(data['reward']):.3f}"
    )
    animate(data, save_path=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
