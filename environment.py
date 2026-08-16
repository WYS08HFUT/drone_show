"""RotorPy vector environment for an Agilicious-style hover task."""

from __future__ import annotations

import numpy as np
import torch
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from rotorpy.wind.default_winds import BatchedConstantWind
from rotorpy.world import World

from .agilicious_params import GRAVITY, KINGFISHER_PARAMS, MASS_KG, THRUST_COEFF


def hover_reward(observation: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Dense reward for returning to a level hover at the world origin."""
    position_error = np.linalg.norm(observation[..., 0:3], axis=-1)
    velocity_error = np.linalg.norm(observation[..., 3:6], axis=-1)
    qx, qy = observation[..., 6], observation[..., 7]
    body_z_alignment = np.clip(1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0)
    rate_error = np.linalg.norm(observation[..., 10:13], axis=-1)
    effort = np.linalg.norm(action, axis=-1)
    return (
        3.0
        - 2.5 * position_error
        - 0.35 * velocity_error
        - 0.75 * (1.0 - body_z_alignment)
        - 0.08 * rate_error
        - 0.01 * effort
    )


class AgiliciousHoverEnv(QuadrotorEnv):
    """CTBR task with action zero defined as physical hover thrust.

    RotorPy normally maps collective action -1..1 linearly to zero..maximum
    thrust, placing hover far from action zero. This residual mapping keeps the
    same physical limits while centering the policy at hover, which makes PPO
    training substantially better conditioned.
    """

    def rescale_action(self, action: np.ndarray) -> dict[str, np.ndarray]:
        if self.control_mode != "cmd_ctbr":
            return super().rescale_action(action)

        action = np.clip(np.asarray(action), -1.0, 1.0)
        collective = action[..., 0]
        hover = np.asarray(self.quad_params.mass.cpu()) * GRAVITY
        min_total = self.quad_params.num_rotors * self.min_thrust
        max_total = self.quad_params.num_rotors * self.max_thrust
        cmd_thrust = np.where(
            collective >= 0.0,
            hover + collective * (max_total - hover),
            hover + collective * (hover - min_total),
        )
        cmd_w = np.column_stack(
            (
                action[..., 1] * self.max_roll_br,
                action[..., 2] * self.max_pitch_br,
                action[..., 3] * self.max_yaw_br,
            )
        )
        return {"cmd_thrust": cmd_thrust.reshape(-1, 1), "cmd_w": cmd_w}

    def step(self, action: np.ndarray):
        observation, reward, dones, infos = super().step(action)
        # Prevent a positive-reward policy from learning to leave the world and
        # exploit RotorPy's automatic reset. Time-limit endings are legitimate.
        for index, done in enumerate(dones):
            if done and not infos[index].get("TimeLimit.truncated", False):
                reward[index] -= 30.0
        return observation, reward, dones, infos


def initial_states(
    num_envs: int,
    device: torch.device,
    position: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    hover_speed = np.sqrt(MASS_KG * GRAVITY / (4.0 * THRUST_COEFF))
    positions = torch.zeros(num_envs, 3, device=device, dtype=torch.float64)
    if position is not None:
        positions[:] = torch.as_tensor(position, dtype=torch.float64, device=device)
    return {
        "x": positions,
        "v": torch.zeros(num_envs, 3, device=device, dtype=torch.float64),
        "q": torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float64).repeat(num_envs, 1),
        "w": torch.zeros(num_envs, 3, device=device, dtype=torch.float64),
        "wind": torch.zeros(num_envs, 3, device=device, dtype=torch.float64),
        "rotor_speeds": torch.full((num_envs, 4), hover_speed, device=device, dtype=torch.float64),
    }


def make_env(
    num_envs: int = 32,
    max_time: float = 5.0,
    sim_rate: int = 100,
    random_reset: bool = True,
    position: np.ndarray | None = None,
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0),
    seed: int = 7,
) -> AgiliciousHoverEnv:
    device = torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    reset_options = {
        "initial_states": "random" if random_reset else "deterministic",
        "pos_bound": 0.6,
        "vel_bound": 0.15,
        "params": "fixed",
    }
    world = World.empty((-8.0, 8.0, -8.0, 8.0, -8.0, 8.0))
    wind_profile = BatchedConstantWind(num_envs, *wind)
    env = AgiliciousHoverEnv(
        num_envs=num_envs,
        initial_states=initial_states(num_envs, device, position),
        control_mode="cmd_ctbr",
        reward_fn=hover_reward,
        quad_params=KINGFISHER_PARAMS,
        device=device,
        max_time=max_time,
        wind_profile=wind_profile,
        world=world,
        sim_rate=sim_rate,
        aero=True,
        render_mode="None",
        reset_options=reset_options,
    )
    # Hover training does not need RotorPy's aggressive ±7 rad/s rate range.
    env.max_roll_br = 3.0
    env.max_pitch_br = 3.0
    env.max_yaw_br = 1.5
    return env
