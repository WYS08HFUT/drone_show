"""End-to-end centralized offline show planner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .assignment import assign_targets
from .config import ShowConfig
from .formations import glyph_formation, launch_grid, parse_sequence, resolve_font
from .mapf import prioritized_mapf, validate_discrete_paths
from .trajectory import minimum_separation, smooth_and_optimize


PALETTE = np.asarray(
    [
        (0.10, 0.65, 1.00),
        (1.00, 0.18, 0.08),
        (1.00, 0.72, 0.05),
        (0.25, 1.00, 0.35),
        (0.85, 0.25, 1.00),
    ],
    dtype=float,
)


@dataclass
class ShowPlan:
    sequence: str
    tokens: list[str]
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    yaw: np.ndarray
    led_rgb: np.ndarray
    phase_labels: np.ndarray
    metrics: dict
    config: ShowConfig
    font_path: str


def _hold(position: np.ndarray, seconds: float, sample_rate: int) -> np.ndarray:
    count = max(2, int(round(seconds * sample_rate)))
    return np.repeat(position[None, :, :], count, axis=0)


def _append_phase(
    positions: list[np.ndarray],
    labels: list[np.ndarray],
    colors: list[np.ndarray],
    phase_positions: np.ndarray,
    label: str,
    color: np.ndarray,
) -> None:
    if positions:
        phase_positions = phase_positions[1:]
    if len(phase_positions) == 0:
        return
    positions.append(phase_positions)
    labels.append(np.full(len(phase_positions), label, dtype="U64"))
    colors.append(np.broadcast_to(color, (len(phase_positions), phase_positions.shape[1], 3)).copy())


def plan_show(
    sequence: str = "2026-龙-马-精-神",
    config: ShowConfig | None = None,
    font_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> ShowPlan:
    config = config or ShowConfig()
    config.validate()
    tokens = parse_sequence(sequence)
    resolved_font = resolve_font(font_path)
    current = launch_grid(config)
    all_positions: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    transitions: list[dict] = []

    _append_phase(
        all_positions,
        all_labels,
        all_colors,
        _hold(current, 1.0, config.sample_rate),
        "launch",
        np.array((0.15, 0.15, 0.15)),
    )

    for index, token in enumerate(tokens):
        if progress is not None:
            progress(f"[{index + 1}/{len(tokens)}] building target formation {token!r}")
        targets = glyph_formation(token, config, resolved_font)
        assigned, assignment_cost, permutation = assign_targets(current, targets)
        start_cells = np.rint(current / config.grid_resolution).astype(int)
        goal_cells = np.rint(assigned / config.grid_resolution).astype(int)
        mapf = prioritized_mapf(
            start_cells,
            goal_cells,
            margin_cells=config.mapf_margin_cells,
            horizon_padding=config.mapf_horizon_padding,
            max_expansions=config.max_astar_expansions,
            safety_radius_cells=config.safe_distance / config.grid_resolution,
            minimum_z_cell=int(
                np.ceil(config.minimum_flight_altitude / config.grid_resolution - 1e-12)
            ),
        )
        if progress is not None:
            progress(
                f"[{index + 1}/{len(tokens)}] MAPF solved: {len(mapf.paths)} steps, "
                f"{mapf.expanded_nodes} expanded nodes"
            )
        validate_discrete_paths(mapf.paths)
        trajectory = smooth_and_optimize(mapf.paths, config)
        if progress is not None:
            progress(
                f"[{index + 1}/{len(tokens)}] trajectory: {trajectory.duration:.1f} s, "
                f"min separation {trajectory.min_separation:.3f} m"
            )
        previous = "launch" if index == 0 else tokens[index - 1]
        _append_phase(
            all_positions,
            all_labels,
            all_colors,
            trajectory.positions,
            f"reconfigure:{previous}->{token}",
            np.array((0.08, 0.12, 0.20)),
        )
        current = assigned
        token_color = PALETTE[index % len(PALETTE)]
        _append_phase(
            all_positions,
            all_labels,
            all_colors,
            _hold(current, config.display_hold, config.sample_rate),
            f"display:{token}",
            token_color,
        )
        if index + 1 < len(tokens):
            _append_phase(
                all_positions,
                all_labels,
                all_colors,
                _hold(current, config.separator_hold, config.sample_rate),
                f"separator:{token}",
                np.array((0.02, 0.02, 0.02)),
            )
        transitions.append(
            {
                "from": previous,
                "to": token,
                "assignment_cost_m": assignment_cost,
                "mapf_steps": int(len(mapf.paths)),
                "mapf_expanded_nodes": int(mapf.expanded_nodes),
                "duration_s": trajectory.duration,
                "min_separation_m": trajectory.min_separation,
                "max_velocity_mps": trajectory.max_velocity,
                "max_acceleration_mps2": trajectory.max_acceleration,
                "max_jerk_mps3": trajectory.max_jerk,
                "smoothing_sigma": trajectory.smoothing_sigma,
                "target_permutation": permutation.tolist(),
            }
        )

    positions = np.concatenate(all_positions, axis=0)
    phase_labels = np.concatenate(all_labels, axis=0)
    led_rgb = np.concatenate(all_colors, axis=0)
    times = np.arange(len(positions), dtype=float) / config.sample_rate
    velocities = np.gradient(positions, 1.0 / config.sample_rate, axis=0, edge_order=1)
    accelerations = np.gradient(velocities, 1.0 / config.sample_rate, axis=0, edge_order=1)
    yaw = np.zeros((len(positions), config.num_drones), dtype=float)
    min_sep = minimum_separation(positions)
    min_altitude = float(positions[:, :, 2].min())
    metrics = {
        "duration_s": float(times[-1]),
        "frames": int(len(times)),
        "num_drones": config.num_drones,
        "minimum_separation_m": min_sep,
        "minimum_altitude_m": min_altitude,
        "maximum_velocity_mps": float(np.linalg.norm(velocities, axis=-1).max()),
        "maximum_acceleration_mps2": float(np.linalg.norm(accelerations, axis=-1).max()),
        "transitions": transitions,
    }
    if min_sep < config.safe_distance - 1e-6:
        raise RuntimeError(f"final plan separation {min_sep:.3f} m violates {config.safe_distance:.3f} m")
    if min_altitude < config.minimum_flight_altitude - 1e-6:
        raise RuntimeError(
            f"final plan altitude {min_altitude:.3f} m violates "
            f"{config.minimum_flight_altitude:.3f} m"
        )
    return ShowPlan(
        sequence=sequence,
        tokens=tokens,
        times=times,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        yaw=yaw,
        led_rgb=led_rgb,
        phase_labels=phase_labels,
        metrics=metrics,
        config=config,
        font_path=str(resolved_font),
    )
