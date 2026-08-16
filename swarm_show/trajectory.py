"""Safety-aware path smoothing and global dynamic time scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from .config import ShowConfig


@dataclass
class TrajectorySegment:
    positions: np.ndarray
    duration: float
    min_separation: float
    max_velocity: float
    max_acceleration: float
    max_jerk: float
    smoothing_sigma: float


def minimum_separation(positions: np.ndarray) -> float:
    minimum = np.inf
    for frame in positions:
        delta = frame[:, None, :] - frame[None, :, :]
        distance_sq = np.sum(delta * delta, axis=-1)
        np.fill_diagonal(distance_sq, np.inf)
        minimum = min(minimum, float(np.sqrt(distance_sq.min())))
    return minimum


def _linear_dense(paths: np.ndarray, factor: int) -> np.ndarray:
    source_time = np.arange(len(paths), dtype=float)
    dense_time = np.linspace(0.0, len(paths) - 1, (len(paths) - 1) * factor + 1)
    output = np.empty((len(dense_time), paths.shape[1], 3), dtype=float)
    for agent in range(paths.shape[1]):
        for axis in range(3):
            output[:, agent, axis] = np.interp(dense_time, source_time, paths[:, agent, axis])
    return output


def _candidate(paths: np.ndarray, sigma: float, factor: int) -> np.ndarray:
    padded = np.concatenate((np.repeat(paths[:1], 3, axis=0), paths, np.repeat(paths[-1:], 3, axis=0)))
    if sigma > 0:
        padded = gaussian_filter1d(padded, sigma=sigma, axis=0, mode="nearest")
    source_time = np.arange(len(padded), dtype=float)
    dense_time = np.linspace(0.0, len(padded) - 1, (len(padded) - 1) * factor + 1)
    spline = CubicSpline(source_time, padded, axis=0, bc_type="clamped")
    dense = spline(dense_time)
    crop = 3 * factor
    dense = dense[crop : len(dense) - crop]
    dense[0] = paths[0]
    dense[-1] = paths[-1]
    return dense


def _kinematics(positions: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity = np.gradient(positions, dt, axis=0, edge_order=1)
    acceleration = np.gradient(velocity, dt, axis=0, edge_order=1)
    jerk = np.gradient(acceleration, dt, axis=0, edge_order=1)
    return velocity, acceleration, jerk


def smooth_and_optimize(grid_paths: np.ndarray, config: ShowConfig) -> TrajectorySegment:
    paths = np.asarray(grid_paths, dtype=float) * config.grid_resolution
    dense_factor = 4
    sigmas = [config.smoothing_sigma, config.smoothing_sigma / 2.0, 0.2]
    selected = None
    selected_sigma = 0.0
    selected_separation = 0.0
    for sigma in sigmas:
        candidate = _candidate(paths, sigma, dense_factor)
        separation = minimum_separation(candidate)
        altitude_is_safe = bool(
            np.all(candidate[:, :, 2] >= config.minimum_flight_altitude - 1e-9)
        )
        if separation >= config.safe_distance and altitude_is_safe:
            selected, selected_sigma, selected_separation = candidate, sigma, separation
            break
    if selected is None:
        selected = _linear_dense(paths, dense_factor)
        selected_sigma = 0.0
        selected_separation = minimum_separation(selected)
    if selected_separation < config.safe_distance - 1e-6:
        raise RuntimeError(
            f"continuous smoothing violates separation: {selected_separation:.3f} < {config.safe_distance:.3f} m"
        )

    base_dt = config.nominal_grid_dt / dense_factor
    velocity, acceleration, jerk = _kinematics(selected, base_dt)
    vmax = float(np.linalg.norm(velocity, axis=-1).max())
    amax = float(np.linalg.norm(acceleration, axis=-1).max())
    jmax = float(np.linalg.norm(jerk, axis=-1).max())
    scale = max(
        1.0,
        vmax / config.max_velocity,
        np.sqrt(amax / config.max_acceleration),
        np.cbrt(jmax / config.max_jerk),
    )
    duration = (len(selected) - 1) * base_dt * scale
    source_parameter = np.linspace(0.0, 1.0, len(selected))
    spatial_spline = CubicSpline(source_parameter, selected, axis=0, bc_type="clamped")
    use_cubic = True
    for _ in range(5):
        sample_count = max(2, int(np.ceil(duration * config.sample_rate)) + 1)
        parameter = np.linspace(0.0, 1.0, sample_count)
        if use_cubic:
            resampled = spatial_spline(parameter)
        else:
            resampled = np.empty((sample_count, selected.shape[1], 3), dtype=float)
            for agent in range(selected.shape[1]):
                for axis in range(3):
                    resampled[:, agent, axis] = np.interp(
                        parameter, source_parameter, selected[:, agent, axis]
                    )
        if np.any(resampled[:, :, 2] < config.minimum_flight_altitude - 1e-9):
            if use_cubic:
                use_cubic = False
                continue
            raise RuntimeError(
                "resampled trajectory violates minimum_flight_altitude: "
                f"{resampled[:, :, 2].min():.3f} < {config.minimum_flight_altitude:.3f} m"
            )
        separation = minimum_separation(resampled)
        if separation < config.safe_distance - 1e-6 and use_cubic:
            use_cubic = False
            continue
        if separation < config.safe_distance - 1e-6:
            raise RuntimeError(
                f"resampled trajectory violates separation: {separation:.3f} < {config.safe_distance:.3f} m"
            )
        dt = duration / max(len(resampled) - 1, 1)
        velocity, acceleration, jerk = _kinematics(resampled, dt)
        vmax_actual = float(np.linalg.norm(velocity, axis=-1).max())
        amax_actual = float(np.linalg.norm(acceleration, axis=-1).max())
        jmax_actual = float(np.linalg.norm(jerk, axis=-1).max())
        correction = max(
            1.0,
            vmax_actual / config.max_velocity,
            np.sqrt(amax_actual / config.max_acceleration),
            np.cbrt(jmax_actual / config.max_jerk),
        )
        if correction <= 1.001:
            break
        duration *= correction * 1.01
    else:
        raise RuntimeError("dynamic time scaling failed to converge")
    return TrajectorySegment(
        positions=resampled,
        duration=duration,
        min_separation=separation,
        max_velocity=float(np.linalg.norm(velocity, axis=-1).max()),
        max_acceleration=float(np.linalg.norm(acceleration, axis=-1).max()),
        max_jerk=float(np.linalg.norm(jerk, axis=-1).max()),
        smoothing_sigma=selected_sigma,
    )
