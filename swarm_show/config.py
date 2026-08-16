"""Configuration for the centralized drone-show planner."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ShowConfig:
    num_drones: int = 64
    grid_resolution: float = 0.40
    safe_distance: float = 0.32
    formation_height: float = 7.20
    formation_altitude: float = 4.80
    launch_spacing: float = 0.80
    launch_altitude: float = 0.80
    minimum_flight_altitude: float = 0.40
    mapf_margin_cells: int = 6
    mapf_horizon_padding: int = 100
    max_astar_expansions: int = 300_000
    display_hold: float = 2.5
    separator_hold: float = 0.8
    sample_rate: int = 20
    nominal_grid_dt: float = 0.22
    max_velocity: float = 3.0
    max_acceleration: float = 4.0
    max_jerk: float = 14.0
    smoothing_sigma: float = 0.8
    seed: int = 2026

    def validate(self) -> None:
        if self.num_drones < 2:
            raise ValueError("num_drones must be at least 2")
        if self.grid_resolution <= self.safe_distance:
            raise ValueError("grid_resolution must be greater than safe_distance")
        if min(self.max_velocity, self.max_acceleration, self.max_jerk) <= 0:
            raise ValueError("dynamic limits must be positive")
        if self.minimum_flight_altitude < 0:
            raise ValueError("minimum_flight_altitude must be non-negative")
        if self.launch_altitude < self.minimum_flight_altitude:
            raise ValueError("launch_altitude must be at least minimum_flight_altitude")
        if self.sample_rate < 5:
            raise ValueError("sample_rate must be at least 5 Hz")

    def to_dict(self) -> dict:
        return asdict(self)
