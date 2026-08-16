from __future__ import annotations

import numpy as np

from rotorpy_agilicious.swarm_show.assignment import assign_targets
from rotorpy_agilicious.swarm_show.config import ShowConfig
from rotorpy_agilicious.swarm_show.formations import glyph_formation
from rotorpy_agilicious.swarm_show.mapf import prioritized_mapf, validate_discrete_paths


def test_glyph_has_unique_targets() -> None:
    config = ShowConfig(num_drones=16)
    points = glyph_formation("龙", config)
    assert points.shape == (16, 3)
    assert len(np.unique(points, axis=0)) == 16


def test_assignment_is_bijective() -> None:
    starts = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float)
    goals = np.asarray([[1, 0, 0], [0, 0, 0]], dtype=float)
    assigned, cost, permutation = assign_targets(starts, goals)
    assert np.allclose(assigned, starts)
    assert cost == 0.0
    assert sorted(permutation.tolist()) == [0, 1]


def test_mapf_resolves_head_on_swap() -> None:
    starts = np.asarray([[0, 0, 0], [2, 0, 0]])
    goals = starts[::-1].copy()
    result = prioritized_mapf(starts, goals, horizon_padding=20)
    validate_discrete_paths(result.paths)
    assert np.array_equal(result.paths[-1], goals)


def test_mapf_protects_stationary_goal() -> None:
    starts = np.asarray([[0, 0, 0], [-2, 0, 0], [2, 0, 0]])
    goals = np.asarray([[0, 0, 0], [2, 0, 0], [-2, 0, 0]])
    result = prioritized_mapf(starts, goals, horizon_padding=30)
    validate_discrete_paths(result.paths)
    assert np.array_equal(result.paths[-1], goals)
    assert np.all(result.paths[:, 0] == np.array([0, 0, 0]))


def test_mapf_respects_minimum_flight_altitude() -> None:
    starts = np.asarray([[0, 0, 2], [2, 0, 2]])
    goals = starts[::-1].copy()
    result = prioritized_mapf(
        starts,
        goals,
        margin_cells=3,
        horizon_padding=20,
        minimum_z_cell=1,
    )
    validate_discrete_paths(result.paths)
    assert np.all(result.paths[:, :, 2] >= 1)
