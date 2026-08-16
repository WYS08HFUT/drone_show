"""Prioritized 3D space-time A* with vertex and edge reservations."""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


Cell = tuple[int, int, int]
MOVES: tuple[Cell, ...] = (
    (0, 0, 0),
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


class PlanningFailure(RuntimeError):
    pass


@dataclass
class MapfResult:
    paths: np.ndarray  # (time, agent, xyz), integer cells
    planning_order: np.ndarray
    expanded_nodes: int


def _manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def _reconstruct(parent: dict, state: tuple[Cell, int]) -> list[Cell]:
    path = []
    while state is not None:
        path.append(state[0])
        state = parent[state]
    return path[::-1]


def _edge_separation(a: Cell, b: Cell, c: Cell, d: Cell) -> float:
    """Minimum separation of two synchronous linear moves over one time step."""
    relative_start = np.asarray(a, dtype=float) - np.asarray(c, dtype=float)
    relative_velocity = (
        np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
        - np.asarray(d, dtype=float)
        + np.asarray(c, dtype=float)
    )
    speed_sq = float(relative_velocity @ relative_velocity)
    if speed_sq <= 1e-12:
        fraction = 0.0
    else:
        fraction = float(np.clip(-(relative_start @ relative_velocity) / speed_sq, 0.0, 1.0))
    return float(np.linalg.norm(relative_start + fraction * relative_velocity))


def _astar(
    start: Cell,
    goal: Cell,
    bounds: tuple[np.ndarray, np.ndarray],
    horizon: int,
    vertex_reservations: dict[int, set[Cell]],
    edge_reservations: dict[int, set[tuple[Cell, Cell]]],
    safety_radius_cells: float,
    max_expansions: int,
) -> tuple[list[Cell], int]:
    lower, upper = bounds
    start_state = (start, 0)
    parent: dict[tuple[Cell, int], tuple[Cell, int] | None] = {start_state: None}
    best_g = {start_state: 0}
    queue: list[tuple[int, int, int, Cell]] = []
    counter = 0
    heapq.heappush(queue, (_manhattan(start, goal), 0, counter, start))
    expanded = 0

    while queue:
        _, time, _, cell = heapq.heappop(queue)
        state = (cell, time)
        if best_g.get(state) != time:
            continue
        expanded += 1
        if expanded > max_expansions:
            raise PlanningFailure(f"A* exceeded {max_expansions} node expansions")
        if cell == goal:
            future_clear = all(goal not in vertex_reservations.get(t, ()) for t in range(time, horizon + 1))
            if future_clear:
                return _reconstruct(parent, state), expanded
        if time >= horizon:
            continue

        for delta in MOVES:
            neighbor = (cell[0] + delta[0], cell[1] + delta[1], cell[2] + delta[2])
            neighbor_array = np.asarray(neighbor)
            if np.any(neighbor_array < lower) or np.any(neighbor_array > upper):
                continue
            next_time = time + 1
            if neighbor in vertex_reservations.get(next_time, ()):
                continue
            if (neighbor, cell) in edge_reservations.get(time, ()):
                continue
            if any(
                _edge_separation(cell, neighbor, reserved_start, reserved_end) < safety_radius_cells - 1e-9
                for reserved_start, reserved_end in edge_reservations.get(time, ())
            ):
                continue
            next_state = (neighbor, next_time)
            if next_time >= best_g.get(next_state, 10**12):
                continue
            best_g[next_state] = next_time
            parent[next_state] = state
            counter += 1
            heuristic = _manhattan(neighbor, goal)
            heapq.heappush(queue, (next_time + heuristic, next_time, counter, neighbor))
    raise PlanningFailure(f"no space-time path from {start} to {goal} within horizon {horizon}")


def prioritized_mapf(
    starts: np.ndarray,
    goals: np.ndarray,
    margin_cells: int = 6,
    horizon_padding: int = 100,
    max_expansions: int = 300_000,
    safety_radius_cells: float = 0.8,
    minimum_z_cell: int | None = None,
) -> MapfResult:
    starts = np.asarray(starts, dtype=int)
    goals = np.asarray(goals, dtype=int)
    if starts.shape != goals.shape or starts.ndim != 2 or starts.shape[1] != 3:
        raise ValueError("starts and goals must have shape (N, 3)")
    if len(np.unique(starts, axis=0)) != len(starts) or len(np.unique(goals, axis=0)) != len(goals):
        raise ValueError("all start cells and all goal cells must be unique")

    distances = np.abs(goals - starts).sum(axis=1)
    stationary = np.flatnonzero(distances == 0)
    moving = np.flatnonzero(distances != 0)
    moving = moving[np.argsort(-distances[moving], kind="stable")]
    # Lock drones that already occupy their targets before routing long paths.
    # Otherwise a long high-priority path may cross a stationary drone's cell
    # in the future and leave the stationary agent with no legal forever-hold.
    order = np.concatenate((stationary, moving))
    lower = np.minimum(starts.min(axis=0), goals.min(axis=0)) - margin_cells
    upper = np.maximum(starts.max(axis=0), goals.max(axis=0)) + margin_cells
    if minimum_z_cell is not None:
        lower[2] = max(int(lower[2]), int(minimum_z_cell))
        if np.any(starts[:, 2] < minimum_z_cell) or np.any(goals[:, 2] < minimum_z_cell):
            raise ValueError("starts and goals must be at or above minimum_z_cell")
    horizon = int(distances.max() + horizon_padding + len(starts))
    vertex: dict[int, set[Cell]] = defaultdict(set)
    edges: dict[int, set[tuple[Cell, Cell]]] = defaultdict(set)
    paths: list[list[Cell] | None] = [None] * len(starts)
    total_expanded = 0

    for agent in order:
        start = tuple(int(value) for value in starts[agent])
        goal = tuple(int(value) for value in goals[agent])
        path, expanded = _astar(
            start,
            goal,
            (lower, upper),
            horizon,
            vertex,
            edges,
            safety_radius_cells,
            max_expansions,
        )
        total_expanded += expanded
        paths[int(agent)] = path
        for time, cell in enumerate(path):
            vertex[time].add(cell)
            if time + 1 < len(path):
                edges[time].add((cell, path[time + 1]))
        for time in range(len(path), horizon + 1):
            vertex[time].add(goal)
        for time in range(len(path) - 1, horizon):
            edges[time].add((goal, goal))

    makespan = max(len(path) for path in paths if path is not None)
    padded = np.empty((makespan, len(starts), 3), dtype=int)
    for agent, path in enumerate(paths):
        assert path is not None
        padded[: len(path), agent] = np.asarray(path)
        padded[len(path) :, agent] = np.asarray(path[-1])
    return MapfResult(padded, order, total_expanded)


def validate_discrete_paths(paths: np.ndarray) -> None:
    """Raise if a vertex collision or head-on edge swap exists."""
    for time in range(len(paths)):
        if len(np.unique(paths[time], axis=0)) != paths.shape[1]:
            raise AssertionError(f"vertex collision at discrete step {time}")
        if time + 1 == len(paths):
            continue
        transitions = {(tuple(paths[time, i]), tuple(paths[time + 1, i])) for i in range(paths.shape[1])}
        for start, end in transitions:
            if start != end and (end, start) in transitions:
                raise AssertionError(f"edge swap at discrete step {time}: {start} <-> {end}")
