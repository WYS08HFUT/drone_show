"""Export a show plan as simulator data and per-drone command streams."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .planner import ShowPlan
from .config import ShowConfig


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON encode {type(value)!r}")


def export_plan(plan: ShowPlan, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    command_dir = output / "drone_commands"
    command_dir.mkdir(parents=True, exist_ok=True)
    archive = output / "show_plan.npz"
    np.savez_compressed(
        archive,
        times=plan.times,
        positions=plan.positions,
        velocities=plan.velocities,
        accelerations=plan.accelerations,
        yaw=plan.yaw,
        led_rgb=plan.led_rgb,
        phase_labels=plan.phase_labels,
    )

    header = [
        "time_s",
        "x_m",
        "y_m",
        "z_m",
        "vx_mps",
        "vy_mps",
        "vz_mps",
        "ax_mps2",
        "ay_mps2",
        "az_mps2",
        "yaw_rad",
        "led_r",
        "led_g",
        "led_b",
        "phase",
    ]
    for drone in range(plan.config.num_drones):
        path = command_dir / f"drone_{drone:03d}.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            for frame, time in enumerate(plan.times):
                writer.writerow(
                    [
                        f"{time:.3f}",
                        *[f"{value:.6f}" for value in plan.positions[frame, drone]],
                        *[f"{value:.6f}" for value in plan.velocities[frame, drone]],
                        *[f"{value:.6f}" for value in plan.accelerations[frame, drone]],
                        f"{plan.yaw[frame, drone]:.6f}",
                        *[f"{value:.4f}" for value in plan.led_rgb[frame, drone]],
                        plan.phase_labels[frame],
                    ]
                )

    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "aerial-show-plan/v1",
                "coordinate_frame": "ENU (x east, y north, z up)",
                "execution": "fully centralized and precomputed",
                "sequence": plan.sequence,
                "tokens": plan.tokens,
                "font": plan.font_path,
                "config": plan.config.to_dict(),
                "metrics": plan.metrics,
                "command_columns": header,
                "warning": "Simulation output only; real flight requires localization, synchronization, geofence, health monitoring, emergency abort, and regulatory review.",
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    return {"archive": archive, "manifest": manifest, "commands": command_dir}


def load_plan(output_dir: str | Path) -> ShowPlan:
    output = Path(output_dir)
    manifest_data = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    with np.load(output / "show_plan.npz") as data:
        arrays = {key: data[key] for key in data.files}
    return ShowPlan(
        sequence=manifest_data["sequence"],
        tokens=list(manifest_data["tokens"]),
        times=arrays["times"],
        positions=arrays["positions"],
        velocities=arrays["velocities"],
        accelerations=arrays["accelerations"],
        yaw=arrays["yaw"],
        led_rgb=arrays["led_rgb"],
        phase_labels=arrays["phase_labels"],
        metrics=manifest_data["metrics"],
        config=ShowConfig(**manifest_data["config"]),
        font_path=manifest_data["font"],
    )
