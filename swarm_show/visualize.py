"""Matplotlib 3D visualization for offline show plans."""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PROJECT_ROOT / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .planner import ShowPlan


def animate_plan(
    plan: ShowPlan,
    save_path: str | Path | None = None,
    show: bool = False,
    fps: int = 24,
    playback_speed: float = 4.0,
    view: str = "3d",
    drone_model: str = "quadcopter",
) -> None:
    if view not in {"3d", "front", "top"}:
        raise ValueError("view must be one of: 3d, front, top")
    if drone_model not in {"quadcopter", "point"}:
        raise ValueError("drone_model must be one of: quadcopter, point")
    step = max(1, int(round(plan.config.sample_rate * playback_speed / fps)))
    frame_indices = np.arange(0, len(plan.times), step)
    positions = plan.positions
    minimum = positions.min(axis=(0, 1))
    maximum = positions.max(axis=(0, 1))
    center = (minimum + maximum) / 2.0
    half_width = max(float((maximum - minimum).max()) / 2.0 + 1.0, 2.0)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(center[0] - half_width, center[0] + half_width)
    ax.set_ylim(center[1] - half_width, center[1] + half_width)
    ax.set_zlim(max(0.0, center[2] - half_width), center[2] + half_width)
    spans = np.maximum(maximum - minimum, 1.0)
    ax.set_box_aspect(spans)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    camera = {"3d": (25, -55), "front": (8, -90), "top": (90, -90)}
    ax.view_init(*camera[view])
    ax.set_proj_type("persp", focal_length=0.9)
    ax.set_title(f"Centralized 3D MAPF + formation control · {view.upper()} view")
    scatter = ax.scatter([], [], [], s=22, depthshade=True)
    placeholder = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    body_segments = Line3DCollection(placeholder, linewidths=2.2, alpha=0.95)
    rotor_segments = Line3DCollection(placeholder, linewidths=0.75, alpha=0.85)
    ax.add_collection3d(body_segments)
    ax.add_collection3d(rotor_segments)
    trails = [ax.plot([], [], [], lw=0.7, alpha=0.25)[0] for _ in range(plan.config.num_drones)]
    unicode_font = FontProperties(fname=plan.font_path)
    status = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, fontproperties=unicode_font)
    safety = ax.text2D(0.02, 0.91, "", transform=ax.transAxes)
    trail_frames = max(2, int(plan.config.sample_rate * 1.2))
    arm = min(0.16, plan.config.safe_distance * 0.38)
    rotor_radius = arm * 0.35
    rotor_angles = np.linspace(0.0, 2.0 * np.pi, 9)
    rotor_offsets = np.asarray(
        [[arm, 0.0, 0.0], [-arm, 0.0, 0.0], [0.0, arm, 0.0], [0.0, -arm, 0.0]]
    )

    def quadcopter_geometry(xyz: np.ndarray, rgb: np.ndarray):
        bodies, rotors, body_colors, rotor_colors = [], [], [], []
        for center_xyz, color in zip(xyz, rgb, strict=True):
            bodies.extend(
                [[center_xyz - rotor_offsets[0], center_xyz + rotor_offsets[0]],
                 [center_xyz - rotor_offsets[2], center_xyz + rotor_offsets[2]]]
            )
            body_colors.extend([color, color])
            for offset in rotor_offsets:
                hub = center_xyz + offset
                rotors.append(
                    np.column_stack(
                        (hub[0] + rotor_radius * np.cos(rotor_angles),
                         hub[1] + rotor_radius * np.sin(rotor_angles),
                         np.full_like(rotor_angles, hub[2]))
                    )
                )
                rotor_colors.append(color)
        return bodies, rotors, body_colors, rotor_colors

    def update(animation_frame: int):
        index = int(frame_indices[animation_frame])
        xyz = positions[index]
        scatter._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
        scatter.set_color(plan.led_rgb[index])
        scatter.set_visible(drone_model == "point")
        body_segments.set_visible(drone_model == "quadcopter")
        rotor_segments.set_visible(drone_model == "quadcopter")
        if drone_model == "quadcopter":
            bodies, rotors, body_colors, rotor_colors = quadcopter_geometry(
                xyz, plan.led_rgb[index]
            )
            body_segments.set_segments(bodies)
            body_segments.set_color(body_colors)
            rotor_segments.set_segments(rotors)
            rotor_segments.set_color(rotor_colors)
        start = max(0, index - trail_frames)
        for drone, line in enumerate(trails):
            trail = positions[start : index + 1, drone]
            line.set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])
            line.set_color(plan.led_rgb[index, drone])
        status.set_text(
            f"t={plan.times[index]:6.2f} s   {plan.phase_labels[index]}   "
            f"drones={plan.config.num_drones}"
        )
        safety.set_text(
            f"validated min separation={plan.metrics['minimum_separation_m']:.2f} m   "
            f"vmax={plan.metrics['maximum_velocity_mps']:.2f} m/s"
        )
        return (scatter, body_segments, rotor_segments, status, safety, *trails)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        animation.save(str(path), writer="ffmpeg", fps=fps, dpi=120)
        print(f"Saved 3D show video: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
