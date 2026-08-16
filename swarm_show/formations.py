"""Convert user text, including Chinese glyphs, into drone target formations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import ShowConfig


FONT_CANDIDATES = (
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/Supplemental/Songti.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)


def parse_sequence(text: str) -> list[str]:
    tokens = [token.strip() for token in text.split("-") if token.strip()]
    if not tokens:
        raise ValueError("show text must contain at least one non-empty token")
    return tokens


def resolve_font(font_path: str | Path | None = None) -> Path:
    if font_path is not None:
        path = Path(font_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"font not found: {path}")
        return path
    for path in FONT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("no Unicode font found; pass --font /path/to/font.ttf")


def _rasterize(token: str, font_path: Path, size: int = 256) -> Image.Image:
    font = ImageFont.truetype(str(font_path), size=size)
    probe = Image.new("L", (16, 16), 0)
    draw = ImageDraw.Draw(probe)
    bbox = draw.textbbox((0, 0), token, font=font, stroke_width=2)
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    image = Image.new("L", (width + 24, height + 24), 0)
    draw = ImageDraw.Draw(image)
    draw.text((12 - bbox[0], 12 - bbox[1]), token, fill=255, font=font, stroke_width=2)
    crop = image.getbbox()
    if crop is None:
        raise ValueError(f"font {font_path} cannot render token {token!r}")
    return image.crop(crop)


def _farthest_sample(candidates: np.ndarray, intensity: np.ndarray, count: int) -> np.ndarray:
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} glyph cells for {count} drones")
    center = candidates.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(candidates - center, axis=1) * (0.5 + intensity)))
    chosen = [first]
    min_distance_sq = np.sum((candidates - candidates[first]) ** 2, axis=1)
    for _ in range(1, count):
        score = min_distance_sq * (0.35 + 0.65 * intensity)
        score[chosen] = -1.0
        index = int(np.argmax(score))
        chosen.append(index)
        distance_sq = np.sum((candidates - candidates[index]) ** 2, axis=1)
        min_distance_sq = np.minimum(min_distance_sq, distance_sq)
    return candidates[np.asarray(chosen)]


def glyph_formation(
    token: str,
    config: ShowConfig,
    font_path: str | Path | None = None,
) -> np.ndarray:
    """Return exactly N unique target points in a vertical x-z glyph plane."""
    font = resolve_font(font_path)
    image = _rasterize(token, font)
    aspect = min(max(image.width / image.height, 0.65), 3.8)
    rows = max(16, int(round(config.formation_height / config.grid_resolution)))
    cols = max(12, int(round(rows * aspect)))
    grid_image = image.resize((cols, rows), Image.Resampling.LANCZOS)
    pixels = np.asarray(grid_image, dtype=np.float64) / 255.0

    threshold = 0.18
    occupied = np.argwhere(pixels >= threshold)
    if len(occupied) < config.num_drones:
        flat_order = np.argsort(pixels.ravel())[::-1][: config.num_drones]
        occupied = np.column_stack(np.unravel_index(flat_order, pixels.shape))
    values = pixels[occupied[:, 0], occupied[:, 1]]
    sampled = _farthest_sample(occupied.astype(np.float64), values, config.num_drones)

    row = sampled[:, 0].astype(int)
    col = sampled[:, 1].astype(int)
    # Build integer grid cells first. Half-cell centering followed by NumPy's
    # bankers rounding can otherwise collapse two distinct glyph pixels.
    x_cell = col - cols // 2
    y_cell = np.zeros_like(x_cell)
    altitude_cell = int(round(config.formation_altitude / config.grid_resolution))
    z_cell = altitude_cell + rows // 2 - row
    cells = np.column_stack((x_cell, y_cell, z_cell))
    if len(np.unique(cells, axis=0)) != config.num_drones:
        raise RuntimeError(f"glyph sampling for {token!r} produced duplicate cells")
    return cells.astype(np.float64) * config.grid_resolution


def launch_grid(config: ShowConfig) -> np.ndarray:
    """Create a compact, collision-free launch formation near the ground."""
    columns = int(np.ceil(np.sqrt(config.num_drones)))
    rows = int(np.ceil(config.num_drones / columns))
    points = []
    for index in range(config.num_drones):
        row, column = divmod(index, columns)
        x = (column - (columns - 1) / 2.0) * config.launch_spacing
        y = (row - (rows - 1) / 2.0) * config.launch_spacing
        points.append((x, y, config.launch_altitude))
    points = np.asarray(points, dtype=np.float64)
    return np.round(points / config.grid_resolution) * config.grid_resolution
