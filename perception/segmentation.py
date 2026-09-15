"""Perception module: ASCII grid formatting and 4-connected component segmentation."""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

# Standard ARC color-to-ASCII map
# 0: Black/Background, 1: Blue, 2: Red, 3: Green, 4: Yellow,
# 5: Grey, 6: Magenta, 7: Orange, 8: Cyan, 9: Maroon
COLOR_TO_CHAR: dict[int, str] = {
    0: ".",  # background / empty
    1: "B",  # Blue (player in GridWorld)
    2: "R",  # Red
    3: "G",  # Green (goal)
    4: "Y",  # Yellow (key)
    5: "#",  # Grey (wall / obstacle)
    6: "M",  # Magenta
    7: "O",  # Orange (door)
    8: "C",  # Cyan (movable block)
    9: "W",  # White / Maroon
}

CHAR_TO_COLOR: dict[str, int] = {v: k for k, v in COLOR_TO_CHAR.items()}


def format_grid_ascii(grid: list[list[int]], show_coordinates: bool = True) -> str:
    """Render a 2D integer grid into a clean ASCII string for the LLM.

    Example output:
       0 1 2 3 4
     0 . . . . .
     1 . B . . .
     2 . . . G .
     3 . . . . .
    """
    if not grid or not grid[0]:
        return "(empty grid)"

    h = len(grid)
    w = len(grid[0])
    lines: list[str] = []

    if show_coordinates and w <= 30:
        # Header with column indices
        col_header = "   " + " ".join(str(c % 10) for c in range(w))
        lines.append(col_header)

    for r in range(h):
        row_chars = [COLOR_TO_CHAR.get(val, str(val % 10)) for val in grid[r]]
        if show_coordinates and w <= 30:
            lines.append(f"{r:2d} " + " ".join(row_chars))
        else:
            lines.append("".join(row_chars))

    return "\n".join(lines)


@dataclass
class ObjectNode:
    """A segmented 4-connected component of identical color."""

    id: int
    color: int
    color_name: str
    pixels: list[tuple[int, int]]  # [(row, col), ...]
    bbox: tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)
    size: int
    shape: tuple[int, int]  # (height, width)
    centroid: tuple[float, float]
    shape_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "color": self.color,
            "color_name": self.color_name,
            "bbox": self.bbox,
            "size": self.size,
            "shape": self.shape,
            "centroid": self.centroid,
            "shape_hash": self.shape_hash,
            "pixel_count": len(self.pixels),
        }

    def __repr__(self) -> str:
        return (
            f"ObjectNode(id={self.id}, color={self.color_name}({self.color}), "
            f"size={self.size}, bbox={self.bbox}, shape={self.shape})"
        )


COLOR_NAMES: dict[int, str] = {
    0: "background",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "magenta",
    7: "orange",
    8: "cyan",
    9: "maroon",
}


def connected_components(
    grid: list[list[int]],
    ignore_background: bool = True,
    background_color: int = 0,
) -> list[ObjectNode]:
    """Segment the grid into 4-connected same-color object components.

    Extracts bounding boxes, shape hashes, and centroids.
    Inspired by Tufa Labs' Duck segmentation technique.
    """
    if not grid or not grid[0]:
        return []

    h = len(grid)
    w = len(grid[0])
    visited: set[tuple[int, int]] = set()
    components: list[ObjectNode] = []
    comp_id = 0

    for r in range(h):
        for c in range(w):
            if (r, c) in visited:
                continue

            color = grid[r][c]
            if ignore_background and color == background_color:
                visited.add((r, c))
                continue

            # BFS for 4-connected component
            pixels: list[tuple[int, int]] = []
            queue = deque([(r, c)])
            visited.add((r, c))

            min_r, max_r = r, r
            min_c, max_c = c, c

            while queue:
                cr, cc = queue.popleft()
                pixels.append((cr, cc))

                min_r = min(min_r, cr)
                max_r = max(max_r, cr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                        if grid[nr][nc] == color:
                            visited.add((nr, nc))
                            queue.append((nr, nc))

            # Compute relative normalized shape hash
            pixels_sorted = sorted(pixels)
            rel_pixels = tuple((pr - min_r, pc - min_c) for pr, pc in pixels_sorted)
            shape_repr = f"{max_r - min_r + 1}x{max_c - min_c + 1}:{rel_pixels}"
            shape_hash = hashlib.md5(shape_repr.encode("utf-8")).hexdigest()[:8]

            # Centroid
            avg_r = sum(pr for pr, _ in pixels) / len(pixels)
            avg_c = sum(pc for _, pc in pixels) / len(pixels)

            node = ObjectNode(
                id=comp_id,
                color=color,
                color_name=COLOR_NAMES.get(color, f"color_{color}"),
                pixels=pixels_sorted,
                bbox=(min_r, min_c, max_r, max_c),
                size=len(pixels),
                shape=(max_r - min_r + 1, max_c - min_c + 1),
                centroid=(round(avg_r, 2), round(avg_c, 2)),
                shape_hash=shape_hash,
            )
            components.append(node)
            comp_id += 1

    return components
