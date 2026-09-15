"""Tests for perception module (ASCII formatting and 4-connected segmentation)."""

import sys
from pathlib import Path

pkg_root = Path(__file__).resolve().parent.parent
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from perception.segmentation import connected_components, format_grid_ascii


def test_format_grid_ascii():
    grid = [
        [0, 1, 0],
        [0, 3, 5],
    ]
    rendered = format_grid_ascii(grid, show_coordinates=False)
    lines = rendered.strip().split("\n")
    assert lines[0] == ".B."
    assert lines[1] == ".G#"


def test_connected_components_segmentation():
    # 5x5 grid with two distinct objects:
    # A 2x2 square of color 1 (Blue) at top-left
    # A 3-pixel L-shape of color 2 (Red) at bottom-right
    grid = [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 0, 2, 2],
    ]

    objects = connected_components(grid, ignore_background=True)
    assert len(objects) == 2

    # Verify blue square
    blue_obj = next(obj for obj in objects if obj.color == 1)
    assert blue_obj.size == 4
    assert blue_obj.bbox == (0, 0, 1, 1)
    assert blue_obj.shape == (2, 2)
    assert blue_obj.color_name == "blue"

    # Verify red L-shape
    red_obj = next(obj for obj in objects if obj.color == 2)
    assert red_obj.size == 3
    assert red_obj.bbox == (3, 3, 4, 4)
    assert (3, 3) in red_obj.pixels
    assert (4, 3) in red_obj.pixels
    assert (4, 4) in red_obj.pixels
