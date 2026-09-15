"""Root extension entrypoint for tau discovery."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure tau-game root is on sys.path
_pkg_root = Path(__file__).resolve().parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from extensions.game.extension import EXTENSION, GameExtension

__all__ = ["EXTENSION", "GameExtension"]
