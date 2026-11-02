"""``InkBox``: a positioned piece of handwriting (a glyph, word or formula).

Coordinates are PDF points with the origin at the box's left edge on its
baseline and y pointing *down* (so ink above the baseline has negative y).

Two kinds of ink are supported:

* ``fills`` -- filled outline strokes (each a list of closed cubic sub-paths,
  arrays of shape (K, 4, 2)), exactly like the writer's real pen strokes;
* ``lines`` -- centre-line strokes ``(points (N, 2), width)`` drawn with a
  round pen, used for synthetic marks (fallback glyphs, fraction bars,
  root signs, underlines...).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .geometry import transform_points


@dataclass
class InkBox:
    fills: List[List[np.ndarray]] = field(default_factory=list)
    lines: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    width: float = 0.0
    ascent: float = 0.0  # extent above the baseline (positive)
    descent: float = 0.0  # extent below the baseline (positive)
    text: str = ""

    # ------------------------------------------------------------------
    def transformed(self, m: np.ndarray, width_scale: Optional[float] = None) -> "InkBox":
        """Apply a 2x3 affine matrix to all ink (metrics are recomputed)."""
        fills = [[transform_points(sp.reshape(-1, 2), m).reshape(sp.shape) for sp in st] for st in self.fills]
        if width_scale is None:
            width_scale = float(np.sqrt(abs(np.linalg.det(m[:, :2]))))
        lines = [(transform_points(p, m), w * width_scale) for p, w in self.lines]
        out = InkBox(fills, lines, self.width, self.ascent, self.descent, self.text)
        out.refit(keep_width=False)
        return out

    def shifted(self, dx: float, dy: float) -> "InkBox":
        fills = [[sp + np.array([dx, dy]) for sp in st] for st in self.fills]
        lines = [(p + np.array([dx, dy]), w) for p, w in self.lines]
        return InkBox(fills, lines, self.width, self.ascent - dy, self.descent + dy, self.text)

    def add(self, other: "InkBox", dx: float = 0.0, dy: float = 0.0) -> None:
        """Merge ``other`` placed with its origin at (dx, dy) of this box."""
        sh = np.array([dx, dy])
        self.fills.extend([[sp + sh for sp in st] for st in other.fills])
        self.lines.extend([(p + sh, w) for p, w in other.lines])
        self.width = max(self.width, dx + other.width)
        self.ascent = max(self.ascent, other.ascent - dy)
        self.descent = max(self.descent, other.descent + dy)

    def ink_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        pts = []
        for st in self.fills:
            for sp in st:
                pts.append(sp.reshape(-1, 2))
        for p, w in self.lines:
            if len(p):
                pts.append(p - w / 2)
                pts.append(p + w / 2)
        if not pts:
            return None
        a = np.vstack(pts)
        return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())

    def refit(self, keep_width: bool = True) -> None:
        b = self.ink_bounds()
        if b is None:
            return
        self.ascent = max(0.0, -b[1])
        self.descent = max(0.0, b[3])
        if not keep_width:
            self.width = max(self.width, b[2])

    def is_empty(self) -> bool:
        return not self.fills and not self.lines


def hstack(boxes: List[InkBox], gaps: List[float]) -> InkBox:
    """Concatenate boxes left to right; ``gaps[i]`` is added after box i."""
    out = InkBox()
    x = 0.0
    for i, b in enumerate(boxes):
        if i > 0:
            x += gaps[i - 1] if i - 1 < len(gaps) else 0.0
        out.add(b, x, 0.0)
        x += b.width
    out.width = x
    out.text = "".join(b.text for b in boxes)
    return out
