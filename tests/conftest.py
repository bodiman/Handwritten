"""Shared fixtures for the test suite.

* ``toy_style`` -- a small synthetic :class:`~handwritten.style.Style` whose
  glyphs are simple filled outlines (bars and rings made of cubic Beziers).
  It needs no external files, so geometry/layout/render tests always run.
* ``real_style`` -- the user's trained style (``me.hwstyle``).  Tests that
  need it are skipped when it is not available.  Set ``HANDWRITTEN_TEST_STYLE``
  to point the suite at another trained style file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

from handwritten.geometry import line_as_cubic
from handwritten.style import BankGlyph, Style, StyleStats

STYLE_CANDIDATES = [
    os.environ.get("HANDWRITTEN_TEST_STYLE", ""),
    "/tmp/claude-0/-home-user-Handwritten/525beb19-0610-5fa1-818c-aaa2915c4eca/scratchpad/me.hwstyle",
]
REAL_STYLE_PATH: Optional[str] = next((p for p in STYLE_CANDIDATES if p and os.path.exists(p)), None)

# Characters of the toy style.  'O' is deliberately missing (it has a '0'
# analogue) and so is everything outside this set (Hershey fallback).
TOY_CHARS = "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJ" + "0123456789" + ".,!?-"
KAPPA = 0.5522847498


# ----------------------------------------------------------------- outlines
def rect_subpath(x0: float, y0: float, x1: float, y1: float, ccw: bool = False) -> np.ndarray:
    """Closed rectangle as 4 straight cubic segments, shape (4, 4, 2)."""
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if ccw:
        pts = pts[::-1]
    return np.stack([line_as_cubic(pts[i], pts[(i + 1) % 4]) for i in range(4)])


def ellipse_subpath(cx: float, cy: float, rx: float, ry: float, reverse: bool = False) -> np.ndarray:
    """Closed ellipse as 4 cubic arcs, shape (4, 4, 2)."""
    k = KAPPA
    p = [(cx + rx, cy), (cx, cy + ry), (cx - rx, cy), (cx, cy - ry)]
    segs = []
    for i in range(4):
        a = np.array(p[i])
        b = np.array(p[(i + 1) % 4])
        ta = np.array([[0, k * ry], [-k * rx, 0], [0, -k * ry], [k * rx, 0]][i])
        tb = np.array([[k * rx, 0], [0, k * ry], [-k * rx, 0], [0, -k * ry]][i])
        segs.append(np.stack([a, a + ta, b + tb, b]))
    sp = np.stack(segs)
    if reverse:
        sp = sp[::-1, ::-1, :].copy()
    return sp


def ring_stroke(cx: float, cy: float, rx: float, ry: float, pen: float) -> List[np.ndarray]:
    """A pen-drawn 'o': outer ellipse plus an oppositely wound inner one."""
    return [ellipse_subpath(cx, cy, rx, ry), ellipse_subpath(cx, cy, rx - pen, ry - pen, reverse=True)]


def toy_glyph(ch: str, variant: int = 0, gid: int = 0, pen: float = 0.16) -> BankGlyph:
    """A crude but well-formed glyph in the bank's frame (x-height units:
    x from the left ink edge, baseline at y = 0, y down)."""
    w = 0.55 * (1.0 + 0.12 * variant)
    strokes: List[List[np.ndarray]] = []
    if ch in ".,":
        strokes.append([rect_subpath(0.0, -0.18, 0.18, 0.0 if ch == "." else 0.25)])
        w = 0.18
    elif ch == "-":
        strokes.append([rect_subpath(0.0, -0.55, 0.5, -0.55 + pen)])
        w = 0.5
    elif ch in "!?":
        strokes.append([rect_subpath(0.0, -1.6, pen, -0.45)])
        strokes.append([rect_subpath(0.0, -0.18, pen, 0.0)])
        w = pen
    elif ch in "aceosu":
        strokes.append(ring_stroke(w / 2, -0.5, w / 2, 0.5, pen))
    elif ch in "bdfhklt":
        strokes.append([rect_subpath(0.0, -1.7, pen, 0.0)])
        strokes.append(ring_stroke(w / 2 + pen, -0.45, w / 2, 0.45, pen))
        w = w + pen
    elif ch in "gjpqy":
        strokes.append([rect_subpath(w - pen, -1.0, w, 0.65)])
        strokes.append(ring_stroke(w / 2, -0.5, w / 2, 0.5, pen))
    elif ch.isupper() or ch.isdigit():
        strokes.append([rect_subpath(0.0, -1.6, pen, 0.0)])
        strokes.append([rect_subpath(0.0, -1.6, w, -1.6 + pen)])
    else:  # i m n r v w x z ...
        strokes.append([rect_subpath(0.0, -1.0, pen, 0.0)])
        strokes.append([rect_subpath(w - pen, -1.0, w, 0.0)])
        if ch in "ij":
            strokes.append([rect_subpath(0.0, -1.45, pen, -1.25)])
    pts = np.vstack([sp.reshape(-1, 2) for st in strokes for sp in st])
    x0 = pts[:, 0].min()
    strokes = [[sp - np.array([x0, 0.0]) for sp in st] for st in strokes]
    pts = pts - np.array([x0, 0.0])
    return BankGlyph(ch, strokes, float(pts[:, 0].max()), float(-pts[:, 1].min()), float(-pts[:, 1].max()),
                     source=f"toy:{variant}", gid=gid)


def make_toy_style(samples: int = 2, chars: str = TOY_CHARS, xh_pt: float = 10.0) -> Style:
    bank: Dict[str, List[BankGlyph]] = {}
    gid = 0
    for ch in chars:
        bank[ch] = []
        for v in range(samples):
            bank[ch].append(toy_glyph(ch, v, gid))
            gid += 1
    stats = StyleStats(xh_pt=xh_pt, slant=0.15, pen_width=0.16, ink=(0.1, 0.12, 0.3))
    stats.char_counts = {c: len(v) for c, v in bank.items()}
    return Style(stats=stats, bank=bank, meta={"sources": ["toy"]})


# ----------------------------------------------------------------- fixtures
@pytest.fixture
def toy_style() -> Style:
    """Fresh toy style (2 samples per character); safe to mutate."""
    return make_toy_style()


@pytest.fixture
def toy_style_path(tmp_path: Path) -> Path:
    p = tmp_path / "toy.hwstyle"
    make_toy_style().save(p)
    return p


@pytest.fixture(scope="session")
def real_style_path() -> str:
    if REAL_STYLE_PATH is None:
        pytest.skip("trained style (me.hwstyle) not available; set HANDWRITTEN_TEST_STYLE")
    return REAL_STYLE_PATH


@pytest.fixture(scope="session")
def real_style(real_style_path: str) -> Style:
    """The user's trained style.  Shared across the session: do not mutate."""
    return Style.load(real_style_path)
