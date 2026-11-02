"""Geometry helpers for ink strokes.

A *stroke* is the filled outline of one pen stroke, stored the way PencilKit
exports it: a list of closed sub-paths whose union (non-zero winding) is the
inked area.  Each sub-path is an array of cubic Bezier segments with shape
``(K, 4, 2)`` (straight segments are stored as degenerate cubics).  Keeping the
original control points means affine transforms are exact and the rendered ink
is indistinguishable from the source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import numpy as np

Subpath = np.ndarray  # (K, 4, 2) float
BBox = Tuple[float, float, float, float]


def line_as_cubic(p0: Sequence[float], p3: Sequence[float]) -> np.ndarray:
    p0 = np.asarray(p0, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    return np.stack([p0, p0 + (p3 - p0) / 3.0, p0 + 2.0 * (p3 - p0) / 3.0, p3])


_T_CACHE: dict = {}


def _bernstein(n: int) -> np.ndarray:
    if n not in _T_CACHE:
        t = np.linspace(0.0, 1.0, n)[:, None]
        _T_CACHE[n] = np.concatenate(
            [(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3], axis=1
        )  # (n, 4)
    return _T_CACHE[n]


def flatten_subpath(sp: Subpath, n: int = 6) -> np.ndarray:
    """Polyline approximation of a sub-path (closed implicitly)."""
    if len(sp) == 0:
        return np.zeros((0, 2))
    B = _bernstein(n)  # (n,4)
    pts = np.einsum("nk,skd->snd", B, sp)  # (K, n, 2)
    pts = pts[:, :-1, :].reshape(-1, 2)  # drop duplicated segment end points
    return np.vstack([pts, sp[-1, 3][None, :]])


def flatten_stroke(subpaths: Sequence[Subpath], n: int = 6) -> List[np.ndarray]:
    return [flatten_subpath(sp, n) for sp in subpaths]


def subpaths_bbox(subpaths: Sequence[Subpath]) -> BBox:
    """Tight-ish bbox (uses flattened curve points, not control polygon)."""
    pts = np.vstack([flatten_subpath(sp, 5) for sp in subpaths if len(sp)])
    return (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))


def bbox_union(boxes: Iterable[BBox]) -> BBox:
    boxes = list(boxes)
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def affine(a: float = 1, b: float = 0, c: float = 0, d: float = 1, e: float = 0, f: float = 0) -> np.ndarray:
    """2x3 affine matrix mapping (x, y) -> (a x + b y + e, c x + d y + f)."""
    return np.array([[a, b, e], [c, d, f]], dtype=float)


def compose(*mats: np.ndarray) -> np.ndarray:
    """compose(A, B, C) applies C first, then B, then A."""
    out = np.eye(3)
    for m in mats:
        mm = np.vstack([m, [0, 0, 1]])
        out = out @ mm
    return out[:2]


def translate(dx: float, dy: float) -> np.ndarray:
    return affine(e=dx, f=dy)


def scale(sx: float, sy: float | None = None) -> np.ndarray:
    return affine(a=sx, d=sx if sy is None else sy)


def rotate(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return affine(a=c, b=-s, c=s, d=c)


def shear_x(k: float) -> np.ndarray:
    """x' = x + k * y.  With y pointing *down* (PDF page space) a negative k
    leans tall letters to the right."""
    return affine(b=k)


def transform_points(pts: np.ndarray, m: np.ndarray) -> np.ndarray:
    return pts @ m[:, :2].T + m[:, 2]


def transform_subpaths(subpaths: Sequence[Subpath], m: np.ndarray) -> List[Subpath]:
    return [transform_points(sp.reshape(-1, 2), m).reshape(sp.shape) for sp in subpaths]


def winding_mask(polys: Sequence[np.ndarray], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Non-zero winding fill of closed polygons sampled on grid ``xs`` x ``ys``.

    Returns a boolean array of shape (len(ys), len(xs)).
    """
    X, Y = np.meshgrid(xs, ys)
    px = X.ravel()
    py = Y.ravel()
    wn = np.zeros(px.shape, dtype=np.int32)
    for poly in polys:
        if len(poly) < 3:
            continue
        a = poly
        b = np.roll(poly, -1, axis=0)
        # Only consider edges whose y-span covers some sample rows.
        for (x0, y0), (x1, y1) in zip(a, b):
            if y0 == y1:
                continue
            ylo, yhi = (y0, y1) if y0 < y1 else (y1, y0)
            sel = (py >= ylo) & (py < yhi)
            if not sel.any():
                continue
            t = (py[sel] - y0) / (y1 - y0)
            xc = x0 + t * (x1 - x0)
            right = xc > px[sel]
            if y1 > y0:
                wn[np.flatnonzero(sel)[right]] += 1
            else:
                wn[np.flatnonzero(sel)[right]] -= 1
    return (wn != 0).reshape(X.shape)


def rasterize(subpath_lists: Sequence[Sequence[Subpath]], box: BBox, res: float, n: int = 4) -> np.ndarray:
    """Rasterize the union of several strokes inside ``box`` at ``res`` px/unit."""
    x0, y0, x1, y1 = box
    w = max(1, int(np.ceil((x1 - x0) * res)))
    h = max(1, int(np.ceil((y1 - y0) * res)))
    xs = x0 + (np.arange(w) + 0.5) / res
    ys = y0 + (np.arange(h) + 0.5) / res
    polys: List[np.ndarray] = []
    for sps in subpath_lists:
        polys.extend(flatten_stroke(sps, n))
    return _winding_fast(polys, xs, ys)


def _winding_fast(polys: Sequence[np.ndarray], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Vectorised scanline non-zero winding rasteriser."""
    H, W = len(ys), len(xs)
    wn = np.zeros((H, W), dtype=np.int32)
    if not polys:
        return wn != 0
    edges_a = np.vstack([p for p in polys if len(p) >= 3]) if any(len(p) >= 3 for p in polys) else None
    if edges_a is None:
        return wn != 0
    A = []
    B = []
    for p in polys:
        if len(p) < 3:
            continue
        A.append(p)
        B.append(np.roll(p, -1, axis=0))
    A = np.vstack(A)
    B = np.vstack(B)
    dy = B[:, 1] - A[:, 1]
    keep = dy != 0
    A, B, dy = A[keep], B[keep], dy[keep]
    direction = np.where(dy > 0, 1, -1).astype(np.int32)
    ylo = np.minimum(A[:, 1], B[:, 1])
    yhi = np.maximum(A[:, 1], B[:, 1])
    x0s = xs[0]
    dx = xs[1] - xs[0] if W > 1 else 1.0
    for r, y in enumerate(ys):
        sel = (ylo <= y) & (yhi > y)
        if not sel.any():
            continue
        a = A[sel]
        b = B[sel]
        t = (y - a[:, 1]) / (b[:, 1] - a[:, 1])
        xc = a[:, 0] + t * (b[:, 0] - a[:, 0])
        d = direction[sel]
        # pixel columns strictly left of the crossing get the contribution
        col = np.ceil((xc - x0s) / dx).astype(int)  # first column with x >= xc
        col = np.clip(col, 0, W)
        acc = np.zeros(W + 1, dtype=np.int32)
        np.add.at(acc, np.zeros_like(col), d)
        np.add.at(acc, col, -d)
        wn[r] += np.cumsum(acc)[:W]
    return wn != 0


@dataclass
class Stroke:
    """One pen stroke as extracted from a PDF."""

    subpaths: List[Subpath]
    page: int = 0
    seq: int = 0  # drawing order on the page
    source: str = ""
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox: BBox = field(default=(0.0, 0.0, 0.0, 0.0))

    def __post_init__(self) -> None:
        if self.bbox == (0.0, 0.0, 0.0, 0.0) and self.subpaths:
            self.bbox = subpaths_bbox(self.subpaths)

    @property
    def cx(self) -> float:
        return 0.5 * (self.bbox[0] + self.bbox[2])

    @property
    def cy(self) -> float:
        return 0.5 * (self.bbox[1] + self.bbox[3])

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def transformed(self, m: np.ndarray) -> "Stroke":
        sps = transform_subpaths(self.subpaths, m)
        return Stroke(sps, self.page, self.seq, self.source, self.color)

    def n_segments(self) -> int:
        return int(sum(len(s) for s in self.subpaths))
