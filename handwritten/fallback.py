"""Fallback glyphs for characters the writer has never written.

Shapes come from the public-domain Hershey single-stroke fonts (Roman
"futural" for Latin/digits/punctuation, Hershey Greek) and from procedural
constructions for maths symbols.  They are restyled to blend in with the
writer's real PencilKit glyphs:

* metrics are calibrated on the writer's own bank: x-height, ascender, cap
  and digit heights and the descender are *ink* extents (like the real
  glyphs), widths are scaled per class (lower case, capitals, digits) so
  the fallback has the writer's proportions, and maths operators take the
  size and axis height of the writer's own ``+ = -``;
* the writer's slant, rounded corners, a gentle low-frequency wobble and
  small stroke-to-stroke misplacement (crossbars that do not quite meet);
* strokes become *filled outlines* drawn with a pressure-sensitive pen:
  slightly tapered round ends, gentle pressure modulation along the stroke
  and stroke-to-stroke weight variation, calibrated so that the measured
  pen width (ink area / skeleton length) matches ``StyleStats.pen_width``.
  Outlines are closed cubic sub-paths (K, 4, 2) wound like PencilKit's;
* where the writer has the parts, punctuation and accented letters are
  composed from real samples (``:`` and ``;`` from the writer's ``.`` and
  ``,``, ``é`` from the writer's ``e`` plus a synthetic accent).

Output is in x-height units in the glyph frame of real glyphs (x from the
left ink edge, baseline at y = 0, y down).

API: ``HersheyFallback.for_style(style)``; ``glyph(ch, rng)`` returns
centre-line strokes ``(lines, width, top, bottom)`` drawn with a round pen;
``glyph_ink(ch, rng)`` returns ``(fills, lines, width, top, bottom)`` with
the ink as PencilKit-like filled outlines (``lines`` is empty).
``pen_stroke(points, pen, rng, unit)`` turns any synthetic centre-line
(fraction bars, radicals...) into the same kind of filled pen stroke.
"""
from __future__ import annotations

import math
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .priors import BASE, UNICODE_HERSHEY, XH, _hershey

# Symbols mapped to (Hershey font, character in that font).  Where a symbol
# is absent it is drawn procedurally in ``_procedural``.
SYMBOL_HERSHEY: Dict[str, Tuple[str, str]] = {"ϑ": ("greek", "q"), "ϕ": ("greek", "f"), "Υ": ("greek", "U")}

# Characters that look like another one: drawn as (and, when the writer
# has it, taken from the writer's real) look-alike.
HOMOGLYPHS: Dict[str, str] = {
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Χ": "X", "ο": "o", "∗": "*", "∣": "|", "∥": "‖", "ϖ": "π", "ϱ": "ρ", "ς": "σ",
    "ϵ": "ε",
}

PROCEDURAL = set("×÷±∓·≤≥≠≈≡→←↑↓↔⇒⇐⇔∞∑∏∫√∂∇∈∉⊂⊆⊃⊇∪∩∅∀∃¬∧∨°′″…⋯|ℕℤℚℝℂ∘⟨⟩‖"
                 "+=<>−-~∼≃≅∝↦∋•θ{}⌈⌉⌊⌋⟶⟵⟷⟹⟸⟺✓ℓ†‡⋆ℏ")

# Maths operators / relations: drawn in a canonical design centred on the
# axis (y = 0.5 above the baseline, about 0.8 x-heights wide) and scaled to
# the writer's own operators.
OPERATORS = set("+−-=<>×÷±∓·≤≥≠≈≡∼~≃≅∝→←↔⇒⇐⇔↦∈∉∋⊂⊆⊃⊇∪∩∧∨¬∘∞•⟶⟵⟷⟹⟸⟺⋆")
BIG_OPS = set("∑∏∫√")
BRACKETS = set("()[]{}⌈⌉⌊⌋⟨⟩")
DOUBLE_STRUCK = {"ℕ": "N", "ℤ": "Z", "ℚ": "Q", "ℝ": "R", "ℂ": "C"}
XHEIGHT_REF = "acemnorsuvwxz"
SCRIPT_SCALE = 0.62  # size of super/subscript characters (², ₙ ...)

# Design metrics of the Hershey fonts / procedural designs (x-height units,
# measured on centre-lines): cap height and ascender 1.5, descender 0.5.
D_CAP = 1.5
D_DESC = 0.5
D_BRACKET = (1.79, -0.5)  # top, bottom of Hershey brackets

# Mid-stroke pen diameter relative to the target pen width; calibrated so
# that the measured pen (ink area / skeleton length, as in style.py) of the
# fallback equals StyleStats.pen_width (tapers make the mean a bit thinner).
PEN_MID = 0.95
FLAT_WEIGHT = 0.76  # pen weight of horizontal strokes relative to the others
DOT_SIZE = 1.35  # dot diameter / pen width (real dots are taps, a bit fatter)


# ----------------------------------------------------------------------------
# polyline helpers
# ----------------------------------------------------------------------------
def _chaikin(p: np.ndarray, n: int = 2) -> np.ndarray:
    """Corner-cutting smoothing; keeps end points."""
    for _ in range(n):
        if len(p) < 3:
            return p
        q = 0.75 * p[:-1] + 0.25 * p[1:]
        r = 0.25 * p[:-1] + 0.75 * p[1:]
        mid = np.empty((2 * (len(p) - 1), 2))
        mid[0::2] = q
        mid[1::2] = r
        p = np.vstack([p[:1], mid, p[-1:]])
    return p


def _dedup(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    if len(p) < 2:
        return p
    d = np.hypot(*np.diff(p, axis=0).T)
    keep = np.r_[True, d > eps]
    return p[keep]


def _densify(p: np.ndarray, h: float) -> np.ndarray:
    """Insert points so no segment is longer than ``h`` (corners are kept)."""
    if len(p) < 2:
        return p
    out = [p[:1]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(math.ceil(float(np.hypot(*(b - a))) / h)))
        t = (np.arange(1, n + 1) / n)[:, None]
        out.append(a + (b - a) * t)
    return np.vstack(out)


def _arclen(p: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))] if len(p) > 1 else np.zeros(len(p))


def _resample(p: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform arc-length resampling; returns (points, arclength)."""
    p = _dedup(p)
    d = _arclen(p)
    L = float(d[-1]) if len(d) else 0.0
    if L < 1e-9:
        return p[:1].copy(), np.zeros(1)
    n = max(2, int(math.ceil(L / h)) + 1)
    s = np.linspace(0.0, L, n)
    return np.column_stack([np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])]), s


def _tangents(p: np.ndarray) -> np.ndarray:
    t = np.empty_like(p)
    if len(p) == 1:
        t[:] = (1.0, 0.0)
        return t
    t[1:-1] = p[2:] - p[:-2]
    t[0] = p[1] - p[0]
    t[-1] = p[-1] - p[-2]
    n = np.hypot(t[:, 0], t[:, 1])
    n[n < 1e-12] = 1.0
    return t / n[:, None]


def _arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 16) -> np.ndarray:
    t = np.linspace(a0, a1, n)
    return np.column_stack([cx + r * np.cos(t), cy - r * np.sin(t)])


def _circle(cx: float, cy: float, r: float, n: int = 28) -> np.ndarray:
    return _arc(cx, cy, r, 0.0, 2 * np.pi, n)


# ----------------------------------------------------------------------------
# filled outlines (PencilKit-like)
# ----------------------------------------------------------------------------
def _arc_cubics(c: np.ndarray, r: float, a0: float, a1: float) -> List[np.ndarray]:
    """Circular arc from angle a0 to a1 (numeric x/y axes) as cubic segments."""
    n = max(1, int(math.ceil(abs(a1 - a0) / (np.pi / 2) - 1e-9)))
    out = []
    for k in range(n):
        t0 = a0 + (a1 - a0) * k / n
        t1 = a0 + (a1 - a0) * (k + 1) / n
        kk = 4.0 / 3.0 * math.tan((t1 - t0) / 4.0)
        p0 = c + r * np.array([math.cos(t0), math.sin(t0)])
        p3 = c + r * np.array([math.cos(t1), math.sin(t1)])
        d0 = r * np.array([-math.sin(t0), math.cos(t0)])
        d3 = r * np.array([-math.sin(t1), math.cos(t1)])
        out.append(np.stack([p0, p0 + kk * d0, p3 - kk * d3, p3]))
    return out


def dot_outline(c: Sequence[float], d: float, elong: float = 0.0, angle: float = 0.0) -> np.ndarray:
    """A pen tap: closed cubic outline of a dot of diameter ``d`` (optionally
    drawn out by ``elong`` along ``angle``)."""
    c = np.asarray(c, float)
    if elong > 1e-6:
        u = np.array([math.cos(angle), math.sin(angle)])
        return np.stack(_piece_outline(np.stack([c - 0.5 * elong * u, c + 0.5 * elong * u]),
                                       np.full(2, 0.5 * d), np.stack([u, u])))
    return np.stack(_arc_cubics(c, 0.5 * d, 0.0, -2 * np.pi))


def _chain(pts: np.ndarray, tans: np.ndarray) -> List[np.ndarray]:
    out = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        L = float(np.hypot(*(b - a))) / 3.0
        out.append(np.stack([a, a + tans[i] * L, b - tans[i + 1] * L, b]))
    return out


def _piece_outline(p: np.ndarray, r: np.ndarray, t: np.ndarray) -> List[np.ndarray]:
    """One closed outline (list of cubics) around a polyline with per-vertex
    radius ``r`` and unit tangents ``t``: one side forward, round end cap,
    other side backward, round start cap.  Negative shoelace area in the y
    down glyph frame, like PencilKit's outlines."""
    n = np.column_stack([-t[:, 1], t[:, 0]])
    left = p + n * r[:, None]
    right = p - n * r[:, None]
    segs = _chain(left, t)
    ae = math.atan2(t[-1, 1], t[-1, 0])
    segs += _arc_cubics(p[-1], float(r[-1]), ae + np.pi / 2, ae - np.pi / 2)
    segs += _chain(right[::-1], -t[::-1])
    a0 = math.atan2(t[0, 1], t[0, 0])
    segs += _arc_cubics(p[0], float(r[0]), a0 - np.pi / 2, a0 - 3 * np.pi / 2)
    return segs


def stroke_outline(p: np.ndarray, widths: np.ndarray) -> List[np.ndarray]:
    """Filled outline of a variable-width stroke along polyline ``p`` with
    pen diameters ``widths`` (one per vertex, points roughly uniformly
    spaced).  Returns closed cubic sub-paths (K, 4, 2) whose non-zero union
    is the ink: the polyline is split at corners too sharp for a clean
    inner offset, the pieces meet with round joins (like PencilKit, which
    also exports a stroke as several overlapping sub-paths)."""
    p = np.asarray(p, float)
    w = np.asarray(widths, float)
    keep = np.r_[True, np.hypot(*np.diff(p, axis=0).T) > 1e-9] if len(p) > 1 else np.ones(len(p), bool)
    p, w = p[keep], w[keep]
    r = 0.5 * np.maximum(w, 1e-4)
    if len(p) < 2:
        return [dot_outline(p[0], 2 * r[0])]
    seg = np.diff(p, axis=0)
    ang = np.arctan2(seg[:, 1], seg[:, 0])
    turn = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)  # at interior vertices 1..n-2
    h = np.hypot(seg[:, 0], seg[:, 1])
    hloc = 0.5 * (h[:-1] + h[1:])
    bad = turn * r[1:-1] / np.maximum(hloc, 1e-9) > 0.8  # inner offset would fold over
    cuts = [0] + [i + 1 for i in np.flatnonzero(bad)] + [len(p) - 1]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b <= a:
            continue
        q = p[a:b + 1]
        t = _tangents(q)
        if len(q) == 2:
            t[:] = t[0]
        out.append(np.stack(_piece_outline(q, r[a:b + 1], t)))
    return out


def _smooth01(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def pen_profile(s: np.ndarray, base: float, rng: Optional[np.random.Generator], var: float = 1.0) -> np.ndarray:
    """Pen diameter along a stroke at arclengths ``s`` (x-heights): tapered
    round ends (the end thinner than the start, as the pen lifts) and gentle
    pressure modulation, as measured on the writer's PencilKit strokes."""
    L = float(s[-1]) if len(s) else 0.0
    if rng is None:
        f0, f1, amp, lam, ph, trend = 0.82, 0.58, 0.0, 2.0, 0.0, 0.0
    else:
        f0 = float(np.clip(rng.normal(0.82, 0.1 * var), 0.55, 1.05))
        f1 = float(np.clip(rng.normal(0.58, 0.1 * var), 0.35, 0.9))
        amp = abs(float(rng.normal(0.1, 0.03))) * var
        lam = float(rng.uniform(1.2, 2.6))
        ph = float(rng.uniform(0, 2 * np.pi))
        trend = float(rng.normal(0.0, 0.12)) * var
    ls = min(0.14, 0.3 * L)
    le = min(0.22, 0.35 * L)
    f = np.ones_like(s)
    if ls > 1e-6:
        f *= f0 + (1 - f0) * _smooth01(s / ls)
    if le > 1e-6:
        f *= f1 + (1 - f1) * _smooth01((L - s) / le)
    if L > 1e-6:
        f *= 1.0 + amp * np.sin(2 * np.pi * s / lam + ph) + trend * (s / L - 0.5)
    return base * np.clip(f, 0.35, 1.6)


# ----------------------------------------------------------------------------
# shapes
# ----------------------------------------------------------------------------
def _hershey_strokes(ch: str) -> Optional[List[np.ndarray]]:
    font, key = "futural", ch
    if ch in UNICODE_HERSHEY:
        font, key = UNICODE_HERSHEY[ch]
    elif ch in SYMBOL_HERSHEY:
        font, key = SYMBOL_HERSHEY[ch]
    elif ord(ch) > 126:
        return None  # futural is ASCII only (it maps unknown code points to junk)
    try:
        f = _hershey(font)
        strokes = [np.asarray(s, float) for s in f.strokes_for_text(key) if len(s)]
    except Exception:
        return None
    if not strokes:
        return None
    # Hershey: y down, baseline at BASE, x-height XH units
    return [np.column_stack([(s[:, 0]) / XH, (s[:, 1] - BASE) / XH]) for s in strokes]


def _procedural(ch: str, cap: float = D_CAP) -> Optional[List[np.ndarray]]:
    """Simple constructions (x-height units, y down, baseline 0).

    Operators are designed centred on the axis ``m`` = 0.5 at a canonical
    size (``+`` is 0.8 wide); other symbols use the Hershey design metrics
    (x-height 1, cap ``cap`` = 1.5)."""
    m = 0.5  # maths axis height
    A = np.array
    if ch == "+":
        return [A([[0.0, -m], [0.9, -m]]), A([[0.45, -m - 0.45], [0.45, -m + 0.45]])]
    if ch == "−":
        return [A([[0.05, -m], [0.85, -m]])]
    if ch == "-":
        return [A([[0.05, -m], [0.6, -m]])]
    if ch == "=":
        return [A([[0.05, -m - 0.17], [0.85, -m - 0.17]]), A([[0.05, -m + 0.17], [0.85, -m + 0.17]])]
    if ch in "<>":
        p = A([[0.8, -m - 0.38], [0.1, -m], [0.8, -m + 0.38]])
        return [p if ch == "<" else A([0.9, 0.0]) + p * A([-1.0, 1.0])]
    if ch in "~∼":
        return [_chaikin(A([[0.05, -m + 0.05], [0.25, -m - 0.12], [0.45, -m + 0.0], [0.65, -m + 0.12],
                            [0.85, -m - 0.05]]), 2)]
    if ch == "≃":
        w = _chaikin(A([[0.05, -m - 0.1], [0.25, -m - 0.25], [0.45, -m - 0.1], [0.65, -m + 0.05], [0.85, -m - 0.1]]))
        return [w, A([[0.05, -m + 0.22], [0.85, -m + 0.22]])]
    if ch == "≅":
        w = _chaikin(A([[0.05, -m - 0.36], [0.25, -m - 0.48], [0.45, -m - 0.36], [0.65, -m - 0.24], [0.85, -m - 0.36]]))
        return [w, A([[0.05, -m + 0.02], [0.85, -m + 0.02]]), A([[0.05, -m + 0.36], [0.85, -m + 0.36]])]
    if ch == "∝":
        return [_chaikin(A([[0.95, -m - 0.28], [0.7, -m - 0.25], [0.5, -m], [0.3, -m + 0.25], [0.1, -m + 0.2],
                            [0.05, -m], [0.1, -m - 0.2], [0.3, -m - 0.25], [0.5, -m], [0.7, -m + 0.25],
                            [0.95, -m + 0.28]]), 2)]
    if ch == "•":
        return [_circle(0.22, -m, 0.14, 14)]
    if ch == "⋆":
        a = np.pi / 2 + np.arange(6) * 4 * np.pi / 5
        return [np.column_stack([0.4 + 0.36 * np.cos(a), -m - 0.36 * np.sin(a)])]
    if ch == "×":
        return [A([[0.15, -m - 0.3], [0.75, -m + 0.3]]), A([[0.75, -m - 0.3], [0.15, -m + 0.3]])]
    if ch == "÷":
        return [A([[0.05, -m], [0.85, -m]]), A([[0.45, -m - 0.3]]), A([[0.45, -m + 0.3]])]
    if ch in "±∓":
        if ch == "±":
            return [A([[0.05, -m - 0.05], [0.85, -m - 0.05]]), A([[0.45, -m - 0.45], [0.45, -m + 0.35]]),
                    A([[0.05, -m + 0.45], [0.85, -m + 0.45]])]
        return [A([[0.05, -m - 0.45], [0.85, -m - 0.45]]), A([[0.45, -m - 0.35], [0.45, -m + 0.45]]),
                A([[0.05, -m + 0.05], [0.85, -m + 0.05]])]
    if ch == "·":
        return [A([[0.1, -m]])]
    if ch in "≤≥":
        s = 1 if ch == "≥" else -1
        x0, x1 = (0.1, 0.8) if s == 1 else (0.8, 0.1)
        return [A([[x0, -m - 0.4], [x1, -m - 0.05], [x0, -m + 0.25]]), A([[0.1, -m + 0.45], [0.8, -m + 0.45]])]
    if ch == "≠":
        return [A([[0.05, -m - 0.17], [0.85, -m - 0.17]]), A([[0.05, -m + 0.17], [0.85, -m + 0.17]]),
                A([[0.62, -m - 0.45], [0.28, -m + 0.45]])]
    if ch == "≈":
        w1 = _chaikin(A([[0.05, -m - 0.1], [0.25, -m - 0.22], [0.45, -m - 0.1], [0.65, -m + 0.02], [0.85, -m - 0.1]]))
        return [w1 - A([0.0, 0.1]), w1 + A([0.0, 0.28])]
    if ch == "≡":
        return [A([[0.05, -m - 0.34], [0.85, -m - 0.34]]), A([[0.05, -m], [0.85, -m]]),
                A([[0.05, -m + 0.34], [0.85, -m + 0.34]])]
    if ch in "⟶⟵⟷⟹⟸⟺":
        L = 1.8
        lo = 0.0 if ch in "⟶⟹" else 0.3
        hi = L if ch in "⟵⟸" else L - 0.3
        out = ([A([[0.0, -m], [L, -m]])] if ch in "⟶⟵⟷" else
               [A([[lo * 0.7, -m - 0.12], [L - (L - hi) * 0.7, -m - 0.12]]),
                A([[lo * 0.7, -m + 0.12], [L - (L - hi) * 0.7, -m + 0.12]])])
        if ch not in "⟵⟸":
            out.append(A([[L - 0.3, -m - 0.28], [L, -m], [L - 0.3, -m + 0.28]]))
        if ch not in "⟶⟹":
            out.append(A([[0.3, -m - 0.28], [0.0, -m], [0.3, -m + 0.28]]))
        return out
    if ch in "→⇒←⇐↔⇔↦":
        L = 1.45 if ch in "→⇒←⇐↦" else 1.6
        left, right = ch in "←⇐↔⇔", ch not in "←⇐"
        if ch in "⇒⇐⇔":
            a, b = (0.22 if left else 0.0), (L - 0.22 if right else L)
            out = [A([[a, -m - 0.13], [b, -m - 0.13]]), A([[a, -m + 0.13], [b, -m + 0.13]])]
            hw, hl = 0.34, 0.36
        else:
            out = [A([[0.0, -m], [L, -m]])]
            hw, hl = 0.25, 0.3
        if right:
            out.append(A([[L - hl, -m - hw], [L, -m], [L - hl, -m + hw]]))
        if left:
            out.append(A([[hl, -m - hw], [0.0, -m], [hl, -m + hw]]))
        if ch == "↦":
            out.append(A([[0.0, -m - 0.22], [0.0, -m + 0.22]]))
        return out
    if ch == "↑":
        return [A([[0.3, 0.0], [0.3, -cap]]), A([[0.05, -cap + 0.3], [0.3, -cap], [0.55, -cap + 0.3]])]
    if ch == "↓":
        return [A([[0.3, -cap], [0.3, 0.0]]), A([[0.05, -0.3], [0.3, 0.0], [0.55, -0.3]])]
    if ch == "∞":
        t = np.linspace(0, 2 * np.pi, 40)
        x = 0.58 + 0.56 * np.sin(t)
        y = -m + 0.5 * np.sin(t) * np.cos(t)
        return [np.column_stack([x, y])]
    if ch == "∑":
        h = cap * 1.15
        return [A([[0.95, -h + 0.12], [0.92, -h], [0.0, -h], [0.55, -h / 2], [0.0, 0.0], [0.95, 0.0],
                   [0.98, -0.12]])]
    if ch == "∏":
        h = cap * 1.15
        return [A([[0.0, -h], [1.0, -h]]), A([[0.2, -h], [0.2, 0.0]]), A([[0.8, -h], [0.8, 0.0]])]
    if ch == "∫":
        h = cap * 1.3
        return [_chaikin(A([[0.62, -h + 0.08], [0.5, -h - 0.02], [0.35, -h + 0.15], [0.3, -h / 2],
                            [0.25, 0.25], [0.1, 0.45], [0.0, 0.35]]), 3)]
    if ch == "√":
        return [A([[0.0, -0.45], [0.15, -0.55], [0.4, 0.0], [0.75, -cap]])]
    if ch == "∂":
        return [_chaikin(A([[0.2, -cap + 0.2], [0.5, -cap], [0.75, -cap + 0.35], [0.7, -0.4], [0.45, 0.0],
                            [0.15, -0.15], [0.15, -0.5], [0.45, -0.75], [0.72, -0.6]]), 3)]
    if ch == "∇":
        return [A([[0.0, -cap], [1.0, -cap], [0.5, 0.0], [0.0, -cap]])]
    if ch in "∈∉":
        out = [_chaikin(A([[0.8, -m - 0.4], [0.3, -m - 0.4], [0.1, -m], [0.3, -m + 0.4], [0.8, -m + 0.4]]), 2),
               A([[0.1, -m], [0.7, -m]])]
        if ch == "∉":
            out.append(A([[0.65, -m - 0.6], [0.3, -m + 0.6]]))
        return out
    if ch == "∋":
        return [_chaikin(A([[0.1, -m - 0.4], [0.6, -m - 0.4], [0.8, -m], [0.6, -m + 0.4], [0.1, -m + 0.4]]), 2),
                A([[0.2, -m], [0.8, -m]])]
    if ch in "⊂⊆":
        if ch == "⊆":
            return [_chaikin(A([[0.8, -m - 0.5], [0.3, -m - 0.5], [0.1, -m - 0.15], [0.3, -m + 0.2],
                                [0.8, -m + 0.2]]), 2), A([[0.1, -m + 0.45], [0.8, -m + 0.45]])]
        return [_chaikin(A([[0.8, -m - 0.4], [0.3, -m - 0.4], [0.1, -m], [0.3, -m + 0.4], [0.8, -m + 0.4]]), 2)]
    if ch in "⊃⊇":
        if ch == "⊇":
            return [_chaikin(A([[0.1, -m - 0.5], [0.6, -m - 0.5], [0.8, -m - 0.15], [0.6, -m + 0.2],
                                [0.1, -m + 0.2]]), 2), A([[0.1, -m + 0.45], [0.8, -m + 0.45]])]
        return [_chaikin(A([[0.1, -m - 0.4], [0.6, -m - 0.4], [0.8, -m], [0.6, -m + 0.4], [0.1, -m + 0.4]]), 2)]
    if ch == "∪":
        return [_chaikin(A([[0.05, -0.95], [0.05, -0.35], [0.45, 0.0], [0.85, -0.35], [0.85, -0.95]]), 2)]
    if ch == "∩":
        return [_chaikin(A([[0.05, 0.0], [0.05, -0.6], [0.45, -0.95], [0.85, -0.6], [0.85, 0.0]]), 2)]
    if ch == "∅":
        t = np.linspace(0, 2 * np.pi, 36)
        return [np.column_stack([0.45 + 0.4 * np.cos(t), -0.62 + 0.55 * np.sin(t)]), A([[0.9, -1.3], [0.0, 0.1]])]
    if ch == "∀":
        return [A([[0.0, -cap], [0.45, 0.0], [0.9, -cap]]), A([[0.2, -cap / 2], [0.7, -cap / 2]])]
    if ch == "∃":
        return [A([[0.0, -cap], [0.7, -cap], [0.7, 0.0], [0.0, 0.0]]), A([[0.1, -cap / 2], [0.7, -cap / 2]])]
    if ch == "¬":
        return [A([[0.05, -m - 0.1], [0.8, -m - 0.1], [0.8, -m + 0.2]])]
    if ch == "∧":
        return [A([[0.05, 0.0], [0.45, -0.95], [0.85, 0.0]])]
    if ch == "∨":
        return [A([[0.05, -0.95], [0.45, 0.0], [0.85, -0.95]])]
    if ch == "°":
        return [_circle(0.2, -cap + 0.2, 0.17, 20)]
    if ch == "′":
        return [A([[0.25, -cap], [0.1, -cap + 0.45]])]
    if ch == "″":
        return [A([[0.25, -cap], [0.1, -cap + 0.45]]), A([[0.5, -cap], [0.35, -cap + 0.45]])]
    if ch in "…⋯":
        y = 0.0 if ch == "…" else -m
        return [A([[x, y]]) for x in (0.1, 0.72, 1.34)]
    if ch == "|":
        return [A([[0.1, -cap - 0.15], [0.1, 0.35]])]
    if ch == "‖":
        return [A([[0.1, -cap - 0.15], [0.1, 0.35]]), A([[0.46, -cap - 0.15], [0.46, 0.35]])]
    if ch == "∘":
        return [_circle(0.25, -m, 0.18, 20)]
    # brackets use the Hershey bracket design range (top 1.79, bottom -0.5)
    bt, bb = -D_BRACKET[0], -D_BRACKET[1]
    bm = 0.5 * (bt + bb)
    if ch in "⟨⟩":
        p = A([[0.45, bt], [0.05, bm], [0.45, bb]])
        return [p if ch == "⟨" else A([0.5, 0.0]) + p * A([-1.0, 1.0])]
    if ch in "{}":
        p = _chaikin(A([[0.6, bt], [0.38, bt + 0.08], [0.3, bt + 0.3], [0.3, bm - 0.35], [0.24, bm - 0.1],
                        [0.04, bm], [0.24, bm + 0.1], [0.3, bm + 0.35], [0.3, bb - 0.3], [0.38, bb - 0.08],
                        [0.6, bb]]), 2)
        return [p if ch == "{" else A([0.64, 0.0]) + p * A([-1.0, 1.0])]
    if ch in "⌈⌉⌊⌋":
        p = A([[0.6, bt], [0.15, bt], [0.15, bb]]) if ch in "⌈⌉" else A([[0.15, bt], [0.15, bb], [0.6, bb]])
        return [p if ch in "⌈⌊" else A([0.75, 0.0]) + p * A([-1.0, 1.0])]
    if ch == "θ":
        t = np.linspace(0.5 * np.pi, 2.5 * np.pi, 40)
        return [np.column_stack([0.42 + 0.36 * np.cos(t), -0.75 - 0.75 * np.sin(t)]),
                A([[0.1, -0.78], [0.76, -0.74]])]
    if ch == "✓":
        return [A([[0.0, -0.55], [0.3, 0.0], [1.0, -1.25]])]
    if ch == "ℓ":
        return [_chaikin(A([[0.0, -0.25], [0.4, -0.7], [0.7, -1.2], [0.68, -1.5], [0.42, -1.52], [0.24, -1.05],
                            [0.17, -0.4], [0.3, -0.02], [0.72, -0.15]]), 2)]
    if ch in "†‡":
        out = [A([[0.35, -cap], [0.35, 0.35]]), A([[0.02, -cap + 0.45], [0.68, -cap + 0.45]])]
        if ch == "‡":
            out.append(A([[0.02, -0.2], [0.68, -0.2]]))
        return out
    if ch == "ℏ":
        h = _hershey_strokes("h")
        return None if h is None else h + [A([[0.15, -1.08], [0.75, -1.28]])]
    return None


# Combining accents (Unicode NFD) drawn above / below a base letter.  Design
# coordinates: x centred on 0, y down, y = 0 is the mark's base line.
def _mark(code: int) -> Optional[Tuple[List[np.ndarray], str]]:
    A = np.array
    if code == 0x0301:
        return [A([[-0.07, 0.0], [0.13, -0.22]])], "above"
    if code == 0x0300:
        return [A([[-0.13, -0.22], [0.07, 0.0]])], "above"
    if code == 0x030B:
        return [A([[-0.17, 0.0], [0.0, -0.22]]), A([[0.06, 0.0], [0.23, -0.22]])], "above"
    if code == 0x0302:
        return [A([[-0.17, 0.0], [0.0, -0.2], [0.17, 0.0]])], "above"
    if code == 0x030C:
        return [A([[-0.17, -0.2], [0.0, 0.0], [0.17, -0.2]])], "above"
    if code == 0x0306:
        return [_chaikin(A([[-0.17, -0.2], [-0.1, -0.02], [0.1, -0.02], [0.17, -0.2]]), 2)], "above"
    if code == 0x0308:
        return [A([[-0.15, -0.06]]), A([[0.15, -0.06]])], "above"
    if code == 0x0307:
        return [A([[0.0, -0.06]])], "above"
    if code == 0x0303:
        return [_chaikin(A([[-0.22, -0.02], [-0.11, -0.14], [0.0, -0.08], [0.11, -0.02], [0.22, -0.14]]), 2)], "above"
    if code == 0x0304:
        return [A([[-0.2, -0.08], [0.2, -0.08]])], "above"
    if code == 0x030A:
        return [_circle(0.0, -0.11, 0.1, 16)], "above"
    if code == 0x0327:
        return [_chaikin(A([[0.0, 0.0], [0.04, 0.12], [0.14, 0.2], [0.06, 0.32], [-0.1, 0.3]]), 2)], "below"
    if code == 0x0328:
        return [_chaikin(A([[0.1, 0.0], [-0.02, 0.14], [0.02, 0.3], [0.16, 0.3]]), 2)], "below_right"
    return None


def _decompose(ch: str) -> Optional[Tuple[str, List[int]]]:
    if ch.isascii():
        return None
    d = unicodedata.normalize("NFD", ch)
    if len(d) < 2 or not d[0].isascii() or not d[0].isalpha():
        return None
    marks = [ord(c) for c in d[1:]]
    if not all(_mark(c) is not None for c in marks):
        return None
    return d[0], marks


def _script(ch: str) -> Optional[Tuple[str, str]]:
    """(base, "super" | "sub") for super/subscript characters like ² or ₙ."""
    d = unicodedata.decomposition(ch)
    for tag, kind in (("<super> ", "super"), ("<sub> ", "sub")):
        if d.startswith(tag):
            try:
                return chr(int(d[len(tag):].split()[0], 16)), kind
            except ValueError:
                return None
    return None


def _kind(ch: str) -> str:
    if ch in DOUBLE_STRUCK:
        return "upper"
    if ch in OPERATORS:
        return "op"
    if ch in BIG_OPS:
        return "big"
    if ch in BRACKETS:
        return "bracket"
    if ch.isdigit():
        return "digit"
    if ch.isalpha():
        return "upper" if ch.isupper() else "lower"
    return "punct"


def _is_dot(p: np.ndarray, size: float = 0.2) -> bool:
    return len(p) == 1 or float(np.ptp(p[:, 0])) < size and float(np.ptp(p[:, 1])) < size


def _natural_direction(p: np.ndarray) -> np.ndarray:
    """Draw strokes the way hands do (down strokes downward, horizontals to
    the right) so that the end taper lands at the natural end."""
    if len(p) < 2:
        return p
    d = p[-1] - p[0]
    if np.hypot(*d) < 0.25:
        return p
    if abs(d[1]) >= abs(d[0]):
        return p[::-1] if d[1] < 0 else p  # y down: d[1] < 0 means drawn upward
    return p[::-1] if d[0] < 0 else p


def pen_stroke(points: np.ndarray, pen: float, rng: Optional[np.random.Generator] = None, unit: float = 1.0,
               var: float = 1.0) -> List[np.ndarray]:
    """Filled PencilKit-like ink for one centre-line stroke.

    ``points`` (N, 2) and the target ``pen`` width are in any units whose
    x-height is ``unit`` (1.0 for the glyph frame, the x-height in points for
    page space).  The stroke gets tapered round ends, pressure modulation, a
    random stroke weight and lighter flat strokes (crossbars, bars); a single
    point becomes a pen tap.  The measured pen (ink area / skeleton length)
    is about ``pen``.  Without ``rng`` the result is noise-free.  Returns
    closed cubic sub-paths (K, 4, 2) in the input units, for ``InkBox.fills``.
    """
    p = np.asarray(points, float).reshape(-1, 2) / unit
    pen = pen / unit
    v = var if rng is not None else 0.0
    if len(p) and len(_dedup(p)) == 1:
        p = p[:1]
    if len(p) == 0:
        return []
    if len(p) == 1:
        d = pen * DOT_SIZE * (math.exp(rng.normal(0.0, 0.1 * v)) if v else 1.0)
        el = float(rng.uniform(0.0, 0.5 * pen)) if v else 0.0
        ang = float(rng.uniform(0, np.pi)) if v else 0.0
        return [dot_outline(p[0], d, el, ang) * unit]
    base = pen * PEN_MID * (math.exp(rng.normal(0.0, 0.08 * v)) if v else 1.0)
    # flat strokes (crossbars, operator bars) are quick, light flicks: on the
    # writer's pages they are ~25% thinner than the other strokes
    ext = np.ptp(p, axis=0)
    ang = math.degrees(math.atan2(float(ext[1]), float(ext[0]) + 1e-9))
    base *= FLAT_WEIGHT + (1.0 - FLAT_WEIGHT) * float(_smooth01(np.array((ang - 10.0) / 22.0)))
    p = _natural_direction(p)
    L = float(_arclen(p)[-1])
    q, s = _resample(p, float(np.clip(L / 6.0, 0.02, 0.07)))
    return [sp * unit for sp in stroke_outline(q, pen_profile(s, base, rng if v else None, max(v, 1e-9)))]


def _wmedian(v: Sequence[float], w: Sequence[float]) -> float:
    v = np.asarray(v, float)
    w = np.asarray(w, float)
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, 0.5 * c[-1])])


# ----------------------------------------------------------------------------
class HersheyFallback:
    """Produce a restyled glyph for any character the writer never wrote."""

    def __init__(self, cap_height: float = 1.6, slant: float = 0.15, width_ratio: float = 1.0,
                 pen_width: float = 0.18, ascender: float = 1.7, descender: float = 0.65, *,
                 x_top: float = 1.0, upper_top: Optional[float] = None, digit_top: Optional[float] = None,
                 bracket: Optional[Tuple[float, float]] = None, op_axis: float = 0.5,
                 op_scale: Tuple[float, float] = (1.0, 1.0), group_ratio: Optional[Dict[str, float]] = None,
                 bank: Optional[Dict[str, list]] = None, variation: float = 1.0):
        """All vertical metrics are *ink* extents in x-heights (like the real
        glyphs and ``StyleStats``); ``width_ratio`` scales all widths unless
        a per-class ``group_ratio`` (lower/upper/digit/punct) is given."""
        self.cap = float(cap_height)
        self.slant = float(slant)
        self.width_ratio = float(width_ratio)
        self.pen = float(pen_width)
        self.asc = float(ascender)
        self.desc = float(descender)
        self.x_top = float(x_top)
        self.upper_top = float(upper_top if upper_top is not None else cap_height)
        self.digit_top = float(digit_top if digit_top is not None else cap_height)
        self.bracket = bracket if bracket is not None else (self.cap + 0.12, -min(0.35, 0.5 * self.desc))
        self.op_axis = float(op_axis)
        self.op_scale = (float(op_scale[0]), float(op_scale[1]))
        self.group_ratio = dict(group_ratio or {})
        self.bank = bank or {}
        self.var = float(variation)

    # ------------------------------------------------------------ calibration
    @classmethod
    def for_style(cls, style) -> "HersheyFallback":
        st = style.stats
        bank = {c: gs for c, gs in (getattr(style, "bank", None) or {}).items() if gs}
        pen = float(st.pen_width) if st.pen_width and st.pen_width > 0 else 0.15

        def cmed(chars, attr, min_n=2):
            """Median over classes of the class-median ``attr``, shrunk by count."""
            vals, ws = [], []
            for c in chars:
                gs = bank.get(c, [])
                if len(gs) >= min_n:
                    vals.append(float(np.median([getattr(g, attr) for g in gs])))
                    ws.append(math.log1p(len(gs)))
            return (_wmedian(vals, ws), len(vals)) if vals else (None, 0)

        x_top, _ = cmed(XHEIGHT_REF, "top")
        x_top = float(np.clip(x_top if x_top is not None else 1.0, 0.8, 1.2))
        cap = float(st.cap_height)
        up, nu = cmed([c for c in bank if c.isalpha() and c.isupper()], "top")
        upper_top = cap if up is None else (nu * up + 1.0 * cap) / (nu + 1.0)
        dg, nd = cmed([c for c in bank if c.isdigit()], "top")
        digit_top = cap if dg is None else (nd * dg + 1.0 * cap) / (nd + 1.0)
        bt, _ = cmed("()", "top")
        bb, _ = cmed("()", "bottom")
        bracket = None
        if bt is not None:
            bracket = (float(np.clip(bt, cap, cap + 0.5)), float(np.clip(bb, -0.55, 0.05)))

        # operators: size and axis of the writer's own + = - < >
        sx, sy, axes, nops = [], [], [], 0
        for c in "+=-<>×":
            gs = bank.get(c, [])
            if len(gs) < 3:
                continue
            p = _procedural(c)
            a = np.vstack(p)
            dw = float(np.ptp(a[:, 0]))
            dh = float(np.ptp(a[:, 1]))
            w = float(np.median([g.width for g in gs])) - pen
            h = float(np.median([g.top - g.bottom for g in gs])) - pen
            sx.append(w / dw)
            if dh > 0.3:
                sy.append(h / dh)
            axes.append(float(np.median([0.5 * (g.top + g.bottom) for g in gs])))
            nops += 1
        op_axis, op_scale = 0.5, (1.0, 1.0)
        if nops:
            k = nops / (nops + 1.5)
            mx = float(np.median(sx))
            my = float(np.median(sy)) if sy else mx
            s = math.sqrt(max(mx, 0.05) * max(my, 0.05))
            s = float(np.clip(1.0 + k * (s - 1.0), 0.6, 1.4))
            asp = float(np.clip((mx / my) ** 0.25, 0.85, 1.15)) if sy else 1.0
            op_scale = (s * asp, s / asp)
            op_axis = float(np.clip(0.5 + k * (float(np.median(axes)) - 0.5), 0.4, 0.75))

        slant = float(np.clip(st.slant, -0.4, 0.6))
        fb = cls(cap, slant, 1.0, pen, st.ascender, st.descender, x_top=x_top, upper_top=upper_top,
                 digit_top=digit_top, bracket=bracket, op_axis=op_axis, op_scale=op_scale, bank=bank)
        fb.group_ratio = fb._calibrate_widths()
        return fb

    def _calibrate_widths(self) -> Dict[str, float]:
        """Per-class width ratios so fallback glyphs have the writer's
        proportions: compare the writer's median ink width per character with
        the fallback's own (noise-free) ink width."""
        per: Dict[str, List[Tuple[float, float]]] = {"lower": [], "upper": [], "digit": []}
        for c, gs in self.bank.items():
            k = _kind(c)
            if k not in per or len(gs) < 2 or c in "Iijl1|" or _decompose(c) is not None:
                continue
            sh = self._shape(c)
            if sh is None:
                continue
            w0 = self._ink_width(self._place(sh[0], sh[1], 0.0))
            w1 = self._ink_width(self._place(sh[0], sh[1], 1.0))
            if w1 - w0 < 0.35:
                continue
            real = float(np.median([g.width for g in gs]))
            per[k].append(((real - w0) / (w1 - w0), math.log1p(len(gs))))
        allv = [v for vs in per.values() for v in vs]
        if not allv:
            return {}
        glob = float(np.clip(_wmedian([v for v, _ in allv], [w for _, w in allv]), 0.5, 1.6))
        out = {"punct": glob}
        for k, vs in per.items():
            if vs:
                g = _wmedian([v for v, _ in vs], [w for _, w in vs])
                n = len(vs)
                out[k] = float(np.clip((n * g + 1.5 * glob) / (n + 1.5), 0.5, 1.6))
            else:
                out[k] = glob
        return out

    def _ratio(self, kind: str) -> float:
        if kind in ("lower", "upper", "digit", "punct"):
            return self.group_ratio.get(kind, self.width_ratio)
        if kind == "big":
            return self.group_ratio.get("upper", self.width_ratio)
        if kind == "bracket":
            return self.group_ratio.get("punct", self.width_ratio)
        return self.width_ratio

    def _ink_width(self, strokes: List[np.ndarray]) -> float:
        a = np.vstack(strokes)
        return float(np.ptp(a[:, 0])) + self.pen

    # ------------------------------------------------------------ geometry
    def _shape(self, ch: str) -> Optional[Tuple[List[np.ndarray], str]]:
        """Raw design centre-lines (x-height units, y down) and their kind."""
        if ch in HOMOGLYPHS:
            return self._shape(HOMOGLYPHS[ch])
        if ch in DOUBLE_STRUCK:
            base = self._shape(DOUBLE_STRUCK[ch])
            if base is None:
                return None
            s = base[0]
            xs = np.concatenate([p[:, 0] for p in s])
            stem = min(s, key=lambda p: abs(p[:, 0].mean() - xs.min()) + abs(np.ptp(p[:, 0])))
            # second stem one pen width clear of the first (in design units)
            dx = (1.7 * self.pen + 0.04) / max(0.3, self._ratio("upper"))
            if np.ptp(stem[:, 0]) > 0.25 * np.ptp(stem[:, 1]):  # round letter (C, Q): a bar inside the bowl
                ys = np.concatenate([p[:, 1] for p in s])
                top, bot = ys.min(), min(ys.max(), 0.0)
                stem = np.array([[xs.min(), top + 0.2 * (bot - top)], [xs.min(), bot - 0.2 * (bot - top)]])
                dx *= 1.15
            return s + [stem + np.array([dx, 0.0])], "upper"
        sc = _script(ch)
        if sc is not None:
            base = self._shape(sc[0])
            if base is None:
                return None
            dy = -0.82 if sc[1] == "super" else 0.32
            return [q * SCRIPT_SCALE + np.array([0.0, dy]) for q in base[0]], "punct"
        dec = _decompose(ch)
        if dec is not None:
            return self._accented_shape(*dec)
        if ch in PROCEDURAL:
            p = _procedural(ch)
            if p is not None:
                return [np.asarray(q, float) for q in p], _kind(ch)
        s = _hershey_strokes(ch)
        if s is None:
            p = _procedural(ch)
            return None if p is None else ([np.asarray(q, float) for q in p], _kind(ch))
        return s, _kind(ch)

    def _accented_shape(self, base: str, marks: List[int]) -> Optional[Tuple[List[np.ndarray], str]]:
        sh = self._shape(base)
        if sh is None:
            return None
        s, kind = list(sh[0]), sh[1]
        if base in "ij" and any(_mark(m)[1] == "above" for m in marks):
            s = [p for p in s if not (_is_dot(p, 0.25) and -p[:, 1].min() > 1.1)]  # dotless
        a = np.vstack(s)
        top_d = D_CAP if kind == "upper" else 1.0
        ya, yb = -(top_d + 0.22), 0.0
        x_top = 0.5 * (a[:, 0].min() + a[:, 0].max())
        x_bot = x_top
        for m in marks:
            strokes, where = _mark(m)
            if where == "above":
                s += [q + np.array([x_top, ya]) for q in strokes]
                ya -= 0.32
            else:
                x = a[:, 0].max() - 0.1 if where == "below_right" else x_bot
                s += [q + np.array([x, yb]) for q in strokes]
        return s, kind

    def _vmap(self, y: np.ndarray, kind: str, top_d: float, bot_d: float) -> np.ndarray:
        """Map design heights (y down) to the writer's, placing centre-lines
        so that the *ink* reaches the writer's ink extents."""
        h = 0.5 * self.pen
        yu = -y
        if kind == "lower":
            B, X = h, self.x_top - h
            out = B + yu * (X - B)
            if top_d > 1.15:
                T = (self.asc - h) * (top_d / D_CAP)
                out = np.where(yu > 1.0, X + (yu - 1.0) * (T - X) / (top_d - 1.0), out)
            else:
                out = np.where(yu > 1.0, X + (yu - 1.0) * (X - B), out)
            if bot_d < -0.2:
                D = -(self.desc - h) * (-bot_d / D_DESC)
                out = np.where(yu < 0.0, B + yu / bot_d * (D - B), out)
            return -out
        if kind == "bracket":
            T, Bt = self.bracket[0] - h, self.bracket[1] + h
            k = (T - Bt) / (D_BRACKET[0] - D_BRACKET[1])
            return -(Bt + (yu - D_BRACKET[1]) * k)
        top = {"upper": self.upper_top, "digit": self.digit_top}.get(kind, self.cap)
        k = (top - 2 * h) / D_CAP
        return -(h + yu * k)

    def _place(self, strokes: List[np.ndarray], kind: str, wr: Optional[float] = None, kx: float = 1.0,
               dslant: float = 0.0) -> List[np.ndarray]:
        """Restyled centre-lines: heights, width ratio (times ``kx``), slant
        (plus ``dslant``)."""
        if kind == "op":
            sx, sy = self.op_scale
            out = [np.column_stack([p[:, 0] * sx * kx, -self.op_axis + (p[:, 1] + 0.5) * sy]) for p in strokes]
        else:
            a = np.vstack(strokes)
            top_d, bot_d = float(-a[:, 1].min()), float(-a[:, 1].max())
            r = (self._ratio(kind) if wr is None else wr) * kx
            out = [np.column_stack([p[:, 0] * r, self._vmap(p[:, 1], kind, top_d, bot_d)]) for p in strokes]
        sl = self.slant + dslant
        return [np.column_stack([p[:, 0] - sl * p[:, 1], p[:, 1]]) for p in out]

    @staticmethod
    def _warp(strokes: List[np.ndarray], rng: np.random.Generator, amp: float) -> List[np.ndarray]:
        """A smooth random displacement field over the glyph: every instance
        gets slightly different proportions (bowls, arms, crossings) while
        joined strokes stay joined."""
        comps = []
        for ax in (0, 1):
            for _ in range(2):
                lam = rng.uniform(1.2, 2.6)
                th = rng.uniform(0, 2 * np.pi)
                kvec = 2 * np.pi / lam * np.array([math.cos(th), math.sin(th)])
                comps.append((ax, kvec, rng.uniform(0, 2 * np.pi), rng.normal(0.0, amp)))
        out = []
        for p in strokes:
            q = p.copy()
            for ax, kvec, ph, a in comps:
                q[:, ax] += a * np.sin(p @ kvec + ph)
            out.append(q)
        return out

    def strokes(self, ch: str) -> Optional[List[np.ndarray]]:
        """Restyled noise-free centre-line polylines for ``ch`` (x-height
        units, y down, baseline 0), or None if ``ch`` cannot be drawn."""
        sh = self._shape(ch)
        if sh is None:
            return None
        return self._place(*sh)

    # -------------------------------------------------------------- noise
    def _hand(self, strokes: List[np.ndarray], rng: Optional[np.random.Generator]) -> List[np.ndarray]:
        """Hand-drawn centre-lines: stroke misplacement, rounded corners,
        low-frequency wobble.  Single points stay single points (dots)."""
        v = self.var if rng is not None else 0.0
        multi = sum(not _is_dot(p) for p in strokes) > 1
        out = []
        for p in strokes:
            p = np.asarray(p, float)
            if _is_dot(p, 0.17):
                c = p.mean(axis=0)
                if v:
                    c = c + rng.normal(0.0, 0.02 * v, 2)
                out.append(c[None, :])
                continue
            if v and multi:
                c = p.mean(axis=0)
                th = rng.normal(0.0, 0.03 * v)
                k = math.exp(rng.normal(0.0, 0.03 * v))
                R = k * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
                p = (p - c) @ R.T + c + rng.normal(0.0, 0.018 * v, 2)
            p = _chaikin(_densify(_dedup(p), 0.1), 2)
            if v and len(p) >= 2:
                s = _arclen(p)
                for ax in (0, 1):
                    lam = rng.uniform(0.9, 2.4)
                    amp = rng.normal(0.0, 0.016 * v)
                    p[:, ax] += amp * np.sin(2 * np.pi * s / lam + rng.uniform(0, 2 * np.pi))
            out.append(p)
        return out

    def _ink(self, ch: str, rng: Optional[np.random.Generator]):
        """Hand-drawn centre-lines of ``ch`` (glyph frame, before left
        alignment): per-instance proportions, slant and warp, then ``_hand``."""
        sh = self._shape(ch)
        if sh is None:
            return None
        v = self.var if rng is not None else 0.0
        if not v:
            return self._hand(self._place(*sh), None)
        kx = math.exp(rng.normal(0.0, 0.05 * v))
        ds = float(np.clip(rng.normal(0.0, 0.04 * v), -0.1, 0.1))
        strokes = self._warp(self._place(sh[0], sh[1], kx=kx, dslant=ds), rng, 0.03 * v)
        return self._hand(strokes, rng)

    def _outline(self, p: np.ndarray, rng: Optional[np.random.Generator], weight: float = 1.0) -> List[np.ndarray]:
        return pen_stroke(p, self.pen * weight, rng if self.var else None, var=self.var)

    # ------------------------------------------------------------ public API
    def glyph(self, ch: str, rng: Optional[np.random.Generator] = None):
        """Return ``(lines, width, top, bottom)`` in x-height units, or None.

        ``lines`` is a list of ``(points, pen_width)`` centre-line strokes
        drawn with a round pen of constant width."""
        rng = rng or np.random.default_rng()
        lines = self._ink(ch, rng)
        if not lines:
            return None
        out = []
        for p in lines:
            if len(p) == 1:
                p = np.vstack([p, p + np.array([0.01, 0.0])])
            out.append((p, self.pen))
        allp = np.vstack([p for p, _ in out])
        x0 = allp[:, 0].min() - self.pen / 2
        out = [(p - np.array([x0, 0.0]), w) for p, w in out]
        allp = np.vstack([p for p, _ in out])
        width = float(allp[:, 0].max() + self.pen / 2)
        top = float(-(allp[:, 1].min() - self.pen / 2))
        bottom = float(-(allp[:, 1].max() + self.pen / 2))
        return out, width, top, bottom

    def glyph_ink(self, ch: str, rng: Optional[np.random.Generator] = None):
        """Return ``(fills, lines, width, top, bottom)`` in x-height units, or
        None.  ``fills`` holds the ink as PencilKit-like filled outlines
        (strokes -> closed cubic sub-paths (K, 4, 2)); ``lines`` is empty.
        Metrics follow the real bank glyphs (control-point bounds, x from the
        left ink edge)."""
        rng = rng or np.random.default_rng()
        fills = self._composed(ch, rng)
        if fills is None:
            lines = self._ink(ch, rng)
            if not lines:
                return None
            weight = math.exp(rng.normal(0.0, 0.08 * self.var))  # this glyph's pen pressure
            fills = [self._outline(p, rng, weight) for p in lines]
        fills = [st for st in fills if st]
        if not fills:
            return None
        a = np.vstack([sp.reshape(-1, 2) for st in fills for sp in st])
        x0 = float(a[:, 0].min())
        fills = [[sp - np.array([x0, 0.0]) for sp in st] for st in fills]
        return fills, [], float(a[:, 0].max() - x0), float(-a[:, 1].min()), float(-a[:, 1].max())

    # ------------------------------------------------------- real-part reuse
    def _sample(self, ch: str, rng: np.random.Generator):
        gs = self.bank.get(ch) or []
        if len(gs) < 2:
            return None
        # prefer typical samples: the writer's usual pen weight and a size
        # near the class median
        cand = [g for g in gs if getattr(g, "pen", 0.0) <= 0 or abs(g.pen - self.pen) < 0.35 * self.pen] or gs
        if len(cand) >= 4:
            w = np.array([g.width for g in cand])
            h = np.array([g.top - g.bottom for g in cand])
            d = np.abs(np.log(w / np.median(w))) + np.abs(np.log(np.maximum(h, 1e-3) / np.median(h)))
            cand = [cand[i] for i in np.argsort(d)[:max(2, (len(cand) * 2) // 3)]]
        return cand[int(rng.integers(len(cand)))]

    def _composed(self, ch: str, rng: np.random.Generator) -> Optional[List[List[np.ndarray]]]:
        """Glyphs assembled from the writer's real parts, or None."""
        if ch in HOMOGLYPHS:
            g = self._sample(HOMOGLYPHS[ch], rng)
            return None if g is None else [list(st) for st in g.strokes]
        if ch in ":;":
            top = self._sample(".", rng)
            low = self._sample("," if ch == ";" else ".", rng)
            if top is None or low is None:
                return None
            sh = np.array([0.0, 0.0])
            out = [[sp + sh for sp in st] for st in low.strokes]
            # the upper dot sits ~2/3 x-height up, leaning with the slant
            cy = -(0.62 * self.x_top + rng.normal(0.0, 0.04 * self.var))
            tc = 0.5 * (top.top + top.bottom)
            lx = 0.5 * low.width if ch == ":" else min(0.5 * low.width, 0.18)
            dx = lx - 0.5 * top.width + self.slant * (-cy) * 0.8 + rng.normal(0.0, 0.03 * self.var)
            sh = np.array([dx, cy + tc])
            out += [[sp + sh for sp in st] for st in top.strokes]
            return out
        dec = _decompose(ch)
        if dec is not None:
            base, marks = dec
            g = self._sample(base, rng)
            if g is None:
                return None
            strokes = list(g.strokes)
            above = [m for m in marks if _mark(m)[1] == "above"]
            if base in "ij" and above and len(strokes) > 1:
                # drop the dot: the smallest stroke lying above the x-height
                def bb(st):
                    a = np.vstack([sp.reshape(-1, 2) for sp in st])
                    return a.min(0), a.max(0)
                cands = [st for st in strokes if -bb(st)[1][1] > 0.75 * self.x_top]
                if cands:
                    dot = min(cands, key=lambda st: float(np.ptp(np.vstack([sp.reshape(-1, 2) for sp in st]))))
                    strokes = [st for st in strokes if st is not dot]
            a = np.vstack([sp.reshape(-1, 2) for st in strokes for sp in st])
            upper = base.isupper()
            ref_top = float(-a[:, 1].min())
            ref_top = max(ref_top, self.upper_top if upper else (0.8 if base in "ij" else 1.0) * self.x_top)
            # centre of the ink near the top, so marks follow the slant
            band = a[:, 1] < -0.6 * ref_top
            cx_top = 0.5 * (a[band, 0].min() + a[band, 0].max()) if band.any() else 0.5 * g.width
            out = [list(st) for st in strokes]
            y = -(ref_top + 0.2 + 0.5 * self.pen)
            for m in marks:
                ms, where = _mark(m)
                if where == "above":
                    x = cx_top + self.slant * (-y - ref_top) + 0.04
                    sh = np.array([x, y])
                    y -= 0.32
                else:
                    x = g.width - 0.15 if where == "below_right" else 0.5 * g.width
                    sh = np.array([x - self.slant * 0.1, 0.05])
                pts = [np.column_stack([q[:, 0] - self.slant * q[:, 1], q[:, 1]]) + sh for q in ms]
                out += [self._outline(p, rng) for p in self._hand(pts, rng)]
            return out
        return None
