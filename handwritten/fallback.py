"""Fallback glyphs for characters the writer has never written.

Uses the public-domain Hershey single-stroke fonts (Roman "futural" for
Latin/digits/punctuation, Hershey Greek, and the maths symbol sets), restyled
to the writer: x-height and cap height, width proportions, slant, pen width,
rounded joins and a little hand wobble.  Output is in x-height units with
the same glyph frame as real glyphs (x from 0, baseline at y = 0, y down).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .priors import BASE, UNICODE_HERSHEY, XH, _hershey

# Symbols mapped to (Hershey font, character in that font).  The "symbolic"
# and "mathlow" Hershey sets contain operators; where a symbol is absent it is
# drawn procedurally in ``_procedural``.
SYMBOL_HERSHEY: Dict[str, Tuple[str, str]] = {}

PROCEDURAL = set("×÷±∓·≤≥≠≈≡→←↑↓↔⇒⇐⇔∞∑∏∫√∂∇∈∉⊂⊆⊃⊇∪∩∅∀∃¬∧∨°′″…⋯|ℕℤℚℝℂ∘⟨⟩‖")


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


def _hershey_strokes(ch: str) -> Optional[List[np.ndarray]]:
    font, key = "futural", ch
    if ch in UNICODE_HERSHEY:
        font, key = UNICODE_HERSHEY[ch]
    elif ch in SYMBOL_HERSHEY:
        font, key = SYMBOL_HERSHEY[ch]
    try:
        f = _hershey(font)
        strokes = [np.asarray(s, float) for s in f.strokes_for_text(key) if len(s)]
    except Exception:
        return None
    if not strokes:
        return None
    # Hershey: y down, baseline at BASE, x-height XH units
    return [np.column_stack([(s[:, 0]) / XH, (s[:, 1] - BASE) / XH]) for s in strokes]


def _arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 16) -> np.ndarray:
    t = np.linspace(a0, a1, n)
    return np.column_stack([cx + r * np.cos(t), cy - r * np.sin(t)])


def _procedural(ch: str, cap: float) -> Optional[List[np.ndarray]]:
    """Simple constructions (x-height units, y down, baseline 0)."""
    m = 0.5  # maths axis height
    A = np.array
    if ch == "×":
        return [A([[0.1, -m - 0.32], [0.74, -m + 0.32]]), A([[0.74, -m - 0.32], [0.1, -m + 0.32]])]
    if ch == "÷":
        return [A([[0.05, -m], [0.85, -m]]), A([[0.45, -m - 0.32], [0.46, -m - 0.3]]),
                A([[0.45, -m + 0.3], [0.46, -m + 0.32]])]
    if ch in "±∓":
        s = 1 if ch == "±" else -1
        return [A([[0.05, -m - 0.05 * s], [0.85, -m - 0.05 * s]]), A([[0.45, -m - 0.45], [0.45, -m + 0.35]]),
                A([[0.05, -m + 0.45 * s], [0.85, -m + 0.45 * s]])] if s == 1 else \
            [A([[0.05, -m - 0.45], [0.85, -m - 0.45]]), A([[0.45, -m - 0.35], [0.45, -m + 0.45]]),
             A([[0.05, -m + 0.05], [0.85, -m + 0.05]])]
    if ch == "·":
        return [A([[0.1, -m], [0.12, -m + 0.01]])]
    if ch in "≤≥":
        s = 1 if ch == "≥" else -1
        x0, x1 = (0.1, 0.8) if s == 1 else (0.8, 0.1)
        return [A([[x0, -m - 0.4], [x1, -m], [x0, -m + 0.25]]), A([[0.1, -m + 0.45], [0.8, -m + 0.38]])]
    if ch == "≠":
        return [A([[0.05, -m - 0.17], [0.85, -m - 0.17]]), A([[0.05, -m + 0.17], [0.85, -m + 0.17]]),
                A([[0.65, -m - 0.5], [0.25, -m + 0.5]])]
    if ch == "≈":
        w1 = _chaikin(A([[0.05, -m - 0.1], [0.25, -m - 0.25], [0.45, -m - 0.1], [0.65, -m + 0.05], [0.85, -m - 0.1]]))
        w2 = w1 + A([0.0, 0.32])
        return [w1, w2]
    if ch == "≡":
        return [A([[0.05, -m - 0.3], [0.85, -m - 0.3]]), A([[0.05, -m], [0.85, -m]]),
                A([[0.05, -m + 0.3], [0.85, -m + 0.3]])]
    if ch in "→⇒":
        out = [A([[0.0, -m], [1.2, -m]]), A([[0.9, -m - 0.25], [1.2, -m], [0.9, -m + 0.25]])]
        if ch == "⇒":
            out = [A([[0.0, -m - 0.12], [1.05, -m - 0.12]]), A([[0.0, -m + 0.12], [1.05, -m + 0.12]]),
                   A([[0.85, -m - 0.35], [1.2, -m], [0.85, -m + 0.35]])]
        return out
    if ch in "←⇐":
        return [A([[0.0, -m], [1.2, -m]]), A([[0.3, -m - 0.25], [0.0, -m], [0.3, -m + 0.25]])]
    if ch == "↔" or ch == "⇔":
        return [A([[0.0, -m], [1.3, -m]]), A([[0.3, -m - 0.25], [0.0, -m], [0.3, -m + 0.25]]),
                A([[1.0, -m - 0.25], [1.3, -m], [1.0, -m + 0.25]])]
    if ch == "↑":
        return [A([[0.3, 0.0], [0.3, -cap]]), A([[0.05, -cap + 0.3], [0.3, -cap], [0.55, -cap + 0.3]])]
    if ch == "↓":
        return [A([[0.3, -cap], [0.3, 0.0]]), A([[0.05, -0.3], [0.3, 0.0], [0.55, -0.3]])]
    if ch == "∞":
        t = np.linspace(0, 2 * np.pi, 40)
        x = 0.55 + 0.5 * np.sin(t)
        y = -m + 0.3 * np.sin(t) * np.cos(t)
        return [np.column_stack([x, y])]
    if ch == "∑":
        h = cap * 1.15
        return [A([[0.95, -h + 0.1], [0.95, -h], [0.0, -h], [0.55, -h / 2], [0.0, 0.0], [0.95, 0.0], [0.95, -0.1]])]
    if ch == "∏":
        h = cap * 1.15
        return [A([[0.0, -h], [1.0, -h]]), A([[0.2, -h], [0.2, 0.0]]), A([[0.8, -h], [0.8, 0.0]])]
    if ch == "∫":
        h = cap * 1.3
        return [_chaikin(A([[0.6, -h + 0.05], [0.5, -h - 0.05], [0.35, -h + 0.15], [0.3, -h / 2],
                             [0.25, 0.25], [0.1, 0.45], [0.0, 0.35]]), 3)]
    if ch == "√":
        return [A([[0.0, -0.45], [0.15, -0.55], [0.4, 0.05], [0.75, -cap]])]
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
    if ch in "⊂⊆":
        out = [_chaikin(A([[0.8, -m - 0.4], [0.3, -m - 0.4], [0.1, -m], [0.3, -m + 0.4], [0.8, -m + 0.4]]), 2)]
        if ch == "⊆":
            out.append(A([[0.1, -m + 0.65], [0.8, -m + 0.65]]))
        return out
    if ch in "⊃⊇":
        out = [_chaikin(A([[0.1, -m - 0.4], [0.6, -m - 0.4], [0.8, -m], [0.6, -m + 0.4], [0.1, -m + 0.4]]), 2)]
        if ch == "⊇":
            out.append(A([[0.1, -m + 0.65], [0.8, -m + 0.65]]))
        return out
    if ch == "∪":
        return [_chaikin(A([[0.05, -1.0], [0.05, -0.35], [0.45, 0.0], [0.85, -0.35], [0.85, -1.0]]), 2)]
    if ch == "∩":
        return [_chaikin(A([[0.05, 0.0], [0.05, -0.65], [0.45, -1.0], [0.85, -0.65], [0.85, 0.0]]), 2)]
    if ch == "∅":
        t = np.linspace(0, 2 * np.pi, 36)
        return [np.column_stack([0.45 + 0.4 * np.cos(t), -0.6 + 0.55 * np.sin(t)]), A([[0.95, -1.3], [-0.05, 0.1]])]
    if ch == "∀":
        return [A([[0.0, -cap], [0.45, 0.0], [0.9, -cap]]), A([[0.2, -cap / 2], [0.7, -cap / 2]])]
    if ch == "∃":
        return [A([[0.0, -cap], [0.7, -cap], [0.7, 0.0], [0.0, 0.0]]), A([[0.1, -cap / 2], [0.7, -cap / 2]])]
    if ch == "¬":
        return [A([[0.05, -m - 0.1], [0.8, -m - 0.1], [0.8, -m + 0.2]])]
    if ch == "∧":
        return [A([[0.05, 0.0], [0.45, -1.0], [0.85, 0.0]])]
    if ch == "∨":
        return [A([[0.05, -1.0], [0.45, 0.0], [0.85, -1.0]])]
    if ch == "°":
        t = np.linspace(0, 2 * np.pi, 20)
        return [np.column_stack([0.2 + 0.18 * np.cos(t), -cap + 0.2 + 0.18 * np.sin(t)])]
    if ch == "′":
        return [A([[0.25, -cap], [0.1, -cap + 0.45]])]
    if ch == "″":
        return [A([[0.25, -cap], [0.1, -cap + 0.45]]), A([[0.5, -cap], [0.35, -cap + 0.45]])]
    if ch in "…⋯":
        y = 0.0 if ch == "…" else -m
        return [A([[x, y], [x + 0.02, y + 0.01]]) for x in (0.1, 0.5, 0.9)]
    if ch == "|":
        return [A([[0.1, -cap - 0.15], [0.1, 0.35]])]
    if ch == "‖":
        return [A([[0.1, -cap - 0.15], [0.1, 0.35]]), A([[0.35, -cap - 0.15], [0.35, 0.35]])]
    if ch == "∘":
        t = np.linspace(0, 2 * np.pi, 20)
        return [np.column_stack([0.25 + 0.2 * np.cos(t), -m + 0.2 * np.sin(t)])]
    if ch == "⟨":
        return [A([[0.4, -cap - 0.1], [0.05, -m + 0.1], [0.4, 0.35]])]
    if ch == "⟩":
        return [A([[0.05, -cap - 0.1], [0.4, -m + 0.1], [0.05, 0.35]])]
    return None


DOUBLE_STRUCK = {"ℕ": "N", "ℤ": "Z", "ℚ": "Q", "ℝ": "R", "ℂ": "C"}


class HersheyFallback:
    """Produce a restyled single-stroke glyph for any character."""

    def __init__(self, cap_height: float = 1.6, slant: float = 0.15, width_ratio: float = 1.0,
                 pen_width: float = 0.18, ascender: float = 1.7, descender: float = 0.65):
        self.cap = cap_height
        self.slant = slant
        self.width_ratio = width_ratio
        self.pen = pen_width
        self.asc = ascender
        self.desc = descender

    @classmethod
    def for_style(cls, style) -> "HersheyFallback":
        st = style.stats
        # width ratio: how wide the writer's glyphs are compared with futural
        from .priors import hershey_metrics

        ratios = []
        for c, gs in style.bank.items():
            m = hershey_metrics(c)
            if m is None or m[2] < 0.3 or len(gs) < 3:
                continue
            ratios.append(float(np.median([g.width for g in gs])) / m[2])
        wr = float(np.clip(np.median(ratios), 0.6, 1.6)) if ratios else 1.0
        return cls(st.cap_height, st.slant, wr, st.pen_width * 0.95, st.ascender, st.descender)

    def strokes(self, ch: str) -> Optional[List[np.ndarray]]:
        """Centre-line polylines for ``ch`` in x-height units (unstyled)."""
        if ch in DOUBLE_STRUCK:
            base = self.strokes(DOUBLE_STRUCK[ch])
            if base is None:
                return None
            # add a second vertical stroke next to the first stem
            xs = np.concatenate([s[:, 0] for s in base])
            stem = min(base, key=lambda s: abs(s[:, 0].mean() - xs.min()) + abs(np.ptp(s[:, 0])))
            return base + [stem + np.array([0.18, 0.0])]
        if ch in PROCEDURAL:
            p = _procedural(ch, self.cap)
            if p is not None:
                return p
        s = _hershey_strokes(ch)
        if s is None:
            p = _procedural(ch, self.cap)
            return p
        # vertical restyle: map Hershey cap (1.5) / ascender to the writer's
        allp = np.vstack(s)
        top = -allp[:, 1].min()
        if top > 1.15:
            target = self.cap if (ch.isupper() or ch.isdigit() or not ch.isalpha()) else self.asc
            k = target / top
            s = [np.column_stack([p[:, 0], np.where(p[:, 1] < -1.0, -1.0 + (p[:, 1] + 1.0) * (target - 1.0) / (top - 1.0), p[:, 1])])
                 if ch.islower() else p * np.array([1.0, k]) for p in s]
            if not ch.islower():
                s = [p * np.array([k ** 0.5 / 1.0, 1.0]) for p in s]
        bottom = allp[:, 1].max()
        if bottom > 0.3 and ch.islower():
            k = self.desc / bottom
            s = [np.column_stack([p[:, 0], np.where(p[:, 1] > 0, p[:, 1] * k, p[:, 1])]) for p in s]
        return s

    def glyph(self, ch: str, rng: Optional[np.random.Generator] = None):
        """Return ``(lines, width, top, bottom)`` in x-height units, or None.

        ``lines`` is a list of ``(points, pen_width)``."""
        s = self.strokes(ch)
        if not s:
            return None
        rng = rng or np.random.default_rng()
        out = []
        for p in s:
            p = np.asarray(p, float)
            if len(p) == 1:
                p = np.vstack([p, p + 0.01])
            p = p * np.array([self.width_ratio, 1.0])
            # slant: shift x by slant * height above baseline (y is negative up)
            p = np.column_stack([p[:, 0] - self.slant * p[:, 1], p[:, 1]])
            if len(p) >= 3:
                p = _chaikin(p, 2)
            # low-frequency wobble
            n = len(p)
            if n >= 2:
                t = np.linspace(0, 1, n)
                ph = rng.uniform(0, 2 * np.pi, size=2)
                amp = rng.normal(0, 0.012, size=2)
                p = p + np.column_stack([amp[0] * np.sin(2 * np.pi * t + ph[0]), amp[1] * np.sin(2 * np.pi * t + ph[1])])
            out.append((p, self.pen))
        allp = np.vstack([p for p, _ in out])
        x0 = allp[:, 0].min() - self.pen / 2
        out = [(p - np.array([x0, 0.0]), w) for p, w in out]
        allp = np.vstack([p for p, _ in out])
        width = float(allp[:, 0].max() + self.pen / 2)
        top = float(-(allp[:, 1].min() - self.pen / 2))
        bottom = float(-(allp[:, 1].max() + self.pen / 2))
        return out, width, top, bottom
