"""Turn text into handwriting ``InkBox`` objects using a trained ``Style``.

For every character the synthesiser picks one of the writer's real samples
(avoiding recent repeats), applies small random variation learned from the
data (size, baseline, rotation, shear), and spaces letters with the
writer's pair-specific letter gaps.  Characters the writer never wrote come
from, in order: a case/shape analogue in the writer's own bank (``C`` from
``c``, ``0`` from ``O`` ...), then a restyled Hershey single-stroke glyph.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .box import InkBox
from .fallback import HersheyFallback
from .geometry import compose, rotate, scale, shear_x, translate
from .style import BankGlyph, Style

# char -> list of (substitute, mode); mode "cap" scales to cap height,
# "xh" scales to x-height, "same" keeps size.
ANALOGS: Dict[str, List[Tuple[str, str]]] = {
    # Scaling a real outline also scales its pen width, so analogues are only
    # used at (nearly) the same size or smaller; lowercase -> capital
    # substitutes would look bold and fall through to the Hershey fallback.
    "O": [("0", "same")], "0": [("O", "same")], "K": [("k", "same")], "k": [("K", "same")],
    "c": [("C", "xh")], "o": [("O", "xh"), ("0", "xh")], "s": [("S", "xh")], "v": [("V", "xh")],
    "w": [("W", "xh")], "x": [("X", "xh")], "z": [("Z", "xh")], "u": [("U", "xh")],
    "l": [("1", "same"), ("|", "same")], "1": [("l", "same")], "|": [("l", "same"), ("1", "same")],
    "×": [("x", "same")], "−": [("-", "same")], "–": [("-", "same")],
    "—": [("-", "wide")], "’": [("'", "same")], "‘": [("'", "same")], "“": [('"', "same")],
    "”": [('"', "same")], "′": [("'", "same")],
}

PUNCT_AFTER = set(",.;:!?)]}'\"’”")


@dataclass
class GlyphChoice:
    fills: List[List[np.ndarray]]  # x-height units, glyph frame
    lines: List[Tuple[np.ndarray, float]]
    width: float
    top: float
    bottom: float
    kind: str  # "real" | "analog" | "fallback" | "missing"


class Synthesizer:
    def __init__(self, style: Style, seed: Optional[int] = 0, x_height: Optional[float] = None,
                 variation: float = 1.0, neatness: float = 0.0):
        """
        x_height: target x-height in points (default: the writer's natural size)
        variation: multiplier on all learned random variation (0 = none)
        neatness: 0..1, shrinks spacing and baseline variability toward the mean
        """
        self.style = style
        self.st = style.stats
        self.rng = np.random.default_rng(seed)
        self.xh = float(x_height or self.st.xh_pt)
        self.var = float(variation)
        self.neat = float(np.clip(neatness, 0.0, 1.0))
        self.fallback = HersheyFallback.for_style(style)
        self._recent: Dict[str, Deque[int]] = defaultdict(lambda: deque(maxlen=3))
        self.missing: Dict[str, int] = defaultdict(int)
        self.used_kind: Dict[str, str] = {}

    # ------------------------------------------------------------ glyphs
    def _pick(self, ch: str) -> Optional[BankGlyph]:
        gs = self.style.bank.get(ch)
        if not gs:
            return None
        if len(gs) == 1:
            return gs[0]
        recent = self._recent[ch]
        cand = [g for g in gs if g.gid not in recent] or gs
        # A real pen has a constant width; samples written at a different size
        # look bolder/thinner once normalised, so prefer typical pen weights.
        target = self.st.pen_width
        if target > 0 and any(g.pen > 0 for g in cand):
            w = np.array([math.exp(-0.5 * ((g.pen - target) / (0.12 * target)) ** 2) if g.pen > 0 else 0.3
                          for g in cand])
            w = w + 1e-3
            g = cand[int(self.rng.choice(len(cand), p=w / w.sum()))]
        else:
            g = cand[int(self.rng.integers(len(cand)))]
        recent.append(g.gid)
        return g

    def choose(self, ch: str) -> GlyphChoice:
        g = self._pick(ch)
        if g is not None:
            self.used_kind.setdefault(ch, "real")
            return GlyphChoice(g.strokes, [], g.width, g.top, g.bottom, "real")
        for sub, mode in ANALOGS.get(ch, []):
            g = self._pick(sub)
            if g is None:
                continue
            if mode == "cap":
                k = self.st.cap_height / max(0.3, g.top)
            elif mode == "xh":
                k = 1.0 / max(0.3, g.top)
            else:
                k = 1.0
            if mode == "wide":
                m = scale(1.8, 1.0)
            else:
                m = scale(k)
            fills = [[(sp.reshape(-1, 2) @ m[:, :2].T).reshape(sp.shape) for sp in st] for st in g.strokes]
            self.used_kind.setdefault(ch, "analog")
            w = g.width * (1.8 if mode == "wide" else k)
            return GlyphChoice(fills, [], w, g.top * (1.0 if mode == "wide" else k),
                               g.bottom * (1.0 if mode == "wide" else k), "analog")
        fb = self.fallback.glyph(ch, self.rng)
        if fb is not None:
            lines, w, top, bottom = fb
            self.used_kind.setdefault(ch, "fallback")
            return GlyphChoice([], lines, w, top, bottom, "fallback")
        self.missing[ch] += 1
        return GlyphChoice([], [], 0.6, 0.0, 0.0, "missing")

    def glyph_box(self, ch: str, size: float = 1.0) -> InkBox:
        """One character as an InkBox in points (x-height = self.xh * size)."""
        c = self.choose(ch)
        v = self.var
        s = self.xh * size * math.exp(self.rng.normal(0.0, self.st.glyph_scale["sd"] * 0.7 * v))
        rot = self.rng.normal(0.0, 0.025 * v)
        sh = self.rng.normal(0.0, 0.03 * v)
        dy = self.rng.normal(0.0, self.st.glyph_baseline["sd"] * v * (1 - 0.5 * self.neat))
        cx = 0.5 * c.width
        # rotate / shear about the glyph's baseline centre, then scale to points
        m = compose(scale(s), translate(cx, -dy), rotate(rot), shear_x(sh), translate(-cx, 0.0))
        fills = [[_tf(sp, m) for sp in st] for st in c.fills]
        lines = [(_tfp(p, m), w * s) for p, w in c.lines]
        box = InkBox(fills, lines, c.width * s, max(0.0, c.top * s), max(0.0, -c.bottom * s), ch)
        box.refit(keep_width=True)
        return box

    # -------------------------------------------------------------- spacing
    def letter_gap(self, a: str, b: str) -> float:
        """Gap between two letters, in x-heights."""
        st = self.st
        g_mean, g_sd = st.letter_gap["mean"], st.letter_gap["sd"]
        pair = st.letter_gap_pair.get(a + b)
        if b in PUNCT_AFTER and (pair is None or pair[2] < 3):
            mean, sd = st.punct_gap["mean"], st.punct_gap["sd"]
        elif pair is not None and pair[2] >= 3:
            w = pair[2] / (pair[2] + 4.0)
            mean = w * pair[0] + (1 - w) * g_mean
            sd = w * pair[1] + (1 - w) * g_sd
        else:
            mean = g_mean
            for table, key in ((st.letter_gap_left, a), (st.letter_gap_right, b)):
                e = table.get(key)
                if e is not None:
                    w = e[2] / (e[2] + 6.0)
                    mean += w * (e[0] - g_mean)
            sd = g_sd
        sd *= 0.55 * self.var * (1 - 0.6 * self.neat)
        x = self.rng.normal(mean, sd)
        return float(np.clip(x, mean - 2 * sd - 0.02, mean + 2 * sd + 0.02))

    def word_gap(self, size: float = 1.0) -> float:
        st = self.st
        sd = min(st.word_gap["sd"], 0.45) * 0.6 * self.var * (1 - 0.6 * self.neat)
        mean = float(np.clip(st.word_gap["mean"], 0.8, 2.6))
        x = self.rng.normal(mean, sd)
        return float(np.clip(x, 0.6 * mean, 1.6 * mean)) * self.xh * size

    # ---------------------------------------------------------------- words
    def word(self, text: str, size: float = 1.0) -> InkBox:
        """Render a word (no spaces) with per-word size and baseline jitter."""
        if not text:
            return InkBox(width=0.0)
        v = self.var * (1 - 0.6 * self.neat)
        wscale = size * math.exp(self.rng.normal(0.0, self.st.word_scale["sd"] * 0.6 * v))
        wdy = self.rng.normal(0.0, self.st.word_baseline["sd"] * 0.8 * v) * self.xh
        out = InkBox(text=text)
        x = 0.0
        prev: Optional[str] = None
        for ch in text:
            if prev is not None:
                x += self.letter_gap(prev, ch) * self.xh * wscale
            gb = self.glyph_box(ch, wscale)
            out.add(gb, x, wdy)
            x += gb.width
            prev = ch
        out.width = x
        out.refit(keep_width=True)
        return out


def _tf(sp: np.ndarray, m: np.ndarray) -> np.ndarray:
    pts = sp.reshape(-1, 2) @ m[:, :2].T + m[:, 2]
    return pts.reshape(sp.shape)


def _tfp(p: np.ndarray, m: np.ndarray) -> np.ndarray:
    return p @ m[:, :2].T + m[:, 2]
