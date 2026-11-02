"""The trained handwriting style model.

A ``Style`` holds

* a bank of the writer's real glyphs (outline strokes in a normalised glyph
  frame: x from the left ink edge, baseline at y = 0, y down, units = points
  at the writer's natural size), and
* statistics learned from how the writer lays text out: x-height, ascender
  and cap height, letter spacing per character pair (with back-off), word
  spacing, line spacing, baseline slope and drift, per-word size jitter,
  slant, pen width and ink colour.

It is saved as a ``.hwstyle`` file: a zip archive with ``style.json`` and
``glyphs.npz``.
"""
from __future__ import annotations

import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import rasterize
from .priors import DESCENDERS, XHEIGHT_CHARS
from .segment import BASELINE_CHARS, Glyph, Segmentation

FORMAT_VERSION = 1


@dataclass
class BankGlyph:
    """One real glyph sample, normalised to x-height 1.0 (units of x-height)."""

    char: str
    strokes: List[List[np.ndarray]]  # strokes -> subpaths (K,4,2), units of x-height
    width: float
    top: float  # positive up, x-heights
    bottom: float
    source: str = ""  # "file:page" for provenance
    gid: int = -1  # stable id within the bank
    pen: float = 0.0  # measured pen width (x-heights); 0 = unknown

    @property
    def n_segments(self) -> int:
        return sum(len(sp) for st in self.strokes for sp in st)


@dataclass
class Dist:
    """A 1-D distribution summary (robust)."""

    mean: float
    sd: float
    n: int = 0
    lo: float = -math.inf
    hi: float = math.inf

    @staticmethod
    def fit(values: Sequence[float], default: Tuple[float, float] = (0.0, 0.1), min_sd: float = 1e-3,
            clip_q: Tuple[float, float] = (0.02, 0.98)) -> "Dist":
        v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
        if len(v) < 3:
            return Dist(default[0], default[1], len(v))
        lo, hi = np.quantile(v, clip_q)
        vv = v[(v >= lo) & (v <= hi)]
        med = float(np.median(vv))
        sd = float(1.4826 * np.median(np.abs(vv - med)))
        return Dist(med, max(sd, min_sd), len(v), float(lo), float(hi))

    def sample(self, rng: np.random.Generator, scale: float = 1.0) -> float:
        x = rng.normal(self.mean, self.sd * scale)
        return float(np.clip(x, self.lo, self.hi)) if np.isfinite(self.lo) else float(x)


@dataclass
class StyleStats:
    xh_pt: float = 9.0  # natural x-height in points
    ascender: float = 1.7  # in x-heights
    cap_height: float = 1.65
    descender: float = 0.65
    slant: float = 0.0  # dx/dy of upright strokes (positive leans right)
    pen_width: float = 0.15  # in x-heights
    ink: Tuple[float, float, float] = (0.118, 0.106, 0.106)
    letter_gap: Dict[str, float] = field(default_factory=lambda: {"mean": 0.12, "sd": 0.08})
    letter_gap_left: Dict[str, Tuple[float, float, int]] = field(default_factory=dict)
    letter_gap_right: Dict[str, Tuple[float, float, int]] = field(default_factory=dict)
    letter_gap_pair: Dict[str, Tuple[float, float, int]] = field(default_factory=dict)
    word_gap: Dict[str, float] = field(default_factory=lambda: {"mean": 1.4, "sd": 0.35})
    punct_gap: Dict[str, float] = field(default_factory=lambda: {"mean": 0.15, "sd": 0.1})
    line_spacing: Dict[str, float] = field(default_factory=lambda: {"mean": 3.2, "sd": 0.25})
    line_slope: Dict[str, float] = field(default_factory=lambda: {"mean": 0.0, "sd": 0.01})
    word_baseline: Dict[str, float] = field(default_factory=lambda: {"mean": 0.0, "sd": 0.06})
    word_scale: Dict[str, float] = field(default_factory=lambda: {"mean": 0.0, "sd": 0.06})  # log scale
    glyph_scale: Dict[str, float] = field(default_factory=lambda: {"mean": 0.0, "sd": 0.05})  # log
    glyph_baseline: Dict[str, float] = field(default_factory=lambda: {"mean": 0.0, "sd": 0.05})
    line_start: Dict[str, float] = field(default_factory=lambda: {"mean": 75.0, "sd": 4.0})  # pt
    line_end: Dict[str, float] = field(default_factory=lambda: {"mean": 540.0, "sd": 15.0})  # pt
    page_top: float = 72.0
    char_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class Style:
    stats: StyleStats
    bank: Dict[str, List[BankGlyph]]
    meta: Dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        glyphs = [g for gs in self.bank.values() for g in gs]
        seg_arrays: List[np.ndarray] = []
        sub_off = [0]  # subpath -> segment offsets
        stroke_off = [0]  # stroke -> subpath offsets
        glyph_off = [0]  # glyph -> stroke offsets
        for g in glyphs:
            for st in g.strokes:
                for sp in st:
                    seg_arrays.append(sp.astype(np.float32))
                    sub_off.append(sub_off[-1] + len(sp))
                stroke_off.append(len(sub_off) - 1)
            glyph_off.append(len(stroke_off) - 1)
        segs = np.concatenate(seg_arrays) if seg_arrays else np.zeros((0, 4, 2), np.float32)
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            segs=segs.astype(np.float16),
            sub_off=np.asarray(sub_off, np.int64),
            stroke_off=np.asarray(stroke_off, np.int64),
            glyph_off=np.asarray(glyph_off, np.int64),
        )
        info = [{"char": g.char, "width": g.width, "top": g.top, "bottom": g.bottom, "source": g.source,
                 "gid": g.gid, "pen": g.pen} for g in glyphs]
        doc = {"format": FORMAT_VERSION, "meta": self.meta, "stats": asdict(self.stats), "glyphs": info}
        with zipfile.ZipFile(str(path), "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("style.json", json.dumps(doc, indent=1, default=float))
            z.writestr("glyphs.npz", buf.getvalue())

    @staticmethod
    def load(path: str | Path) -> "Style":
        with zipfile.ZipFile(str(path)) as z:
            doc = json.loads(z.read("style.json"))
            arr = np.load(io.BytesIO(z.read("glyphs.npz")))
            segs = arr["segs"].astype(np.float64)
            sub_off, stroke_off, glyph_off = arr["sub_off"], arr["stroke_off"], arr["glyph_off"]
        if doc.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported style format {doc.get('format')}")
        st = doc["stats"]
        st["ink"] = tuple(st["ink"])
        for k in ("letter_gap_left", "letter_gap_right", "letter_gap_pair"):
            st[k] = {kk: tuple(v) for kk, v in st[k].items()}
        stats = StyleStats(**st)
        bank: Dict[str, List[BankGlyph]] = defaultdict(list)
        for gi, info in enumerate(doc["glyphs"]):
            strokes = []
            for si in range(glyph_off[gi], glyph_off[gi + 1]):
                sps = []
                for pi in range(stroke_off[si], stroke_off[si + 1]):
                    sps.append(segs[sub_off[pi]:sub_off[pi + 1]])
                strokes.append(sps)
            bank[info["char"]].append(BankGlyph(info["char"], strokes, info["width"], info["top"], info["bottom"],
                                                info.get("source", ""), info.get("gid", gi), info.get("pen", 0.0)))
        return Style(stats=stats, bank=dict(bank), meta=doc.get("meta", {}))

    # ------------------------------------------------------------- helpers
    def coverage(self) -> Dict[str, int]:
        return {c: len(v) for c, v in sorted(self.bank.items())}

    def merge(self, other: "Style", weight_other_stats: float = 0.0) -> "Style":
        """Add another style's glyphs (e.g. from a calibration sheet)."""
        bank = {c: list(v) for c, v in self.bank.items()}
        next_id = 1 + max([g.gid for gs in bank.values() for g in gs] + [0])
        for c, gs in other.bank.items():
            for g in gs:
                bank.setdefault(c, []).append(BankGlyph(g.char, g.strokes, g.width, g.top, g.bottom, g.source,
                                                        next_id, g.pen))
                next_id += 1
        stats = self.stats
        stats.char_counts = {c: len(v) for c, v in bank.items()}
        return Style(stats=stats, bank=bank, meta=dict(self.meta))


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
def _glyph_to_bank(g: Glyph, gid: int) -> BankGlyph:
    s = 1.0 / g.xh
    strokes = [[sp * s for sp in st] for st in g.strokes]
    return BankGlyph(g.char, strokes, g.width * s, g.top * s, g.bottom * s, f"{g.source}:{g.page + 1}", gid)


def glyph_pen_width(g: BankGlyph, res: float = 60.0) -> float:
    """Pen width of one glyph in x-heights (ink area / skeleton length)."""
    from skimage.morphology import skeletonize

    box = (-0.1, -g.top - 0.1, g.width + 0.1, -g.bottom + 0.1)
    m = rasterize(g.strokes, box, res, n=4)
    sk = int(skeletonize(m).sum())
    return float(m.sum() / sk / res) if sk > 5 else 0.0


def estimate_pen_width(glyphs: Sequence[BankGlyph], n: int = 150, seed: int = 0) -> float:
    """Mean pen width in x-heights: ink area / skeleton length."""
    from skimage.morphology import skeletonize

    rng = np.random.default_rng(seed)
    cand = [g for g in glyphs if g.char in "acemnorsuvwxz"] or list(glyphs)
    if not cand:
        return 0.15
    pick = rng.choice(len(cand), size=min(n, len(cand)), replace=False)
    ws = []
    res = 60.0  # px per x-height
    for i in pick:
        g = cand[int(i)]
        box = (-0.1, -g.top - 0.1, g.width + 0.1, -g.bottom + 0.1)
        m = rasterize(g.strokes, box, res, n=4)
        area = m.sum()
        sk = skeletonize(m).sum()
        if sk > 5:
            ws.append(area / sk / res)
    return float(np.median(ws)) if ws else 0.15


def estimate_slant(glyphs: Sequence[BankGlyph]) -> float:
    """Average lean of upright single-stroke glyphs (l, 1, |), dx/dy (right = +)."""
    vals = []
    for g in glyphs:
        if g.char not in "l1|" or len(g.strokes) != 1:
            continue
        pts = np.vstack([sp.reshape(-1, 2) for sp in g.strokes[0]])
        x = pts[:, 0] - pts[:, 0].mean()
        y = pts[:, 1] - pts[:, 1].mean()
        vy = (y * y).mean()
        if vy < 1e-6:
            continue
        # x = k * y regression; y down so a right lean has x decreasing with y
        k = (x * y).mean() / vy
        vals.append(-k)
    return float(np.median(vals)) if vals else 0.0


def _pair_stats(values: Dict[str, List[float]], min_n: int) -> Dict[str, Tuple[float, float, int]]:
    out = {}
    for k, v in values.items():
        if len(v) >= min_n:
            a = np.asarray(v)
            med = float(np.median(a))
            sd = float(1.4826 * np.median(np.abs(a - med))) if len(a) >= 3 else 0.08
            out[k] = (med, max(sd, 0.02), len(a))
    return out


DOTTED = set("ij!?;:%\"'")


def _bbox_of(strokes) -> Tuple[float, float, float, float]:
    pts = np.vstack([sp.reshape(-1, 2) for st in strokes for sp in st])
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def _refit_bank_glyph(g: BankGlyph) -> None:
    x0, y0, x1, y1 = _bbox_of(g.strokes)
    if abs(x0) > 1e-6:
        g.strokes = [[sp - np.array([x0, 0.0]) for sp in st] for st in g.strokes]
    g.width, g.top, g.bottom = x1 - x0, -y0, -y1


def _strip_stray_marks(bank: Dict[str, List[BankGlyph]]) -> None:
    """Remove tiny detached marks (a neighbour's i-dot, pen taps) from glyphs
    of characters that have no dots."""
    for c, gs in bank.items():
        if c in DOTTED or not c.isalnum():
            continue
        for g in gs:
            if len(g.strokes) < 2:
                continue
            boxes = [_bbox_of([st]) for st in g.strokes]
            sizes = [max(b[2] - b[0], b[3] - b[1]) for b in boxes]
            big = max(sizes)
            keep = []
            for st, b, sz in zip(g.strokes, boxes, sizes):
                if sz < 0.22 and sz < 0.3 * big:
                    # detached? (does not touch any other stroke's bbox)
                    touches = any(not (b[2] < o[0] - 0.05 or b[0] > o[2] + 0.05 or b[3] < o[1] - 0.05 or
                                       b[1] > o[3] + 0.05) for o in boxes if o is not b)
                    if not touches:
                        continue
                keep.append(st)
            if keep and len(keep) < len(g.strokes):
                g.strokes = keep
                _refit_bank_glyph(g)


def _bank_feature(g: BankGlyph, res: float = 8.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    W, H = 2.6, 3.0
    cx = 0.5 * g.width
    m = rasterize(g.strokes, (cx - W / 2, -2.0, cx + W / 2, 1.0), res, n=3).astype(np.float32)
    f = gaussian_filter(m, 1.0).ravel()
    n = np.linalg.norm(f)
    return f / n if n > 1e-9 else f


def _prune_outliers(bank: Dict[str, List[BankGlyph]], z: float = 3.0, min_n: int = 8) -> None:
    """Drop samples that do not look like the rest of their class.

    Two robust tests per class (after size normalisation):
    * local outlier: mean distance to the 3 nearest same-class samples;
    * shape proportion: log aspect ratio (width / height).
    """
    for c, gs in list(bank.items()):
        if len(gs) < min_n:
            continue
        F = np.stack([_bank_feature(g) for g in gs])
        D = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (F @ F.T)))
        np.fill_diagonal(D, np.inf)
        k = min(3, len(gs) - 1)
        knn = np.sort(D, axis=1)[:, :k].mean(axis=1)
        asp = np.log(np.array([max(g.width, 0.05) / max(g.top - g.bottom, 0.05) for g in gs]))
        keep = np.ones(len(gs), bool)
        for v in (knn, asp):
            med = float(np.median(v))
            mad = 1.4826 * float(np.median(np.abs(v - med))) + 1e-6
            keep &= np.abs(v - med) / mad <= (z if v is knn else z + 0.5)
        bank[c] = [g for g, kk in zip(gs, keep) if kk]


NO_SIZE_NORM = set(".,'`-_~\"=:;")


def _normalise_sizes(bank: Dict[str, List[BankGlyph]], shrink: float) -> None:
    """Pull each glyph's height toward its class median (by ``shrink`` in log
    space).  Removes size errors inherited from mis-measured lines while
    keeping some natural variation."""
    if shrink <= 0:
        return
    for c, gs in bank.items():
        if c in NO_SIZE_NORM or len(gs) < 3:
            continue
        h = np.array([g.top - g.bottom for g in gs])
        med = float(np.median(h))
        for g, hh in zip(gs, h):
            if hh <= 1e-3:
                continue
            k = float(np.clip((med / hh) ** shrink, 0.7, 1.43))
            if abs(k - 1.0) < 1e-3:
                continue
            g.strokes = [[sp * k for sp in st] for st in g.strokes]
            g.width *= k
            g.top *= k
            g.bottom *= k
        # vertical re-anchoring: pull each glyph's bottom (baseline letters) or
        # top (descender letters) toward the class median
        on_base = c in BASELINE_CHARS and c not in DESCENDERS
        ref = np.array([g.bottom if on_base else g.top for g in gs])
        mref = float(np.median(ref))
        for g, r in zip(gs, ref):
            dy = (mref - r) * shrink  # positive = move up
            if abs(dy) < 1e-4:
                continue
            g.strokes = [[sp + np.array([0.0, -dy]) for sp in st] for st in g.strokes]
            g.top += dy
            g.bottom += dy


def _gate_sparse_classes(bank: Dict[str, List[BankGlyph]], min_samples: int) -> List[str]:
    """Drop classes with too few samples to trust their OCR label.

    Labels from free text are noisy, so a character seen only once is
    usually a mis-recognition; such characters fall back to analogues or
    Hershey glyphs instead (a calibration sheet supplies trusted samples)."""
    dropped = []
    for c in list(bank):
        if len(bank[c]) < min_samples:
            dropped.append(c)
            del bank[c]
    return dropped


def train_style(seg: Segmentation, ink: Optional[Tuple[float, float, float]] = None,
                meta: Optional[Dict[str, object]] = None, min_samples: int = 2,
                size_shrink: float = 0.8) -> Style:
    glyphs = [g for g in seg.glyphs if g.ok]
    if not glyphs:
        raise ValueError("no usable glyphs were found in the training documents")
    bank: Dict[str, List[BankGlyph]] = defaultdict(list)
    for gid, g in enumerate(glyphs):
        bank[g.char].append(_glyph_to_bank(g, gid))
    _strip_stray_marks(bank)
    dropped = _gate_sparse_classes(bank, min_samples)
    _normalise_sizes(bank, size_shrink)
    _prune_outliers(bank)
    allb = [b for v in bank.values() for b in v]
    for b in allb:
        b.pen = glyph_pen_width(b)

    st = StyleStats()
    words, lines = seg.words, seg.lines
    xhs = [ln.xh for ln in lines if any(words[i].ok for i in ln.words)]
    st.xh_pt = float(np.median(xhs)) if xhs else 9.0
    tall = [b.top for b in allb if b.char in "bdhkl"]
    caps = [b.top for b in allb if b.char.isupper() or b.char.isdigit()]
    desc = [-b.bottom for b in allb if b.char in "gpqy"]
    st.ascender = float(np.median(tall)) if tall else 1.7
    st.cap_height = float(np.median(caps)) if len(caps) >= 3 else st.ascender * 0.95
    st.descender = float(np.median(desc)) if desc else 0.65
    st.slant = estimate_slant(allb)
    pens = [b.pen for b in allb if b.pen > 0 and b.char.isalnum()]
    st.pen_width = float(np.median(pens)) if pens else estimate_pen_width(allb)
    if ink is not None:
        st.ink = tuple(float(x) for x in ink)  # type: ignore[assignment]

    # ---- letter spacing (in x-heights), only between two accepted glyphs
    gaps_all: List[float] = []
    gl: Dict[str, List[float]] = defaultdict(list)
    gr: Dict[str, List[float]] = defaultdict(list)
    gp: Dict[str, List[float]] = defaultdict(list)
    by_pos = {(g.word_id, g.index): g for g in seg.glyphs}
    for g in seg.glyphs:
        if g.gap_next is None or not g.ok:
            continue
        nxt = by_pos.get((g.word_id, g.index + 1))
        if nxt is None or not nxt.ok:
            continue
        v = g.gap_next / g.xh
        if not (-0.8 < v < 2.0):
            continue
        gaps_all.append(v)
        gl[g.char].append(v)
        gr[nxt.char].append(v)
        gp[g.char + nxt.char].append(v)
    if gaps_all:
        d = Dist.fit(gaps_all, (0.12, 0.08), 0.02)
        st.letter_gap = {"mean": d.mean, "sd": d.sd}
    st.letter_gap_left = _pair_stats(gl, 4)
    st.letter_gap_right = _pair_stats(gr, 4)
    st.letter_gap_pair = _pair_stats(gp, 3)

    # ---- word gaps, punctuation gaps, line geometry
    wgaps: List[float] = []
    pgaps: List[float] = []
    slopes: List[float] = []
    wbase: List[float] = []
    wscale: List[float] = []
    starts: List[float] = []
    ends: List[float] = []
    spacing: List[float] = []
    for li, ln in enumerate(lines):
        ws = sorted([words[i] for i in ln.words], key=lambda w: w.ink_bbox[0])
        if not ws:
            continue
        if len(ws) >= 3:
            slopes.append(ln.slope)
        starts.append(ws[0].ink_bbox[0])
        ends.append(ws[-1].ink_bbox[2])
        for a, b in zip(ws, ws[1:]):
            gap = (b.ink_bbox[0] - a.ink_bbox[2]) / ln.xh
            if b.text in {",", ".", ";", ":", "!", "?", ")"} and len(b.text) == 1:
                if -0.5 < gap < 1.0:
                    pgaps.append(gap)
            elif a.text not in {"(", "-"} and 0.3 < gap < 6.0:
                wgaps.append(gap)
        for w in ws:
            if not w.ok or not w.groups:
                continue
            bots, tops = [], []
            for c, (k, i) in zip(w.chars, w.groups):
                ss = w.strokes[k:i]
                cx = 0.5 * (min(s.bbox[0] for s in ss) + max(s.bbox[2] for s in ss))
                if c in BASELINE_CHARS and c not in DESCENDERS:
                    bots.append((ln.baseline(cx) - max(s.bbox[3] for s in ss)) / ln.xh)
                if c in XHEIGHT_CHARS:
                    tops.append((ln.baseline(cx) - min(s.bbox[1] for s in ss)) / ln.xh)
            if len(bots) >= 2:
                wbase.append(float(np.median(bots)))
            if len(tops) >= 2:
                wscale.append(math.log(max(0.3, float(np.median(tops)))))
        # line spacing to the next line on the same page
        if li + 1 < len(lines):
            nx = lines[li + 1]
            if nx.source == ln.source and nx.page == ln.page:
                x = 300.0
                dy = (nx.baseline(x) - ln.baseline(x)) / ln.xh
                if 1.5 < dy < 6.0:
                    spacing.append(dy)
    d = Dist.fit(wgaps, (1.4, 0.35), 0.05)
    st.word_gap = {"mean": d.mean, "sd": d.sd}
    d = Dist.fit(pgaps, (0.15, 0.1), 0.03)
    st.punct_gap = {"mean": d.mean, "sd": d.sd}
    d = Dist.fit(spacing, (3.2, 0.25), 0.05)
    st.line_spacing = {"mean": d.mean, "sd": d.sd}
    d = Dist.fit(slopes, (0.0, 0.01), 0.002)
    st.line_slope = {"mean": d.mean, "sd": d.sd}
    d = Dist.fit(wbase, (0.0, 0.06), 0.01)
    st.word_baseline = {"mean": 0.0, "sd": d.sd}
    d = Dist.fit(wscale, (0.0, 0.06), 0.01)
    st.word_scale = {"mean": 0.0, "sd": d.sd}
    if starts:
        # natural left margin: the writer's usual line start (indented / maths
        # lines start further right, so use a low quantile, not the median)
        a = np.asarray(starts)
        left = float(np.quantile(a, 0.3))
        near = a[np.abs(a - left) < 20.0]
        sd = float(1.4826 * np.median(np.abs(near - np.median(near)))) if len(near) >= 3 else 4.0
        st.line_start = {"mean": left, "sd": float(np.clip(sd, 1.0, 8.0))}
    if ends:
        st.line_end = {"mean": float(np.quantile(np.asarray(ends), 0.85)), "sd": 10.0}
    st.char_counts = {c: len(v) for c, v in sorted(bank.items())}

    m = dict(meta or {})
    m.setdefault("n_glyphs", len(allb))
    m.setdefault("n_words_used", int(sum(w.ok for w in words)))
    m.setdefault("dropped_sparse_classes", "".join(sorted(dropped)))
    return Style(stats=st, bank=dict(bank), meta=m)
