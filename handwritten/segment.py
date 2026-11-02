"""Segment handwriting into labelled character glyphs.

Input: ``InkPage`` objects (strokes in writing order + noisy OCR words).
Output: ``Glyph`` objects -- the strokes of one character, normalised to a
glyph frame (x from the left ink edge, baseline at y = 0, y pointing down,
units of PDF points) -- plus per-word / per-line geometry used to learn
spacing statistics.

Algorithm
---------
1. Assign every stroke to the OCR word box it overlaps most.
2. Group words into text lines; estimate a baseline (robust line fit) and
   x-height per line from stroke geometry.
3. Re-insert *delayed strokes* (i-dots, t-bars written after the word) next
   to the stroke they belong to.
4. Align strokes to characters with dynamic programming.  Each character
   takes a contiguous run of 1..6 strokes; the cost combines
   * vertical extent / width likelihood relative to baseline & x-height,
   * a stroke-count prior,
   * horizontal coherence (strokes of one letter overlap, letters do not),
   * distance to a raster template of that character (after iteration 0).
5. EM: re-estimate per-character statistics, templates, baselines and
   x-heights from the alignment and repeat.
6. Reject words whose alignment is implausible (usually OCR errors) and
   individual glyphs that are outliers for their class.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import BBox, Stroke, compose, rasterize, rotate, subpaths_bbox, transform_subpaths, translate
from .ink import InkPage, OcrWord
from .priors import BASELINE_CHARS, DESCENDERS, XHEIGHT_CHARS, char_prior, stroke_count_logprior

MAX_STROKES_PER_CHAR = 6
FEAT_H, FEAT_W = 24, 12  # raster template size (rows: 2.0 xh above .. 1.0 below baseline)
RASTER_RES = 2.5  # px per pt for per-stroke rasters

SUPPORTED_EXTRA = set("αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ")


def is_trainable_char(c: str) -> bool:
    return (33 <= ord(c) < 127) or c in SUPPORTED_EXTRA


_PROSE = re.compile(r"^[(\[\"']?[A-Za-z][A-Za-z']+[.,;:!?)\]\"']{0,2}$")
_LETTER = re.compile(r"^[(]?[A-Za-z][.,;:!?)]{0,2}$")
_NUMBER = re.compile(r"^[(]?\d{1,4}[.,;:)]{0,2}$")
_PUNCT = re.compile(r"^[.,!()=+\-]$")


def token_trust(text: str) -> float:
    """Prior belief that an OCR token's label is right.

    iOS handwriting recognition is good on prose but mangles mathematics
    (``3^n`` becomes ``3"``, ``(p-1)!`` becomes ``(P-1) !`` ...).  Only tokens
    that look like ordinary words, plain numbers or lone punctuation are used
    to learn glyph shapes."""
    t = text.strip()
    if _PROSE.match(t):
        return 1.0
    if _LETTER.match(t) or _NUMBER.match(t):
        return 0.8
    if _PUNCT.match(t):
        return 0.7
    return 0.2


MIN_TRUST = 0.5

# allowed stroke counts and width/height aspect range for simple symbols
SYMBOL_RULES: Dict[str, Tuple[Tuple[int, ...], float, float]] = {
    "-": ((1,), 2.0, 99.0), "_": ((1,), 3.0, 99.0), "=": ((2,), 1.0, 99.0), "+": ((2,), 0.5, 2.0),
    ".": ((1,), 0.2, 5.0), ",": ((1,), 0.1, 1.5), "(": ((1,), 0.1, 0.9), ")": ((1,), 0.1, 0.9),
    "!": ((2,), 0.05, 0.8), ":": ((2,), 0.05, 1.2), ";": ((2,), 0.05, 1.2), "/": ((1,), 0.1, 1.5),
    "|": ((1,), 0.0, 0.4), "'": ((1,), 0.05, 1.5), '"': ((2,), 0.3, 3.0),
}

# Characters that may legitimately look alike; the cross-class check never
# rejects a glyph because it resembles a member of its own group.
CONFUSABLE = ["oO0", "lI1|!/", "sS5", "cC(", "vV", "wW", "xX", "zZ2", "kK", "pP", "uU", "-_~", ".,'",
              "jJ", "yY", "9gqa", "6b", "7T", ")]", "([", "tf+"]


def confusable(a: str, b: str) -> bool:
    return any(a in grp and b in grp for grp in CONFUSABLE)


# --------------------------------------------------------------------------
# data classes
# --------------------------------------------------------------------------
@dataclass
class WordInk:
    text: str
    strokes: List[Stroke]
    source: str
    page: int
    bbox: BBox
    order: int
    line: int = -1
    trust: float = 1.0  # how much we believe the OCR label (see token_trust)
    # alignment result
    groups: Optional[List[Tuple[int, int]]] = None  # [k, i) stroke ranges per char
    cost: float = math.inf
    char_costs: Optional[List[float]] = None
    ok: bool = False
    # cached per-stroke rasters
    _masks: Optional[List[Tuple[np.ndarray, int, int]]] = None  # (mask, row0, col0) on page grid
    _grid0: Tuple[float, float] = (0.0, 0.0)

    @property
    def chars(self) -> List[str]:
        return [c for c in self.text if not c.isspace()]

    @property
    def ink_bbox(self) -> BBox:
        return (
            min(s.bbox[0] for s in self.strokes),
            min(s.bbox[1] for s in self.strokes),
            max(s.bbox[2] for s in self.strokes),
            max(s.bbox[3] for s in self.strokes),
        )


@dataclass
class TextLine:
    source: str
    page: int
    words: List[int]  # indices into Segmentation.words
    slope: float = 0.0
    intercept: float = 0.0  # baseline y = intercept + slope * x
    xh: float = 8.0

    def baseline(self, x: float) -> float:
        return self.intercept + self.slope * x

    @property
    def angle(self) -> float:
        return math.atan(self.slope)


@dataclass
class Glyph:
    char: str
    strokes: List[List[np.ndarray]]  # list of strokes, each a list of (K,4,2) subpaths, glyph frame
    xh: float  # x-height (pt) of the line this glyph came from
    width: float  # ink width (pt)
    top: float  # highest ink above baseline (pt, positive up)
    bottom: float  # lowest ink relative to baseline (pt, negative = below)
    source: str = ""
    page: int = 0
    word: str = ""
    index: int = 0
    prev_char: Optional[str] = None
    next_char: Optional[str] = None
    gap_prev: Optional[float] = None  # ink gap to previous glyph in the word (pt)
    gap_next: Optional[float] = None
    cost: float = 0.0
    template_dist: float = 0.0
    seqs: Tuple[int, ...] = ()
    ok: bool = True
    reject_reason: str = ""
    word_id: int = -1  # index into Segmentation.words


@dataclass
class Segmentation:
    words: List[WordInk]
    lines: List[TextLine]
    glyphs: List[Glyph] = field(default_factory=list)
    char_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    templates: Dict[str, np.ndarray] = field(default_factory=dict)
    unassigned: List[Stroke] = field(default_factory=list)
    log: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _robust_line(xs: np.ndarray, ys: np.ndarray, ws: Optional[np.ndarray] = None,
                 max_slope: float = 0.15) -> Tuple[float, float]:
    """Robust (IRLS / Huber) fit of y = a + b x.  Returns (slope, intercept)."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    ws = np.ones_like(xs) if ws is None else np.asarray(ws, float)
    if len(xs) == 0:
        return 0.0, 0.0
    if len(xs) < 3 or np.ptp(xs) < 1e-6:
        return 0.0, float(np.average(ys, weights=ws)) if ws.sum() > 0 else float(np.median(ys))
    b, a = 0.0, float(np.median(ys))
    for _ in range(12):
        r = ys - (a + b * xs)
        s = 1.4826 * np.median(np.abs(r)) + 1e-3
        hub = np.minimum(1.0, 1.5 * s / (np.abs(r) + 1e-9))
        w = ws * hub
        W = w.sum()
        mx = (w * xs).sum() / W
        my = (w * ys).sum() / W
        vx = (w * (xs - mx) ** 2).sum()
        if vx < 1e-9:
            b = 0.0
        else:
            b = float(np.clip((w * (xs - mx) * (ys - my)).sum() / vx, -max_slope, max_slope))
        a = float(my - b * mx)
    return b, a


def _resize_box(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """Area-average resize of a 2D float array."""
    from PIL import Image

    if mask.size == 0:
        return np.zeros((h, w), np.float32)
    im = Image.fromarray(mask.astype(np.float32), mode="F")
    return np.asarray(im.resize((w, h), Image.BOX), dtype=np.float32)


def _blur(a: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(a, sigma)


def _cosdist(f: np.ndarray, t: np.ndarray) -> float:
    nf = np.linalg.norm(f)
    nt = np.linalg.norm(t)
    if nf < 1e-9 or nt < 1e-9:
        return 1.0
    return float(1.0 - (f * t).sum() / (nf * nt))


# --------------------------------------------------------------------------
# stage 1/2: words and lines
# --------------------------------------------------------------------------
def assign_strokes(page: InkPage, min_frac: float = 0.5) -> Tuple[Dict[int, List[Stroke]], List[Stroke]]:
    words = [w for w in page.words]
    if not words or not page.strokes:
        return {}, list(page.strokes)
    wb = np.array([w.bbox for w in words], float)
    wb[:, 0] -= 1.0
    wb[:, 1] -= 1.5
    wb[:, 2] += 1.0
    wb[:, 3] += 1.5
    out: Dict[int, List[Stroke]] = defaultdict(list)
    rest: List[Stroke] = []
    for s in page.strokes:
        x0, y0, x1, y1 = s.bbox
        area = max((x1 - x0) * (y1 - y0), 1e-3)
        ox = np.clip(np.minimum(x1, wb[:, 2]) - np.maximum(x0, wb[:, 0]), 0, None)
        oy = np.clip(np.minimum(y1, wb[:, 3]) - np.maximum(y0, wb[:, 1]), 0, None)
        frac = ox * oy / area
        # thin strokes (dots/bars): fall back to centre containment
        inside = (s.cx >= wb[:, 0]) & (s.cx <= wb[:, 2]) & (s.cy >= wb[:, 1]) & (s.cy <= wb[:, 3])
        frac = np.maximum(frac, inside * 0.75)
        j = int(np.argmax(frac))
        if frac[j] >= min_frac:
            out[j].append(s)
        else:
            rest.append(s)
    return out, rest


def build_words(pages: Sequence[InkPage]) -> Tuple[List[WordInk], List[Stroke]]:
    words: List[WordInk] = []
    unassigned: List[Stroke] = []
    order = 0
    for p in pages:
        amap, rest = assign_strokes(p)
        unassigned.extend(rest)
        for j, w in enumerate(p.words):
            strokes = sorted(amap.get(j, []), key=lambda s: s.seq)
            text = w.text.strip()
            if not strokes:
                continue
            words.append(WordInk(text=text, strokes=strokes, source=p.source, page=p.page, bbox=w.bbox, order=order,
                                 trust=token_trust(text)))
            order += 1
    return words, unassigned


def build_lines(words: List[WordInk]) -> List[TextLine]:
    lines: List[TextLine] = []
    by_page: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for i, w in enumerate(words):
        by_page[(w.source, w.page)].append(i)
    for (src, pg), idx in by_page.items():
        idx.sort(key=lambda i: words[i].order)
        heights = [words[i].ink_bbox[3] - words[i].ink_bbox[1] for i in idx]
        med_h = float(np.median(heights)) if heights else 10.0
        cur: List[int] = []
        for i in idx:
            b = words[i].ink_bbox
            if cur:
                last = words[cur[-1]].ink_bbox
                cy = np.median([0.5 * (words[k].ink_bbox[1] + words[k].ink_bbox[3]) for k in cur[-3:]])
                same = abs(0.5 * (b[1] + b[3]) - cy) < 0.55 * med_h and b[0] > last[0] - 0.5 * med_h
                if not same:
                    lines.append(TextLine(src, pg, cur))
                    cur = []
            cur.append(i)
        if cur:
            lines.append(TextLine(src, pg, cur))
    for li, ln in enumerate(lines):
        for i in ln.words:
            words[i].line = li
        # A line made mostly of non-prose tokens is maths: the recogniser's
        # labels there are unreliable even for letter-only tokens ("anx").
        # Digits and symbols keep their trust: they pass strict size and
        # construction checks of their own in extract_glyphs.
        toks = [words[i] for i in ln.words if len(words[i].text) > 0]
        prose = sum(1 for w in toks if w.trust >= 1.0)
        if len(toks) >= 2 and prose / len(toks) < 0.5:
            for w in toks:
                if any(ch.isalpha() for ch in w.text):
                    w.trust = min(w.trust, 0.3)
    return lines


def init_line_geometry(words: List[WordInk], lines: List[TextLine]) -> None:
    """Initial baseline/x-height from stroke geometry only (no labels)."""
    for ln in lines:
        strokes = [s for i in ln.words for s in words[i].strokes]
        hs = np.array([s.height for s in strokes])
        if len(hs) == 0:
            continue
        H = float(np.median(hs))
        big = [s for s in strokes if s.height > 0.45 * H]
        xs = np.array([s.cx for s in big])
        ys = np.array([s.bbox[3] for s in big])
        ws = np.array([s.height for s in big])
        # A descender stroke pulls the bottom down: down-weight the lowest third
        ln.slope, ln.intercept = _robust_line(xs, ys, ws)
        rel_top = np.array([ln.baseline(s.cx) - s.bbox[1] for s in big])
        rel_bot = np.array([ln.baseline(s.cx) - s.bbox[3] for s in big])
        near = np.abs(rel_bot) < 0.3 * H
        cand = rel_top[near] if near.sum() >= 2 else rel_top
        ln.xh = float(max(2.0, np.percentile(cand, 40)))


# --------------------------------------------------------------------------
# stage 3: delayed strokes
# --------------------------------------------------------------------------
def reorder_delayed(strokes: List[Stroke], xh: float) -> List[Stroke]:
    """Move strokes written *after* moving on (dots, crossbars) back next to the
    stroke they overlap horizontally."""
    out: List[Stroke] = []
    for s in strokes:
        if out:
            prev = out[-1]
            small = s.height < 0.7 * xh or s.width < 0.35 * xh or s.height < 0.35 * s.width
            went_back = s.cx < prev.bbox[0] - 0.05 * xh
            if small and went_back:
                # best horizontally overlapping earlier stroke
                best, best_ov = None, 0.0
                for k, t in enumerate(out):
                    ov = _overlap(s.bbox[0], s.bbox[2], t.bbox[0], t.bbox[2])
                    ov /= max(1e-3, min(s.width, t.width) + 0.2 * xh)
                    if ov > best_ov + 1e-6:
                        best, best_ov = k, ov
                if best is None:
                    # nearest by centre
                    best = int(np.argmin([abs(t.cx - s.cx) for t in out]))
                out.insert(best + 1, s)
                continue
        out.append(s)
    return out


# --------------------------------------------------------------------------
# stage 4: alignment
# --------------------------------------------------------------------------
class CharModel:
    """Per-character statistics used by the alignment cost."""

    def __init__(self) -> None:
        self.stats: Dict[str, Dict[str, float]] = {}
        self.stroke_counts: Dict[str, Counter] = {}
        self.templates: Dict[str, np.ndarray] = {}
        self.tdist_scale: Dict[str, float] = {}
        self.global_tdist: float = 0.25
        self.use_templates = False

    def stat(self, c: str) -> Dict[str, float]:
        return self.stats.get(c) or char_prior(c)

    def stroke_logp(self, c: str, n: int) -> float:
        cnt = self.stroke_counts.get(c)
        prior = math.exp(stroke_count_logprior(c, n))
        if not cnt:
            return math.log(prior + 1e-6)
        tot = sum(cnt.values())
        a = 2.0  # pseudo-count weight of prior
        return math.log((cnt.get(n, 0) + a * prior) / (tot + a))


@dataclass
class _Group:
    left: float
    right: float
    top: float  # page y of highest ink
    bottom: float  # page y of lowest ink
    gap: float  # internal horizontal gap (pt)
    n: int


def _word_masks(w: WordInk) -> None:
    if w._masks is not None:
        return
    x0, y0, x1, y1 = w.ink_bbox
    x0 -= 1
    y0 -= 1
    w._grid0 = (x0, y0)
    masks = []
    for s in w.strokes:
        bx = (s.bbox[0] - 0.5, s.bbox[1] - 0.5, s.bbox[2] + 0.5, s.bbox[3] + 0.5)
        m = rasterize([s.subpaths], bx, RASTER_RES, n=3)
        r0 = int(round((bx[1] - y0) * RASTER_RES))
        c0 = int(round((bx[0] - x0) * RASTER_RES))
        masks.append((m, r0, c0))
    w._masks = masks


def group_feature(w: WordInk, k: int, i: int, baseline: float, xh: float) -> np.ndarray:
    """Raster template feature of strokes [k, i) of word ``w``."""
    _word_masks(w)
    gx0, gy0 = w._grid0
    left = min(s.bbox[0] for s in w.strokes[k:i])
    right = max(s.bbox[2] for s in w.strokes[k:i])
    top_y = baseline - 2.0 * xh
    bot_y = baseline + 1.0 * xh
    R = RASTER_RES
    H = max(4, int(math.ceil((bot_y - top_y) * R)))
    W = max(2, int(math.ceil((right - left) * R)) + 2)
    canvas = np.zeros((H, W), np.float32)
    oy = int(round((gy0 - top_y) * R))
    ox = int(round((gx0 - left) * R)) + 1
    for (m, r0, c0) in w._masks[k:i]:  # type: ignore[index]
        rr = r0 + oy
        cc = c0 + ox
        h, ww = m.shape
        ra, rb = max(0, rr), min(H, rr + h)
        ca, cb = max(0, cc), min(W, cc + ww)
        if ra < rb and ca < cb:
            canvas[ra:rb, ca:cb] = np.maximum(canvas[ra:rb, ca:cb], m[ra - rr:rb - rr, ca - cc:cb - cc])
    f = _resize_box(canvas, FEAT_H, FEAT_W)
    return _blur(f, 0.7).ravel()


KNN_RES = 8.0  # px per x-height for kNN features
KNN_W, KNN_H = 2.6, 3.0  # window (x-heights): centred horizontally, 2 xh above .. 1 xh below baseline


def fixed_feature(w: WordInk, k: int, i: int, baseline: float, xh: float) -> np.ndarray:
    """Fixed-scale raster of strokes [k, i) (no width normalisation), unit L2."""
    ss = w.strokes[k:i]
    left = min(s.bbox[0] for s in ss)
    right = max(s.bbox[2] for s in ss)
    cx = 0.5 * (left + right)
    res = KNN_RES / xh
    box = (cx - 0.5 * KNN_W * xh, baseline - 2.0 * xh, cx + 0.5 * KNN_W * xh, baseline + 1.0 * xh)
    m = rasterize([s.subpaths for s in ss], box, res, n=3).astype(np.float32)
    H, W = int(round(KNN_H * KNN_RES)), int(round(KNN_W * KNN_RES))
    out = np.zeros((H, W), np.float32)
    out[:min(H, m.shape[0]), :min(W, m.shape[1])] = m[:H, :W]
    f = _blur(out, 1.0).ravel()
    n = np.linalg.norm(f)
    return f / n if n > 1e-9 else f


def _groups_for_word(w: WordInk) -> Dict[Tuple[int, int], _Group]:
    n = len(w.strokes)
    out = {}
    for k in range(n):
        for i in range(k + 1, min(n, k + MAX_STROKES_PER_CHAR) + 1):
            ss = w.strokes[k:i]
            iv = sorted((s.bbox[0], s.bbox[2]) for s in ss)
            gap = 0.0
            reach = iv[0][1]
            for a, b in iv[1:]:
                if a > reach:
                    gap += a - reach
                reach = max(reach, b)
            out[(k, i)] = _Group(
                left=iv[0][0], right=max(b for _, b in iv), top=min(s.bbox[1] for s in ss),
                bottom=max(s.bbox[3] for s in ss), gap=gap, n=i - k,
            )
    return out


def align_word(w: WordInk, ln: TextLine, model: CharModel) -> None:
    chars = w.chars
    m = len(chars)
    n = len(w.strokes)
    w.groups, w.cost, w.char_costs, w.ok = None, math.inf, None, False
    if m == 0 or n < m or n > m * MAX_STROKES_PER_CHAR or not all(is_trainable_char(c) for c in chars):
        return
    xh = ln.xh
    G = _groups_for_word(w)
    feat_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def char_cost(j: int, k: int, i: int) -> float:
        c = chars[j]
        g = G[(k, i)]
        cx = 0.5 * (g.left + g.right)
        base = ln.baseline(cx)
        top = (base - g.top) / xh
        bot = (base - g.bottom) / xh
        wid = (g.right - g.left) / xh
        st = model.stat(c)
        cost = 0.5 * ((top - st["top"]) / st["top_sd"]) ** 2
        cost += 0.5 * ((bot - st["bottom"]) / st["bottom_sd"]) ** 2
        cost += 0.5 * ((wid - st["width"]) / st["width_sd"]) ** 2
        cost -= model.stroke_logp(c, g.n)
        cost += 6.0 * (g.gap / xh)  # strokes of one char should touch/overlap
        if model.use_templates and c in model.templates:
            key = (k, i)
            if key not in feat_cache:
                feat_cache[key] = group_feature(w, k, i, base, xh)
            d = _cosdist(feat_cache[key], model.templates[c])
            sc = model.tdist_scale.get(c, model.global_tdist)
            cost += 2.0 * d / max(sc, 0.03)
        return cost

    def trans_cost(prev: Tuple[int, int], cur: Tuple[int, int]) -> float:
        a = G[prev]
        b = G[cur]
        ov = _overlap(a.left, a.right, b.left, b.right)
        rel = ov / max(1e-3, min(a.right - a.left, b.right - b.left))
        cost = 3.0 * max(0.0, rel - 0.25) + 2.0 * max(0.0, ov / xh - 0.3)
        # letters should progress to the right
        if b.left + b.right < a.left + a.right:
            cost += 4.0
        return cost

    INF = math.inf
    # best[(j, k, i)] -> (cost, back) ; char j covers strokes [k, i)
    best: Dict[Tuple[int, int, int], Tuple[float, Optional[Tuple[int, int, int]], float]] = {}
    for i in range(1, min(n, MAX_STROKES_PER_CHAR) + 1):
        if n - i < (m - 1):
            break
        best[(0, 0, i)] = (char_cost(0, 0, i), None, char_cost(0, 0, i))
    for j in range(1, m):
        for (jj, kp, ip), (cprev, _, _) in list(best.items()):
            if jj != j - 1:
                continue
            k = ip
            for i in range(k + 1, min(n, k + MAX_STROKES_PER_CHAR) + 1):
                if n - i < (m - 1 - j):
                    break
                if (m - 1 - j) * MAX_STROKES_PER_CHAR < n - i:
                    continue
                cc = char_cost(j, k, i)
                tot = cprev + cc + trans_cost((kp, ip), (k, i))
                key = (j, k, i)
                if key not in best or tot < best[key][0]:
                    best[key] = (tot, (jj, kp, ip), cc)
    finals = [(v[0], key) for key, v in best.items() if key[0] == m - 1 and key[2] == n]
    if not finals:
        return
    tot, key = min(finals)
    groups: List[Tuple[int, int]] = []
    ccs: List[float] = []
    cur: Optional[Tuple[int, int, int]] = key
    while cur is not None:
        groups.append((cur[1], cur[2]))
        ccs.append(best[cur][2])
        cur = best[cur][1]
    w.groups = groups[::-1]
    w.char_costs = ccs[::-1]
    w.cost = tot / m
    w.ok = True


# --------------------------------------------------------------------------
# stage 5: EM re-estimation
# --------------------------------------------------------------------------
def _glyph_measures(w: WordInk, ln: TextLine, k: int, i: int) -> Tuple[float, float, float, float, float]:
    ss = w.strokes[k:i]
    left = min(s.bbox[0] for s in ss)
    right = max(s.bbox[2] for s in ss)
    top = min(s.bbox[1] for s in ss)
    bot = max(s.bbox[3] for s in ss)
    base = ln.baseline(0.5 * (left + right))
    return left, right, (base - top) / ln.xh, (base - bot) / ln.xh, (right - left) / ln.xh


def reestimate_lines(words: List[WordInk], lines: List[TextLine]) -> None:
    """Refit baselines from bottoms of baseline chars and x-height from tops of
    x-height chars, using the current alignment."""
    for ln in lines:
        xs, ys, ws, tops = [], [], [], []
        for wi in ln.words:
            w = words[wi]
            if not w.ok or w.groups is None or w.trust < MIN_TRUST:
                continue
            for c, (k, i) in zip(w.chars, w.groups):
                ss = w.strokes[k:i]
                left = min(s.bbox[0] for s in ss)
                right = max(s.bbox[2] for s in ss)
                bot = max(s.bbox[3] for s in ss)
                top = min(s.bbox[1] for s in ss)
                cx = 0.5 * (left + right)
                if c in BASELINE_CHARS and c not in DESCENDERS:
                    xs.append(cx)
                    ys.append(bot)
                    ws.append(1.0)
                if c in XHEIGHT_CHARS:
                    tops.append((cx, top))
        if len(xs) >= 2:
            ln.slope, ln.intercept = _robust_line(np.array(xs), np.array(ys), np.array(ws))
        if len(tops) >= 2:
            rel = [ln.baseline(x) - t for x, t in tops]
            ln.xh = float(max(2.0, np.median(rel)))


def reestimate_model(words: List[WordInk], lines: List[TextLine], model: CharModel, use_templates: bool,
                     word_ok_quantile: float = 0.85) -> None:
    oks = [w for w in words if w.ok and w.trust >= MIN_TRUST]
    if not oks:
        return
    costs = np.array([w.cost for w in oks])
    thr = float(np.quantile(costs, word_ok_quantile))
    meas: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    counts: Dict[str, Counter] = defaultdict(Counter)
    feats: Dict[str, List[np.ndarray]] = defaultdict(list)
    for w in oks:
        if w.cost > thr:
            continue
        ln = lines[w.line]
        for c, (k, i) in zip(w.chars, w.groups or []):
            _, _, top, bot, wid = _glyph_measures(w, ln, k, i)
            meas[c].append((top, bot, wid))
            counts[c][i - k] += 1
            if use_templates:
                left = min(s.bbox[0] for s in w.strokes[k:i])
                right = max(s.bbox[2] for s in w.strokes[k:i])
                feats[c].append(group_feature(w, k, i, ln.baseline(0.5 * (left + right)), ln.xh))
    stats = {}
    for c, arr in meas.items():
        a = np.array(arr)
        pr = char_prior(c)
        n = len(a)
        kp = 4.0  # prior pseudo-count
        med = np.median(a, axis=0)
        mad = 1.4826 * np.median(np.abs(a - med), axis=0) if n >= 3 else np.zeros(3)
        mean = (n * med + kp * np.array([pr["top"], pr["bottom"], pr["width"]])) / (n + kp)
        sd_prior = np.array([pr["top_sd"], pr["bottom_sd"], pr["width_sd"]])
        sd = np.sqrt((n * mad ** 2 + kp * sd_prior ** 2) / (n + kp))
        sd = np.maximum(sd, [0.12, 0.12, 0.12])
        stats[c] = {"top": mean[0], "bottom": mean[1], "width": mean[2],
                    "top_sd": sd[0], "bottom_sd": sd[1], "width_sd": sd[2], "n": float(n)}
    model.stats = stats
    model.stroke_counts = dict(counts)
    if use_templates:
        model.templates = {}
        model.tdist_scale = {}
        all_d = []
        for c, fs in feats.items():
            if len(fs) < 2:
                continue
            F = np.stack(fs)
            T = np.median(F, axis=0)
            d = np.array([_cosdist(f, T) for f in F])
            model.templates[c] = T
            model.tdist_scale[c] = float(np.median(d)) + 0.02
            all_d.extend(d.tolist())
        model.global_tdist = float(np.median(all_d)) + 0.02 if all_d else 0.25
        model.use_templates = True


# --------------------------------------------------------------------------
# stage 6: glyph extraction
# --------------------------------------------------------------------------
def _calibrated_prior(c: str, stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Hershey prior rescaled to this writer's ascender / capital heights."""
    pr = dict(char_prior(c))

    def ratio(chars: str) -> float:
        r = []
        for k in chars:
            st = stats.get(k)
            if st and st.get("n", 0) >= 5:
                r.append(st["top"] / max(0.3, char_prior(k)["top"]))
        return float(np.median(r)) if r else 1.0

    if c.isupper() or c.isdigit():
        f = ratio("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        pr["top"] *= f
    elif pr["top"] > 1.3:
        pr["top"] *= ratio("bdfhklt")
    return pr


def extract_glyphs(words: List[WordInk], lines: List[TextLine], model: CharModel,
                   word_cost_quantile: float = 0.9, glyph_z: float = 3.0, metric_z: float = 3.0) -> List[Glyph]:
    oks = [w for w in words if w.ok and w.trust >= MIN_TRUST]
    for w in words:
        if w.ok and w.trust < MIN_TRUST:
            w.ok = False
    if not oks:
        return []
    costs = np.array([w.cost for w in oks])
    med = float(np.median(costs))
    mad = 1.4826 * float(np.median(np.abs(costs - med))) + 1e-6
    thr = min(float(np.quantile(costs, word_cost_quantile)), med + 3.0 * mad)
    glyphs: List[Glyph] = []
    feats: List[np.ndarray] = []
    kfeats: List[np.ndarray] = []
    id_word: List[int] = []
    word_index = {id(w): n for n, w in enumerate(words)}
    for wi_, w in enumerate(oks):
        if w.cost > thr:
            w.ok = False
            continue
        ln = lines[w.line]
        chars = w.chars
        spans = []
        for (k, i) in w.groups or []:
            ss = w.strokes[k:i]
            spans.append((min(s.bbox[0] for s in ss), max(s.bbox[2] for s in ss)))
        for j, (c, (k, i)) in enumerate(zip(chars, w.groups or [])):
            ss = w.strokes[k:i]
            left, right = spans[j]
            cx = 0.5 * (left + right)
            base = ln.baseline(cx)
            # glyph frame: origin at (left ink edge, baseline), de-rotated by line angle
            M = compose(rotate(-ln.angle), translate(-left, -base))
            gstrokes = [transform_subpaths(s.subpaths, M) for s in ss]
            bb = subpaths_bbox([sp for st in gstrokes for sp in st])
            dx = bb[0]
            if abs(dx) > 1e-6:
                gstrokes = [[sp - np.array([dx, 0.0]) for sp in st] for st in gstrokes]
                bb = (0.0, bb[1], bb[2] - dx, bb[3])
            g = Glyph(
                char=c, strokes=gstrokes, xh=ln.xh, width=bb[2] - bb[0], top=-bb[1], bottom=-bb[3],
                source=w.source, page=w.page, word=w.text, index=j,
                prev_char=chars[j - 1] if j > 0 else None,
                next_char=chars[j + 1] if j + 1 < len(chars) else None,
                gap_prev=(left - spans[j - 1][1]) if j > 0 else None,
                gap_next=(spans[j + 1][0] - right) if j + 1 < len(chars) else None,
                cost=(w.char_costs or [0.0] * len(chars))[j],
                seqs=tuple(s.seq for s in ss),
                word_id=word_index[id(w)],
            )
            f = group_feature(w, k, i, base, ln.xh)
            if c in model.templates:
                g.template_dist = _cosdist(f, model.templates[c])
            glyphs.append(g)
            feats.append(f)
            kfeats.append(fixed_feature(w, k, i, base, ln.xh))
            id_word.append(wi_)

    def reject(g: Glyph, why: str) -> None:
        if g.ok:
            g.ok = False
            g.reject_reason = why

    by_c: Dict[str, List[int]] = defaultdict(list)
    for gi, g in enumerate(glyphs):
        by_c[g.char].append(gi)
    tall = [g.top / g.xh for g in glyphs if g.char in "bdhklABDEFHKLMNPRTXZ"]
    cap_h = float(np.median(tall)) if len(tall) >= 5 else 1.65
    for c, idx in by_c.items():
        gs = [glyphs[i] for i in idx]
        # (a) template distance / alignment cost outliers
        if len(gs) >= 4:
            for attr in ("template_dist", "cost"):
                v = np.array([getattr(g, attr) for g in gs])
                md = np.median(v)
                sd = 1.4826 * np.median(np.abs(v - md)) + 1e-6
                for g, x in zip(gs, v):
                    if (x - md) / sd > glyph_z:
                        reject(g, attr)
        # (b) size / vertical placement outliers (super/subscripts, wrong case)
        M_ = np.array([[g.top / g.xh, g.bottom / g.xh, g.width / g.xh] for g in gs])
        if c in BASELINE_CHARS and c not in DESCENDERS:
            for g, (t, b, wd) in zip(gs, M_):
                if b > 0.35 or b < -0.4:
                    reject(g, "off-baseline")
        if c.isdigit() or c.isupper():
            for g, (t, b, wd) in zip(gs, M_):
                if not (0.72 * cap_h <= t <= 1.35 * cap_h):
                    reject(g, "cap-height")
        # simple symbols have a fixed construction
        rule = SYMBOL_RULES.get(c)
        if rule is not None:
            n_ok_strokes, min_aspect, max_aspect = rule
            for g, (t, b, wd) in zip(gs, M_):
                asp = wd / max(1e-3, t - b)
                if len(g.strokes) not in n_ok_strokes or not (min_aspect <= asp <= max_aspect):
                    reject(g, "construction")
        if len(gs) >= 5:
            md = np.median(M_, axis=0)
            sd = np.maximum(1.4826 * np.median(np.abs(M_ - md), axis=0), [0.12, 0.12, 0.12])
            z = np.abs(M_ - md) / sd
            for g, zz in zip(gs, z):
                if zz.max() > metric_z:
                    reject(g, "size")
        else:
            pr = _calibrated_prior(c, model.stats)
            for g, (t, b, wd) in zip(gs, M_):
                if abs(t - pr["top"]) > 0.55 or abs(b - pr["bottom"]) > 0.55 or \
                        abs(wd - pr["width"]) > max(0.6, 0.6 * pr["width"]):
                    reject(g, "prior")

    # (c) cross-class kNN check: reject glyphs whose nearest neighbours (from
    # other words) overwhelmingly carry one other, non-confusable label.
    ok_idx = [gi for gi, g in enumerate(glyphs) if g.ok]
    if len(ok_idx) > 20:
        F = np.stack([kfeats[gi] for gi in ok_idx])
        labels = [glyphs[gi].char for gi in ok_idx]
        wid = [id_word[gi] for gi in ok_idx]
        n_cls = Counter(labels)
        D = 2.0 - 2.0 * (F @ F.T)  # squared L2 between unit vectors
        np.fill_diagonal(D, np.inf)
        K = 7
        for a, gi in enumerate(ok_idx):
            row = D[a].copy()
            same_word = [b for b in range(len(ok_idx)) if wid[b] == wid[a]]
            row[same_word] = np.inf
            nn = np.argsort(row)[:K]
            lab = Counter(labels[b] for b in nn)
            own = lab.get(labels[a], 0)
            other, cnt = max(((c2, n2) for c2, n2 in lab.items() if c2 != labels[a]), key=lambda t: t[1],
                             default=(None, 0))
            if other is None or confusable(other, labels[a]):
                continue
            if n_cls[labels[a]] >= 5:
                bad = own == 0 and cnt >= 5
            else:
                bad = cnt >= 6
            if bad:
                reject(glyphs[gi], f"looks like {other}")

    # (d) class spread: a class may not be more scattered around its medoid
    # than the clean, well-populated classes are.
    ok_by_c: Dict[str, List[int]] = defaultdict(list)
    for gi, g in enumerate(glyphs):
        if g.ok:
            ok_by_c[g.char].append(gi)
    med_d: Dict[str, np.ndarray] = {}
    for c, idx in ok_by_c.items():
        if len(idx) < 3:
            continue
        F = np.stack([kfeats[i] for i in idx])
        D = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (F @ F.T)))
        med = int(np.argmin(D.sum(axis=1)))
        med_d[c] = D[med]
    ref = [d for c, d in med_d.items() if len(d) >= 20]
    if ref:
        tau = float(np.percentile(np.concatenate(ref), 95))
        for c, d in med_d.items():
            lim = min(tau, 2.5 * float(np.median(d)) + 1e-6) if len(d) >= 20 else tau
            for gi, dd in zip(ok_by_c[c], d):
                if dd > lim:
                    reject(glyphs[gi], "spread")
    return glyphs


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def segment(pages: Sequence[InkPage], n_iter: int = 5, verbose: bool = False) -> Segmentation:
    words, unassigned = build_words(pages)
    lines = build_lines(words)
    init_line_geometry(words, lines)
    seg = Segmentation(words=words, lines=lines, unassigned=unassigned)
    model = CharModel()
    for it in range(n_iter):
        for w in words:
            ln = lines[w.line]
            if it == 0:
                w.strokes = reorder_delayed(w.strokes, ln.xh)
                w._masks = None
            align_word(w, ln, model)
        n_ok = sum(w.ok for w in words)
        mc = float(np.median([w.cost for w in words if w.ok])) if n_ok else float("nan")
        msg = f"iter {it}: aligned {n_ok}/{len(words)} words, median cost/char {mc:.2f}"
        seg.log.append(msg)
        if verbose:
            print(msg)
        reestimate_lines(words, lines)
        reestimate_model(words, lines, model, use_templates=it >= 0)
    seg.glyphs = extract_glyphs(words, lines, model)
    seg.char_stats = model.stats
    seg.templates = model.templates
    n_ok = sum(g.ok for g in seg.glyphs)
    msg = f"glyphs: {n_ok} accepted / {len(seg.glyphs)} from {sum(w.ok for w in words)} words"
    seg.log.append(msg)
    if verbose:
        print(msg)
    return seg
