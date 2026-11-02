"""Calibration sheets: write the characters your free-text sample lacks.

``make_template`` writes PDF pages of labelled writing cells.  Every character
gets one group: the character printed small in grey followed by ``repeats``
boxes, each with faint guides (solid baseline, dashed x-height line, dotted
cap/ascender and descender lines).  Fill the boxes on an iPad (Files/Markup,
Notes, GoodNotes, ...) and export the PDF.

``style_from_sheets`` reads filled sheets back.  The sheet layout is embedded
three ways because iPad apps differ in what they keep when they export:

1. an invisible text marker on every page (``HWTEMPLATE v1 k/n <base64>``,
   zlib-compressed JSON) -- page content survives almost everything;
2. the same data in the document metadata (``keywords``);
3. the page itself: the cell boxes are vector rectangles in a distinctive
   colour and every group has a printed label, so a layout can be rebuilt
   from the page graphics, or -- as a last resort when the background was
   flattened to an image -- recomputed from the default parameters.

User ink is every vector drawing that is not the template's own (the
template draws only straight stroked lines in its guide colours, plus text).
Both PencilKit-style filled outlines and stroked paths (converted to filled
outlines: a union of round capsules made of cubic Beziers) are accepted.
Ink is grouped into touching clusters, assigned to the cell holding the
cluster's centre and normalised with that cell's guides into the glyph frame
used by ``style.BankGlyph`` (x from the left ink edge, baseline at y = 0,
y down, units of x-height).
"""
from __future__ import annotations

import base64
import copy
import json
import re
import warnings
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pymupdf

from .geometry import subpaths_bbox
from .ink import _items_to_subpaths
from .render import PAPER_SIZES
from .style import BankGlyph, Style, StyleStats, estimate_pen_width, estimate_slant

try:  # per-glyph pen width (newer style module)
    from .style import glyph_pen_width
except ImportError:  # pragma: no cover
    glyph_pen_width = None

TEMPLATE_VERSION = 1
MARKER = "HWTEMPLATE"

LOWER = "abcdefghijklmnopqrstuvwxyz"
UPPER = LOWER.upper()
DIGITS = "0123456789"
PUNCT = ".,;:!?'\"()[]{}-+=*/\\<>%&#@$_~^|"
GREEK = "αβγδεθλμπσφψωΔΣΠΩΓ"
MATHS = "∞∑∏∫√≤≥≠≈≡±×÷→⇒∈⊂∪∩∂∇∀∃"
DEFAULT_CHARS = LOWER + UPPER + DIGITS + PUNCT + GREEK + MATHS

# ---------------------------------------------------------------- geometry
# Vertical guides, in guide x-heights above the baseline.
ASC = 1.7  # cap height / ascender (dotted)
DESC = 0.7  # descender, below the baseline (dotted)
CELL_UP = 2.4  # cell top above the baseline
CELL_DOWN = 1.2  # cell bottom below the baseline
ROW_GAP = 0.75  # vertical gap between rows
SECTION_GAP = 1.3  # extra space above a section caption
LABEL_W = 2.1  # label column width
GROUP_GAP = 1.2  # minimum gap between groups on a row
GUIDE_INSET = 0.25  # guides stop short of the cell's side edges

CELL_NARROW, CELL_NORMAL, CELL_WIDE = 2.6, 3.3, 3.7
NARROW = set("ijl1!|.,;:'\"`()[]{}Itfr")
WIDE = set("mwMW@%&ωΣΠΩ∞∑∏≈≡±÷→⇒")

MARGIN_X = 40.0
MARGIN_TOP = 40.0
MARGIN_BOTTOM = 34.0
HEADER_H_FIRST = 62.0  # title + instructions on page 1
HEADER_H_OTHER = 20.0

# Template colours.  Guides are only ever straight, stroked lines in these
# colours; ingestion skips drawings that match.
CELL_COLOR = (0.784, 0.839, 0.910)
BASE_COLOR = (0.431, 0.596, 0.835)
XH_COLOR = (0.580, 0.710, 0.886)
ASC_COLOR = (0.580, 0.710, 0.886)
TEMPLATE_COLORS = (CELL_COLOR, BASE_COLOR, XH_COLOR, ASC_COLOR)
LABEL_COLOR = (0.45, 0.45, 0.45)
TEXT_COLOR = (0.33, 0.33, 0.33)

# Hints printed under the label of easily confused characters.
HINTS = {
    "l": "ell", "I": "cap i", "1": "one", "|": "bar", "O": "cap o", "0": "zero",
    "'": "apostr.", '"': "quotes", "`": "backtick", "-": "hyphen", "_": "underscore", "~": "tilde",
    "^": "caret", ".": "period", ",": "comma", ";": "semicolon", ":": "colon", "×": "times",
    "/": "slash", "\\": "backslash", "*": "star", "ε": "epsilon", "∈": "elem. of", "∂": "partial",
    "∇": "nabla", "∑": "sum", "∏": "product", "Σ": "Sigma", "Π": "Pi", "∪": "union", "∩": "intersect",
    "ν": "nu", "ψ": "psi", "φ": "phi", "θ": "theta", "≡": "equiv", "≈": "approx",
}

# Characters whose ink is legitimately small or made of separate dots.
SMALL_OK = set(".,'`\"-_~^·:;=")
DOTTED_OK = set("ij!?;:%\"'÷…=≡≈≠≤≥±∴∵äëöü")

SPILL_TOL = 0.6  # reject a cell whose ink leaves the box by more than this (x-heights)
KAPPA = 0.5522847498

FONT_FILES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    "/usr/local/share/fonts/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "C:/Windows/Fonts/seguisym.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


# ===========================================================================
# layout model
# ===========================================================================
@dataclass
class Cell:
    """One writing box, in the sheet's layout coordinates (points, y down)."""

    char: str
    x0: float
    y0: float
    x1: float
    y1: float
    baseline: float
    xh: float
    group: int = 0
    k: int = 0  # repeat index within the group

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


@dataclass
class Group:
    """A label followed by ``n`` adjacent cells of width ``cell_w``."""

    char: str
    x0: float  # left edge of the first cell
    baseline: float
    cell_w: float
    n: int

    def label_box(self, xh: float) -> Tuple[float, float, float, float]:
        return (self.x0 - LABEL_W * xh, self.baseline - CELL_UP * xh, self.x0, self.baseline + CELL_DOWN * xh)


@dataclass
class SheetLayout:
    """Cell layout of one template page.

    ``transform`` = (sx, sy, tx, ty) maps layout coordinates to the coordinates
    of the page it was read from (identity for an unmodified template)."""

    page: int
    n_pages: int
    width: float
    height: float
    xh: float
    groups: List[Group]
    params: Dict[str, object] = field(default_factory=dict)
    sections: List[Tuple[str, float, float]] = field(default_factory=list)  # (caption, x, y) for drawing
    transform: Tuple[float, float, float, float] = (1.0, 1.0, 0.0, 0.0)
    origin: str = "computed"  # computed | marker | metadata | guides | recomputed

    def cells(self) -> List[Cell]:
        out: List[Cell] = []
        xh = self.xh
        for gi, g in enumerate(self.groups):
            for k in range(g.n):
                x0 = g.x0 + k * g.cell_w
                out.append(Cell(g.char, x0, g.baseline - CELL_UP * xh, x0 + g.cell_w, g.baseline + CELL_DOWN * xh,
                                g.baseline, xh, gi, k))
        return out

    # ------------------------------------------------------------ encoding
    def payload(self) -> dict:
        return {
            "v": TEMPLATE_VERSION, "p": self.page, "n": self.n_pages, "W": round(self.width, 2),
            "H": round(self.height, 2), "xh": round(self.xh, 3), "prm": self.params,
            "G": [[g.char, round(g.x0, 2), round(g.baseline, 2), round(g.cell_w, 3), g.n] for g in self.groups],
        }

    @staticmethod
    def from_payload(d: dict, origin: str = "marker") -> "SheetLayout":
        if int(d.get("v", 0)) != TEMPLATE_VERSION:
            raise ValueError(f"unsupported calibration sheet version {d.get('v')}")
        groups = [Group(str(c), float(x), float(b), float(w), int(n)) for c, x, b, w, n in d["G"]]
        return SheetLayout(int(d["p"]), int(d["n"]), float(d["W"]), float(d["H"]), float(d["xh"]), groups,
                           dict(d.get("prm", {})), origin=origin)

    # --------------------------------------------------------- transform
    def to_page(self, x: float, y: float) -> Tuple[float, float]:
        sx, sy, tx, ty = self.transform
        return sx * x + tx, sy * y + ty


def _encode(obj) -> str:
    raw = json.dumps(obj, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode("ascii").rstrip("=")


def _decode(s: str):
    s = s.strip()
    s += "=" * (-len(s) % 4)
    return json.loads(zlib.decompress(base64.urlsafe_b64decode(s.encode("ascii"))).decode("ascii"))


# ---------------------------------------------------------------- planning
def normalize_chars(chars: Optional[Iterable[str]]) -> str:
    """Characters for a sheet, in order, without whitespace or duplicates."""
    if chars is None:
        return DEFAULT_CHARS
    seen = []
    for c in "".join(chars):
        if c.isspace() or c in seen:
            continue
        seen.append(c)
    if not seen:
        raise ValueError("no characters given for the calibration sheet")
    return "".join(seen)


def cell_width(ch: str) -> float:
    """Cell width in guide x-heights (wide glyphs get wider boxes)."""
    if ch in WIDE:
        return CELL_WIDE
    if ch in NARROW:
        return CELL_NARROW
    return CELL_NORMAL


_SECTIONS = [
    ("lower", "lowercase"), ("upper", "CAPITALS"), ("digit", "digits"), ("punct", "punctuation & symbols"),
    ("greek", "Greek"), ("letter", "other letters"), ("maths", "maths symbols"),
]


def _category(ch: str) -> str:
    if ch in LOWER:
        return "lower"
    if ch in UPPER:
        return "upper"
    if ch in DIGITS:
        return "digit"
    if 0x370 <= ord(ch) <= 0x3FF:
        return "greek"
    if 0x2100 <= ord(ch) <= 0x214F or 0x1D400 <= ord(ch) <= 0x1D7FF:
        return "maths"  # letterlike symbols (ℝ, ℕ...) and maths alphanumerics
    if ch.isalpha():
        return "letter"
    if ord(ch) < 128:
        return "punct"
    return "maths"


def layout_sheet(chars: Optional[Iterable[str]] = None, repeats: int = 4, paper: str = "letter",
                 x_height: float = 10.0) -> List[SheetLayout]:
    """Deterministic page layout of a calibration sheet."""
    chars = normalize_chars(chars)
    repeats = int(repeats)
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if paper not in PAPER_SIZES:
        raise ValueError(f"unknown paper size {paper!r} (use one of {', '.join(PAPER_SIZES)})")
    W, H = PAPER_SIZES[paper]
    xh = float(x_height)
    params = {"chars": chars, "repeats": repeats, "paper": paper, "x_height": xh}

    # group characters by section, keeping first-appearance order
    order: List[str] = []
    by_cat: Dict[str, List[str]] = defaultdict(list)
    for c in chars:
        cat = _category(c)
        if cat not in by_cat:
            order.append(cat)
        by_cat[cat].append(c)
    names = dict(_SECTIONS)

    left, right = MARGIN_X, W - MARGIN_X
    usable = right - left
    label_w = LABEL_W * xh
    gap = GROUP_GAP * xh
    if usable < label_w + CELL_WIDE * xh:
        raise ValueError(f"x-height {xh:g} pt is too large for {paper} paper")

    # a character whose boxes do not fit on one row is split over several groups
    def chunks(c: str) -> List[int]:
        per_row = max(1, int((usable - label_w) // (cell_width(c) * xh)))
        return [min(per_row, repeats - i) for i in range(0, repeats, per_row)]

    # column grid: every group starts at a column, so labels line up
    widest = max(label_w + n * cell_width(c) * xh for c in chars for n in chunks(c))
    ncols = max(1, int((usable + gap) // (widest + gap)))
    slot = (usable - (ncols - 1) * gap) / ncols

    bottom_limit = H - MARGIN_BOTTOM
    pages: List[Tuple[List[Group], List[Tuple[str, float, float]]]] = []
    groups: List[Group] = []
    sections: List[Tuple[str, float, float]] = []
    row_pitch = (CELL_UP + CELL_DOWN + ROW_GAP) * xh

    def first_baseline(page_no: int) -> float:
        return MARGIN_TOP + (HEADER_H_FIRST if page_no == 0 else HEADER_H_OTHER) + CELL_UP * xh

    page_no = 0
    y = first_baseline(0) - row_pitch  # baseline of the previous row
    for cat in order:
        units = [(c, n) for c in by_cat[cat] for n in chunks(c)]
        # section caption above the first row of the section
        if y + SECTION_GAP * xh + row_pitch + CELL_DOWN * xh > bottom_limit and groups:
            pages.append((groups, sections))
            groups, sections = [], []
            page_no += 1
            y = first_baseline(page_no) - row_pitch
        y += SECTION_GAP * xh if groups else 0.6 * xh
        y += row_pitch
        sections.append((names.get(cat, cat), left, y - CELL_UP * xh - 0.35 * xh))
        col = 0
        for c, n in units:
            if col == ncols:
                col = 0
                y += row_pitch
                if y + CELL_DOWN * xh > bottom_limit:
                    pages.append((groups, sections))
                    groups, sections = [], []
                    page_no += 1
                    y = first_baseline(page_no)
            x = left + col * (slot + gap)
            groups.append(Group(c, x + label_w, y, cell_width(c) * xh, n))
            col += 1
    if groups:
        pages.append((groups, sections))
    n = len(pages)
    return [SheetLayout(i, n, W, H, xh, gs, dict(params), secs) for i, (gs, secs) in enumerate(pages)]


# ===========================================================================
# drawing
# ===========================================================================
class _LabelFonts:
    """Pick, per character, the first available font that has a glyph."""

    def __init__(self) -> None:
        self.fonts: List[Tuple[str, pymupdf.Font, dict]] = []
        for path in FONT_FILES:
            if Path(path).is_file():
                try:
                    self.fonts.append(("hwlab%d" % len(self.fonts), pymupdf.Font(fontfile=path),
                                       {"fontfile": path}))
                except Exception:
                    continue
                break  # one good system Unicode font is enough
        try:
            f = pymupdf.Font("cjk")  # Droid Sans Fallback, bundled with PyMuPDF: Greek + maths
            self.fonts.append(("hwlab%d" % len(self.fonts), f, {"fontbuffer": f.buffer}))
        except Exception:
            pass
        self.fonts.append(("helv", pymupdf.Font("helv"), {}))
        self._inserted: Dict[int, set] = defaultdict(set)

    def pick(self, text: str) -> Optional[Tuple[str, pymupdf.Font, dict]]:
        for entry in self.fonts:
            if all(entry[1].has_glyph(ord(c)) for c in text if not c.isspace()):
                return entry
        return None

    def use(self, page: pymupdf.Page, entry) -> str:
        name, _, src = entry
        if name != "helv" and name not in self._inserted[page.number]:
            page.insert_font(fontname=name, **src)
            self._inserted[page.number].add(name)
        return name

    def wrap(self, text: str, size: float, max_w: float) -> List[str]:
        entry = self.pick(text) or self.fonts[-1]
        lines: List[str] = []
        cur = ""
        for word in text.split(" "):
            trial = (cur + " " + word) if cur else word
            if cur and entry[1].text_length(trial, fontsize=size) > max_w:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return lines

    def text(self, page: pymupdf.Page, x: float, y: float, s: str, size: float, color, align: str = "left",
             max_w: Optional[float] = None) -> bool:
        entry = self.pick(s)
        if entry is None:
            return False
        w = entry[1].text_length(s, fontsize=size)
        if max_w is not None and w > max_w:
            size *= max_w / w
            w = max_w
        if align == "center":
            x -= w / 2
        elif align == "right":
            x -= w
        page.insert_text((x, y), s, fontname=self.use(page, entry), fontsize=size, color=color)
        return True


def _hershey_label(page: pymupdf.Page, ch: str, cx: float, base: float, cap_pt: float) -> None:
    """Draw a label with the Hershey fallback when no font has the glyph."""
    from .fallback import HersheyFallback

    hf = HersheyFallback(cap_height=1.45, slant=0.0, pen_width=0.12, variation=0.0)  # clean labels
    g = hf.glyph(ch, np.random.default_rng(0))
    if g is None:
        return
    lines, w, _, _ = g
    s = cap_pt / 1.39
    sh = page.new_shape()
    for pts, pw in lines:
        p = [(cx + (q[0] - w / 2) * s, base + q[1] * s) for q in pts]
        if len(p) >= 2:
            sh.draw_polyline(p)
    sh.finish(color=LABEL_COLOR, width=0.12 * s, lineCap=1, lineJoin=1, closePath=False)
    sh.commit()


def _draw_page(page: pymupdf.Page, lay: SheetLayout, fonts: _LabelFonts, markers: bool) -> None:
    xh = lay.xh
    W, H = lay.width, lay.height
    # ---- header
    if lay.page == 0:
        fonts.text(page, MARGIN_X, MARGIN_TOP + 10, "Handwriting calibration sheet", 14, TEXT_COLOR)
        ins = ("Write the grey character once in each box beside it, in your normal handwriting. Sit "
               "letters on the solid line: small letters reach the dashed line, capitals and tall letters the upper "
               "dotted line, tails (g, p, y) the lower dotted line. Leave a box empty to skip it. Fill it in on an "
               "iPad (Files > Markup, Notes, GoodNotes...) without resizing the page, export as PDF, then run: "
               "handwritten train <your pdfs> --calibration <this pdf>")
        for i, s in enumerate(fonts.wrap(ins, 7.2, W - 2 * MARGIN_X)[:4]):
            fonts.text(page, MARGIN_X, MARGIN_TOP + 24 + 8.6 * i, s, 7.2, TEXT_COLOR, max_w=W - 2 * MARGIN_X)
        right_note = f"page 1 of {lay.n_pages}"
        fonts.text(page, W - MARGIN_X, MARGIN_TOP + 10, right_note, 7.5, LABEL_COLOR, align="right")
    else:
        fonts.text(page, MARGIN_X, MARGIN_TOP + 8, "Handwriting calibration sheet", 9, TEXT_COLOR)
        fonts.text(page, W - MARGIN_X, MARGIN_TOP + 8, f"page {lay.page + 1} of {lay.n_pages}", 7.5,
                   LABEL_COLOR, align="right")
    for cap, x, y in lay.sections:
        fonts.text(page, x, y, cap, 7.0, LABEL_COLOR)

    cells = lay.cells()
    ins = GUIDE_INSET * xh
    # ---- cell boxes
    sh = page.new_shape()
    for c in cells:
        sh.draw_rect(pymupdf.Rect(c.x0, c.y0, c.x1, c.y1))
    sh.finish(color=CELL_COLOR, width=0.5, closePath=False)
    # ---- baselines (solid)
    for c in cells:
        sh.draw_line((c.x0 + ins, c.baseline), (c.x1 - ins, c.baseline))
    sh.finish(color=BASE_COLOR, width=0.6, closePath=False)
    # ---- x-height (dashed)
    for c in cells:
        sh.draw_line((c.x0 + ins, c.baseline - xh), (c.x1 - ins, c.baseline - xh))
    sh.finish(color=XH_COLOR, width=0.45, dashes="[1.8 1.4] 0", closePath=False)
    # ---- cap / ascender and descender (dotted)
    for c in cells:
        sh.draw_line((c.x0 + ins, c.baseline - ASC * xh), (c.x1 - ins, c.baseline - ASC * xh))
        sh.draw_line((c.x0 + ins, c.baseline + DESC * xh), (c.x1 - ins, c.baseline + DESC * xh))
    sh.finish(color=ASC_COLOR, width=0.75, dashes="[0 1.7] 0", lineCap=1, closePath=False)
    sh.commit()

    # ---- labels
    size = 1.5 * xh
    for g in lay.groups:
        cx = g.x0 - 0.5 * LABEL_W * xh - 0.1 * xh
        if not fonts.text(page, cx, g.baseline, g.char, size, LABEL_COLOR, align="center", max_w=LABEL_W * xh):
            _hershey_label(page, g.char, cx, g.baseline, 1.1 * xh)
        hint = HINTS.get(g.char, "")
        if hint:
            fonts.text(page, cx, g.baseline + 0.95 * xh, hint, 0.5 * xh, LABEL_COLOR, align="center",
                       max_w=LABEL_W * xh)

    # ---- footer + machine-readable marker
    fonts.text(page, MARGIN_X, H - 18, f"handwritten calibration sheet v{TEMPLATE_VERSION} | guide x-height "
               f"{xh:g} pt | {lay.params.get('repeats')} boxes per character", 6.0, LABEL_COLOR)
    if markers:
        data = _encode(lay.payload())
        n = 110
        chunks = [data[i:i + n] for i in range(0, len(data), n)] or [""]
        x = W - MARGIN_X - 170.0
        for i, ch in enumerate(chunks):
            y = H - 10.0 - 2.4 * (len(chunks) - 1 - i)
            page.insert_text((x, y), f"{MARKER} v{TEMPLATE_VERSION} {i + 1}/{len(chunks)} {ch}",
                             fontname="helv", fontsize=2.0, render_mode=3)


def make_template(path: Union[str, Path], chars: Optional[Iterable[str]] = None, repeats: int = 4,
                  paper: str = "letter", x_height: float = 10.0, markers: bool = True) -> List[SheetLayout]:
    """Write a calibration sheet PDF and return its page layouts.

    chars: characters to include (default ``DEFAULT_CHARS``)
    repeats: boxes per character
    paper: ``letter`` or ``a4``
    x_height: guide x-height in points (the writer's natural size is ~10 pt)
    markers: embed the machine-readable layout (keep True; False is for tests)
    """
    layouts = layout_sheet(chars, repeats, paper, x_height)
    doc = pymupdf.open()
    fonts = _LabelFonts()
    for lay in layouts:
        page = doc.new_page(width=lay.width, height=lay.height)
        _draw_page(page, lay, fonts, markers)
    meta = {"title": "Handwriting calibration sheet", "creator": "handwritten", "producer": "handwritten",
            "subject": "handwritten calibration sheet"}
    if markers:
        meta["keywords"] = f"{MARKER} v{TEMPLATE_VERSION} " + _encode([l.payload() for l in layouts])
    doc.set_metadata(meta)
    try:
        doc.subset_fonts()
    except Exception:  # pragma: no cover - subsetting is an optimisation only
        pass
    doc.save(str(path), garbage=3, deflate=True)
    doc.close()
    return layouts


def chars_needed(style: Style, chars: Optional[Iterable[str]] = None, min_samples: int = 3,
                 anchors: str = "aenos") -> str:
    """Characters of ``chars`` (default: the full sheet set) that ``style``
    has fewer than ``min_samples`` real samples of.

    ``anchors`` -- common letters the style already knows well -- are put
    first: written on the same sheet they tell ``style_from_sheets`` how much
    bigger than usual the writer wrote in the boxes."""
    need = [c for c in normalize_chars(chars) if len(style.bank.get(c, [])) < min_samples]
    if not need:
        return ""
    anchor = [c for c in anchors if len(style.bank.get(c, [])) >= min_samples]
    return normalize_chars(anchor + need)


# ===========================================================================
# reading a sheet's layout back
# ===========================================================================
_MARKER_RE = re.compile(MARKER + r" v(\d+) (\d+)/(\d+) ([A-Za-z0-9_\-]*)")


def _marker_layout(page: pymupdf.Page) -> Optional[SheetLayout]:
    try:
        text = page.get_text("text")
    except Exception:
        return None
    chunks: Dict[int, str] = {}
    total = 0
    for m in _MARKER_RE.finditer(text):
        if int(m.group(1)) != TEMPLATE_VERSION:
            continue
        chunks[int(m.group(2))] = m.group(4)
        total = int(m.group(3))
    if not total or len(chunks) < total:
        return None
    try:
        return SheetLayout.from_payload(_decode("".join(chunks[i] for i in range(1, total + 1))), "marker")
    except Exception:
        return None


def _metadata_layouts(doc: pymupdf.Document) -> List[SheetLayout]:
    kw = (doc.metadata or {}).get("keywords") or ""
    m = re.search(MARKER + r" v(\d+) ([A-Za-z0-9_\-]+)", kw)
    if not m or int(m.group(1)) != TEMPLATE_VERSION:
        return []
    try:
        return [SheetLayout.from_payload(d, "metadata") for d in _decode(m.group(2))]
    except Exception:
        return []


def _close_color(c, ref, tol: float = 0.03) -> bool:
    return c is not None and len(c) == 3 and max(abs(float(a) - b) for a, b in zip(c, ref)) <= tol


def _is_template_color(c, tol: float = 0.03) -> bool:
    return any(_close_color(c, ref, tol) for ref in TEMPLATE_COLORS)


def _observed_cells(page: pymupdf.Page) -> List[Tuple[float, float, float, float]]:
    """Cell boxes found among the page's vector graphics."""
    rects = []
    for d in page.get_drawings():
        if d.get("type") != "s" or not _close_color(d.get("color"), CELL_COLOR):
            continue
        items = d.get("items", [])
        lines: List = []
        for it in items:
            if it[0] == "re":
                r = it[1]
                rects.append((r.x0, r.y0, r.x1, r.y1))
            elif it[0] == "qu":
                q = it[1].rect
                rects.append((q.x0, q.y0, q.x1, q.y1))
            elif it[0] == "l":
                lines.append(it)
        # rectangles re-encoded as 4 lines
        for i in range(0, len(lines) - 3, 4):
            pts = [(p.x, p.y) for it in lines[i:i + 4] for p in it[1:3]]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            rects.append((min(xs), min(ys), max(xs), max(ys)))
    out = []
    for r in rects:
        x0, y0, x1, y1 = min(r[0], r[2]), min(r[1], r[3]), max(r[0], r[2]), max(r[1], r[3])
        if x1 - x0 > 3 and y1 - y0 > 3:
            out.append((x0, y0, x1, y1))
    return sorted(set((round(a, 2), round(b, 2), round(c, 2), round(d, 2)) for a, b, c, d in out),
                  key=lambda r: (r[1], r[0]))


def _visible_chars(page: pymupdf.Page) -> List[Tuple[str, Tuple[float, float, float, float], float]]:
    out = []
    try:
        raw = page.get_text("rawdict")
    except Exception:
        return out
    for b in raw.get("blocks", []):
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                if span.get("size", 0) < 3.0:
                    continue  # markers / hairline text
                for c in span.get("chars", []):
                    if not c["c"].isspace():
                        out.append((c["c"], tuple(c["bbox"]), float(span["size"])))
    return out


def _register(lay: SheetLayout, observed: Sequence[Tuple[float, float, float, float]], page_w: float,
              page_h: float) -> Tuple[Tuple[float, float, float, float], float]:
    """Fit layout -> page (sx, sy, tx, ty) by matching cell boxes.

    Starts from two guesses (page size ratio; bounding box of all cell
    boxes), refines each by nearest-box matching + least squares and keeps
    the better one.  Returns the transform and the fraction of expected
    cells matched (0 when there is nothing to match against)."""
    k = page_w / lay.width
    if abs(page_h / lay.height - k) > 0.01:
        k = min(page_w / lay.width, page_h / lay.height)
    T0 = (k, k, 0.5 * (page_w - k * lay.width), 0.5 * (page_h - k * lay.height))
    cells = lay.cells()
    if not cells or not observed:
        return T0, 0.0
    E = np.array([[c.x0, c.y0, c.x1, c.y1] for c in cells])
    O = np.asarray(observed, float)
    guesses = [T0]
    ew, eh = E[:, 2].max() - E[:, 0].min(), E[:, 3].max() - E[:, 1].min()
    ow, oh = O[:, 2].max() - O[:, 0].min(), O[:, 3].max() - O[:, 1].min()
    if ew > 1 and eh > 1 and ow > 1 and oh > 1:
        sx, sy = ow / ew, oh / eh
        guesses.append((sx, sy, O[:, 0].min() - sx * E[:, 0].min(), O[:, 1].min() - sy * E[:, 1].min()))
    best = (T0, 0.0, np.inf)
    for T in guesses:
        T, frac, res = _refine(E, O, T)
        if frac > best[1] + 1e-9 or (abs(frac - best[1]) < 1e-9 and res < best[2]):
            best = (T, frac, res)
    return best[0], best[1]


def _refine(E: np.ndarray, O: np.ndarray, T: Tuple[float, float, float, float], iters: int = 4):
    Oc = np.column_stack([(O[:, 0] + O[:, 2]) / 2, (O[:, 1] + O[:, 3]) / 2])
    frac, res = 0.0, np.inf
    for it in range(iters + 1):
        sx, sy, tx, ty = T
        M = np.column_stack([E[:, 0] * sx + tx, E[:, 1] * sy + ty, E[:, 2] * sx + tx, E[:, 3] * sy + ty])
        Mc = np.column_stack([(M[:, 0] + M[:, 2]) / 2, (M[:, 1] + M[:, 3]) / 2])
        D = np.hypot(Mc[:, None, 0] - Oc[None, :, 0], Mc[:, None, 1] - Oc[None, :, 1])
        j = np.argmin(D, axis=1)
        d = D[np.arange(len(E)), j]
        ok = d < 0.3 * np.minimum(M[:, 2] - M[:, 0], M[:, 3] - M[:, 1])
        frac = float(ok.mean())
        res = float(d[ok].mean()) if ok.any() else np.inf
        if ok.sum() < 3 or it == iters:
            break
        ex = np.concatenate([E[ok, 0], E[ok, 2]])
        ox = np.concatenate([O[j[ok], 0], O[j[ok], 2]])
        ey = np.concatenate([E[ok, 1], E[ok, 3]])
        oy = np.concatenate([O[j[ok], 1], O[j[ok], 3]])
        if np.ptp(ex) < 1e-3 or np.ptp(ey) < 1e-3:
            break
        sx, tx = np.polyfit(ex, ox, 1)
        sy, ty = np.polyfit(ey, oy, 1)
        T = (float(sx), float(sy), float(tx), float(ty))
    return T, frac, res


def _label_char(chars, g: Group, xh: float) -> Optional[str]:
    """The printed label of a group (page coordinates), if it is text."""
    lx0, ly0, lx1, _ = g.label_box(xh)
    best = None
    for c, bb, size in chars:
        if size < 1.0 * xh:
            continue  # hints, captions
        cx, cy = 0.5 * (bb[0] + bb[2]), 0.5 * (bb[1] + bb[3])
        if lx0 - 0.5 * xh <= cx <= lx1 + 0.1 * xh and ly0 <= cy <= g.baseline + 0.3 * xh:
            if best is None or size > best[1]:
                best = (c, size)
    return best[0] if best else None


def _guide_layout(page: pymupdf.Page, observed, page_no: int, n_pages: int,
                  hint_fn=None) -> Optional[SheetLayout]:
    """Rebuild a layout from the cell boxes and printed labels on the page."""
    if not observed:
        return None
    rows: List[List[Tuple[float, float, float, float]]] = []
    for r in sorted(observed, key=lambda r: (round(r[1]), r[0])):
        if rows and abs(rows[-1][0][1] - r[1]) < 1.0 and abs(rows[-1][0][3] - r[3]) < 1.0:
            rows[-1].append(r)
        else:
            rows.append([r])
    chars = _visible_chars(page)
    groups: List[Group] = []
    xhs = []
    hint = None
    for row in rows:
        row.sort(key=lambda r: r[0])
        runs: List[List[Tuple[float, float, float, float]]] = []
        for r in row:
            if runs and abs(runs[-1][-1][2] - r[0]) < 1.0 and abs((runs[-1][-1][2] - runs[-1][-1][0]) -
                                                                  (r[2] - r[0])) < 1.0:
                runs[-1].append(r)
            else:
                runs.append([r])
        for run in runs:
            x0, y0, _, y1 = run[0]
            xh = (y1 - y0) / (CELL_UP + CELL_DOWN)
            base = y0 + CELL_UP * xh
            cw = float(np.mean([r[2] - r[0] for r in run]))
            g = Group("", x0, base, cw, len(run))
            label = _label_char(chars, g, xh)
            if label is not None:
                g.char = label
            elif hint_fn is not None:
                # unreadable label (drawn as graphics): take the recomputed guess
                if hint is None:
                    hint = hint_fn() or False
                for hg in (hint.groups if hint else []):
                    hx, hy = hint.to_page(hg.x0, hg.baseline)
                    if abs(hx - x0) < 0.3 * cw and abs(hy - base) < 0.5 * xh:
                        g.char = hg.char
                        break
            if g.char:
                groups.append(g)
                xhs.append(xh)
    if not groups:
        return None
    xh = float(np.median(xhs))
    return SheetLayout(page_no, n_pages, page.rect.width, page.rect.height, xh, groups, {}, origin="guides")


def read_layouts(src: Union[str, Path, pymupdf.Document]) -> List[Optional[SheetLayout]]:
    """The calibration-sheet layout of every page (None for pages that are
    not part of a sheet).  Layouts carry a ``transform`` from layout to page
    coordinates."""
    own = not isinstance(src, pymupdf.Document)
    doc = pymupdf.open(str(src)) if own else src
    try:
        meta = _metadata_layouts(doc)
        out: List[Optional[SheetLayout]] = []
        n = len(doc)
        for pno, page in enumerate(doc):
            pw, ph = page.rect.width, page.rect.height
            observed = _observed_cells(page)
            lay = _marker_layout(page)
            if lay is None and meta and pno < len(meta) and len(meta) == n:
                lay = meta[pno]
            if lay is not None:
                lay.transform, _ = _register(lay, observed, pw, ph)
                out.append(lay)
                continue
            # no embedded layout: rebuild it from the page graphics, or recompute
            # the default sheet and check it against the ink
            paper = min(PAPER_SIZES, key=lambda p: abs(PAPER_SIZES[p][0] / PAPER_SIZES[p][1] - pw / ph))
            if observed:
                lay = _guide_layout(page, observed, pno, n, lambda: _guess_from_guides(page, observed, paper))
            else:
                lay = _guess_from_ink(page, pno, n, paper)
            out.append(lay)
        return out
    finally:
        if own:
            doc.close()


REPEAT_GUESSES = (4, 3, 5, 2, 6, 1, 7, 8)


def _guess_from_guides(page: pymupdf.Page, observed, paper: str) -> Optional[SheetLayout]:
    """The default-sheet page (any of REPEAT_GUESSES) whose cells coincide
    with the page's cell boxes and whose labels agree with the printed ones."""
    pw, ph = page.rect.width, page.rect.height
    chars = _visible_chars(page)
    O = np.asarray(observed, float)
    best: Tuple[float, Optional[SheetLayout]] = (-1.0, None)
    for rep in REPEAT_GUESSES:
        for lay in layout_sheet(None, rep, paper):
            T, frac_e = _register(lay, observed, pw, ph)
            if frac_e < 0.9:
                continue
            sx, sy, tx, ty = T
            ms = np.array([[g.x0 * sx + tx, g.baseline * sy + ty] for g in lay.groups])
            n_cells = sum(g.n for g in lay.groups)
            if len(O) > 1.1 * n_cells:
                continue  # the page has many more boxes than this layout
            agree = seen = 0
            for g, (gx, gy) in zip(lay.groups, ms):
                c = _label_char(chars, Group(g.char, gx, gy, g.cell_w * sx, g.n), lay.xh * sy)
                if c is not None:
                    seen += 1
                    agree += c == g.char
            if seen and agree < 0.8 * seen:
                continue
            score = frac_e + (agree / seen if seen else 0.0)
            if score > best[0]:
                lay.transform, lay.origin = T, "recomputed"
                best = (score, lay)
    return best[1]


def _ink_fit(lay: SheetLayout, boxes: np.ndarray) -> float:
    """Fraction of ink clusters (page coordinates) that sit inside one cell."""
    if not len(boxes):
        return 0.0
    sx, sy, tx, ty = lay.transform
    R = np.array([[c.x0 * sx + tx, c.y0 * sy + ty, c.x1 * sx + tx, c.y1 * sy + ty] for c in lay.cells()])
    tol = SPILL_TOL * lay.xh * sy
    good = 0
    for b in boxes:
        cx, cy = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
        hit = np.nonzero((R[:, 0] <= cx) & (cx <= R[:, 2]) & (R[:, 1] <= cy) & (cy <= R[:, 3]))[0]
        if len(hit):
            r = R[hit[0]]
            if max(r[0] - b[0], b[2] - r[2], r[1] - b[1], b[3] - r[3]) <= tol:
                good += 1
    return good / len(boxes)


def _guess_from_ink(page: pymupdf.Page, pno: int, n_pages: int, paper: str,
                    min_fit: float = 0.75) -> Optional[SheetLayout]:
    """Last resort when the template background is gone (flattened to an
    image): the default-sheet page whose cells the ink fits best."""
    ink = user_ink(page)
    if not ink:
        return None
    sboxes = np.array([_bbox([sps]) for sps, _ in ink])
    boxes = np.array([[sboxes[cl, 0].min(), sboxes[cl, 1].min(), sboxes[cl, 2].max(), sboxes[cl, 3].max()]
                      for cl in _clusters(sboxes)])
    pw, ph = page.rect.width, page.rect.height
    best: Tuple[float, Optional[SheetLayout]] = (-1.0, None)
    for rep in REPEAT_GUESSES:
        lays = layout_sheet(None, rep, paper)
        for lay in lays:
            lay.transform, _ = _register(lay, [], pw, ph)
            fit = _ink_fit(lay, boxes)
            if fit < min_fit:
                continue
            score = fit + (0.05 if len(lays) == n_pages and lay.page == pno else 0.0) + (0.02 if rep == 4 else 0.0)
            if score > best[0]:
                lay.origin = "recomputed"
                best = (score, lay)
    return best[1]


# ===========================================================================
# user ink
# ===========================================================================
def _quarter(c: np.ndarray, a: np.ndarray, b: np.ndarray, r: float) -> np.ndarray:
    """Cubic quarter arc around ``c`` from direction ``a`` to ``b`` (unit vectors)."""
    p0, p3 = c + r * a, c + r * b
    return np.stack([p0, p0 + KAPPA * r * b, p3 + KAPPA * r * a, p3])


def _line(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.stack([p, p + (q - p) / 3.0, p + 2.0 * (q - p) / 3.0, q])


def capsule(p: Sequence[float], q: Sequence[float], width: float) -> np.ndarray:
    """Closed cubic outline of a round-capped segment p-q (shape (K, 4, 2)).

    All capsules share one orientation, so a union of them fills correctly
    under the non-zero winding rule."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    r = max(1e-3, 0.5 * float(width))
    d = q - p
    L = float(np.hypot(*d))
    u = d / L if L > 1e-6 else np.array([1.0, 0.0])
    n = np.array([-u[1], u[0]])
    segs = []
    if L > 1e-6:
        segs.append(_line(p + r * n, q + r * n))
    segs.append(_quarter(q, n, u, r))
    segs.append(_quarter(q, u, -n, r))
    if L > 1e-6:
        segs.append(_line(q - r * n, p - r * n))
    segs.append(_quarter(p, -n, -u, r))
    segs.append(_quarter(p, -u, n, r))
    return np.stack(segs)


def _rdp(pts: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker polyline simplification."""
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        ab = b - a
        L = float(np.hypot(*ab))
        seg = pts[i + 1:j]
        if L < 1e-9:
            dist = np.hypot(*(seg - a).T)
        else:
            dist = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / L
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return pts[keep]


def polyline_outline(points: np.ndarray, width: float, eps: Optional[float] = None) -> List[np.ndarray]:
    """Filled outline (list of closed cubic sub-paths) of a polyline drawn
    with a round pen of the given width: a union of capsules."""
    pts = np.asarray(points, float).reshape(-1, 2)
    if len(pts) == 0:
        return []
    if eps is None:
        eps = min(0.05, 0.06 * width)
    pts = _rdp(pts, eps)
    if len(pts) == 1:
        return [capsule(pts[0], pts[0], width)]
    return [capsule(a, b, width) for a, b in zip(pts[:-1], pts[1:])]


def _items_to_polylines(items, closed: bool) -> List[np.ndarray]:
    t = np.linspace(0.0, 1.0, 9)[:, None]
    B = np.concatenate([(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3], axis=1)
    polys: List[List[np.ndarray]] = []
    cur: List[np.ndarray] = []
    for it in items:
        kind = it[0]
        if kind == "l":
            seg = np.array([[it[1].x, it[1].y], [it[2].x, it[2].y]])
        elif kind == "c":
            seg = B @ np.array([[p.x, p.y] for p in it[1:5]])
        elif kind in ("re", "qu"):
            if kind == "re":
                r = it[1]
                pts = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1), (r.x0, r.y0)]
            else:
                q = it[1]
                pts = [(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y), (q.ul.x, q.ul.y)]
            if cur:
                polys.append(cur)
                cur = []
            polys.append([np.asarray(pts, float)])
            continue
        else:
            continue
        if cur and np.hypot(*(cur[-1][-1] - seg[0])) > 1e-3:
            polys.append(cur)
            cur = []
        cur.append(seg if not cur else seg[1:])
    if cur:
        polys.append(cur)
    out = []
    for parts in polys:
        p = np.vstack(parts)
        if closed and len(p) > 2 and np.hypot(*(p[0] - p[-1])) > 1e-3:
            p = np.vstack([p, p[:1]])
        out.append(p)
    return out


def _all_straight(items) -> bool:
    return all(it[0] in ("l", "re", "qu") for it in items)


def _opacity(v) -> float:
    return 1.0 if v is None else float(v)


def user_ink(page: pymupdf.Page) -> List[Tuple[List[np.ndarray], Tuple[float, float, float]]]:
    """Vector ink on a page that is not part of the template: a list of
    (stroke as closed cubic sub-paths in page coordinates, colour)."""
    out = []
    area = page.rect.width * page.rect.height
    for d in page.get_drawings():
        items = d.get("items") or []
        if not items:
            continue
        r = d.get("rect")
        if r is not None and r.width * r.height > 0.25 * area:
            continue  # page backgrounds, frames
        kind = d.get("type")
        if kind in ("f", "fs") and d.get("fill") is not None:
            col = tuple(float(x) for x in d["fill"][:3])
            if (_is_template_color(col) and _all_straight(items)) or min(col) > 0.95:
                continue
            if _opacity(d.get("fill_opacity")) < 0.5:
                continue  # highlighter
            sps = _items_to_subpaths(items)
            if sps:
                out.append((sps, col))
        elif kind == "s" and d.get("color") is not None:
            col = tuple(float(x) for x in d["color"][:3])
            if (_is_template_color(col) and _all_straight(items)) or min(col) > 0.95:
                continue
            if _close_color(col, LABEL_COLOR, 0.004):
                continue  # labels drawn with the Hershey fallback
            if _opacity(d.get("stroke_opacity")) < 0.5:
                continue
            w = float(d.get("width") or 1.0)
            sps = []
            for poly in _items_to_polylines(items, bool(d.get("closePath"))):
                sps.extend(polyline_outline(poly, w))
            if sps:
                out.append((sps, col))
    return out


# ===========================================================================
# ingestion
# ===========================================================================
def _bbox(strokes: Sequence[Sequence[np.ndarray]]) -> Tuple[float, float, float, float]:
    return subpaths_bbox([sp for st in strokes for sp in st])


def _clusters(boxes: np.ndarray) -> List[List[int]]:
    """Group strokes whose bounding boxes intersect (union-find)."""
    n = len(boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n:
        ov = ((boxes[:, None, 0] <= boxes[None, :, 2]) & (boxes[None, :, 0] <= boxes[:, None, 2]) &
              (boxes[:, None, 1] <= boxes[None, :, 3]) & (boxes[None, :, 1] <= boxes[:, None, 3]))
        for i, j in zip(*np.nonzero(np.triu(ov, 1))):
            a, b = find(int(i)), find(int(j))
            if a != b:
                parent[a] = b
    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def _strip_strays(char: str, strokes: List[List[np.ndarray]], xh: float) -> List[List[np.ndarray]]:
    """Drop tiny detached marks (pen taps) from glyphs that have no dots."""
    if char in DOTTED_OK or len(strokes) < 2 or not (char.isalnum() or char in MATHS):
        return strokes
    boxes = np.array([_bbox([st]) for st in strokes])
    size = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    big = size.max()
    keep = [st for st, s in zip(strokes, size) if not (s < 0.12 * xh and s < 0.3 * big)]
    return keep or strokes


@dataclass
class SheetReport:
    pages: int = 0
    cells: int = 0
    filled: int = 0
    accepted: int = 0
    rejected: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    origins: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


def sheet_glyphs(src: Union[str, Path], report: Optional[SheetReport] = None
                 ) -> Tuple[List[Tuple[BankGlyph, float]], Counter]:
    """Glyphs written into the cells of one filled sheet.

    Returns ``[(glyph in guide x-height units, guide x-height pt)]`` and a
    Counter of ink colours."""
    report = report if report is not None else SheetReport()
    colors: Counter = Counter()
    out: List[Tuple[BankGlyph, float]] = []
    doc = pymupdf.open(str(src))
    try:
        if any(True for p in doc for _ in p.annots()):
            try:
                doc.bake()  # ink annotations -> page content
            except Exception:
                pass
        for page in doc:
            if page.rotation:
                page.set_rotation(0)  # work in the template's (unrotated) coordinates
        layouts = read_layouts(doc)
        if not any(layouts):
            warnings.warn(f"{Path(str(src)).name}: no calibration-sheet layout found "
                          f"(make one with 'handwritten template')")
        for pno, page in enumerate(doc):
            lay = layouts[pno] if pno < len(layouts) else None
            if lay is None:
                continue
            report.pages += 1
            report.origins[lay.origin] += 1
            out.extend(_page_glyphs(page, pno, lay, report, colors))
    finally:
        doc.close()
    return out, colors


def _page_glyphs(page: pymupdf.Page, pno: int, lay: SheetLayout, report: SheetReport,
                 colors: Counter) -> List[Tuple[BankGlyph, float]]:
    cells = lay.cells()
    report.cells += len(cells)
    ink = user_ink(page)
    if not ink or not cells:
        return []
    sx, sy, tx, ty = lay.transform
    A = np.array([[1.0 / sx, 0.0, -tx / sx], [0.0, 1.0 / sy, -ty / sy]])  # page -> layout
    strokes = []
    cols = []
    for sps, col in ink:
        strokes.append([(sp.reshape(-1, 2) @ A[:, :2].T + A[:, 2]).reshape(sp.shape) for sp in sps])
        cols.append(col)
    boxes = np.array([_bbox([st]) for st in strokes])
    R = np.array([[c.x0, c.y0, c.x1, c.y1] for c in cells])
    per_cell: Dict[int, List[int]] = defaultdict(list)
    for cl in _clusters(boxes):
        b = boxes[cl]
        cx = 0.5 * (b[:, 0].min() + b[:, 2].max())
        cy = 0.5 * (b[:, 1].min() + b[:, 3].max())
        hit = np.nonzero((R[:, 0] <= cx) & (cx <= R[:, 2]) & (R[:, 1] <= cy) & (cy <= R[:, 3]))[0]
        if len(hit):
            per_cell[int(hit[0])].extend(cl)
    out = []
    for ci in sorted(per_cell):
        c = cells[ci]
        idx = sorted(per_cell[ci])
        report.filled += 1
        xh = c.xh
        sts = _strip_strays(c.char, [strokes[i] for i in idx], xh)
        x0, y0, x1, y1 = _bbox(sts)
        over = max(c.x0 - x0, x1 - c.x1, c.y0 - y0, y1 - c.y1) / xh
        if over > SPILL_TOL:
            report.rejected["spill"] += 1
            continue
        size = max(x1 - x0, y1 - y0) / xh
        if size < 0.08 or (size < 0.3 and c.char not in SMALL_OK and (c.char.isalnum() or c.char in MATHS)):
            report.rejected["tiny"] += 1
            continue
        off = np.array([x0, c.baseline])
        gst = [[(sp - off) / xh for sp in st] for st in sts]
        g = BankGlyph(c.char, gst, (x1 - x0) / xh, (c.baseline - y0) / xh, (c.baseline - y1) / xh,
                      f"sheet:{pno + 1}", -1)
        out.append((g, xh))
        report.accepted += 1
        for i in idx:
            colors[tuple(round(v, 3) for v in cols[i])] += 1
    return out


def _height(g: BankGlyph) -> float:
    return g.top - g.bottom


def _scale_glyph(g: BankGlyph, k: float) -> None:
    if abs(k - 1.0) < 1e-6:
        return
    g.strokes = [[sp * k for sp in st] for st in g.strokes]
    g.width *= k
    g.top *= k
    g.bottom *= k


NO_RESCALE = set(".,'`-_~\"=:;·^")


def style_from_sheets(paths: Sequence[Union[str, Path]], base: Optional[Style] = None,
                      rescale: bool = True, report: Optional[SheetReport] = None) -> Style:
    """Learn glyphs from filled calibration sheets.

    Glyph sizes: sheets are normalised with their printed guides.  People
    tend to write bigger (or smaller) in boxes than in free text, so when
    ``base`` has >= 3 samples of a character its sheet samples are rescaled to
    the base class median height; the median of those per-character factors
    (or, without a base, the measured x-height of the sheet's small letters)
    rescales the remaining characters.

    The returned style's stats are copied from ``base`` (or defaults with the
    sheet's x-height) and its bank holds only the sheet glyphs; combine with
    ``base.merge(result)``.
    """
    report = report if report is not None else SheetReport()
    items: List[Tuple[BankGlyph, float]] = []
    colors: Counter = Counter()
    for p in paths:
        g, c = sheet_glyphs(p, report)
        items.extend(g)
        colors.update(c)
    bank: Dict[str, List[BankGlyph]] = defaultdict(list)
    for g, _ in items:
        bank[g.char].append(g)
    guide_xh = float(np.median([x for _, x in items])) if items else 10.0

    # ---- size calibration
    k_global = 1.0
    if rescale and bank:
        ratios: Dict[str, float] = {}
        if base is not None:
            for c, gs in bank.items():
                bs = base.bank.get(c, [])
                if len(bs) >= 3 and c not in NO_RESCALE:
                    h_base = float(np.median([_height(b) for b in bs]))
                    h_sheet = float(np.median([_height(g) for g in gs]))
                    if h_base > 0.05 and h_sheet > 0.05:
                        ratios[c] = float(np.clip(h_base / h_sheet, 0.5, 2.0))
        alnum = [v for c, v in ratios.items() if c.isalnum()]
        if len(alnum) >= 3:
            k_global = float(np.median(alnum))
        else:
            tops = [g.top for c, gs in bank.items() if c in "acemnorsuvwxz" for g in gs]
            if len(tops) >= 3:
                k_global = 1.0 / float(np.median(tops))
        k_global = float(np.clip(k_global, 0.6, 1.6))
        for c, gs in bank.items():
            k = ratios.get(c, k_global)
            for g in gs:
                _scale_glyph(g, k)

    gid = 0
    for c in bank:
        for g in bank[c]:
            g.gid = gid
            gid += 1
            if glyph_pen_width is not None and hasattr(g, "pen"):
                g.pen = glyph_pen_width(g)

    if base is not None:
        stats = copy.deepcopy(base.stats)
    else:
        stats = StyleStats()
        stats.xh_pt = guide_xh / k_global
        allg = [g for gs in bank.values() for g in gs]
        if allg:
            caps = [g.top for g in allg if g.char.isupper() or g.char.isdigit()]
            tall = [g.top for g in allg if g.char in "bdhkl"]
            desc = [-g.bottom for g in allg if g.char in "gpqy"]
            if len(tall) >= 2:
                stats.ascender = float(np.median(tall))
            if len(caps) >= 3:
                stats.cap_height = float(np.median(caps))
            if len(desc) >= 2:
                stats.descender = float(np.median(desc))
            stats.slant = estimate_slant(allg)
            pens = [g.pen for g in allg if getattr(g, "pen", 0.0) > 0 and g.char.isalnum()]
            stats.pen_width = float(np.median(pens)) if pens else estimate_pen_width(allg)
        if colors:
            stats.ink = tuple(float(v) for v in colors.most_common(1)[0][0])  # type: ignore[assignment]
    stats.char_counts = {c: len(v) for c, v in sorted(bank.items())}
    meta = {
        "calibration_sheets": [Path(str(p)).name for p in paths],
        "n_glyphs": sum(len(v) for v in bank.values()),
        "sheet_scale": round(k_global, 4),
        "sheet_guide_xh_pt": round(guide_xh, 3),
        "sheet_cells_filled": report.filled,
        "sheet_rejected": dict(report.rejected),
        "sheet_layout_origin": dict(report.origins),
    }
    return Style(stats=stats, bank=dict(bank), meta=meta)


def merge_calibration(base: Style, cal: Style, replace_below: int = 5, min_sheet: int = 3) -> Style:
    """``base.merge(cal)`` where trusted sheet samples replace scarce free-text ones.

    Free-text labels come from handwriting recognition, and for characters
    seen only a few times a sample is often a mis-recognised neighbour (an
    ``f`` filed as ``F``...).  For a character with fewer than
    ``replace_below`` free-text samples and at least ``min_sheet`` sheet
    samples only the sheet samples are kept."""
    bank = {}
    replaced = []
    for c, gs in base.bank.items():
        if len(gs) < replace_below and len(cal.bank.get(c, [])) >= min_sheet:
            replaced.append(c)
            continue
        bank[c] = list(gs)
    meta = dict(base.meta)
    if replaced:
        meta["replaced_by_sheet"] = "".join(sorted(replaced))
    out = Style(stats=copy.deepcopy(base.stats), bank=bank, meta=meta).merge(cal)
    out.meta.setdefault("calibration_sheets", cal.meta.get("calibration_sheets", []))
    return out


__all__ = [
    "DEFAULT_CHARS", "Cell", "Group", "SheetLayout", "SheetReport", "capsule", "cell_width", "chars_needed",
    "layout_sheet", "make_template", "merge_calibration", "normalize_chars", "polyline_outline", "read_layouts",
    "sheet_glyphs",
    "style_from_sheets", "user_ink",
]
