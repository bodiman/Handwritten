"""Extract handwriting ink and its recognised text from PDFs.

Designed for PDFs exported from iPad apps that use Apple PencilKit (Files /
Markup, Notes, ...): every pen stroke is a separate filled vector path drawn in
writing order, and iOS adds an invisible text layer with its handwriting
recognition result.  Both are recovered here:

* ``Stroke`` objects (outline Beziers, page, drawing order, colour)
* ``OcrWord`` objects (recognised text + bounding box) from invisible text
* the bounding boxes of the *printed* page content (typeset text, rules...),
  used when writing onto an existing PDF.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pymupdf

from .geometry import BBox, Stroke, line_as_cubic

INVISIBLE = 3  # PDF text render mode "neither fill nor stroke"


@dataclass
class OcrWord:
    text: str
    bbox: BBox
    page: int
    source: str = ""
    order: int = 0


@dataclass
class InkPage:
    source: str
    page: int
    width: float
    height: float
    strokes: List[Stroke] = field(default_factory=list)
    words: List[OcrWord] = field(default_factory=list)
    printed: List[BBox] = field(default_factory=list)  # visible non-ink content
    other_ink: List[Stroke] = field(default_factory=list)  # e.g. highlighter


def _items_to_subpaths(items) -> List[np.ndarray]:
    """Split a pymupdf drawing item list into closed cubic sub-paths."""
    subpaths: List[List[np.ndarray]] = []
    cur: List[np.ndarray] = []
    last: Optional[np.ndarray] = None
    for it in items:
        kind = it[0]
        if kind == "c":
            seg = np.array([[p.x, p.y] for p in it[1:5]], dtype=float)
        elif kind == "l":
            seg = line_as_cubic((it[1].x, it[1].y), (it[2].x, it[2].y))
        elif kind == "re":
            r = it[1]
            pts = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1), (r.x0, r.y0)]
            if cur:
                subpaths.append(cur)
            subpaths.append([line_as_cubic(pts[i], pts[i + 1]) for i in range(4)])
            cur, last = [], None
            continue
        elif kind == "qu":
            q = it[1]
            pts = [(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y), (q.ul.x, q.ul.y)]
            if cur:
                subpaths.append(cur)
            subpaths.append([line_as_cubic(pts[i], pts[i + 1]) for i in range(4)])
            cur, last = [], None
            continue
        else:
            continue
        if last is not None and np.hypot(*(seg[0] - last)) > 1e-3:
            subpaths.append(cur)
            cur = []
        cur.append(seg)
        last = seg[3]
    if cur:
        subpaths.append(cur)
    return [np.stack(sp) for sp in subpaths if len(sp)]


def _is_ink_like(d) -> bool:
    """Filled paths made mostly of curves look like digital ink."""
    if d.get("type") not in ("f", "fs") or d.get("fill") is None:
        return False
    items = d.get("items", [])
    n_curves = sum(1 for it in items if it[0] == "c")
    return n_curves >= 4


def _color_key(c) -> Tuple[float, float, float]:
    return tuple(round(float(x), 3) for x in c)  # type: ignore[return-value]


def detect_ink_colors(pdf_paths: Sequence[str | Path], top: int = 1) -> List[Tuple[float, float, float]]:
    """Most common fill colours of ink-like paths across documents."""
    cnt: Counter = Counter()
    for path in pdf_paths:
        with pymupdf.open(str(path)) as doc:
            for page in doc:
                for d in page.get_drawings():
                    if _is_ink_like(d):
                        cnt[_color_key(d["fill"])] += 1
    return [c for c, _ in cnt.most_common(top)]


def _invisible_fonts(page) -> set:
    fonts = set()
    try:
        for t in page.get_texttrace():
            if t.get("type") == INVISIBLE:
                fonts.add(t.get("font"))
    except Exception:  # pragma: no cover - very old pymupdf
        pass
    return fonts


def extract_page(page, page_no: int, source: str, ink_colors: Optional[Sequence[Tuple[float, float, float]]] = None,
                 color_tol: float = 0.08) -> InkPage:
    out = InkPage(source=source, page=page_no, width=page.rect.width, height=page.rect.height)
    seq = 0
    for d in page.get_drawings():
        if _is_ink_like(d):
            col = _color_key(d["fill"])
            sps = _items_to_subpaths(d["items"])
            if not sps:
                continue
            st = Stroke(sps, page=page_no, seq=seq, source=source, color=col)
            seq += 1
            if ink_colors is None or any(max(abs(a - b) for a, b in zip(col, ic)) <= color_tol for ic in ink_colors):
                out.strokes.append(st)
            else:
                out.other_ink.append(st)
        else:
            r = d.get("rect")
            if r is not None and r.width * r.height > 0 or (r is not None and (r.width > 0 or r.height > 0)):
                out.printed.append((r.x0, r.y0, r.x1, r.y1))

    inv = _invisible_fonts(page)
    raw = page.get_text("rawdict")
    order = 0
    for b in raw.get("blocks", []):
        if b.get("type") == 1:  # image block
            x0, y0, x1, y1 = b["bbox"]
            out.printed.append((x0, y0, x1, y1))
            continue
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                chars = span.get("chars", [])
                invisible = span.get("font") in inv
                if not invisible:
                    txt = "".join(c["c"] for c in chars).strip()
                    if txt:
                        out.printed.append(tuple(span["bbox"]))  # type: ignore[arg-type]
                    continue
                # split the span into whitespace separated words
                cur: List[dict] = []
                for c in chars + [{"c": " ", "bbox": None}]:
                    if c["c"].isspace():
                        if cur:
                            text = "".join(cc["c"] for cc in cur)
                            xs0 = min(cc["bbox"][0] for cc in cur)
                            xs1 = max(cc["bbox"][2] for cc in cur)
                            y0 = span["bbox"][1]
                            y1 = span["bbox"][3]
                            out.words.append(OcrWord(text, (xs0, y0, xs1, y1), page_no, source, order))
                            order += 1
                        cur = []
                    else:
                        cur.append(c)
    return out


def extract_pdf(path: str | Path, ink_colors: Optional[Sequence[Tuple[float, float, float]]] = None) -> List[InkPage]:
    path = str(path)
    if ink_colors is None:
        ink_colors = detect_ink_colors([path])
    pages: List[InkPage] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc):
            pages.append(extract_page(page, i, Path(path).name, ink_colors))
    return pages


def extract_many(paths: Sequence[str | Path]) -> List[InkPage]:
    """Extract all pages; ink colour is detected jointly over all documents."""
    colors = detect_ink_colors(paths, top=1)
    pages: List[InkPage] = []
    for p in paths:
        pages.extend(extract_pdf(p, colors))
    return pages


def summarize(pages: Sequence[InkPage]) -> Dict[str, object]:
    n_strokes = sum(len(p.strokes) for p in pages)
    n_words = sum(len(p.words) for p in pages)
    chars: Counter = Counter()
    for p in pages:
        for w in p.words:
            chars.update(w.text)
    return {"pages": len(pages), "strokes": n_strokes, "ocr_words": n_words, "chars": dict(chars.most_common())}
