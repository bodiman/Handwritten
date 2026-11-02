"""Write laid-out handwriting to vector PDF (and PNG previews)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pymupdf

from .box import InkBox

LETTER = (612.0, 792.0)
A4 = (595.28, 841.89)
PAPER_SIZES = {"letter": LETTER, "a4": A4}


@dataclass
class Placed:
    box: InkBox
    x: float
    y: float  # baseline y (page coords, y down)


@dataclass
class RenderPage:
    width: float
    height: float
    items: List[Placed] = field(default_factory=list)
    base_page: Optional[int] = None  # index into a base PDF to draw on
    paper: str = "plain"  # plain | ruled | grid | dots
    rule_spacing: float = 28.0
    rule_top: float = 72.0
    margin_left: float = 72.0


def _fmt(v: float) -> str:
    s = f"{v:.2f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _subpath_ops(sp: np.ndarray, dx: float, dy: float) -> str:
    a = np.round(sp + np.array([dx, dy]), 2)
    parts = [f"{_fmt(a[0, 0, 0])} {_fmt(a[0, 0, 1])} m"]
    for seg in a:
        parts.append(
            f"{_fmt(seg[1, 0])} {_fmt(seg[1, 1])} {_fmt(seg[2, 0])} {_fmt(seg[2, 1])} "
            f"{_fmt(seg[3, 0])} {_fmt(seg[3, 1])} c"
        )
    parts.append("h")
    return "\n".join(parts)


def ink_stream(items: Sequence[Placed], page_height: float, ink: Tuple[float, float, float]) -> bytes:
    r, g, b = ink
    out = [f"q 1 0 0 -1 0 {_fmt(page_height)} cm", f"{r:.3f} {g:.3f} {b:.3f} rg {r:.3f} {g:.3f} {b:.3f} RG",
           "1 J 1 j"]
    for it in items:
        for st in it.box.fills:
            ops = [_subpath_ops(sp, it.x, it.y) for sp in st if len(sp)]
            if ops:
                out.append("\n".join(ops))
                out.append("f")
        cur_w = None
        for pts, w in it.box.lines:
            if len(pts) < 1:
                continue
            if cur_w is None or abs(cur_w - w) > 1e-3:
                out.append(f"{_fmt(w)} w")
                cur_w = w
            p = np.round(pts + np.array([it.x, it.y]), 2)
            seg = [f"{_fmt(p[0, 0])} {_fmt(p[0, 1])} m"]
            if len(p) == 1:
                seg.append(f"{_fmt(p[0, 0] + 0.01)} {_fmt(p[0, 1])} l")
            for q in p[1:]:
                seg.append(f"{_fmt(q[0])} {_fmt(q[1])} l")
            seg.append("S")
            out.append(" ".join(seg))
    out.append("Q")
    return "\n".join(out).encode("latin-1")


def paper_stream(pg: RenderPage) -> bytes:
    if pg.paper == "plain" or pg.base_page is not None:
        return b""
    W, H = pg.width, pg.height
    out = [f"q 1 0 0 -1 0 {_fmt(H)} cm"]
    if pg.paper == "ruled":
        out.append("0.62 0.78 0.93 RG 0.5 w")
        y = pg.rule_top
        while y < H - 36:
            out.append(f"0 {_fmt(y)} m {_fmt(W)} {_fmt(y)} l S")
            y += pg.rule_spacing
        out.append(f"0.93 0.55 0.55 RG 0.6 w {_fmt(pg.margin_left - 8)} 0 m {_fmt(pg.margin_left - 8)} {_fmt(H)} l S")
    elif pg.paper == "grid":
        out.append("0.78 0.86 0.93 RG 0.35 w")
        step = 14.17  # 5 mm
        x = 0.0
        while x <= W:
            out.append(f"{_fmt(x)} 0 m {_fmt(x)} {_fmt(H)} l S")
            x += step
        y = 0.0
        while y <= H:
            out.append(f"0 {_fmt(y)} m {_fmt(W)} {_fmt(y)} l S")
            y += step
    elif pg.paper == "dots":
        out.append("0.7 0.7 0.75 rg")
        step = 14.17
        y = step
        while y < H:
            x = step
            while x < W:
                out.append(f"{_fmt(x - 0.4)} {_fmt(y - 0.4)} 0.8 0.8 re f")
                x += step
            y += step
    out.append("Q")
    return "\n".join(out).encode("latin-1")


def _append_stream(doc: pymupdf.Document, page: pymupdf.Page, data: bytes, under: bool = False) -> None:
    if not data:
        return
    xref = doc.get_new_xref()
    doc.update_object(xref, "<<>>")
    doc.update_stream(xref, data)
    conts = page.get_contents()
    refs = [f"{x} 0 R" for x in conts]
    if under:
        refs.insert(0, f"{xref} 0 R")
    else:
        refs.append(f"{xref} 0 R")
    doc.xref_set_key(page.xref, "Contents", "[" + " ".join(refs) + "]")


def write_pdf(pages: Sequence[RenderPage], path: str | Path, ink: Tuple[float, float, float],
              base_pdf: Optional[str | Path] = None, title: str = "Handwritten") -> None:
    out = pymupdf.open()
    base = pymupdf.open(str(base_pdf)) if base_pdf else None
    for pg in pages:
        if base is not None and pg.base_page is not None and pg.base_page < len(base):
            out.insert_pdf(base, from_page=pg.base_page, to_page=pg.base_page)
            page = out[-1]
            page.wrap_contents()
        else:
            page = out.new_page(width=pg.width, height=pg.height)
            # an empty page has no contents yet; give it one so we can append
            page.insert_text((0, 0), "", fontsize=1)
        _append_stream(out, page, paper_stream(pg), under=True)
        _append_stream(out, page, ink_stream(pg.items, pg.height, ink))
    out.set_metadata({"title": title, "creator": "handwritten", "producer": "handwritten"})
    out.save(str(path), deflate=True, garbage=3)
    out.close()
    if base is not None:
        base.close()


def pdf_to_pngs(pdf_path: str | Path, out_prefix: str | Path, dpi: int = 150) -> List[str]:
    paths = []
    with pymupdf.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi)
            p = f"{out_prefix}-{i + 1}.png"
            pix.save(p)
            paths.append(p)
    return paths
