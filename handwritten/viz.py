"""Diagnostic visualisations (segmentation overlays, glyph atlases, samples)."""
from __future__ import annotations

import colorsys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import flatten_stroke, rasterize, subpaths_bbox
from .segment import Glyph, Segmentation


def _font(size: int = 14):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _color(i: int):
    h = (i * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.85)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_segmentation(seg: Segmentation, source: str, page: int, region=None, scale: float = 3.0,
                      path: Optional[str] = None) -> Image.Image:
    """Overlay of character groups (coloured), labels and baselines for a page."""
    words = [w for w in seg.words if w.source == source and w.page == page]
    if region is None:
        xs0 = [w.ink_bbox[0] for w in words]
        ys0 = [w.ink_bbox[1] for w in words]
        xs1 = [w.ink_bbox[2] for w in words]
        ys1 = [w.ink_bbox[3] for w in words]
        region = (min(xs0) - 10, min(ys0) - 10, max(xs1) + 10, max(ys1) + 10)
    x0, y0, x1, y1 = region
    W, H = int((x1 - x0) * scale), int((y1 - y0) * scale)
    img = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(img)
    font = _font(int(4 * scale))

    def P(x, y):
        return ((x - x0) * scale, (y - y0) * scale)

    ci = 0
    for w in words:
        ln = seg.lines[w.line]
        wb = w.ink_bbox
        if not (wb[2] >= x0 and wb[0] <= x1 and wb[3] >= y0 and wb[1] <= y1):
            continue
        # baseline + x-height line
        a, b = wb[0] - 2, wb[2] + 2
        dr.line([P(a, ln.baseline(a)), P(b, ln.baseline(b))], fill=(120, 200, 255), width=1)
        dr.line([P(a, ln.baseline(a) - ln.xh), P(b, ln.baseline(b) - ln.xh)], fill=(255, 200, 120), width=1)
        groups = w.groups if w.groups else [(0, len(w.strokes))]
        chars = w.chars if w.groups else ["?"]
        for c, (k, i) in zip(chars, groups):
            col = _color(ci) if w.ok else (170, 170, 170)
            ci += 1
            for s in w.strokes[k:i]:
                for poly in flatten_stroke(s.subpaths, 3):
                    if len(poly) >= 3:
                        dr.polygon([P(*p) for p in poly], fill=col)
            lx = min(s.bbox[0] for s in w.strokes[k:i])
            dr.text(P(lx, ln.baseline(lx) + 0.6 * ln.xh), c, fill=col if w.ok else (200, 0, 0), font=font)
        dr.text(P(wb[0], wb[1] - 5), f"{w.text} {w.cost:.1f}" if w.ok else f"{w.text} X", fill=(0, 0, 0),
                font=_font(int(3 * scale)))
    if path:
        img.save(path)
    return img


def glyph_image(g: Glyph, xh_px: float = 24.0, pad: int = 4) -> Image.Image:
    """Render a glyph so that its x-height is ``xh_px`` pixels, with baseline guides."""
    s = xh_px / g.xh
    top = 2.1 * xh_px
    bot = 1.0 * xh_px
    W = int(g.width * s) + 2 * pad
    H = int(top + bot) + 2 * pad
    img = Image.new("L", (max(W, 4), H), 255)
    m = rasterize(g.strokes, (-pad / s, -(top + pad) / s, (W - pad) / s, (bot + pad) / s), s, n=3)
    arr = np.asarray(img).copy()
    hh, ww = min(arr.shape[0], m.shape[0]), min(arr.shape[1], m.shape[1])
    arr[:hh, :ww][m[:hh, :ww]] = 0
    img = Image.fromarray(arr).convert("RGB")
    dr = ImageDraw.Draw(img)
    yb = int(top + pad)
    dr.line([(0, yb), (img.width, yb)], fill=(120, 200, 255))
    dr.line([(0, yb - xh_px), (img.width, yb - xh_px)], fill=(255, 210, 150))
    return img


def glyph_atlas(glyphs: Sequence[Glyph], chars: Optional[str] = None, per_row: int = 24, max_per_char: int = 48,
                xh_px: float = 22.0, path: Optional[str] = None, show_rejected: bool = True,
                label_ids: bool = False) -> Image.Image:
    by: Dict[str, List[Glyph]] = defaultdict(list)
    for g in glyphs:
        if g.ok or show_rejected:
            by[g.char].append(g)
    keys = sorted(by) if chars is None else [c for c in chars if c in by]
    font = _font(14)
    rows: List[Image.Image] = []
    for c in keys:
        gs = sorted(by[c], key=lambda g: (not g.ok, g.template_dist))[:max_per_char]
        tiles = [glyph_image(g, xh_px) for g in gs]
        if label_ids:
            small = _font(9)
            for t, g in zip(tiles, gs):
                gid = g.seqs[0] if g.seqs else -1
                ImageDraw.Draw(t).text((1, 1), str(gid), fill=(200, 60, 60), font=small)
        for t, g in zip(tiles, gs):
            if not g.ok:
                ImageDraw.Draw(t).rectangle([0, 0, t.width - 1, t.height - 1], outline=(255, 0, 0), width=2)
        line_imgs = []
        for r in range(0, len(tiles), per_row):
            chunk = tiles[r:r + per_row]
            w = sum(t.width + 4 for t in chunk) + 40
            h = max(t.height for t in chunk)
            im = Image.new("RGB", (w, h), "white")
            d = ImageDraw.Draw(im)
            if r == 0:
                n_ok = sum(g.ok for g in by[c])
                d.text((2, 2), f"{c}", fill=(0, 0, 0), font=font)
                d.text((2, 20), f"{n_ok}/{len(by[c])}", fill=(90, 90, 90), font=_font(9))
            x = 40
            for t in chunk:
                im.paste(t, (x, 0))
                x += t.width + 4
            line_imgs.append(im)
        rows.extend(line_imgs)
    if not rows:
        return Image.new("RGB", (10, 10), "white")
    W = max(r.width for r in rows)
    H = sum(r.height + 2 for r in rows)
    out = Image.new("RGB", (W, H), "white")
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height + 2
    if path:
        out.save(path)
    return out


def bank_atlas(style, path: Optional[str] = None, per_row: int = 20, max_per_char: int = 40,
               xh_px: float = 22.0) -> Image.Image:
    """Atlas of a trained style's glyph bank (one row block per character)."""
    glyphs = []
    for c, gs in sorted(style.bank.items()):
        for b in gs[:max_per_char]:
            glyphs.append(Glyph(char=c, strokes=b.strokes, xh=1.0, width=b.width, top=b.top, bottom=b.bottom,
                                seqs=(b.gid,)))
    return glyph_atlas(glyphs, per_row=per_row, max_per_char=max_per_char, xh_px=xh_px, path=path,
                       label_ids=True)
