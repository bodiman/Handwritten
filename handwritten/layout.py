"""Lay out parsed documents as handwritten pages.

Pages are filled top to bottom with greedy line breaking.  Each line gets a
small random baseline slope and start offset, each word its own size and
baseline jitter (see ``Synthesizer.word``), so the page drifts the way real
handwriting does.

Writing onto an existing PDF (``LayoutOptions.base_pdf``): writing starts
below the existing content of the first page; a page break in the document
(``\\newpage``) moves on to the next page of the base PDF, and text that
overflows a page continues on an inserted blank page (so the next section
still lands on its own base page).  ``<!-- page N -->`` jumps to base page N.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .box import InkBox
from .docparse import Block, Span
from .mathlayout import render_math
from .render import PAPER_SIZES, Placed, RenderPage
from .synth import Synthesizer

NO_SPACE_BEFORE = set(",.;:!?)]}%'’”")


@dataclass
class LayoutOptions:
    paper: str = "letter"  # letter | a4
    margin_left: Optional[float] = None  # points; default = writer's natural margin
    margin_right: Optional[float] = None  # points; default = writer's natural margin
    margin_top: float = 72.0
    margin_bottom: float = 60.0
    line_spacing: Optional[float] = None  # in x-heights; default = writer's
    background: str = "plain"  # plain | ruled | grid | dots
    heading_scale: float = 1.3
    base_pdf: Optional[str] = None  # write onto this PDF (below existing content)
    base_gap: float = 1.5  # x-heights between existing content and the writing
    para_gap: float = 0.6  # extra line-heights for a blank line
    indent: float = 2.2  # list hanging indent in x-heights


@dataclass
class _Token:
    box: InkBox
    space_before: bool


class Layout:
    def __init__(self, synth: Synthesizer, opts: Optional[LayoutOptions] = None):
        self.s = synth
        self.o = opts or LayoutOptions()
        st = synth.st
        self.xh = synth.xh
        self.rng = synth.rng
        self.W, self.H = PAPER_SIZES.get(self.o.paper, PAPER_SIZES["letter"])
        self.base_tops: List[float] = []
        self.base_sizes: List[tuple] = []
        if self.o.base_pdf:
            self._read_base(self.o.base_pdf)
            if self.base_sizes:
                self.W, self.H = self.base_sizes[0]
        scale = self.W / 612.0  # natural margins were measured on letter paper
        nat_left = float(np.clip(st.line_start["mean"], 40.0, 110.0)) * scale
        nat_right = float(np.clip(self.W - st.line_end["mean"] * scale, 36.0, 90.0))
        self.left = self.o.margin_left if self.o.margin_left is not None else nat_left
        self.right = self.W - (self.o.margin_right if self.o.margin_right is not None else nat_right)
        if self.right - self.left < 10 * self.xh:
            self.right = min(self.W - 18.0, self.left + 10 * self.xh)
        ls = self.o.line_spacing if self.o.line_spacing is not None else float(
            np.clip(st.line_spacing["mean"], 2.4, 4.2))
        self.line_h = ls * self.xh
        self.pages: List[RenderPage] = []
        self.base_idx = -1
        self.y = 0.0
        self.first_line_on_page = True
        self._prev_desc = 0.3 * self.xh

    # -------------------------------------------------------------- base pdf
    def _read_base(self, path: str) -> None:
        import pymupdf

        with pymupdf.open(path) as doc:
            for page in doc:
                H = page.rect.height
                bottom = 0.0
                for b in page.get_text("blocks"):
                    if b[4].strip() and b[3] < H - 30:
                        bottom = max(bottom, b[3])
                for d in page.get_drawings():
                    r = d.get("rect")
                    # ignore full-page frames/backgrounds and footer rules
                    if r is None or r.height > 0.8 * H or r.y1 > H - 30:
                        continue
                    bottom = max(bottom, r.y1)
                for img in page.get_image_info():
                    r = img["bbox"]
                    if r[3] - r[1] < 0.8 * H:
                        bottom = max(bottom, r[3])
                self.base_tops.append(bottom)
                self.base_sizes.append((page.rect.width, page.rect.height))

    # --------------------------------------------------------------- pages
    def _new_page(self, advance_base: bool = True) -> RenderPage:
        """Start a page.  ``advance_base``: move to the next base-PDF page
        (a section break); otherwise insert a blank continuation page."""
        base = None
        if advance_base:
            self.base_idx += 1
            if self.base_idx < len(self.base_tops):
                base = self.base_idx
        W, H = (self.base_sizes[base] if base is not None else (self.W, self.H))
        pg = RenderPage(W, H, base_page=base, paper=self.o.background, rule_spacing=self.line_h,
                        rule_top=self.o.margin_top + self.line_h, margin_left=self.left)
        self.pages.append(pg)
        if base is not None and self.base_tops[base] > 0:
            self.y = self.base_tops[base] + self.o.base_gap * self.xh + self.s.st.ascender * self.xh
        else:
            self.y = self.o.margin_top + self.s.st.ascender * self.xh
        self.y = self._snap(self.y)
        self.first_line_on_page = True
        self._prev_desc = 0.3 * self.xh
        return pg

    def _goto_base_page(self, n: int) -> None:
        """Jump to base page ``n`` (0-based), adding blank pages if needed."""
        while self.base_idx < n - 1:
            self._new_page(advance_base=True)
        self._new_page(advance_base=True)

    def _snap(self, y: float) -> float:
        """On ruled paper, sit baselines on the rules."""
        pg = self.pages[-1] if self.pages else None
        if pg is None or pg.paper != "ruled" or pg.base_page is not None:
            return y
        k = max(0, math.ceil((y - pg.rule_top - 0.15 * self.xh) / self.line_h))
        return pg.rule_top + k * self.line_h - 0.08 * self.xh

    @property
    def page(self) -> RenderPage:
        if not self.pages:
            self._new_page()
        return self.pages[-1]

    # -------------------------------------------------------------- tokens
    def _tokens(self, spans: Sequence[Span], size: float) -> List[_Token]:
        """Split spans into word / formula tokens.  ``space_before`` records
        whether the source had whitespace before the token."""
        toks: List[_Token] = []
        pending_space = False
        for sp in spans:
            if sp.math:
                box = render_math(sp.text, self.s, display=False, size=size)
                if not box.is_empty() or box.width > 0:
                    toks.append(_Token(box, pending_space and bool(toks)))
                pending_space = False
                continue
            text = sp.text.replace("\t", " ")
            if not text:
                continue
            if text[0].isspace():
                pending_space = True
            for k, word in enumerate(text.split(" ")):
                if k > 0:
                    pending_space = True
                if not word:
                    continue
                sb = pending_space and bool(toks) and not (len(word) == 1 and word in NO_SPACE_BEFORE)
                toks.append(_Token(self.s.word(word, size), sb))
                pending_space = False
            if text[-1].isspace():
                pending_space = True
        return toks

    # ---------------------------------------------------------------- lines
    def _advance(self, ascent: float) -> None:
        jitter = self.rng.normal(0.0, 0.04 * self.line_h * self.s.var * (1 - 0.6 * self.s.neat))
        need = self._prev_desc + ascent + 0.45 * self.xh
        self.y = self._snap(self.y + max(self.line_h + jitter, need))

    def _place_line(self, toks: List[_Token], x0: float, gaps: List[float]) -> float:
        """Place a line of tokens at the current baseline; returns its descent."""
        st = self.s.st
        v = self.s.var * (1 - 0.6 * self.s.neat)
        slope = self.rng.normal(st.line_slope["mean"] * 0.5, min(st.line_slope["sd"], 0.02) * 0.4 * v)
        x = x0
        descent = 0.0
        pg = self.page
        for t, g in zip(toks, gaps):
            x += g
            pg.items.append(Placed(t.box, x, self.y + slope * (x - x0)))
            x += t.box.width
            descent = max(descent, t.box.descent)
        return descent

    def _flow(self, toks: List[_Token], indent: float = 0.0, size: float = 1.0, first_x: Optional[float] = None,
              hang: Optional[float] = None) -> None:
        """Greedy line filling.  ``first_x`` positions the first line;
        continuation lines start at ``left + indent`` (or ``hang`` if given)."""
        i = 0
        v = self.s.var * (1 - 0.6 * self.s.neat)
        start_sd = min(self.s.st.line_start["sd"], 6.0) * 0.4 * v
        first = True
        while i < len(toks):
            if first and first_x is not None:
                x_start = first_x
            else:
                x_start = (hang if hang is not None else self.left + indent) + self.rng.normal(0.0, start_sd)
            first = False
            line: List[_Token] = []
            gaps: List[float] = []
            x = x_start
            while i < len(toks):
                t = toks[i]
                g = (self.s.word_gap(size) if t.space_before else 0.0) if line else 0.0
                if line and x + g + t.box.width > self.right:
                    break
                line.append(t)
                gaps.append(g)
                x += g + t.box.width
                i += 1
            asc = max([t.box.ascent for t in line] + [self.s.st.ascender * self.xh * size * 0.8])
            desc = max([t.box.descent for t in line] + [0.0])
            if not self.first_line_on_page:
                self._advance(asc)
            elif asc > self.s.st.ascender * self.xh * 1.05:
                self.y += asc - self.s.st.ascender * self.xh  # tall first line (display maths)
            if self.y + desc > self.page.height - self.o.margin_bottom and not self.first_line_on_page:
                self._new_page(advance_base=False)
                if asc > self.s.st.ascender * self.xh * 1.05:
                    self.y += asc - self.s.st.ascender * self.xh
            self.first_line_on_page = False
            self._prev_desc = self._place_line(line, x_start, gaps)

    # -------------------------------------------------------------- blocks
    def layout(self, blocks: Sequence[Block]) -> List[RenderPage]:
        self.pages = []
        self.base_idx = -1
        self._new_page(advance_base=True)
        for b in blocks:
            if b.kind == "pagebreak":
                self._new_page(advance_base=True)
            elif b.kind == "gotopage":
                self._goto_base_page(max(0, b.level - 1))
            elif b.kind == "blank":
                if not self.first_line_on_page:
                    self.y += self.o.para_gap * self.line_h
            elif b.kind == "rule":
                self._rule()
            elif b.kind == "displaymath":
                box = render_math(b.text, self.s, display=True)
                x = max(self.left, 0.5 * (self.left + self.right) - 0.5 * box.width)
                self._flow([_Token(box, False)], first_x=x)
            elif b.kind == "heading":
                size = self.o.heading_scale if b.level <= 1 else (1.15 if b.level == 2 else 1.0)
                if not self.first_line_on_page:
                    self.y += 0.4 * self.line_h
                self._flow(self._tokens(b.spans, size), size=size)
            elif b.kind == "item":
                self._item(b)
            else:
                toks = self._tokens(b.spans, 1.0)
                if toks:
                    self._flow(toks)
        return self.pages

    def _item(self, b: Block) -> None:
        mx = self.left + b.level * self.o.indent * self.xh
        text_x = self.left + (b.level + 1) * self.o.indent * self.xh
        marker = b.marker if b.marker not in ("-", "*", "+", "•") else "-"
        mbox = self.s.word(marker)
        toks = self._tokens(b.spans, 1.0)
        gap = max(text_x - mx - mbox.width, 0.6 * self.s.word_gap())
        spacer = InkBox(width=gap)
        self._flow([_Token(mbox, False), _Token(spacer, False)] + [_Token(t.box, t.space_before if k else False)
                                                                   for k, t in enumerate(toks)],
                   first_x=mx, hang=text_x)

    def _rule(self) -> None:
        x0, x1 = self.left, self.right
        n = 12
        xs = np.linspace(0.0, x1 - x0, n)
        ys = self.rng.normal(-0.3 * self.xh, 0.06 * self.xh, n)
        box = InkBox(lines=[(np.column_stack([xs, ys]), self.s.st.pen_width * self.xh * 0.9)], width=x1 - x0)
        box.refit()
        self._flow([_Token(box, False)], first_x=x0)


def layout_document(blocks: Sequence[Block], synth: Synthesizer,
                    opts: Optional[LayoutOptions] = None) -> List[RenderPage]:
    return Layout(synth, opts).layout(blocks)
