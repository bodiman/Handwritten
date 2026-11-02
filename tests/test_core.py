"""Unit tests: geometry, InkBox, document parsing, style files, synthesiser,
layout and PDF rendering.

Everything here runs on the synthetic ``toy_style`` (see conftest.py), so it
needs no personal handwriting data; the few tests that use the real trained
style are skipped without it.
"""
from __future__ import annotations

import json
import math
import zipfile
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pymupdf
import pytest

from conftest import ellipse_subpath, make_toy_style, rect_subpath, ring_stroke
from handwritten import geometry as G
from handwritten.box import InkBox, hstack
from handwritten.docparse import parse_file, parse_text, parse_tex, split_inline, strip_markdown
from handwritten.ink import extract_pdf
from handwritten.layout import Layout, LayoutOptions, layout_document
from handwritten.render import A4, LETTER, Placed, RenderPage, ink_stream, pdf_to_pngs, write_pdf
from handwritten.style import Style
from handwritten.synth import Synthesizer

LOREM = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore "
         "et dolore magna aliqua")


def _blocks(blocks):
    """Compact view of parsed blocks: (kind, level, marker, [(text, math)])."""
    return [(b.kind, b.level, b.marker, [(s.text, s.math) for s in b.spans]) for b in blocks]


def _page_ink_bounds(it: Placed):
    b = it.box.ink_bounds()
    return None if b is None else (b[0] + it.x, b[1] + it.y, b[2] + it.x, b[3] + it.y)


# =========================================================================
# geometry
# =========================================================================
class TestGeometry:
    def test_line_as_cubic_puts_controls_at_thirds(self):
        np.testing.assert_allclose(G.line_as_cubic((0, 0), (3, 6)), [[0, 0], [1, 2], [2, 4], [3, 6]])

    def test_flatten_subpath_is_closed_and_on_curve(self):
        sp = ellipse_subpath(0.0, 0.0, 2.0, 1.0)
        poly = G.flatten_subpath(sp, n=8)
        assert poly.shape == (4 * 7 + 1, 2)
        np.testing.assert_allclose(poly[0], poly[-1], atol=1e-12)
        r = (poly[:, 0] / 2.0) ** 2 + poly[:, 1] ** 2  # = 1 on the ellipse
        assert np.abs(r - 1.0).max() < 2e-3

    def test_bbox_uses_the_curve_not_the_control_polygon(self):
        arch = np.array([[0, 0], [0, -4], [4, -4], [4, 0]], float)  # peaks at y = -3
        sp = np.stack([arch, G.line_as_cubic((4, 0), (0, 0))])
        x0, y0, x1, y1 = G.subpaths_bbox([sp])
        assert (x0, x1, y1) == pytest.approx((0.0, 4.0, 0.0), abs=1e-9)
        assert y0 == pytest.approx(-3.0, abs=1e-6)  # control polygon would say -4

    def test_overlapping_subpaths_fill_their_union(self):
        a = rect_subpath(0, 0, 4, 4)
        b = rect_subpath(2, 2, 6, 6)  # same orientation: winding number 2 in the overlap
        res = 10.0
        for strokes in ([[a, b]], [[a], [b]]):  # one stroke of 2 subpaths, or 2 strokes
            m = G.rasterize(strokes, (0, 0, 6, 6), res)
            assert m.sum() / res ** 2 == pytest.approx(16 + 16 - 4, rel=0.02)
            assert m[30, 30], "overlap (winding 2) must be filled under the non-zero rule"
            assert not m[5, 55] and not m[55, 5]

    def test_opposite_winding_cuts_a_hole_same_winding_does_not(self):
        outer = rect_subpath(0, 0, 6, 6)
        res = 10.0
        holed = G.rasterize([[outer, rect_subpath(2, 2, 4, 4, ccw=True)]], (0, 0, 6, 6), res)
        assert holed.sum() / res ** 2 == pytest.approx(36 - 4, rel=0.02)
        assert not holed[30, 30]
        # non-zero (not even-odd): a nested contour with the same direction stays filled
        filled = G.rasterize([[outer, rect_subpath(2, 2, 4, 4)]], (0, 0, 6, 6), res)
        assert filled.sum() / res ** 2 == pytest.approx(36, rel=0.02)
        assert filled[30, 30]

    def test_pen_ring_is_an_annulus(self):
        m = G.rasterize([ring_stroke(0.0, 0.0, 3.0, 3.0, 1.0)], (-3, -3, 3, 3), 10.0, n=16)
        area = m.sum() / 100.0
        assert area == pytest.approx(math.pi * (9 - 4), rel=0.03)
        assert not m[30, 30] and m[30, 3]

    def test_slow_and_fast_winding_rasterisers_agree(self):
        rng = np.random.default_rng(0)
        polys = [rng.uniform(0, 10, size=(7, 2)) for _ in range(3)]  # self-intersecting
        xs = (np.arange(50) + 0.5) / 5.0
        np.testing.assert_array_equal(G.winding_mask(polys, xs, xs), G._winding_fast(polys, xs, xs))

    def test_compose_applies_the_rightmost_matrix_first(self):
        p = np.array([[1.0, 1.0]])
        np.testing.assert_allclose(G.transform_points(p, G.compose(G.translate(10, 0), G.scale(2))), [[12, 2]])
        np.testing.assert_allclose(G.transform_points(p, G.compose(G.scale(2), G.translate(10, 0))), [[22, 2]])
        m = G.compose(G.translate(5, 0), G.rotate(math.pi / 2), G.scale(3))
        np.testing.assert_allclose(G.transform_points(np.array([[1.0, 0.0]]), m), [[5, 3]], atol=1e-12)
        np.testing.assert_allclose(G.compose(), G.affine())

    def test_shear_and_rotate_conventions(self):
        # y points down: a point above the baseline (y < 0) moves right for k < 0
        np.testing.assert_allclose(G.transform_points(np.array([[0.0, -1.0]]), G.shear_x(-0.2)), [[0.2, -1.0]])
        np.testing.assert_allclose(G.transform_points(np.array([[1.0, 0.0]]), G.rotate(math.pi / 2)), [[0, 1]],
                                   atol=1e-12)
        np.testing.assert_allclose(G.scale(2), G.scale(2, 2))

    def test_transform_subpaths_maps_every_control_point(self):
        sps = [ellipse_subpath(1, 1, 1, 0.5), rect_subpath(0, 0, 2, 1)]
        before = [sp.copy() for sp in sps]
        m = G.compose(G.translate(3, -2), G.rotate(0.3), G.scale(1.5))
        out = G.transform_subpaths(sps, m)
        for sp, o, b in zip(sps, out, before):
            assert o.shape == sp.shape
            np.testing.assert_array_equal(sp, b)  # input untouched
            np.testing.assert_allclose(o.reshape(-1, 2), sp.reshape(-1, 2) @ m[:, :2].T + m[:, 2])
            # Beziers are affine invariant: transform-then-flatten == flatten-then-transform
            np.testing.assert_allclose(G.flatten_subpath(o), G.transform_points(G.flatten_subpath(sp), m),
                                       atol=1e-9)

    def test_stroke_bbox_and_transformed(self):
        st = G.Stroke([rect_subpath(0, 0, 2, 1)], page=3, seq=7, source="x.pdf")
        assert st.bbox == pytest.approx((0, 0, 2, 1))
        assert (st.width, st.height, st.cx, st.cy) == pytest.approx((2, 1, 1, 0.5))
        assert st.n_segments() == 4
        t = st.transformed(G.translate(5, 5))
        assert t.bbox == pytest.approx((5, 5, 7, 6))
        assert (t.page, t.seq, t.source) == (3, 7, "x.pdf")
        assert st.bbox == pytest.approx((0, 0, 2, 1))


# =========================================================================
# InkBox
# =========================================================================
def _bar(width: float, top: float, bottom: float, text: str = "") -> InkBox:
    """An InkBox holding one filled rectangle from y=-top to y=bottom."""
    return InkBox(fills=[[rect_subpath(0, -top, width, bottom)]], width=width, ascent=top, descent=bottom, text=text)


class TestInkBox:
    def test_add_places_ink_and_grows_metrics(self):
        a = _bar(5, 10, 0, "a")
        b = _bar(4, 6, 3, "b")
        a.add(b, dx=8, dy=-6)  # raised by 6
        assert a.width == pytest.approx(12)
        assert a.ascent == pytest.approx(12)
        assert a.descent == pytest.approx(0)
        assert a.ink_bounds() == pytest.approx((0, -12, 12, 0))
        a.add(b, dx=0, dy=4)  # lowered by 4
        assert a.descent == pytest.approx(7)
        assert a.width == pytest.approx(12)  # did not shrink
        assert len(a.fills) == 3
        assert b.ink_bounds() == pytest.approx((0, -6, 4, 3))  # source not modified

    def test_hstack_adds_widths_and_gaps(self):
        boxes = [_bar(5, 10, 0, "x"), _bar(4, 7, 3, "y"), _bar(3, 12, 1, "z")]
        out = hstack(boxes, [1.0, 2.0])
        assert out.width == pytest.approx(5 + 1 + 4 + 2 + 3)
        assert (out.ascent, out.descent) == pytest.approx((12, 3))
        assert out.text == "xyz"
        starts = [float(st[0][..., 0].min()) for st in out.fills]
        assert starts == pytest.approx([0, 6, 12])
        # missing gaps count as zero
        assert hstack(boxes, []).width == pytest.approx(12)

    def test_shifted_moves_ink_and_metrics(self):
        b = _bar(4, 6, 2)
        s = b.shifted(2, -3)
        assert (s.ascent, s.descent, s.width) == pytest.approx((9, -1, 4))
        assert s.ink_bounds() == pytest.approx((2, -9, 6, -1))
        assert b.ink_bounds() == pytest.approx((0, -6, 4, 2))

    def test_transformed_rescales_ink_and_pen(self):
        b = InkBox(fills=[[rect_subpath(0, -5, 2, 0)]], lines=[(np.array([[0.0, 0.0], [4.0, 0.0]]), 0.5)], width=4)
        t = b.transformed(G.scale(2))
        assert t.lines[0][1] == pytest.approx(1.0)
        assert t.ascent == pytest.approx(10)
        assert t.width == pytest.approx(8.5)  # grows to cover the ink (line end + half pen)
        assert b.transformed(G.scale(2), width_scale=1.0).lines[0][1] == pytest.approx(0.5)

    def test_ink_bounds_and_refit(self):
        ln = InkBox(lines=[(np.array([[0.0, 0.0], [10.0, 0.0]]), 2.0)])
        assert ln.ink_bounds() == pytest.approx((-1, -1, 11, 1))
        assert InkBox().ink_bounds() is None and InkBox().is_empty()
        b = _bar(3, 7, 2)
        b.ascent = b.descent = 0.0
        b.refit()
        assert (b.ascent, b.descent) == pytest.approx((7, 2))


# =========================================================================
# document parsing
# =========================================================================
class TestDocparse:
    def test_inline_maths_and_escaped_dollar(self):
        spans = split_inline(r"Let $x^2$ be \(y\) and it costs \$5, not $$z$$.")
        assert [(s.text, s.math) for s in spans] == [
            ("Let ", False), ("x^2", True), (" be ", False), ("y", True), (" and it costs $5, not ", False),
            ("z", True), (".", False)]
        assert [(s.text, s.math) for s in split_inline(r"price \$3 or \$4")] == [("price $3 or $4", False)]
        # an unmatched dollar is literal text
        assert [(s.text, s.math) for s in split_inline("cost $5")] == [("cost $5", False)]

    def test_display_maths_single_and_multi_line(self):
        src = "$$a+b=c$$\n$$\n\\sum_i x_i\n= 1\n$$\n\\[ x^2 \\] then text"
        assert _blocks(parse_text(src)) == [
            ("displaymath", 0, "", [("a+b=c", True)]),
            ("displaymath", 0, "", [("\\sum_i x_i = 1", True)]),
            ("displaymath", 0, "", [("x^2", True)]),
            ("line", 0, "", [("then text", False)]),
        ]

    def test_headings_lists_rules_and_markdown(self):
        src = ("# Title\n\n### Sub **bold**\n- a\n* b\n  - nested\n1. first\n2) second\n(a) part\n(iv) roman\n"
               "---\n`code` and [link](http://x) and _em_ and snake_case_name\n\n\n")
        assert _blocks(parse_text(src)) == [
            ("heading", 1, "", [("Title", False)]),
            ("blank", 0, "", []),
            ("heading", 3, "", [("Sub bold", False)]),
            ("item", 0, "-", [("a", False)]),
            ("item", 0, "-", [("b", False)]),
            ("item", 1, "-", [("nested", False)]),
            ("item", 0, "1.", [("first", False)]),
            ("item", 0, "2)", [("second", False)]),
            ("item", 0, "(a)", [("part", False)]),
            ("item", 0, "(iv)", [("roman", False)]),
            ("rule", 0, "", []),
            ("line", 0, "", [("code and link and em and snake_case_name", False)]),
        ]  # trailing blank lines are dropped
        assert strip_markdown(r"a\_b \# c") == "a_b # c"

    def test_escaped_asterisks_are_literal(self):
        assert strip_markdown(r"\*not italic\*") == "*not italic*"

    def test_page_breaks_and_page_jumps(self):
        src = "one\n\\newpage\ntwo\n<!-- pagebreak -->\nthree\n\\clearpage\n<!-- page 3 -->\nfour\n\fnext"
        kinds = [(b.kind, b.level) for b in parse_text(src)]
        assert kinds[:7] == [("line", 0), ("pagebreak", 0), ("line", 0), ("pagebreak", 0), ("line", 0),
                             ("pagebreak", 0), ("gotopage", 3)]
        assert ("pagebreak", 0) in kinds[7:]

    def test_form_feed_keeps_surrounding_text(self):
        blocks = parse_text("end of page one\n\fstart of page two")
        assert [b.text for b in blocks if b.kind == "line"] == ["end of page one", "start of page two"]

    def test_plain_text_mode_ignores_markdown(self):
        assert _blocks(parse_text("# not heading\n- not item\n**kept**", markdown=False)) == [
            ("line", 0, "", [("# not heading", False)]),
            ("line", 0, "", [("- not item", False)]),
            ("line", 0, "", [("**kept**", False)]),
        ]

    def test_reflow_joins_lines_of_a_paragraph(self):
        src = "line one\nline two\n\npara two\n- item\n  continues"
        flowed = parse_text(src, reflow=True)
        assert [(b.kind, b.text) for b in flowed] == [
            ("line", "line one line two"), ("blank", ""), ("line", "para two"), ("item", "item continues")]
        kept = parse_text(src, reflow=False)
        assert [(b.kind, b.text) for b in kept][:2] == [("line", "line one"), ("line", "line two")]

    def test_parse_file_reflow_defaults_by_extension(self, tmp_path):
        src = "First line\nsecond line with $x$\n\n# Head\n"
        md = tmp_path / "a.md"
        txt = tmp_path / "a.txt"
        md.write_text(src)
        txt.write_text(src)
        md_blocks = parse_file(md)
        assert [(b.kind, b.text) for b in md_blocks] == [
            ("line", "First line second line with x"), ("blank", ""), ("heading", "Head")]
        assert any(s.math for s in md_blocks[0].spans)
        txt_blocks = parse_file(txt)
        assert [(b.kind, b.text) for b in txt_blocks] == [
            ("line", "First line"), ("line", "second line with x"), ("blank", ""), ("line", "# Head")]
        # explicit choice wins over the default
        assert len([b for b in parse_file(md, reflow=False) if b.kind == "line"]) == 2
        assert len([b for b in parse_file(txt, reflow=True) if b.kind == "line"]) == 2

    def test_docx(self, tmp_path):
        docx = pytest.importorskip("docx")
        d = docx.Document()
        d.add_heading("Answers", 1)
        d.add_paragraph("Let $x = 2$ then.")
        d.add_paragraph("first point", style="List Bullet")
        d.add_heading("Part two", 2)
        d.add_page_break()
        d.add_paragraph("after break")
        p = tmp_path / "a.docx"
        d.save(str(p))
        blocks = [b for b in parse_file(p) if b.kind != "blank"]
        assert _blocks(blocks) == [
            ("heading", 1, "", [("Answers", False)]),
            ("line", 0, "", [("Let ", False), ("x = 2", True), (" then.", False)]),
            ("item", 0, "-", [("first point", False)]),
            ("heading", 2, "", [("Part two", False)]),
            ("pagebreak", 0, "", []),
            ("line", 0, "", [("after break", False)]),
        ]

    def test_tex_subset(self, tmp_path):
        src = r"""\documentclass{article}
\usepackage{amsmath}
\title{Ignored}
\begin{document}
\maketitle
\section{Intro}
Some \textbf{bold} and \emph{em} text with $x^2$. % a comment
\begin{itemize}
\item first
\item second
\end{itemize}
\begin{enumerate}
\item[(a)] part a
\end{enumerate}
\begin{align*}
a &= b \\
  &= c \label{eq:1}
\end{align*}
\newpage
\subsection*{More}
Closing line.\\ Next line.
\end{document}
"""
        p = tmp_path / "a.tex"
        p.write_text(src)
        blocks = [b for b in parse_file(p) if b.kind != "blank"]
        assert _blocks(blocks) == [
            ("heading", 1, "", [("Intro", False)]),
            ("line", 0, "", [("Some bold and em text with ", False), ("x^2", True), (".", False)]),
            ("item", 0, "-", [("first", False)]),
            ("item", 0, "-", [("second", False)]),
            ("item", 0, "(a)", [("part a", False)]),
            # multi-line align becomes stacked rows (mathlayout aligns them at &)
            ("displaymath", 0, "", [(r"\begin{aligned}a &= b \\ &= c\end{aligned}", True)]),
            ("pagebreak", 0, "", []),
            ("heading", 2, "", [("More", False)]),
            ("line", 0, "", [("Closing line.", False)]),
            ("line", 0, "", [("Next line.", False)]),
        ]
        assert parse_tex("no document environment, $y$") != []

    def test_pdf_text_layer(self, tmp_path):
        doc = pymupdf.open()
        for i in range(2):
            page = doc.new_page()
            page.insert_text((72, 100), f"Problem {i + 1}", fontsize=12)
            page.insert_text((72, 120), "Show that it holds.", fontsize=12)
        p = tmp_path / "problems.pdf"
        doc.save(str(p))
        blocks = [b for b in parse_file(p) if b.kind != "blank"]
        assert [(b.kind, b.text) for b in blocks] == [
            ("line", "Problem 1"), ("line", "Show that it holds."), ("pagebreak", ""),
            ("line", "Problem 2"), ("line", "Show that it holds.")]


# =========================================================================
# style files
# =========================================================================
class TestStyleIO:
    def test_round_trip(self, toy_style, tmp_path):
        st = toy_style.stats
        st.letter_gap_pair = {"ab": (0.1, 0.02, 5)}
        st.letter_gap_left = {"a": (0.11, 0.03, 9)}
        st.letter_gap_right = {"b": (0.09, 0.04, 7)}
        st.word_gap = {"mean": 1.5, "sd": 0.3}
        toy_style.meta["note"] = "unit test"
        toy_style.bank["a"][0].pen = 0.17
        p = tmp_path / "s.hwstyle"
        toy_style.save(p)

        with zipfile.ZipFile(p) as z:
            assert sorted(z.namelist()) == ["glyphs.npz", "style.json"]
            assert json.loads(z.read("style.json"))["format"] == 1

        s2 = Style.load(p)
        assert asdict(s2.stats) == asdict(toy_style.stats)
        assert s2.meta == toy_style.meta
        assert s2.coverage() == toy_style.coverage()
        for ch, gs in toy_style.bank.items():
            for g, h in zip(gs, s2.bank[ch]):
                assert (h.char, h.gid, h.source) == (g.char, g.gid, g.source)
                assert (h.width, h.top, h.bottom, h.pen) == pytest.approx((g.width, g.top, g.bottom, g.pen))
                assert [len(x) for x in h.strokes] == [len(x) for x in g.strokes]
                for sh, sg in zip(h.strokes, g.strokes):
                    for a, b in zip(sh, sg):
                        assert a.shape == b.shape
                        np.testing.assert_allclose(a, b, atol=1e-3)  # stored as float16

    def test_rejects_unknown_format(self, toy_style, tmp_path):
        p = tmp_path / "s.hwstyle"
        toy_style.save(p)
        with zipfile.ZipFile(p) as z:
            doc = json.loads(z.read("style.json"))
            npz = z.read("glyphs.npz")
        doc["format"] = 99
        bad = tmp_path / "bad.hwstyle"
        with zipfile.ZipFile(bad, "w") as z:
            z.writestr("style.json", json.dumps(doc))
            z.writestr("glyphs.npz", npz)
        with pytest.raises(ValueError):
            Style.load(bad)

    def test_merge_adds_samples_with_fresh_ids(self, toy_style):
        other = make_toy_style(samples=2, chars="xyZ")
        merged = toy_style.merge(other)
        assert len(merged.bank["x"]) == 4 and len(merged.bank["Z"]) == 2
        ids = [g.gid for gs in merged.bank.values() for g in gs]
        assert len(ids) == len(set(ids))
        assert merged.stats.char_counts["x"] == 4

    def test_real_style_round_trip(self, real_style, tmp_path):
        p = tmp_path / "copy.hwstyle"
        real_style.save(p)
        s2 = Style.load(p)
        assert s2.coverage() == real_style.coverage()
        assert asdict(s2.stats) == asdict(real_style.stats)
        g, h = real_style.bank["e"][0], s2.bank["e"][0]
        np.testing.assert_allclose(h.strokes[0][0], g.strokes[0][0], atol=2e-3)


# =========================================================================
# synthesiser
# =========================================================================
def _ink_arrays(box: InkBox):
    return [sp for st in box.fills for sp in st] + [p for p, _ in box.lines]


def _same_ink(a: InkBox, b: InkBox) -> bool:
    xa, xb = _ink_arrays(a), _ink_arrays(b)
    return a.width == b.width and len(xa) == len(xb) and all(
        x.shape == y.shape and np.array_equal(x, y) for x, y in zip(xa, xb))


class TestSynthesizer:
    WORDS = ["hello", "world,", "Zebra", "quick!", "0O"]

    def test_same_seed_same_handwriting(self, toy_style):
        a = Synthesizer(toy_style, seed=42)
        b = Synthesizer(toy_style, seed=42)
        for w in self.WORDS:
            assert _same_ink(a.word(w), b.word(w))
        c = Synthesizer(toy_style, seed=43)
        a2 = Synthesizer(toy_style, seed=42)
        assert not all(_same_ink(a2.word(w), c.word(w)) for w in self.WORDS)

    def test_same_seed_same_document(self, toy_style):
        blocks = parse_text("# Heading\n\n" + LOREM + "\n\n- item one\n- item two", reflow=True)
        runs = []
        for _ in range(2):
            pages = layout_document(blocks, Synthesizer(toy_style, seed=7), LayoutOptions())
            runs.append([ink_stream(p.items, p.height, (0, 0, 0)) for p in pages])
        assert runs[0] == runs[1]

    def test_variation_zero_reproduces_bank_glyphs_exactly(self):
        style = make_toy_style(samples=1)
        xh = 12.0
        sy = Synthesizer(style, seed=1, variation=0.0, x_height=xh)
        for ch in "abgk0!":
            g = style.bank[ch][0]
            box = sy.glyph_box(ch)
            assert box.width == pytest.approx(g.width * xh)
            assert box.ascent == pytest.approx(g.top * xh)
            assert box.descent == pytest.approx(max(0.0, -g.bottom * xh))
            for sb, sg in zip(box.fills, g.strokes):
                for a, b in zip(sb, sg):
                    np.testing.assert_allclose(a, b * xh, atol=1e-9)
        st = style.stats
        assert [sy.letter_gap("a", "b") for _ in range(5)] == pytest.approx([st.letter_gap["mean"]] * 5)
        expect_wg = float(np.clip(st.word_gap["mean"], 0.8, 2.6)) * xh
        assert [sy.word_gap() for _ in range(3)] == pytest.approx([expect_wg] * 3)
        # with one sample per character and no variation, the seed is irrelevant
        other = Synthesizer(style, seed=99, variation=0.0, x_height=xh)
        assert _same_ink(Synthesizer(style, seed=1, variation=0.0, x_height=xh).word("abc"), other.word("abc"))

    def test_variation_and_neatness_scale_the_randomness(self, toy_style):
        def gap_sd(**kw):
            sy = Synthesizer(toy_style, seed=3, **kw)
            return float(np.std([sy.letter_gap("e", "n") for _ in range(400)]))

        loose, tight, neat = gap_sd(variation=1.0), gap_sd(variation=0.3), gap_sd(variation=1.0, neatness=1.0)
        assert tight < 0.5 * loose
        assert neat < 0.6 * loose
        assert gap_sd(variation=0.0) == pytest.approx(0.0, abs=1e-12)

    def test_x_height_option_scales_the_writing(self):
        style = make_toy_style(samples=1)
        small = Synthesizer(style, seed=0, variation=0.0, x_height=10).word("hello")
        big = Synthesizer(style, seed=0, variation=0.0, x_height=20).word("hello")
        assert big.width == pytest.approx(2 * small.width)
        assert big.ascent == pytest.approx(2 * small.ascent)
        assert Synthesizer(style).xh == style.stats.xh_pt

    def test_missing_characters_use_analogue_then_fallback(self, toy_style):
        sy = Synthesizer(toy_style, seed=0)
        assert sy.choose("a").kind == "real"
        assert sy.choose("O").kind == "analog"  # from the writer's '0'
        fb = sy.glyph_box("Z")
        assert fb.fills and not fb.lines and fb.width > 0  # restyled Hershey glyph as filled pen strokes
        assert sy.choose("☃").kind == "missing"  # nothing can draw a snowman
        assert sy.used_kind["a"] == "real" and sy.used_kind["O"] == "analog" and sy.used_kind["Z"] == "fallback"
        assert sy.missing["☃"] == 1


# =========================================================================
# layout
# =========================================================================
class TestLayout:
    # distinct from the toy style's natural margins, so ignoring an option shows
    MARGINS = SimpleNamespace(left=100.0, right=90.0, top=90.0, bottom=80.0)

    def _opts(self, **kw):
        m = self.MARGINS
        return LayoutOptions(margin_left=m.left, margin_right=m.right, margin_top=m.top, margin_bottom=m.bottom, **kw)

    def test_long_text_paginates_within_margins(self, toy_style):
        text = " ".join([LOREM] * 25)
        sy = Synthesizer(toy_style, seed=5)
        pages = layout_document(parse_text(text, reflow=True), sy, self._opts())
        assert len(pages) >= 3
        # every word placed exactly once, in order
        assert [it.box.text for pg in pages for it in pg.items] == text.split()
        W, H = LETTER
        m = self.MARGINS
        xh = sy.xh
        for k, pg in enumerate(pages):
            assert (pg.width, pg.height) == LETTER and pg.items
            ys = [it.y for it in pg.items]
            assert ys[0] < ys[-1]
            boxes = [_page_ink_bounds(it) for it in pg.items]
            for x0, y0, x1, y1 in boxes:
                assert 0 <= x0 and x1 <= W and 0 <= y0 and y1 <= H
                # line starts jitter a little around the margin; ink never crosses the others
                assert x0 >= m.left - 1.0 * xh
                assert x1 <= W - m.right + 0.5 * xh
                assert y0 >= m.top - 0.5 * xh
                assert y1 <= H - m.bottom + 0.3 * xh
            # lines are filled to the right margin, pages down to the bottom margin
            assert max(b[2] for b in boxes) > W - m.right - 4 * xh
            assert min(b[0] for b in boxes) < m.left + 1.0 * xh
            if k < len(pages) - 1:
                assert max(b[3] for b in boxes) > H - m.bottom - 2 * sy.xh * sy.st.line_spacing["mean"]

    def test_page_break_and_a4_paper(self, toy_style):
        blocks = parse_text("one\n\\newpage\ntwo\n\\newpage\nthree")
        pages = layout_document(blocks, Synthesizer(toy_style, seed=0), LayoutOptions(paper="a4"))
        assert len(pages) == 3
        assert all((p.width, p.height) == A4 for p in pages)
        assert [[it.box.text for it in p.items] for p in pages] == [["one"], ["two"], ["three"]]

    def test_list_items_hang_under_the_text(self, toy_style):
        sy = Synthesizer(toy_style, seed=2)
        lay = Layout(sy, self._opts())
        pages = lay.layout(parse_text("- " + " ".join([LOREM] * 2)))
        items = pages[0].items
        assert items[0].box.text == "-"
        text_x = lay.left + lay.o.indent * sy.xh
        words = [it for it in items if it.box.text not in ("-", "")]
        line_starts = [it for prev, it in zip(words, words[1:]) if it.y > prev.y + 0.5 * lay.line_h]
        assert line_starts, "expected the item to wrap"
        for it in line_starts:  # continuation lines start at the hanging indent
            assert it.x == pytest.approx(text_x, abs=0.8 * sy.xh)

    def test_ruled_paper_puts_baselines_on_the_rules(self, toy_style):
        sy = Synthesizer(toy_style, seed=4)
        lay = Layout(sy, self._opts(background="ruled"))
        pages = lay.layout(parse_text(" ".join([LOREM] * 3), reflow=True))
        pg = pages[0]
        assert pg.paper == "ruled" and pg.rule_spacing == pytest.approx(lay.line_h)
        firsts = [it for it in pg.items if it.x < lay.left + 2 * sy.xh]  # line-initial words
        assert len(firsts) >= 3
        for it in firsts:
            k = (it.y + 0.08 * sy.xh - pg.rule_top) / pg.rule_spacing
            assert abs(k - round(k)) * pg.rule_spacing < 0.3 * sy.xh

    def test_onto_base_pdf_writes_below_existing_content(self, toy_style, tmp_path):
        base = tmp_path / "problems.pdf"
        doc = pymupdf.open()
        for i in range(2):
            page = doc.new_page()
            page.insert_text((72, 90), f"Problem {i + 1}.", fontsize=14)
            page.insert_text((72, 140 + 40 * i), "Prove the statement below.", fontsize=12)
        doc.save(str(base))
        bottoms = [143.0, 183.0]

        blocks = parse_text("first answer\n\\newpage\nsecond answer")
        sy = Synthesizer(toy_style, seed=1)
        pages = layout_document(blocks, sy, LayoutOptions(base_pdf=str(base)))
        assert [p.base_page for p in pages] == [0, 1]
        for pg, bottom in zip(pages, bottoms):
            assert pg.items
            assert min(_page_ink_bounds(it)[1] for it in pg.items) > bottom

        out = tmp_path / "answers.pdf"
        write_pdf(pages, out, toy_style.stats.ink, base_pdf=base)
        with pymupdf.open(str(out)) as res:
            assert len(res) == 2
            for i, page in enumerate(res):
                assert f"Problem {i + 1}." in page.get_text()  # original content kept
                assert any(d["type"] == "f" for d in page.get_drawings())

    def test_onto_overflow_inserts_continuation_pages(self, toy_style, tmp_path):
        base = tmp_path / "one.pdf"
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 90), "Problem 1.", fontsize=14)
        doc.new_page().insert_text((72, 90), "Problem 2.", fontsize=14)
        doc.save(str(base))
        long = " ".join([LOREM] * 25)
        pages = layout_document(parse_text(long + "\n\\newpage\nshort", reflow=True), Synthesizer(toy_style, seed=1),
                                LayoutOptions(base_pdf=str(base)))
        base_pages = [p.base_page for p in pages]
        assert base_pages[0] == 0 and base_pages[-1] == 1
        assert None in base_pages[1:-1]  # blank continuation page(s) in between


# =========================================================================
# rendering
# =========================================================================
class TestRender:
    def _pages(self, style, text=None, seed=0, synth_kw=None, **opts):
        text = text if text is not None else " ".join([LOREM] * 20) + " Zigzag"
        return layout_document(parse_text(text, reflow=True), Synthesizer(style, seed=seed, **(synth_kw or {})),
                               LayoutOptions(**opts))

    def test_write_pdf_page_count_and_vector_ink(self, toy_style, tmp_path):
        # centre-line fallback glyphs so that stroked (round-pen) output is exercised too
        pages = self._pages(toy_style, synth_kw={"fallback_fills": False})
        assert len(pages) >= 2
        ink = (0.1, 0.12, 0.3)
        out = tmp_path / "out.pdf"
        write_pdf(pages, out, ink)
        with pymupdf.open(str(out)) as doc:
            assert len(doc) == len(pages)
            assert doc.metadata["creator"] == "handwritten"
            n_lines_total = 0
            for pg, page in zip(pages, doc):
                assert (page.rect.width, page.rect.height) == pytest.approx((pg.width, pg.height))
                assert page.get_images() == [] and page.get_text().strip() == ""
                ds = page.get_drawings()
                fills = [d for d in ds if d["type"] == "f"]
                strokes = [d for d in ds if d["type"] == "s"]
                # one filled path per pen stroke, made of Bezier curves, in the ink colour
                assert len(fills) == sum(len(it.box.fills) for it in pg.items)
                assert all(it[0] == "c" for d in fills for it in d["items"])
                assert all(d["fill"] == pytest.approx(ink, abs=2e-3) for d in fills)
                assert all(d["lineCap"][0] == 1 for d in strokes)  # synthetic lines: round pen
                n_lines_total += len(strokes)
            assert n_lines_total == sum(len(it.box.lines) for pg in pages for it in pg.items) > 0

    def test_rendered_ink_reads_back_as_strokes(self, toy_style, tmp_path):
        pages = self._pages(toy_style, text=LOREM)
        out = tmp_path / "out.pdf"
        write_pdf(pages, out, toy_style.stats.ink)
        (ip,) = extract_pdf(out)
        n_fills = sum(len(it.box.fills) for it in pages[0].items)
        assert len(ip.strokes) == n_fills
        assert [s.seq for s in ip.strokes] == sorted(s.seq for s in ip.strokes)
        assert ip.words == []

    @pytest.mark.parametrize("paper", ["plain", "ruled", "grid", "dots"])
    def test_paper_backgrounds(self, toy_style, tmp_path, paper):
        pages = self._pages(toy_style, text="a few words", background=paper)
        out = tmp_path / f"{paper}.pdf"
        write_pdf(pages, out, toy_style.stats.ink)
        with pymupdf.open(str(out)) as doc:
            ds = doc[0].get_drawings()
        ink_fills = [d for d in ds if d["type"] == "f" and d["fill"] == pytest.approx(toy_style.stats.ink, abs=2e-3)]
        extra = [d for d in ds if d not in ink_fills]
        assert ink_fills
        if paper == "plain":
            assert extra == []
        elif paper == "ruled":
            colours = {tuple(round(c, 2) for c in d["color"]) for d in extra if d["type"] == "s"}
            assert (0.62, 0.78, 0.93) in colours and (0.93, 0.55, 0.55) in colours  # rules + margin line
        else:
            assert len(extra) > 20

    def test_pdf_to_pngs(self, toy_style, tmp_path):
        pages = self._pages(toy_style, text="hello there")
        out = tmp_path / "p.pdf"
        write_pdf(pages, out, (0, 0, 0))
        paths = pdf_to_pngs(out, tmp_path / "p", dpi=40)
        assert len(paths) == 1 and paths[0].endswith("p-1.png")
        from PIL import Image

        im = np.asarray(Image.open(paths[0]).convert("L"))
        assert im.shape == (round(792 * 40 / 72), round(612 * 40 / 72))
        assert (im < 128).sum() > 20
