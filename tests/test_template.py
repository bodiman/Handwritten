"""Calibration sheet: template generation and filled-sheet round trips.

Filled sheets are simulated by writing synthetic glyphs (the Hershey fallback
of an empty style) into the cells with the regular renderer, on top of the
template -- as PencilKit-style filled outlines, as stroked paths and as PDF
ink annotations.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pymupdf
import pytest

from handwritten.box import InkBox
from handwritten.geometry import rasterize
from handwritten.render import Placed, RenderPage, write_pdf
from handwritten.style import BankGlyph, Style, StyleStats
from handwritten.synth import Synthesizer
from handwritten.template import (DEFAULT_CHARS, SheetReport, _LabelFonts, capsule, cell_width, chars_needed,
                                  layout_sheet, make_template, merge_calibration, polyline_outline, read_layouts,
                                  style_from_sheets, user_ink)

CHARS = "agA7?∑("


def _style() -> Style:
    return Style(StyleStats(xh_pt=10.0, slant=0.15, pen_width=0.18), {})


def _to_fills(box: InkBox) -> InkBox:
    fills = list(box.fills) + [polyline_outline(p, w) for p, w in box.lines]
    return InkBox(fills, [], box.width, box.ascent, box.descent, box.text)


def _fill(template, out, mode="filled", size=1.0, seed=0, layouts=None, cells=None):
    """Write a glyph into every cell; returns [(page, cell, box)]."""
    lays = layouts or read_layouts(template)
    synth = Synthesizer(_style(), seed=seed, x_height=lays[0].xh, fallback_fills=(mode != "stroked"))
    pages, truth = [], []
    for lay in lays:
        pg = RenderPage(lay.width, lay.height, base_page=lay.page)
        cs = lay.cells()
        for c in (cs if cells is None else cells(cs)):
            box = synth.glyph_box(c.char, size)  # Hershey centre-line strokes
            if mode == "filled":
                box = _to_fills(box)
            pg.items.append(Placed(box, c.x0 + (c.width - box.width) / 2, c.baseline))
            truth.append((lay.page, c, box))
        pages.append(pg)
    write_pdf(pages, out, (0.1, 0.1, 0.12), base_pdf=template)
    return truth


def _check_metrics(style: Style, truth, tol=0.03):
    exp = defaultdict(list)
    for _, c, box in truth:
        b = box.ink_bounds()
        exp[c.char].append((-b[1] / c.xh, -b[3] / c.xh, (b[2] - b[0]) / c.xh))
    assert set(style.bank) == set(exp)
    for ch, gs in style.bank.items():
        assert len(gs) == len(exp[ch]), ch
        for g, (top, bottom, width) in zip(gs, exp[ch]):
            assert g.char == ch
            assert g.source.startswith("sheet:")
            assert abs(g.top - top) < tol and abs(g.bottom - bottom) < tol and abs(g.width - width) < tol, ch
            pts = np.vstack([sp.reshape(-1, 2) for st in g.strokes for sp in st])
            assert abs(pts[:, 0].min()) < 0.05  # glyph frame: x from the left ink edge
            assert all(sp.ndim == 3 and sp.shape[1:] == (4, 2) for st in g.strokes for sp in st)


# --------------------------------------------------------------------- template
def test_default_chars():
    required = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                ".,;:!?'\"()[]{}-+=*/\\<>%&#@$_~^|αβγδεθλμπσφψωΔΣΠΩΓ∞∑∏∫√≤≥≠≈≡±×÷→⇒∈⊂∪∩∂∇∀∃")
    assert set(required) <= set(DEFAULT_CHARS)
    assert len(set(DEFAULT_CHARS)) == len(DEFAULT_CHARS)
    assert cell_width("m") > cell_width("a") > cell_width("i")
    assert cell_width("W") > cell_width("A")


def test_layout_is_deterministic_and_valid():
    a = layout_sheet()
    b = layout_sheet()
    assert [l.payload() for l in a] == [l.payload() for l in b]
    seen = defaultdict(int)
    for lay in a:
        cells = lay.cells()
        R = np.array([[c.x0, c.y0, c.x1, c.y1] for c in cells])
        assert R[:, 0].min() >= 30 and R[:, 2].max() <= lay.width - 30
        assert R[:, 1].min() >= 60 and R[:, 3].max() <= lay.height - 30
        # no two cells overlap
        ox = np.minimum(R[:, None, 2], R[None, :, 2]) - np.maximum(R[:, None, 0], R[None, :, 0])
        oy = np.minimum(R[:, None, 3], R[None, :, 3]) - np.maximum(R[:, None, 1], R[None, :, 1])
        overlap = (ox > 0.01) & (oy > 0.01)
        np.fill_diagonal(overlap, False)
        assert not overlap.any()
        for c in cells:
            seen[c.char] += 1
            assert abs(c.baseline - c.y0 - 2.4 * c.xh) < 1e-6
    assert all(seen[c] == 4 for c in DEFAULT_CHARS) and len(seen) == len(DEFAULT_CHARS)
    assert abs(a[0].xh - 10.0) < 1e-9
    # big repeat counts wrap instead of running off the page
    for lay in layout_sheet("mW", repeats=20):
        assert max(c.x1 for c in lay.cells()) <= lay.width - 30


def test_make_template(tmp_path):
    path = tmp_path / "t.pdf"
    lays = make_template(path)
    with pymupdf.open(str(path)) as doc:
        assert len(doc) == len(lays) >= 2
        assert "HWTEMPLATE v1" in (doc.metadata.get("keywords") or "")
        visible = set()
        for page in doc:
            assert "HWTEMPLATE v1" in page.get_text()
            for b in page.get_text("rawdict")["blocks"]:
                for line in b.get("lines", []):
                    for span in line["spans"]:
                        if span["size"] >= 5:
                            visible.update(c["c"] for c in span["chars"])
            # the template's own guides and labels are never taken for ink
            assert user_ink(page) == []
            pix = page.get_pixmap(dpi=30)
            assert pix.width > 0
        if _LabelFonts().pick(DEFAULT_CHARS) is not None:  # a Unicode font is available (DejaVu / PyMuPDF CJK)
            assert set(DEFAULT_CHARS) <= visible  # every label is real, legible text
        guides = [d for d in doc[0].get_drawings() if d["type"] == "s"]
        assert any(d.get("dashes") and "0 " not in d["dashes"][:3] for d in guides)  # dashed x-height
    back = read_layouts(path)
    assert [l.origin for l in back] == ["marker"] * len(lays)
    assert [l.payload() for l in back] == [l.payload() for l in lays]
    assert all(np.allclose(l.transform, (1, 1, 0, 0), atol=1e-6) for l in back)


# ------------------------------------------------------------------ round trips
@pytest.mark.parametrize("mode", ["filled", "stroked"])
def test_round_trip(tmp_path, mode):
    tpl, out = tmp_path / "t.pdf", tmp_path / "f.pdf"
    make_template(tpl, chars=CHARS, repeats=3)
    truth = _fill(tpl, out, mode)
    with pymupdf.open(str(out)) as doc:  # write_pdf replaced the metadata: markers must carry the layout
        assert "HWTEMPLATE" not in (doc.metadata.get("keywords") or "")
        kinds = {d["type"] for d in doc[0].get_drawings() if d.get("fill") is not None or d["type"] == "s"}
        assert ("f" in kinds) if mode == "filled" else ("s" in kinds)
    rep = SheetReport()
    st = style_from_sheets([out], rescale=False, report=rep)
    assert rep.origins == {"marker": 1} and rep.accepted == len(CHARS) * 3 and not rep.rejected
    _check_metrics(st, truth)
    assert abs(st.stats.xh_pt - 10.0) < 1e-6
    assert st.stats.ink == pytest.approx((0.1, 0.1, 0.12), abs=0.01)
    pens = [g.pen for gs in st.bank.values() for g in gs if hasattr(g, "pen")]
    if pens:  # measured per glyph: the synthetic pen is 0.18 x-heights (+ variation)
        assert 0.1 < np.median(pens) < 0.3 and 0.1 < st.stats.pen_width < 0.3


def test_layout_recovered_without_markers(tmp_path):
    tpl, out = tmp_path / "t.pdf", tmp_path / "f.pdf"
    make_template(tpl, chars=CHARS, repeats=3, markers=False)
    lays = read_layouts(tpl)
    assert [l.origin for l in lays] == ["guides"]
    assert "".join(g.char for g in lays[0].groups) == "agA7?(∑"  # grouped by section
    truth = _fill(tpl, out, "filled", layouts=lays)
    _check_metrics(style_from_sheets([out], rescale=False), truth)


def test_flattened_background_uses_recomputed_layout(tmp_path):
    tpl, flat, out = tmp_path / "t.pdf", tmp_path / "flat.pdf", tmp_path / "f.pdf"
    lays = make_template(tpl, markers=False)
    with pymupdf.open(str(tpl)) as src, pymupdf.open() as dst:
        for p in src:  # an app that rasterises the imported background
            pg = dst.new_page(width=p.rect.width, height=p.rect.height)
            pg.insert_image(pg.rect, pixmap=p.get_pixmap(dpi=40))
        dst.save(str(flat))
    assert read_layouts(flat) == [None] * len(lays)  # nothing to go on without ink
    truth = _fill(flat, out, "filled", layouts=lays, cells=lambda cs: cs[5:40:3])
    back = read_layouts(out)
    assert [l.origin for l in back] == ["recomputed"] * len(lays)
    _check_metrics(style_from_sheets([out], rescale=False), truth)


def test_scaled_export_is_registered(tmp_path):
    tpl, out, scaled = tmp_path / "t.pdf", tmp_path / "f.pdf", tmp_path / "s.pdf"
    make_template(tpl, chars=CHARS, repeats=3)
    truth = _fill(tpl, out, "filled")
    with pymupdf.open(str(out)) as src, pymupdf.open() as dst:
        pg = dst.new_page(width=800, height=1100)
        pg.show_pdf_page(pymupdf.Rect(15, 40, 785, 1080), src, 0)
        dst.save(str(scaled))
    lay = read_layouts(scaled)[0]
    assert lay.origin == "marker" and abs(lay.transform[0] - 770 / 612) < 0.01
    _check_metrics(style_from_sheets([scaled], rescale=False), truth)


def test_ink_annotations(tmp_path):
    tpl, out = tmp_path / "t.pdf", tmp_path / "f.pdf"
    lays = make_template(tpl, chars=CHARS, repeats=2)
    synth = Synthesizer(_style(), seed=1, x_height=lays[0].xh, fallback_fills=False)
    truth = []
    with pymupdf.open(str(tpl)) as doc:
        for c in lays[0].cells():
            box = synth.glyph_box(c.char)
            x = c.x0 + (c.width - box.width) / 2
            annot = doc[0].add_ink_annot([[(float(q[0] + x), float(q[1] + c.baseline)) for q in p]
                                          for p, _ in box.lines])
            annot.set_border(width=box.lines[0][1])
            annot.set_colors(stroke=(0.0, 0.2, 0.6))
            annot.update()
            truth.append((0, c, box))
        doc.save(str(out))
    _check_metrics(style_from_sheets([out], rescale=False), truth, tol=0.04)


def test_empty_spilled_and_tiny_cells(tmp_path):
    tpl, out = tmp_path / "t.pdf", tmp_path / "f.pdf"
    lays = make_template(tpl, chars="aH", repeats=4)
    cells = lays[0].cells()
    xh = lays[0].xh
    items = []
    synth = Synthesizer(_style(), seed=2, x_height=xh)
    a0, a1, a2, a3, h0 = cells[0], cells[1], cells[2], cells[3], cells[4]
    good = _to_fills(synth.glyph_box("a"))
    items.append(Placed(good, a0.x0 + 5, a0.baseline))
    # a stroke centred in a1 that leaves the box by ~1.5 x-heights
    long = InkBox([polyline_outline(np.array([[-a1.width / 2 - 1.5 * xh, -5.0], [a1.width / 2 + 1.5 * xh, -5.0]]),
                                    2.0)], [], 0, 0, 0)
    items.append(Placed(long, a1.x0 + a1.width / 2, a1.baseline))
    # a pen tap in a2
    items.append(Placed(InkBox([[capsule((0, 0), (0.3, 0), 1.0)]], [], 0, 0, 0), a2.x0 + 10, a2.baseline + 6))
    # a3 left empty; one H
    items.append(Placed(_to_fills(synth.glyph_box("H")), h0.x0 + 5, h0.baseline))
    write_pdf([RenderPage(lays[0].width, lays[0].height, items, base_page=0)], out, (0, 0, 0), base_pdf=tpl)
    rep = SheetReport()
    st = style_from_sheets([out], rescale=False, report=rep)
    assert {c: len(v) for c, v in st.bank.items()} == {"a": 1, "H": 1}
    assert rep.rejected == {"spill": 1, "tiny": 1} and rep.filled == 4


def test_rescale_to_base_style(tmp_path):
    """Writing 30 % bigger in boxes is undone: per character where the base
    style has >= 3 samples, by the median of those factors elsewhere."""
    tpl, out = tmp_path / "t.pdf", tmp_path / "f.pdf"
    make_template(tpl, chars="aceA", repeats=3)
    truth = _fill(tpl, out, "filled", size=1.3)
    ref = style_from_sheets([out], rescale=False)
    base_bank = {}
    want = {}
    for ch, f in (("a", 0.97), ("c", 1.0), ("e", 1.04)):
        h = f * float(np.median([g.top - g.bottom for g in ref.bank[ch]])) / 1.3
        want[ch] = h
        gs = []
        for i in range(4):
            g = ref.bank[ch][i % 3]
            k = h / (g.top - g.bottom)
            gs.append(BankGlyph(ch, [[sp * k for sp in st] for st in g.strokes], g.width * k, g.top * k,
                                g.bottom * k, "base", i))
        base_bank[ch] = gs
    base = Style(StyleStats(xh_pt=9.5, cap_height=1.6), base_bank, {})
    st = style_from_sheets([out], base=base)
    for ch, h in want.items():
        assert np.median([g.top - g.bottom for g in st.bank[ch]]) == pytest.approx(h, rel=0.02)
    k = st.meta["sheet_scale"]
    assert k == pytest.approx(1 / 1.3, rel=0.02)
    ha = np.median([g.top for g in st.bank["A"]])
    assert ha == pytest.approx(np.median([g.top for g in ref.bank["A"]]) * k, rel=1e-3)
    assert st.stats.xh_pt == 9.5  # stats come from the base style
    merged = base.merge(st)
    assert len(merged.bank["a"]) == 7 and len(merged.bank["A"]) == 3
    # scarce free-text classes are replaced by the (trusted) sheet samples
    base.bank["A"] = base.bank["a"][:2]  # e.g. two mis-recognised 'A's
    mc = merge_calibration(base, st, replace_below=3)
    assert [g.source for g in mc.bank["A"]] == ["sheet:1"] * 3 and len(mc.bank["a"]) == 7
    assert mc.meta["replaced_by_sheet"] == "A" and sum(len(v) for v in mc.bank.values()) == 3 * 4 + 12
    # without a base: the sheet's small letters define the x-height
    nb = style_from_sheets([out])
    tops = [g.top for c in "ace" for g in nb.bank[c]]
    assert np.median(tops) == pytest.approx(1.0, abs=0.02)
    assert nb.stats.xh_pt == pytest.approx(10.0 / nb.meta["sheet_scale"], rel=1e-3)


def test_capsule_outline_fills_correctly():
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [3.0, 4.0]])
    sps = polyline_outline(pts, 2.0)
    m = rasterize([sps], (-2, -2, 12, 12), 20.0)
    length = 10 + 10 + np.hypot(7, 6)
    area = m.sum() / 400.0
    assert area == pytest.approx(length * 2.0 + np.pi, rel=0.06)  # overlaps fill once (non-zero rule)
    # one orientation for every direction
    def signed_area(sp):
        p = sp[:, 0, :]
        return 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    signs = {np.sign(signed_area(capsule((0, 0), (np.cos(t), np.sin(t)), 0.5))) for t in np.linspace(0, 6, 13)}
    assert len(signs) == 1 and np.sign(signed_area(capsule((1, 1), (1, 1), 0.5))) in signs


def test_chars_needed_and_cli(tmp_path):
    from handwritten.cli import main

    st = Style(StyleStats(), {"a": [None] * 5, "B": [None] * 1})
    need = chars_needed(st)
    assert need.startswith("abcd") and need.count("a") == 1 and "B" in need  # 'a' kept as a size anchor
    assert chars_needed(st, anchors="").startswith("bcd")
    assert chars_needed(Style(StyleStats(), {c: [None] * 3 for c in DEFAULT_CHARS})) == ""
    out = tmp_path / "cli.pdf"
    assert main(["template", "-o", str(out), "--chars", "x Y z", "--repeats", "2"]) == 0
    lay = read_layouts(out)[0]
    assert [g.char for g in lay.groups] == ["x", "z", "Y"] and lay.groups[0].n == 2
