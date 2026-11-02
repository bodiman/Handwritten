"""Fallback glyphs: coverage, metrics, determinism, pen weight, outlines and
calibration to the writer's bank."""
from __future__ import annotations

import numpy as np
import pytest

from handwritten.fallback import PROCEDURAL, HersheyFallback, dot_outline, pen_profile, pen_stroke, stroke_outline
from handwritten.geometry import flatten_subpath, rasterize
from handwritten.style import BankGlyph, Style, StyleStats, glyph_pen_width
from handwritten.template import DEFAULT_CHARS, polyline_outline

STATS = dict(xh_pt=10.0, ascender=1.7, cap_height=1.55, descender=0.7, slant=0.2, pen_width=0.2)


def _teacher(**kw) -> HersheyFallback:
    """A synthetic 'writer' whose hand we know exactly."""
    args = dict(cap_height=1.4, slant=0.15, width_ratio=0.75, pen_width=0.2, ascender=1.65, descender=0.75,
                x_top=0.9, op_axis=0.6, op_scale=(0.8, 0.8))
    args.update(kw)
    return HersheyFallback(**args)


def _bank_from(fb: HersheyFallback, chars: str, n: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    bank, gid = {}, 0
    for c in chars:
        for _ in range(n):
            fills, _, w, top, bottom = fb.glyph_ink(c, rng)
            g = BankGlyph(c, fills, w, top, bottom, gid=gid)
            g.pen = glyph_pen_width(g)
            bank.setdefault(c, []).append(g)
            gid += 1
    return bank


@pytest.fixture(scope="module")
def writer():
    """A style whose bank is written by the teacher (lowercase, a few capitals,
    digits, operators and punctuation), like a real trained style."""
    t = _teacher()
    bank = _bank_from(t, "acemnorsuvwxzbdhklgpqy" + "SXF" + "237" + "+=-" + ".,")
    st = StyleStats(xh_pt=10.0, ascender=1.65, cap_height=1.4, descender=0.75, slant=0.15, pen_width=0.2)
    return t, Style(st, bank)


@pytest.fixture(scope="module")
def plain():
    return HersheyFallback.for_style(Style(StyleStats(**STATS), {}))


def _area(sp: np.ndarray) -> float:
    p = flatten_subpath(sp, 6)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# ---------------------------------------------------------------- coverage
def test_every_template_char_has_a_plausible_glyph(plain):
    rng = np.random.default_rng(0)
    fb = plain
    for c in DEFAULT_CHARS:
        r = fb.glyph_ink(c, rng)
        assert r is not None, c
        fills, lines, w, top, bottom = r
        assert fills and not lines, c
        assert 0.05 < w < 2.6, (c, w)
        assert top > bottom and -0.6 < top < 2.7 and -1.3 < bottom < 1.3, (c, top, bottom)
        g = fb.glyph(c, rng)
        assert g is not None and g[1] > 0 and g[2] > g[3], c
        assert all(abs(pw - fb.pen) < 1e-12 and len(p) >= 2 for p, pw in g[0]), c
        # ink starts at x = 0 like the real glyphs
        a = np.vstack([sp.reshape(-1, 2) for st in fills for sp in st])
        assert abs(a[:, 0].min()) < 1e-9 and abs(a[:, 0].max() - w) < 1e-9, c


def test_vertical_metrics_follow_the_writer(plain):
    """Ink extents land on the writer's x-height, cap, ascender, descender
    and maths axis (measured like the real glyphs: ink, not centre-lines)."""
    fb = plain
    rng = np.random.default_rng(1)

    def med(chars, i):
        return float(np.median([fb.glyph_ink(c, rng)[i] for c in chars for _ in range(3)]))

    assert med("acemnorsuvwxz", 3) == pytest.approx(fb.x_top, abs=0.08)
    assert med("acemnorsuvwxz", 4) == pytest.approx(0.0, abs=0.06)
    assert med("ABDEHKMNRTUVWXZ", 3) == pytest.approx(fb.upper_top, abs=0.08)
    assert med("ABDEHKMNRTUVWXZ", 4) == pytest.approx(0.0, abs=0.06)
    assert med("456789", 3) == pytest.approx(fb.digit_top, abs=0.1)
    assert med("bdhkl", 3) == pytest.approx(fb.asc, abs=0.1)
    assert -med("gpqy", 4) == pytest.approx(fb.desc, abs=0.12)
    for c in "+=≤≥≠≈±×÷<>∈⊂":
        _, _, _, top, bottom = fb.glyph_ink(c, rng)
        assert 0.5 * (top + bottom) == pytest.approx(fb.op_axis, abs=0.15), c
    # big operators are taller than capitals, the integral descends
    assert fb.glyph_ink("∑", rng)[3] > fb.upper_top
    assert fb.glyph_ink("∫", rng)[4] < -0.2


def test_procedural_and_greek_symbols(plain):
    rng = np.random.default_rng(2)
    chars = sorted(PROCEDURAL | set("αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩϑϕ²³₁ⁿéüñçÉ"))
    for c in chars:
        r = plain.glyph_ink(c, rng)
        assert r is not None, c
        _, _, w, top, bottom = r
        assert 0.1 < w < 2.4 and top > bottom and top < 2.7 and bottom > -1.3, (c, w, top, bottom)
        assert plain.strokes(c) is not None, c
    assert plain.glyph_ink("一", rng) is None  # nothing sensible for CJK: the synthesiser marks it missing


def test_template_label_usage_is_unchanged():
    """template.py draws labels with a hand-configured fallback's centre-lines."""
    hf = HersheyFallback(cap_height=1.45, slant=0.0, pen_width=0.12)
    lines, w, top, bottom = hf.glyph("∑", np.random.default_rng(0))
    assert lines and w > 0.5 and top > 1.45 and all(pw == 0.12 for _, pw in lines)
    lines, w, top, bottom = hf.glyph("A", np.random.default_rng(0))
    assert top == pytest.approx(1.45, abs=0.08) and bottom == pytest.approx(0.0, abs=0.06)


# ------------------------------------------------------------- determinism
def test_deterministic_with_fixed_rng(writer):
    fb = HersheyFallback.for_style(writer[1])
    for c in "Q4?;α≤ℝé":
        a = fb.glyph_ink(c, np.random.default_rng(7))
        b = fb.glyph_ink(c, np.random.default_rng(7))
        assert a[2:] == b[2:]
        assert all(np.array_equal(x, y) for sa, sb in zip(a[0], b[0]) for x, y in zip(sa, sb))
        la = fb.glyph(c, np.random.default_rng(7))
        lb = fb.glyph(c, np.random.default_rng(7))
        assert all(np.array_equal(p, q) for (p, _), (q, _) in zip(la[0], lb[0]))
    a = fb.glyph_ink("Q", np.random.default_rng(1))
    b = fb.glyph_ink("Q", np.random.default_rng(2))
    assert not np.allclose(a[0][0][0][:3], b[0][0][0][:3])  # instances differ like real samples


def test_no_variation_is_noise_free(plain):
    fb = HersheyFallback.for_style(Style(StyleStats(**STATS), {}))
    fb.var = 0.0
    a = fb.glyph_ink("R", np.random.default_rng(1))
    b = fb.glyph_ink("R", np.random.default_rng(2))
    assert a[2:] == pytest.approx(b[2:])


# ----------------------------------------------------------------- the pen
@pytest.mark.parametrize("pen", [0.15, 0.21])
def test_pen_width_matches_style(pen):
    """Measured like the real bank (ink area / skeleton length)."""
    fb = HersheyFallback.for_style(Style(StyleStats(**dict(STATS, pen_width=pen)), {}))
    rng = np.random.default_rng(3)
    vals = []
    for c in "acemnorsuvwxzABCDGHKMNRSUVWZ2345689":
        fills, _, w, top, bottom = fb.glyph_ink(c, rng)
        vals.append(glyph_pen_width(BankGlyph(c, fills, w, top, bottom)))
    assert np.median(vals) == pytest.approx(pen, rel=0.1)
    # stroke-to-stroke and glyph-to-glyph weight varies, like a real pen
    assert np.std(vals) / np.mean(vals) > 0.03


def test_pen_profile_tapers_and_modulates():
    s = np.linspace(0.0, 2.0, 81)
    w = pen_profile(s, 0.2, np.random.default_rng(0))
    mid = np.median(w[20:60])
    assert w[0] < 0.97 * mid and w[-1] < 0.9 * mid  # tapered ends, the lift-off thinner
    assert 0.15 < mid < 0.25 and np.std(w[20:60]) > 0.002  # gentle pressure modulation
    flat = pen_profile(s, 0.2, None)
    assert flat[40] == pytest.approx(0.2)


def test_outlines_are_consistently_wound_without_holes(writer):
    """All sub-paths share PencilKit's orientation, so the non-zero fill of
    a glyph equals the union of its pieces (no cancelled overlaps)."""
    fb = HersheyFallback.for_style(writer[1])
    rng = np.random.default_rng(4)
    for c in "BQ8@&%ß≈∞→ℝ":
        r = fb.glyph_ink(c, rng)
        if r is None:
            continue
        fills, _, w, top, bottom = r
        assert all(_area(sp) < 0 for st in fills for sp in st), c
        box = (-0.2, -top - 0.2, w + 0.2, -bottom + 0.2)
        whole = rasterize(fills, box, 40.0)
        union = np.zeros_like(whole)
        for st in fills:
            for sp in st:
                union |= rasterize([[sp]], box, 40.0)
        assert (whole == union).all(), c


def test_stroke_outline_sharp_corners_match_capsule_union():
    """A zig-zag with corners far sharper than the pen: the outline must not
    fold over (it equals the round-pen union of capsules)."""
    p = np.array([[0.0, 0.0], [0.5, -1.0], [0.62, 0.0], [1.4, -0.3], [0.2, -0.35]])
    dense = np.vstack([np.linspace(a, b, 12, endpoint=False) for a, b in zip(p[:-1], p[1:])] + [p[-1:]])
    w = 0.3
    ours = stroke_outline(dense, np.full(len(dense), w))
    ref = polyline_outline(dense, w, eps=1e-4)
    box = (-0.3, -1.4, 1.8, 0.4)
    a = rasterize([ours], box, 60.0)
    b = rasterize([ref], box, 60.0)
    iou = (a & b).sum() / (a | b).sum()
    assert iou > 0.97
    d = dot_outline((0.0, 0.0), 0.3)
    assert _area(d) < 0 and rasterize([[d]], (-0.2, -0.2, 0.2, 0.2), 100.0).sum() == pytest.approx(
        np.pi * 0.15 ** 2 * 1e4, rel=0.05)


def test_pen_stroke_in_page_units():
    """The public helper works in points (unit = x-height in pt), e.g. for
    synthetic maths strokes: measured pen ~ requested pen, tap for a point."""
    from skimage.morphology import skeletonize

    xh = 10.0
    pts = np.array([[0.0, 0.0], [6.0, -9.0], [14.0, -12.0], [22.0, -6.0]])
    vals = []
    for seed in range(4):
        sps = pen_stroke(pts, 2.0, np.random.default_rng(seed), unit=xh)
        assert sps and all(sp.ndim == 3 and sp.shape[1:] == (4, 2) and _area(sp) < 0 for sp in sps)
        m = rasterize([sps], (-3, -16, 26, 4), 8.0)
        vals.append(m.sum() / skeletonize(m).sum() / 8.0)
    assert np.median(vals) == pytest.approx(2.0, rel=0.15)
    tap = pen_stroke(np.array([[5.0, 5.0]]), 2.0, None, unit=xh)
    assert len(tap) == 1 and np.allclose(tap[0].reshape(-1, 2).mean(0), (5.0, 5.0), atol=0.2)
    assert pen_stroke(np.zeros((0, 2)), 2.0) == []


# ------------------------------------------------------------- calibration
def test_calibration_recovers_the_writers_proportions(writer):
    t, style = writer
    fb = HersheyFallback.for_style(style)
    for k in ("lower", "upper", "digit"):
        assert fb.group_ratio[k] == pytest.approx(t.width_ratio, abs=0.08), k
    assert fb.x_top == pytest.approx(t.x_top, abs=0.05)
    assert fb.op_axis == pytest.approx(t.op_axis, abs=0.06)
    assert np.sqrt(fb.op_scale[0] * fb.op_scale[1]) == pytest.approx(0.8, abs=0.08)
    # characters the writer never wrote get the writer's widths
    rng = np.random.default_rng(5)
    for c in "BDHNRU4569":
        ws = [fb.glyph_ink(c, rng)[2] for _ in range(4)]
        wt = [t.glyph_ink(c, rng)[2] for _ in range(4)]
        assert np.median(ws) == pytest.approx(np.median(wt), rel=0.15), c


def test_composed_from_real_parts(writer):
    _, style = writer
    fb = HersheyFallback.for_style(style)
    rng = np.random.default_rng(6)

    def owners(fills, chars):
        real = {id(sp): c for c in chars for g in style.bank[c] for st in g.strokes for sp in st}
        found = []
        for st in fills:
            for sp in st:
                for c in chars:
                    for g in style.bank[c]:
                        for rst in g.strokes:
                            for rsp in rst:
                                if rsp.shape == sp.shape and np.allclose(rsp - rsp[0, 0], sp - sp[0, 0]):
                                    found.append(c)
        return found, real

    found, _ = owners(fb.glyph_ink(":", rng)[0], ".")
    assert found.count(".") == 2  # both dots are the writer's own
    found, _ = owners(fb.glyph_ink(";", rng)[0], ".,")
    assert sorted(set(found)) == [",", "."]
    fills, _, w, top, bottom = fb.glyph_ink("é", rng)
    found, _ = owners(fills, "e")
    assert "e" in found and top > fb.x_top + 0.2  # the writer's e under a synthetic accent
    assert fb.glyph_ink("Ο", rng) is not None  # Greek Omicron -> Latin O (Hershey, not in the bank)
