"""Tests for the handwritten maths typesetter (``handwritten.mathlayout``)."""
from __future__ import annotations

import os
import random

import numpy as np
import pytest

from handwritten import mathlayout as M
from handwritten.box import InkBox
from handwritten.mathlayout import (BigOp, Fenced, Frac, Matrix, OpName, Row, Scripts, Sqrt, Sym, Text, parse,
                                    render_math, tokenize)

STYLE_CANDIDATES = [
    os.environ.get("HANDWRITTEN_TEST_STYLE", ""),
    "/tmp/claude-0/-home-user-Handwritten/525beb19-0610-5fa1-818c-aaa2915c4eca/scratchpad/me.hwstyle",
]
STYLE_PATH = next((p for p in STYLE_CANDIDATES if p and os.path.exists(p)), None)
needs_style = pytest.mark.skipif(STYLE_PATH is None, reason="trained style not available")

GALLERY = [
    r"n(n+1)/2 \mid n!", r"\frac{n(n+1)}{2}", r"3^n", r"x_{n+1} = x_n + \frac{1}{2^n}",
    r"\varphi_n = \sum_{i=1}^{n-1} \varphi_i\varphi_{n-i}", r"\binom{2k-2}{k-1}",
    r"\lim_{n\to\infty} x_n = x_0 + (x_1 - x_0)S", r"2^n + 1 \equiv 0 \pmod 7",
    r"p_1^{\alpha_1} p_2^{\alpha_2}\cdots p_k^{\alpha_k}", r"\sqrt{2}", r"\int_0^1 f(x)\,dx",
    r"\left(\frac{1}{3}\right)^n", r"\{1,2,\dots,n\}", r"a \le b \ne c", r"n \in \mathbb{N}",
    r"\frac{(2k-2)!}{k!(k-1)!}", r"\sqrt[3]{x^2+1}", r"f(x) = \begin{cases} x^2 & x \ge 0 \\ -x & \text{otherwise} \end{cases}",
    r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}", r"\overline{zw} = \bar z \bar w", r"|x - y| < \varepsilon",
]

JUNK = [
    "", " ", "{", "}", "}}}{{{", "^", "_", "x^", "x_", "^^^", "x^^2", "x__2", "\\", "\\\\", "&", "&&",
    r"\frac", r"\frac{1}", r"\sqrt", r"\sqrt[", r"\sqrt[3", r"\left(", r"\right)", r"\left( x", r"x \right]",
    r"\left", r"\right", r"\middle|", r"\begin{pmatrix}", r"\end{pmatrix}", r"\begin{nonsense} a & b \end{nonsense}",
    r"\foo", r"\foo{bar}", r"\text{", r"\text{unclosed", r"\mathbb{", r"\big", r"\not", r"\over", r"a \over",
    r"$x$", "%", r"\binom{1}", r"\operatorname", r"\hspace{", r"\overset{a}", "{" * 200 + "x" + "}" * 200,
    r"\frac{" * 60 + "x", "x^" * 80, r"\left(" * 50, "α ≤ β → ∞ ∑ ∫", "x² + y₁",
]


# ----------------------------------------------------------------------------- fixtures

@pytest.fixture(autouse=True)
def _centre_line_strokes(monkeypatch):
    """Structural tests inspect synthetic strokes as centre-lines."""
    import handwritten.mathlayout as ml

    monkeypatch.setattr(ml, "PEN_FILLS", False)


@pytest.fixture(scope="module")
def style():
    if STYLE_PATH is None:
        pytest.skip("trained style not available")
    from handwritten.style import Style

    return Style.load(STYLE_PATH)


def make_synth(style, seed=0, **kw):
    from handwritten.synth import Synthesizer

    return Synthesizer(style, seed=seed, **kw)


def strip_points(box: InkBox) -> np.ndarray:
    pts = M._ink_points(box)
    assert pts is not None
    return pts


def fill_points(strokes) -> np.ndarray:
    from handwritten.geometry import flatten_subpath

    return np.vstack([flatten_subpath(sp, 6) for st in strokes for sp in st])


# ----------------------------------------------------------------------------- tokenizer / parser
def test_tokenize_commands_and_symbols():
    toks = tokenize(r"\frac{a}{b}^2_\alpha \, x")
    kinds = [(t.kind, t.value) for t in toks]
    assert kinds[0] == ("cmd", "frac")
    assert ("lbrace", "{") in kinds and ("rbrace", "}") in kinds
    assert ("sup", "^") in kinds and ("sub", "_") in kinds
    assert ("cmd", "alpha") in kinds and ("cmd", ",") in kinds
    assert kinds[-1][0] == "eof"


def test_combined_scripts():
    n = parse("x_i^2")
    assert isinstance(n, Row) and len(n.items) == 1
    s = n.items[0]
    assert isinstance(s, Scripts)
    assert s.base == Sym("x") and s.sub == Sym("i") and s.sup == Sym("2")


def test_nested_scripts_and_primes():
    s = parse("x^{y^{z}}").items[0]
    assert isinstance(s, Scripts) and isinstance(s.sup, Row)
    inner = s.sup.items[0]
    assert isinstance(inner, Scripts) and inner.base == Sym("y") and inner.sup == Row([Sym("z")])
    f = parse("f''(x)").items[0]
    assert isinstance(f, Scripts) and f.primes == 2 and f.sup is None


def test_fraction_and_variants():
    for cmd, style in (("frac", "auto"), ("dfrac", "d"), ("tfrac", "t")):
        f = parse(rf"\{cmd}{{a+b}}{{2}}").items[0]
        assert isinstance(f, Frac) and f.style == style and f.bar
        assert [type(x) for x in f.num.items] == [Sym, Sym, Sym]
    f = parse(r"\frac12").items[0]
    assert f.num == Sym("1") and f.den == Sym("2")
    b = parse(r"\binom{n}{k}").items[0]
    assert isinstance(b, Fenced) and b.left == "(" and b.right == ")"
    assert isinstance(b.body.items[0], Frac) and not b.body.items[0].bar
    o = parse(r"a \over b").items[0]
    assert isinstance(o, Frac)


def test_roots_delimiters_and_operators():
    r = parse(r"\sqrt[3]{x}").items[0]
    assert isinstance(r, Sqrt) and r.index == Row([Sym("3")])
    lr = parse(r"\left( \frac{1}{3} \right)^n").items[0]
    assert isinstance(lr, Scripts) and isinstance(lr.base, Fenced) and not lr.base.auto
    auto = parse("(a+b)^2").items[0]
    assert isinstance(auto, Scripts) and isinstance(auto.base, Fenced) and auto.base.auto
    dot = parse(r"\left. x \right|").items[0]
    assert dot.left == "." and dot.right == "|"
    s = parse(r"\sum_{i=1}^{n} a_i").items[0]
    assert isinstance(s, Scripts) and isinstance(s.base, BigOp) and s.base.ch == "∑"
    lim = parse(r"\lim_{n\to\infty} x_n").items[0]
    assert isinstance(lim.base, OpName) and lim.base.name == "lim" and lim.base.limits
    assert any(isinstance(x, Sym) and x.ch == "→" and x.cls == "rel" for x in lim.sub.items)
    sin = parse(r"\sin x").items[0]
    assert isinstance(sin, OpName) and not sin.limits
    braces = parse(r"\{1,2,\dots,n\}").items[0]
    assert isinstance(braces, Fenced) and braces.left == "{" and braces.right == "}"
    bars = parse("|x| + |y|").items
    assert isinstance(bars[0], Fenced) and bars[0].left == "|" and isinstance(bars[2], Fenced)
    divides = parse(r"2 \mid n").items
    assert [type(x) for x in divides] == [Sym, Sym, Sym] and divides[1].cls == "rel"


def test_symbols_greek_text_and_fonts():
    items = parse(r"\alpha \Omega \le \cdot \infty").items
    assert [x.ch for x in items] == ["α", "Ω", "≤", "·", "∞"]
    assert [x.cls for x in items] == ["ord", "ord", "rel", "bin", "ord"]
    assert parse(r"\mathbb{R}").items[0] == Row([Sym("ℝ")])
    assert parse(r"\mathbb N").items[0] == Sym("ℕ")
    t = parse(r"\text{if } x").items[0]
    assert isinstance(t, Text) and t.text == "if "
    op = parse(r"\operatorname{lcm}(a,b)").items[0]
    assert isinstance(op, OpName) and op.name == "lcm"
    pm = parse(r"2^n \equiv 0 \pmod 7")
    texts = []

    def walk(n):
        if isinstance(n, Text):
            texts.append(n.text)
        for v in getattr(n, "__dict__", {}).values():
            if isinstance(v, list):
                for x in v:
                    walk(x)
            elif isinstance(v, M.Node):
                walk(v)

    walk(pm)
    assert "mod" in texts


def test_environments_and_rows():
    m = parse(r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}").items[0]
    assert isinstance(m, Matrix) and m.kind == "pmatrix"
    assert len(m.rows) == 2 and all(len(r) == 2 for r in m.rows)
    c = parse(r"f = \begin{cases} 1 & x > 0 \\ 0 & \text{else} \end{cases}").items[-1]
    assert isinstance(c, Matrix) and c.kind == "cases" and len(c.rows) == 2
    top = parse(r"a &= b \\ c &= d")
    assert isinstance(top, Row) and isinstance(top.items[0], Matrix) and top.items[0].kind == "aligned"


def test_words_bars_and_unicode_scripts():
    items = parse(r"\mathrm{d}x + \mathrm{Var}(X)").items
    assert items[0] == Row([Sym("d")]) and Text("Var") in items
    assert parse("2 | n").items[1] == Sym("∣", "rel")  # a lone bar between terms means "divides"
    sq = parse("x² + y₁").items
    assert isinstance(sq[0], Scripts) and sq[0].sup == Row([Sym("2")])
    assert isinstance(sq[2], Scripts) and sq[2].sub == Row([Sym("1")])
    assert parse("a{}b").items[1] == Row()


def test_unknown_commands_render_their_name():
    n = parse(r"\foobar x")
    assert isinstance(n.items[0], Text) and n.items[0].text == "foobar"


@pytest.mark.parametrize("src", JUNK)
def test_parser_never_raises(src):
    parse(src)


# ----------------------------------------------------------------------------- layout (needs the trained style)
@needs_style
def test_empty_and_whitespace(style):
    s = make_synth(style)
    for src in ("", "   "):
        b = render_math(src, s)
        assert isinstance(b, InkBox) and b.is_empty() and b.width == 0


@needs_style
def test_junk_renders_without_error(style):
    s = make_synth(style, seed=3)
    for src in JUNK:
        for disp in (False, True):
            b = render_math(src, s, display=disp)
            assert isinstance(b, InkBox)
            assert b.width >= 0 and b.ascent >= 0 and b.descent >= 0


@needs_style
def test_random_junk_never_crashes(style):
    s = make_synth(style, seed=5)
    rng = random.Random(7)
    alphabet = list("\\{}[]()^_&$%~'|<>=+-/.,;:!abxy0123 ") + [r"\frac", r"\sqrt", r"\left", r"\right", r"\\",
                                                              r"\begin{cases}", r"\end{cases}", r"\sum", r"\text{"]
    for _ in range(80):
        src = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 25)))
        b = render_math(src, s, display=rng.random() < 0.5)
        assert b.width >= 0


@needs_style
@pytest.mark.parametrize("display", [False, True])
def test_gallery_metrics_cover_ink(style, display):
    """Width/ascent/descent must enclose all ink so lines can be stacked."""
    s = make_synth(style, seed=11)
    for src in GALLERY:
        b = render_math(src, s, display=display)
        assert b.width > 0, src
        bounds = b.ink_bounds()
        assert bounds is not None, src
        x0, y0, x1, y1 = bounds
        tol = 0.05
        assert x0 >= -tol and x1 <= b.width + tol, (src, bounds, b.width)
        assert -y0 <= b.ascent + tol and y1 <= b.descent + tol, (src, bounds, b.ascent, b.descent)
        assert b.ascent > 0.5 * s.xh, src
        assert s.missing == {}, (src, dict(s.missing))


@needs_style
def test_glyphs_come_from_the_synthesizer(style):
    """Letters and digits must be the writer's real glyphs (outline fills)."""
    s = make_synth(style, seed=2)
    calls = []
    orig = s.glyph_box

    def spy(ch, size=1.0):
        calls.append((ch, size))
        return orig(ch, size)

    s.glyph_box = spy
    b = render_math(r"x_{n+1} = an", s)
    chars = [c for c, _ in calls]
    for ch in "xn1=a":
        assert ch in chars
    sizes = dict((c, sz) for c, sz in calls)
    assert sizes["a"] == pytest.approx(1.0) and sizes["1"] == pytest.approx(M.SCRIPT_SCALE)
    assert b.fills  # real outline strokes


def _scripted(style, seed, base, **kw):
    """Lay out ``base`` alone and ``Scripts(base, ...)`` with identical random draws.
    Returns (synth, base box, scripts box, script-only ink points); the base's
    strokes are drawn first, so the remaining strokes belong to the scripts."""
    s1, s2 = make_synth(style, seed=seed), make_synth(style, seed=seed)
    b1 = M.MathLayout(s1).lay(base, M.Ctx())
    b2 = M.MathLayout(s2).lay(Scripts(base, **kw), M.Ctx())
    extra = InkBox(fills=b2.fills[len(b1.fills):], lines=b2.lines[len(b1.lines):])
    return s1, b1, b2, extra


@needs_style
def test_superscript_is_raised(style):
    for seed in range(6):
        s, base, box, sup = _scripted(style, seed, Sym("x"), sup=Sym("2"))
        pts = strip_points(sup)
        bpts = strip_points(base)
        # the whole superscript sits well above the baseline, its top above the base's top
        assert pts[:, 1].max() < -0.3 * s.xh, seed
        assert pts[:, 1].min() < bpts[:, 1].min(), seed
        # ... and to the right of the base's ink at the same height
        band = bpts[(bpts[:, 1] >= pts[:, 1].min()) & (bpts[:, 1] <= pts[:, 1].max())]
        if len(band):
            assert pts[:, 0].min() > band[:, 0].max() - 0.15 * s.xh, seed
        assert box.width > base.width and box.ascent > base.ascent


@needs_style
def test_subscript_is_lowered(style):
    for seed in range(6):
        s, base, box, sub = _scripted(style, seed, Sym("x"), sub=Sym("n"))
        pts = strip_points(sub)
        assert pts[:, 1].max() > 0.15 * s.xh, seed  # reaches below the baseline
        assert pts[:, 1].min() > -0.85 * s.xh, seed  # top well below the x-height line
        assert pts[:, 0].max() > base.width, seed  # sticks out to the right of the base
        assert box.descent > 0.15 * s.xh


@needs_style
def test_combined_scripts_do_not_collide(style):
    for seed in range(6):
        s, base, box, scripts = _scripted(style, seed, Sym("x"), sup=Sym("2"), sub=Sym("i"))
        groups = [strip_points(InkBox(fills=[st])) for st in scripts.fills] + \
                 [strip_points(InkBox(lines=[ln])) for ln in scripts.lines]
        centres = [0.5 * (g[:, 1].min() + g[:, 1].max()) for g in groups]
        upper = [g for g, c in zip(groups, centres) if c < -0.45 * s.xh]
        lower = [g for g, c in zip(groups, centres) if c >= -0.45 * s.xh]
        assert upper and lower, seed
        sup_bottom = max(g[:, 1].max() for g in upper)
        sub_top = min(g[:, 1].min() for g in lower)
        assert sub_top - sup_bottom > 0.05 * s.xh, seed


@needs_style
@pytest.mark.parametrize("display", [False, True])
def test_fraction_bar_numerator_denominator(style, display):
    for seed in range(4):
        s = make_synth(style, seed=seed)
        lay = M.MathLayout(s)
        box = lay.lay(Frac(Row([Sym("a")]), Row([Sym("b"), Sym("c")])), M.Ctx(display=display))
        assert len(box.lines) == 1, "exactly one synthetic stroke: the fraction bar"
        bar, w = box.lines[0]
        bar_y = float(np.mean(bar[:, 1]))
        assert bar_y == pytest.approx(-M.AXIS * s.xh, abs=0.12 * s.xh)
        assert w == pytest.approx(s.st.pen_width * s.xh, rel=1e-6)
        assert np.ptp(bar[:, 0]) > 0.5 * s.xh
        above = below = 0
        for st in box.fills:
            p = fill_points([st])
            if p[:, 1].max() < bar_y - w / 2:
                above += 1
                cx = 0.5 * (p[:, 0].min() + p[:, 0].max())
                assert bar[:, 0].min() < cx < bar[:, 0].max()
            elif p[:, 1].min() > bar_y + w / 2:
                below += 1
            else:
                pytest.fail("ink crosses the fraction bar")
        assert above >= 1 and below >= 2
        # numerator centred over the bar
        num = np.vstack([fill_points([st]) for st in box.fills if fill_points([st])[:, 1].max() < bar_y])
        mid_num = 0.5 * (num[:, 0].min() + num[:, 0].max())
        mid_bar = 0.5 * (bar[:, 0].min() + bar[:, 0].max())
        assert abs(mid_num - mid_bar) < 0.4 * s.xh


@needs_style
def test_inline_fraction_smaller_than_display(style):
    s = make_synth(style, seed=1)
    inline = render_math(r"\frac{a}{b}", s, display=False)
    s = make_synth(style, seed=1)
    disp = render_math(r"\frac{a}{b}", s, display=True)
    assert disp.ascent + disp.descent > inline.ascent + inline.descent


@needs_style
def test_sqrt_has_radical_with_pen_width(style):
    s = make_synth(style, seed=4)
    lay = M.MathLayout(s)
    box = lay.lay(Sqrt(Row([Sym("2")])), M.Ctx())
    assert len(box.lines) == 1
    rad, w = box.lines[0]
    assert w == pytest.approx(s.st.pen_width * s.xh, rel=1e-6)
    body = fill_points(box.fills)
    # overbar above the body, radical starts left of the body
    assert rad[:, 1].min() < body[:, 1].min()
    assert rad[:, 0].min() < body[:, 0].min()
    assert rad[:, 0].max() > body[:, 0].max()
    idx = lay.lay(Sqrt(Row([Sym("x")]), index=Row([Sym("3")])), M.Ctx())
    assert idx.width > 0


@needs_style
def test_delimiters_grow_with_content(style):
    s = make_synth(style, seed=6)
    small = render_math(r"(x)", s)
    big = render_math(r"\left(\frac{1}{3}\right)", s)
    assert big.ascent + big.descent > 1.3 * (small.ascent + small.descent)
    auto = render_math(r"(\frac{1}{3})", s)  # matched pairs are sized like \left..\right
    assert auto.ascent + auto.descent > 1.3 * (small.ascent + small.descent)


@needs_style
def test_display_operators_have_limits_above_and_below(style):
    s = make_synth(style, seed=8)
    inline = render_math(r"\sum_{i=1}^{n} i", s, display=False)
    s = make_synth(style, seed=8)
    disp = render_math(r"\sum_{i=1}^{n} i", s, display=True)
    assert disp.ascent > inline.ascent and disp.descent > inline.descent
    assert disp.width < inline.width + 0.5 * s.xh  # limits stack instead of trailing
    lim = render_math(r"\lim_{n\to\infty} x_n", make_synth(style, seed=8), display=True)
    assert lim.descent > 0.6 * s.xh


@needs_style
def test_spacing_around_operators(style):
    s = make_synth(style, variation=0.0)
    lay = M.MathLayout(s)
    u = s.xh
    a, plus, eq, b = Sym("a"), Sym("+", "bin"), Sym("=", "rel"), Sym("b")
    top, script = M.Ctx(), M.Ctx().script()
    assert lay.gap("ord", "bin", a, plus, top) == pytest.approx(M.BIN_GAP * u)
    assert lay.gap("bin", "ord", plus, b, top) == pytest.approx(M.BIN_GAP * u)
    assert lay.gap("ord", "rel", a, eq, top) == pytest.approx(M.REL_GAP * u)
    assert 0.2 * u < M.BIN_GAP * u < M.REL_GAP * u < 0.6 * u
    # tighter inside scripts (and scaled with the script size)
    assert lay.gap("ord", "rel", a, eq, script) < 0.5 * lay.gap("ord", "rel", a, eq, top)
    # handwriting letter gaps between ordinary symbols
    g = lay.gap("ord", "ord", a, b, top)
    assert 0.0 < g < 0.5 * u

    # a leading minus is unary: no binary spacing is requested
    seen = []
    orig = lay.gap

    def spy(lc, rc, ln, rn, ctx):
        seen.append((lc, rc))
        return orig(lc, rc, ln, rn, ctx)

    lay.gap = spy
    lay.lay(parse("-b"), M.Ctx())
    assert seen == [("ord", "ord")]
    seen.clear()
    lay.lay(parse("a-b"), M.Ctx())
    assert seen == [("ord", "bin"), ("bin", "ord")]
    seen.clear()
    lay.lay(parse("x=-1"), M.Ctx())
    assert seen == [("ord", "rel"), ("rel", "ord"), ("ord", "ord")]


@needs_style
def test_text_and_operator_names_use_words(style):
    s = make_synth(style, seed=9)
    calls = []
    orig = s.word

    def spy(text, size=1.0):
        calls.append(text)
        return orig(text, size)

    s.word = spy
    render_math(r"\sin x + \text{if and only if} + \operatorname{lcm}(a, b)", s)
    assert "sin" in calls and "if" in calls and "only" in calls and "lcm" in calls


@needs_style
def test_size_scales_output(style):
    a = render_math(r"x^2 + \frac{1}{2}", make_synth(style, seed=3), size=1.0)
    b = render_math(r"x^2 + \frac{1}{2}", make_synth(style, seed=3), size=2.0)
    assert b.width == pytest.approx(2 * a.width, rel=0.25)
    assert b.ascent == pytest.approx(2 * a.ascent, rel=0.25)


@needs_style
def test_matrix_and_cases_layout(style):
    s = make_synth(style, seed=10)
    m = render_math(r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}", s, display=True)
    one_row = render_math(r"\begin{pmatrix} 1 & 2 \end{pmatrix}", s, display=True)
    assert m.ascent + m.descent > 1.5 * (one_row.ascent + one_row.descent) * 0.8
    c = render_math(r"\begin{cases} 1 & x > 0 \\ 0 & x \le 0 \end{cases}", s, display=True)
    assert c.width > 0 and c.ascent > s.xh and c.descent > 0.5 * s.xh


@needs_style
def test_full_pipeline_with_maths(style, tmp_path):
    from handwritten.docparse import parse_text
    from handwritten.layout import layout_document
    from handwritten.render import write_pdf

    src = ("For $n \\ge 1$ we have $x_{n+1} = x_n + \\frac{1}{2^n}$ and $\\sqrt{2} \\notin \\mathbb{Q}$.\n\n"
           "$$\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}$$\n\nDone.")
    s = make_synth(style, seed=1)
    pages = layout_document(parse_text(src), s)
    assert pages and pages[0].items
    out = tmp_path / "m.pdf"
    write_pdf(pages, out, s.st.ink)
    assert out.stat().st_size > 1000
    # lines must not overlap: consecutive baselines leave room for ascent + descent
    items = sorted(pages[0].items, key=lambda p: p.y)
    rows = {}
    for it in items:
        rows.setdefault(round(it.y / 3.0), []).append(it)
    ys = sorted(rows)
    for a, b in zip(ys, ys[1:]):
        ra, rb = rows[a], rows[b]
        bottom_a = max(it.y + it.box.descent for it in ra)
        top_b = min(it.y - it.box.ascent for it in rb)
        if min(it.y for it in rb) - max(it.y for it in ra) > 0.5 * s.xh:
            assert top_b > bottom_a - 0.35 * s.xh


def test_synthetic_strokes_become_pen_fills(style, monkeypatch):
    monkeypatch.setattr(M, "PEN_FILLS", True)
    box = render_math(r"\frac{a}{b} + \sqrt{2}", make_synth(style, seed=3))
    assert box.lines == [], "bars and radicals are drawn as filled pen strokes"
    assert box.fills and box.width > 0
    b = box.ink_bounds()
    assert b is not None and box.ascent >= -b[1] - 1e-6 and box.descent >= b[3] - 1e-6
