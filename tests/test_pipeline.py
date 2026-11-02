"""End-to-end pipeline: learn a style from an iPad-like PDF, then write with it.

A *synthetic iPad export* is built once per session: a known paragraph is
written in the user's trained hand with the regular renderer (filled outline
strokes, one path per pen stroke, in writing order -- like Apple PencilKit),
and an invisible text layer (render mode 3) holds every word at its ink
bounding box, like the handwriting recognition layer iOS adds on export.
Training on it must recover the words, the characters and the size it was
written at.  The CLI is smoke-tested on the same file.

The synthetic PDF needs the real trained style; those tests are skipped
without it (see conftest.py).  CLI option/error tests use the toy style.
"""
from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import pymupdf
import pytest

from conftest import make_toy_style
from handwritten import cli
from handwritten.docparse import parse_text
from handwritten.ink import extract_many
from handwritten.layout import LayoutOptions, layout_document
from handwritten.render import A4, RenderPage, write_pdf
from handwritten.segment import segment
from handwritten.style import Style, train_style
from handwritten.synth import Synthesizer

# Only characters the writer has real samples of (lowercase, comma, full stop),
# so every mark is a filled PencilKit-like outline.
PARAGRAPH = (
    "we can see that the sum of the first terms is equal to the number of ways to write each one "
    "as a product of smaller parts, and the answer does not depend on the order in which the parts "
    "are chosen. since every step uses the same rule, we may assume that the claim holds for all "
    "smaller values and then show that it also holds for the next one. this gives the result for "
    "every number, so the proof is complete. note that the bound is sharp, and there are examples "
    "where the inequality becomes an equality."
)
XH = 12.0  # x-height to write at; deliberately not the writer's natural ~10 pt
SEED = 3
OCR_PAD = 1.0  # recognised-word boxes extend this far beyond the ink (pt)


@dataclass
class SyntheticIpad:
    path: Path
    words: List[str]  # recognised words, in writing order
    boxes: List[Tuple[float, float, float, float]]  # their ink bounding boxes (page points)
    n_strokes: int  # filled pen strokes drawn
    n_pages: int
    xh: float


def _ocr_layer(pages: List[RenderPage], path: Path) -> Tuple[List[str], list]:
    """Blank pages holding each placed word as invisible text fitted to its
    ink box -- the shape of iOS's recognised-text layer."""
    font = pymupdf.Font("helv")
    asc, desc = font.ascender, font.descender
    words, boxes = [], []
    doc = pymupdf.open()
    for i, pg in enumerate(pages):
        page = doc.new_page(width=pg.width, height=pg.height)
        for it in pg.items:
            text = it.box.text.strip()
            b = it.box.ink_bounds()
            if not text or b is None:
                continue
            x0, y0, x1, y1 = b[0] + it.x, b[1] + it.y, b[2] + it.x, b[3] + it.y
            fs = (y1 - y0 + 2 * OCR_PAD) / (asc - desc)  # span height == padded ink height
            org = pymupdf.Point(x0 - OCR_PAD, y0 - OCR_PAD + asc * fs)
            sx = (x1 - x0 + 2 * OCR_PAD) / font.text_length(text, fontsize=fs)  # width == padded ink width
            page.insert_text(org, text, fontsize=fs, fontname="helv", render_mode=3,
                             morph=(org, pymupdf.Matrix(sx, 1.0)))
            words.append(text)
            boxes.append((x0, y0, x1, y1))
        pg.base_page = i
    doc.save(str(path))
    doc.close()
    return words, boxes


def build_synthetic_ipad_pdf(style: Style, path: Path, text: str = PARAGRAPH, xh: float = XH,
                             seed: int = SEED) -> SyntheticIpad:
    synth = Synthesizer(style, seed=seed, x_height=xh, variation=0.8)
    pages = layout_document(parse_text(text, reflow=True), synth, LayoutOptions(margin_left=60, margin_right=50))
    synthetic = sorted(c for c, k in synth.used_kind.items() if k != "real")
    assert not synthetic, f"paragraph uses characters the writer never wrote: {synthetic}"
    layer = path.with_name(path.stem + "-text.pdf")
    words, boxes = _ocr_layer(pages, layer)
    # ink drawn onto the text-layer pages: filled outlines, one path per stroke
    write_pdf(pages, path, style.stats.ink, base_pdf=layer)
    n_strokes = sum(len(it.box.fills) for pg in pages for it in pg.items)
    return SyntheticIpad(path, words, boxes, n_strokes, len(pages), xh)


@pytest.fixture(scope="session")
def ipad(tmp_path_factory, real_style) -> SyntheticIpad:
    return build_synthetic_ipad_pdf(real_style, tmp_path_factory.mktemp("ipad") / "homework.pdf")


@pytest.fixture(scope="session")
def learned(ipad):
    pages = extract_many([ipad.path])
    seg = segment(pages)
    style = train_style(seg, meta={"sources": [ipad.path.name]})
    return SimpleNamespace(pages=pages, seg=seg, style=style)


# =========================================================================
# the synthetic export itself
# =========================================================================
def test_synthetic_pdf_looks_like_a_pencilkit_export(ipad):
    assert len(ipad.words) == len(PARAGRAPH.split()) and ipad.words == PARAGRAPH.split()
    with pymupdf.open(str(ipad.path)) as doc:
        assert len(doc) == ipad.n_pages == 1
        page = doc[0]
        ds = page.get_drawings()
        assert len(ds) == ipad.n_strokes
        assert all(d["type"] == "f" and {it[0] for it in d["items"]} <= {"c", "l"} for d in ds)
        # outlines are Bezier curves (tiny segments may collapse to lines after rounding)
        assert all(sum(it[0] == "c" for it in d["items"]) >= 4 for d in ds)
        trace = page.get_texttrace()
        assert trace and all(t["type"] == 3 for t in trace)  # invisible text only
        assert page.get_text().split() == ipad.words


# =========================================================================
# learning
# =========================================================================
def test_extracts_strokes_and_recognised_words(ipad, learned):
    (pg,) = learned.pages
    assert len(pg.strokes) == ipad.n_strokes
    assert [s.seq for s in pg.strokes] == list(range(ipad.n_strokes))  # writing order kept
    assert [w.text for w in pg.words] == ipad.words
    for w, (x0, y0, x1, y1) in zip(pg.words, ipad.boxes):
        bx0, by0, bx1, by1 = w.bbox
        tol = 0.5
        assert bx0 <= x0 + tol and bx1 >= x1 - tol and by0 <= y0 + tol and by1 >= y1 - tol
        assert bx1 - bx0 <= x1 - x0 + 2 * OCR_PAD + tol
    assert pg.printed == []  # nothing visible besides ink


def test_most_words_align(learned):
    words = learned.seg.words
    assert len(words) == len(PARAGRAPH.split())
    ok = sum(w.ok for w in words) / len(words)
    assert ok >= 0.7, f"only {ok:.0%} of words aligned"
    glyphs = learned.seg.glyphs
    assert sum(g.ok for g in glyphs) >= 0.6 * len(glyphs)


def test_learned_bank_has_the_expected_characters(learned):
    bank = learned.style.bank
    assert set("aehilnorst") <= set(bank)
    assert set(bank) <= set(PARAGRAPH) - {" "}, "learned a character that was never written"
    counts = learned.style.coverage()
    assert counts["e"] >= 20 and counts["t"] >= 10


def test_learned_glyphs_have_the_right_shapes(learned, real_style):
    """Labels are right if each class has the vertical extent of its kind."""
    bank = learned.style.bank

    def med(chars, attr):
        return float(np.median([getattr(g, attr) for c in chars for g in bank.get(c, [])]))

    assert med("acemnorsuw", "top") == pytest.approx(1.0, abs=0.2)  # x-height letters
    assert med("acemnorsuw", "bottom") == pytest.approx(0.0, abs=0.15)  # sit on the baseline
    assert med("bdhl", "top") == pytest.approx(real_style.stats.ascender, rel=0.15)
    assert med("py", "bottom") < -0.4  # descenders


def test_learned_size_and_pen(learned, real_style):
    st = learned.style.stats
    assert st.xh_pt == pytest.approx(XH, rel=0.2)
    # outlines were scaled, so pen width in x-heights and slant carry over
    assert st.pen_width == pytest.approx(real_style.stats.pen_width, rel=0.2)
    assert st.slant == pytest.approx(real_style.stats.slant, abs=0.1)
    assert st.ascender == pytest.approx(real_style.stats.ascender, rel=0.15)
    assert 1.5 < st.line_spacing["mean"] < 6.0 and 0.8 < st.word_gap["mean"] < 3.0


def test_learned_style_saves_and_writes(learned, tmp_path):
    p = tmp_path / "learned.hwstyle"
    learned.style.save(p)
    style = Style.load(p)
    synth = Synthesizer(style, seed=0)
    box = synth.word("the stone")
    assert box.fills and not box.lines
    assert {synth.used_kind[c] for c in "thestone"} == {"real"}


# =========================================================================
# command line
# =========================================================================
ANSWERS_MD = """# Problem one

we see that the answer is $n + 1$ and so the claim holds.

- the first case is easy
- the other one follows
"""


def test_cli_train_render_inspect(ipad, tmp_path, capsys):
    style = tmp_path / "me.hwstyle"
    assert cli.main(["train", str(ipad.path), "-o", str(style)]) == 0
    err = capsys.readouterr().err
    assert f"{ipad.n_strokes} ink strokes" in err and f"{len(ipad.words)} recognised words" in err
    assert style.exists()

    md = tmp_path / "answers.md"
    md.write_text(ANSWERS_MD)
    out = tmp_path / "answers.pdf"
    assert cli.main(["render", str(md), "-s", str(style), "-o", str(out), "--seed", "1", "--png",
                     "--dpi", "40"]) == 0
    err = capsys.readouterr().err
    assert "wrote" in err
    with pymupdf.open(str(out)) as doc:
        assert len(doc) == 1
        assert sum(d["type"] == "f" for d in doc[0].get_drawings()) > 50
    assert (tmp_path / "answers-1.png").exists()

    assert cli.main(["inspect", str(style)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["glyphs"] > 100 and report["coverage"]["e"] >= 20
    assert report["x_height_pt"] == pytest.approx(XH, rel=0.2)
    assert report["meta"]["sources"] == [ipad.path.name]
    assert "z" in report["missing"]


def test_cli_render_options_stdin_and_onto(toy_style_path, tmp_path, monkeypatch, capsys):
    out = tmp_path / "opts.pdf"
    monkeypatch.setattr(sys, "stdin", io.StringIO("hello from stdin\nsecond line\n"))
    assert cli.main(["render", "-", "-s", str(toy_style_path), "-o", str(out), "--paper", "a4",
                     "--paper-style", "ruled", "--size", "14", "--ink", "blue", "--neatness", "0.5",
                     "--variation", "0.5", "--seed", "2", "--keep-lines"]) == 0
    with pymupdf.open(str(out)) as doc:
        assert (doc[0].rect.width, doc[0].rect.height) == pytest.approx(A4)
        fills = [d for d in doc[0].get_drawings() if d["type"] == "f"]
        assert fills and all(d["fill"] == pytest.approx((0.08, 0.2, 0.6), abs=3e-3) for d in fills)
        assert any(d["type"] == "s" for d in doc[0].get_drawings())  # ruled lines

    base = tmp_path / "problems.pdf"
    doc = pymupdf.open()
    for i in range(2):
        doc.new_page().insert_text((72, 90), f"Problem {i + 1}.", fontsize=14)
    doc.save(str(base))
    src = tmp_path / "answers.txt"
    src.write_text("answer one\n\\newpage\nanswer two\n")
    out2 = tmp_path / "onto.pdf"
    assert cli.main(["render", str(src), "-s", str(toy_style_path), "-o", str(out2), "--onto", str(base),
                     "--ink", "#ff0000"]) == 0
    with pymupdf.open(str(out2)) as res:
        assert len(res) == 2
        for i, page in enumerate(res):
            assert f"Problem {i + 1}." in page.get_text()
            fills = [d for d in page.get_drawings() if d["type"] == "f"]
            assert fills and min(d["rect"].y0 for d in fills) > 95
            assert fills[0]["fill"] == pytest.approx((1.0, 0.0, 0.0), abs=3e-3)


def test_cli_train_rejects_unusable_input(tmp_path, capsys):
    assert cli.main(["train", "-o", str(tmp_path / "x.hwstyle")]) == 2

    typed = tmp_path / "typed.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), "Only typed text here.", fontsize=12)
    doc.save(str(typed))
    assert cli.main(["train", str(typed), "-o", str(tmp_path / "x.hwstyle")]) == 2
    assert "no vector ink strokes" in capsys.readouterr().err

    # vector ink but no recognised-text layer
    style = make_toy_style()
    pages = layout_document(parse_text("some ink without text"), Synthesizer(style, seed=0), LayoutOptions())
    inky = tmp_path / "ink_only.pdf"
    write_pdf(pages, inky, style.stats.ink)
    assert cli.main(["train", str(inky), "-o", str(tmp_path / "x.hwstyle")]) == 2
    assert "recognised-text layer" in capsys.readouterr().err
    assert not (tmp_path / "x.hwstyle").exists()


def test_cli_prune(toy_style_path, tmp_path):
    before = Style.load(toy_style_path)
    gone = before.bank["a"][0].gid
    out = tmp_path / "pruned.hwstyle"
    assert cli.main(["prune", str(toy_style_path), "--remove", str(gone), "--drop-chars", "xy",
                     "-o", str(out)]) == 0
    after = Style.load(out)
    assert "x" not in after.bank and "y" not in after.bank
    assert gone not in [g.gid for g in after.bank["a"]] and len(after.bank["a"]) == len(before.bank["a"]) - 1
    assert after.stats.char_counts["a"] == len(after.bank["a"])


def test_console_entry_points():
    r = subprocess.run([sys.executable, "-m", "handwritten.cli", "--help"], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0
    for cmd in ("train", "render", "inspect", "template"):
        assert cmd in r.stdout
    exe = shutil.which("handwritten") or str(Path(sys.executable).with_name("handwritten"))
    if Path(exe).exists():
        r = subprocess.run([exe, "render", "--help"], capture_output=True, text=True, timeout=60)
        assert r.returncode == 0 and "--onto" in r.stdout
