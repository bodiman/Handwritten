"""Command line interface.

    handwritten train  samples/*.pdf -o me.hwstyle      # learn a handwriting style
    handwritten render answer.md -s me.hwstyle -o out.pdf
    handwritten inspect me.hwstyle                       # coverage report + atlas
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional


def _p(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def cmd_train(a: argparse.Namespace) -> int:
    from .ink import detect_ink_colors, extract_many
    from .segment import segment
    from .style import train_style

    t0 = time.time()
    files = [str(p) for p in a.inputs]
    cal_files = [str(p) for p in (a.calibration or [])]
    if not files and not cal_files:
        _p("error: give handwritten PDFs and/or --calibration sheet(s)")
        return 2
    style = None
    seg = None
    if files:
        _p(f"reading {len(files)} PDF(s)...")
        pages = extract_many(files)
        n_strokes = sum(len(p.strokes) for p in pages)
        n_words = sum(len(p.words) for p in pages)
        _p(f"  {len(pages)} pages, {n_strokes} ink strokes, {n_words} recognised words")
        if n_strokes == 0:
            _p("error: no vector ink strokes found. The PDFs must contain the handwriting as vector ink "
               "(e.g. exported from an iPad app using Apple Pencil); scans/photos are not supported.")
            return 2
        if n_words == 0:
            msg = ("no recognised-text layer found (iOS adds one when exporting from Files/Markup or Notes)")
            if not cal_files:
                _p("error: " + msg + "; fill in a calibration sheet instead: handwritten template")
                return 2
            _p("  warning: " + msg + "; using calibration sheets only")
        else:
            _p("segmenting characters...")
            seg = segment(pages, verbose=a.verbose)
            for line in seg.log[-2:]:
                _p("  " + line)
            ink = detect_ink_colors(files)[0]
            style = train_style(seg, ink=ink, meta={"sources": [Path(f).name for f in files]},
                                min_samples=a.min_samples)
    if cal_files:
        from .template import merge_calibration, style_from_sheets

        cal = style_from_sheets(cal_files, base=style)
        n_cal = sum(len(v) for v in cal.bank.values())
        if n_cal == 0:
            _p("  warning: no handwriting found in the calibration sheet(s) - were they filled in and "
               "exported as PDF?")
        else:
            _p(f"  {n_cal} glyphs from calibration sheet(s) (cells filled {cal.meta.get('sheet_cells_filled')}, "
               f"rejected {cal.meta.get('sheet_rejected')})")
        # trusted sheet samples replace scarce (often mis-recognised) free-text samples
        style = cal if style is None else merge_calibration(style, cal)
    assert style is not None
    style.save(a.output)
    cov = style.coverage()
    _p(f"saved {a.output}: {sum(cov.values())} glyphs for {len(cov)} characters "
       f"({time.time() - t0:.1f}s)")
    _p("  characters: " + "".join(sorted(cov)))
    if a.atlas and seg is not None:
        from .viz import glyph_atlas

        glyph_atlas(seg.glyphs, path=a.atlas)
        _p(f"  glyph atlas (red = rejected) written to {a.atlas}")
    return 0


def cmd_render(a: argparse.Namespace) -> int:
    from .docparse import parse_file, parse_text
    from .layout import LayoutOptions, layout_document
    from .render import pdf_to_pngs, write_pdf
    from .style import Style
    from .synth import Synthesizer

    style = Style.load(a.style)
    if a.input == "-":
        blocks = parse_text(sys.stdin.read(), markdown=True, reflow=not a.keep_lines)
    else:
        reflow = None
        if a.keep_lines:
            reflow = False
        if a.reflow:
            reflow = True
        blocks = parse_file(a.input, reflow=reflow)
    synth = Synthesizer(style, seed=a.seed, x_height=a.size, variation=a.variation, neatness=a.neatness)
    opts = LayoutOptions(paper=a.paper, background=a.paper_style, margin_left=a.margin_left,
                         line_spacing=a.line_spacing, base_pdf=a.onto)
    pages = layout_document(blocks, synth, opts)
    write_pdf(pages, a.output, style.stats.ink if a.ink is None else _parse_color(a.ink), base_pdf=a.onto)
    _p(f"wrote {a.output} ({len(pages)} page{'s' if len(pages) != 1 else ''})")
    fb = sorted(c for c, k in synth.used_kind.items() if k != "real")
    if fb:
        _p("  not in your handwriting sample (synthesised): " + "".join(fb) +
           "  -> add them with a calibration sheet: handwritten template")
    if a.png:
        for p in pdf_to_pngs(a.output, Path(a.output).with_suffix(""), dpi=a.dpi):
            _p(f"  preview {p}")
    return 0


INK_NAMES = {"black": (0.1, 0.1, 0.1), "blue": (0.08, 0.2, 0.6), "red": (0.75, 0.1, 0.1),
             "green": (0.1, 0.45, 0.2), "pencil": (0.35, 0.35, 0.37)}


def _parse_color(s: str):
    t = s.strip().lower()
    if t in INK_NAMES:
        return INK_NAMES[t]
    h = t.lstrip("#")
    if len(h) == 6 and all(c in "0123456789abcdef" for c in h):
        return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    raise SystemExit(f"error: unknown ink colour {s!r}; use one of {', '.join(INK_NAMES)} or #rrggbb")


def cmd_inspect(a: argparse.Namespace) -> int:
    from dataclasses import asdict

    from .style import Style

    style = Style.load(a.style)
    cov = style.coverage()
    st = asdict(style.stats)
    report = {
        "glyphs": sum(cov.values()),
        "characters": len(cov),
        "coverage": cov,
        "x_height_pt": round(st["xh_pt"], 2),
        "slant": round(st["slant"], 3),
        "pen_width_xh": round(st["pen_width"], 3),
        "word_gap_xh": round(st["word_gap"]["mean"], 2),
        "line_spacing_xh": round(st["line_spacing"]["mean"], 2),
        "meta": style.meta,
    }
    import string

    want = string.ascii_letters + string.digits + ".,;:!?()-+=/'\""
    report["missing"] = "".join(c for c in want if c not in cov)
    print(json.dumps(report, indent=2, default=str))
    if a.atlas:
        from .viz import bank_atlas

        bank_atlas(style, a.atlas)
        _p(f"atlas written to {a.atlas}")
    return 0


def cmd_prune(a: argparse.Namespace) -> int:
    from .style import Style

    style = Style.load(a.style)
    ids = {int(x) for part in a.remove for x in part.replace(",", " ").split()}
    chars = set(a.drop_chars or "")
    n0 = sum(len(v) for v in style.bank.values())
    bank = {}
    for c, gs in style.bank.items():
        if c in chars:
            continue
        keep = [g for g in gs if g.gid not in ids]
        if keep:
            bank[c] = keep
    style.bank = bank
    style.stats.char_counts = {c: len(v) for c, v in bank.items()}
    out = a.output or a.style
    style.save(out)
    n1 = sum(len(v) for v in bank.values())
    _p(f"removed {n0 - n1} glyph(s); {n1} remain -> {out}")
    return 0


def cmd_template(a: argparse.Namespace) -> int:
    from .template import chars_needed, make_template

    chars = a.chars
    xh = a.size or 10.0
    if a.for_style:
        from .style import Style

        st = Style.load(a.for_style)
        chars = chars_needed(st, chars)
        xh = a.size or round(st.stats.xh_pt, 1)
        if not chars:
            _p("nothing missing: every character already has enough samples")
            return 0
        _p(f"characters to write: {''.join(chars)}")
    make_template(a.output, chars=chars, repeats=a.repeats, paper=a.paper, x_height=xh)
    _p(f"wrote {a.output}: print it or open it on your iPad, write each character in its boxes, "
       f"export as PDF and pass it to 'handwritten train ... --calibration sheet.pdf'")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="handwritten", description="Learn your handwriting and write documents with it.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="learn a style from handwritten PDFs (iPad / PencilKit exports)")
    t.add_argument("inputs", nargs="*", type=Path, help="handwritten PDFs (iPad exports)")
    t.add_argument("-o", "--output", default="style.hwstyle")
    t.add_argument("--calibration", nargs="*", type=Path, help="filled-in calibration sheet PDF(s)")
    t.add_argument("--atlas", help="write a PNG atlas of all segmented glyphs (for review)")
    t.add_argument("--min-samples", type=int, default=2,
                   help="minimum samples for a character learned from free text (default 2)")
    t.add_argument("-v", "--verbose", action="store_true")
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("render", help="write a document in a learned handwriting")
    r.add_argument("input", help=".md/.txt/.tex/.docx/.pdf file, or - for stdin")
    r.add_argument("-s", "--style", required=True)
    r.add_argument("-o", "--output", default="handwritten.pdf")
    r.add_argument("--onto", help="write onto this PDF (below existing content, one section per page)")
    r.add_argument("--paper", default="letter", choices=["letter", "a4"])
    r.add_argument("--paper-style", default="plain", choices=["plain", "ruled", "grid", "dots"])
    r.add_argument("--size", type=float, default=None, help="x-height in points (default: your natural size)")
    r.add_argument("--line-spacing", type=float, default=None, help="line spacing in x-heights")
    r.add_argument("--margin-left", type=float, default=None, help="left margin in points")
    r.add_argument("--variation", type=float, default=1.0, help="randomness multiplier (0 = minimal)")
    r.add_argument("--neatness", type=float, default=0.0, help="0..1: straighter lines, steadier spacing")
    r.add_argument("--seed", type=int, default=None, help="random seed (same seed = same output)")
    r.add_argument("--ink", default=None, help="ink colour: black, blue, red, green, pencil or #rrggbb")
    r.add_argument("--reflow", action="store_true", help="join lines into paragraphs")
    r.add_argument("--keep-lines", action="store_true", help="keep source line breaks")
    r.add_argument("--png", action="store_true", help="also write PNG previews")
    r.add_argument("--dpi", type=int, default=150)
    r.set_defaults(func=cmd_render)

    i = sub.add_parser("inspect", help="show what a style file contains")
    i.add_argument("style")
    i.add_argument("--atlas", help="write a PNG atlas of the glyph bank")
    i.set_defaults(func=cmd_inspect)

    pr = sub.add_parser("prune", help="remove bad glyph samples (ids from 'inspect --atlas') or whole characters")
    pr.add_argument("style")
    pr.add_argument("--remove", nargs="*", default=[], help="glyph ids to remove, e.g. 12 57 301")
    pr.add_argument("--drop-chars", default=None, help="characters to remove entirely (use the fallback instead)")
    pr.add_argument("-o", "--output", default=None, help="output file (default: overwrite)")
    pr.set_defaults(func=cmd_prune)

    tp = sub.add_parser("template", help="make a calibration sheet to fill in by hand")
    tp.add_argument("-o", "--output", default="calibration.pdf")
    tp.add_argument("--chars", default=None, help="characters to include (default: full set)")
    tp.add_argument("--repeats", type=int, default=4)
    tp.add_argument("--paper", default="letter", choices=["letter", "a4"])
    tp.add_argument("--for-style", default=None, help="only the characters this .hwstyle is missing or short of")
    tp.add_argument("--size", type=float, default=None,
                    help="guide x-height in pt (default 10, or the style's natural x-height)")
    tp.set_defaults(func=cmd_template)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.func(a) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
