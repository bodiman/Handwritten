# handwritten

Learn your handwriting from a few pages you wrote on an iPad, then turn any
document (Markdown with LaTeX maths, plain text, a LaTeX subset, Word, PDF)
into a PDF that looks handwritten by you.

The output is built from your own pen strokes. Every letter on the page is one
you actually wrote: it is taken from your sample pages, given small random
variations, and laid out with the letter spacing, word spacing, line spacing,
slant, size and baseline drift measured from your writing. The result is a
vector PDF, so it stays sharp at any zoom and prints cleanly.

```
iPad PDFs ──► train ──► me.hwstyle ──► render answers.md ──► answers.pdf
(Apple Pencil)          (your glyphs +         (+ $maths$)      (in your hand)
                         layout statistics)
```

## Quick start

```bash
pip install -e .            # Python >= 3.10; add ".[dev]" for pytest

# 1. learn your handwriting from iPad exports (PDFs with Apple Pencil ink)
handwritten train samples/*.pdf -o me.hwstyle

# 2. write a document in it
handwritten render answers.md -s me.hwstyle -o answers.pdf

# write your answers below the printed problems of an existing PDF
handwritten render answers.md -s me.hwstyle -o hw3.pdf --onto problems.pdf

# see what was learned (JSON report) and a picture of every stored glyph
handwritten inspect me.hwstyle --atlas bank.png
```

`render` lists any characters that are not in your sample and had to be
synthesised. You can add them with a calibration sheet (see
[Calibration sheets](#calibration-sheets)).

### Training input

Training needs **vector ink**. Export pages written with an Apple Pencil from
an app that uses Apple PencilKit, such as Files/Markup or Notes, as PDF. In
those exports every pen stroke is a separate filled vector path, stored in
writing order. iOS also adds an invisible text layer with its handwriting
recognition result. `train` uses both. Ordinary homework or notes work well.
For example, two homework sets (16 pages, mostly prose with some maths) gave
about 1,500 usable letters for 43 characters, and training took about 20
seconds on a laptop CPU.

### `render` options

| option | meaning |
| --- | --- |
| `-s, --style FILE` | trained `.hwstyle` file (required) |
| `-o, --output FILE` | output PDF (default `handwritten.pdf`) |
| `--onto PDF` | write onto this PDF, below its existing content (see below) |
| `--paper letter\|a4` | page size (default `letter`) |
| `--paper-style plain\|ruled\|grid\|dots` | page background (default `plain`); on ruled paper baselines sit on the rules |
| `--size PT` | x-height in points (default: your natural size, measured from the samples) |
| `--line-spacing N` | line spacing in x-heights (default: yours) |
| `--margin-left PT` | left margin in points (default: your usual margin) |
| `--variation X` | multiplier on all learned randomness; `0` = minimal, `1` = as measured (default) |
| `--neatness X` | `0..1`: straighter lines and steadier spacing (default `0`) |
| `--seed N` | random seed; the same seed and input give the same output |
| `--ink COLOUR` | `black`, `blue`, `red` or `#rrggbb` (default: the colour you wrote in) |
| `--reflow` / `--keep-lines` | join source lines into paragraphs / keep source line breaks |
| `--png` `--dpi N` | also write PNG previews (`answers-1.png`, ...) at `N` dpi (default 150) |

Use `-` as the input file to read Markdown from standard input.

**Writing onto an existing PDF (`--onto`).** Writing starts below the
existing content of the first page. A page break in your document
(`\newpage`) moves to the next page of the PDF, so one answer goes under each
problem. Text that overflows a page continues on an inserted blank page, so
the next answer still lands on its own page. `<!-- page N -->` jumps to page
`N` of the base PDF. The paper background is not drawn on base pages.

### Other commands

* `handwritten train ... --atlas segments.png` writes a picture of every
  segmented glyph, with rejected ones outlined in red. `--min-samples N`
  (default 2) is how many free-text samples a character needs before it is
  trusted. `-v` prints progress.
* `handwritten inspect me.hwstyle [--atlas bank.png]` prints coverage, size,
  slant, pen width, spacing and the common characters still missing. The atlas
  labels each glyph with an id.
* `handwritten prune me.hwstyle --remove 12 57 301 --drop-chars "%" [-o out.hwstyle]`
  removes bad samples (ids from the atlas) or whole characters. Dropped
  characters then use the fallback glyphs.
* `handwritten template` makes a calibration sheet (see below).

## Input formats

| file | what is understood |
| --- | --- |
| `.md`, `.markdown`, no extension | `#` headings; `-`/`*`/`+` bullets and numbered items (`1.`, `2)`, `(a)`, `(iv)`), with nesting; `---` rules; inline maths `$...$` or `\(...\)`; display maths `$$...$$` or `\[...\]` (may span lines); `\$` for a literal dollar. Emphasis, code and link markup is removed. Single line breaks are joined into paragraphs, as in Markdown; use `--keep-lines` to keep them. |
| `.txt` | Lines are kept as written (`--reflow` joins them). Markdown syntax is not interpreted, but `$...$` maths is. |
| `.tex` | Document body only: `\section`/`\subsection`/`\subsubsection` (also starred), `itemize`/`enumerate` with `\item` and `\item[(a)]`, `equation`/`align`/`gather`/`multline` (also starred) as display maths, inline `$...$`, `\textbf`/`\textit`/`\emph`/`\underline`/`\text` unwrapped, `\\` line breaks, `%` comments. Preamble and layout commands are dropped. |
| `.docx` | Paragraphs, `Heading N` styles, list styles and page breaks. Type maths as `$...$` in the text; Word equation-editor objects are not read. |
| `.pdf` | The text layer, line by line, one page break per page. A scanned PDF without a text layer yields nothing. |

In Markdown, text and LaTeX input, a line containing only `\newpage`,
`\pagebreak`, `\clearpage` or `<!-- pagebreak -->` starts a new page. Word
documents use their own page breaks.

## Maths

Maths is typeset from a practical subset of LaTeX, in your handwriting. It
covers super- and subscripts (nested, primes), fractions and binomials,
square and n-th roots, sums, products and integrals with limits, `\lim` and
other operator names, Greek letters, relations, arrows and binary operators
with TeX-like spacing, `\left`/`\right` delimiters and automatically sized
brackets, `\text`, `\mathbb`, accents and matrix-like environments such as
`pmatrix` and `cases`. The supported set is still growing; see the docstring
of `handwritten/mathlayout.py` for the current list.

Letters, digits and symbols in formulas are drawn with your glyphs where you
have them. Structural marks a person draws by hand, such as fraction bars,
root signs, large brackets and accents, are drawn as pen strokes with your
pen width and slant. A formula that cannot be parsed is written out as plain
handwriting instead of failing the document.

## Calibration sheets

Free text rarely contains every character: capitals, digits, punctuation,
Greek letters and maths symbols are often missing or too rare. iOS
recognition also mangles maths, so letters inside formulas are not trusted for
training. To fill these gaps:

```bash
handwritten template -o sheet.pdf                 # all characters
handwritten template -o sheet.pdf --chars "XYZ?&∑∫" --repeats 6
```

1. Open `sheet.pdf` on the iPad (Files/Markup, Notes, GoodNotes, ...) or print
   it.
2. Write each character into its boxes, sitting on the solid baseline. The
   dashed line is the x-height, the dotted lines are the cap height and the
   descender.
3. Export the result as PDF and train with it:

```bash
handwritten train samples/*.pdf --calibration sheet-filled.pdf -o me.hwstyle
handwritten train --calibration sheet-filled.pdf -o me.hwstyle   # sheets only
```

The sheet carries its own layout as an invisible marker and in its metadata,
and it can also be recognised from its printed boxes. This lets it survive
apps that rewrite PDFs on export. Ink may be PencilKit-style filled outlines,
stroked paths or PDF ink annotations. When you train sheets together with free
text, sheet glyphs are rescaled to your free-text size, because people tend
to write bigger in boxes.

## How it works

1. **Ink extraction** (`ink.py`). Each PencilKit stroke is one filled vector
   path: the outline of the stroke as closed cubic Bezier sub-paths, in writing
   order. Ink is told apart from printed content by being filled, curve-heavy
   and in the most common ink colour. The invisible text layer (PDF text
   render mode 3) gives the recognised words and their bounding boxes.
2. **Stroke-to-character alignment** (`segment.py`). Strokes are assigned to
   recognised words, and words are grouped into lines. Each line gets a robust
   baseline and x-height. Delayed strokes (i-dots, t-bars written after the
   word) are moved back next to their letter. Dynamic programming then splits
   each word's strokes into one contiguous run of 1-6 strokes per character.
   The cost combines vertical extent relative to baseline and x-height,
   stroke-count priors, horizontal coherence and distance to a raster template
   of the character. An EM loop re-estimates per-character statistics,
   templates, baselines and x-heights and aligns again.
   Recognition labels are weighted by how trustworthy they look: prose words
   are trusted, maths-like tokens and mostly-maths lines are not.
   Implausible words and glyphs that are outliers for their class are
   rejected.
3. **Style model** (`style.py`). The accepted glyphs form a glyph bank,
   normalised to x-height 1, with your original outlines. Training removes
   stray marks, drops characters seen too rarely, pulls sizes toward each
   class's median, prunes shape outliers and measures each glyph's pen width.
   Alongside the bank it stores statistics: x-height, ascender, cap height and
   descender, slant, pen width, ink colour, letter gaps per character pair
   (with back-off to per-character and global values), word and punctuation
   gaps, line spacing and slope, per-word size and baseline jitter, and your
   natural margins. Everything is saved as a `.hwstyle` file (a zip with JSON
   and compressed arrays).
4. **Synthesis and layout** (`synth.py`, `layout.py`, `mathlayout.py`). For
   each character, one of your real samples is picked, avoiding recent repeats
   and preferring typical pen weight. It gets small learned variations in
   size, rotation, shear and baseline. Lines are filled greedily, and each
   line gets a slight slope and start offset.
5. **Fallback chain** (`synth.py`, `fallback.py`). For a character you never
   wrote, the synthesiser first tries an analogue from your own bank: `0` for
   `O` and vice versa, `c` from a shrunk `C`, `l` from `1`, and so on. Next it
   uses a Hershey single-stroke glyph restyled to your x-height, cap height,
   width, slant and pen width, with a little wobble. Maths symbols missing
   from the Hershey sets are drawn procedurally.
6. **Rendering** (`render.py`). Your strokes are written to the PDF as filled
   outlines, exactly as recorded. Synthetic marks are written as round-pen
   strokes.

### Why not a neural network?

A neural handwriting generator needs far more data than two homework sets.
Such models are trained on hours of pen trajectories from many writers. Your
samples give roughly 1,500 usable letters across about 40 characters, and many
characters appear only a handful of times. A model trained on that little data
would overfit, or produce blurred, averaged strokes. Reusing real strokes
avoids both problems: the letters are authentic because they are yours.
Training takes about 20 seconds on a laptop CPU, needs no GPU, and the result
is easy to inspect and correct (`inspect --atlas`, `prune`).

More data would change this. Many more pages and calibration sheets would
first make the bank richer, with more variants per letter and fewer
fallbacks. With enough samples, a generative model of glyph shape could create
new variants of each letter rather than replaying stored ones, and a model of
pen trajectories could learn joins between letters.

## Limitations

* Training needs **vector ink PDFs from iPad apps** (Apple PencilKit-style
  filled stroke outlines). Free-text training also needs the iOS
  recognised-text layer; without it, use calibration sheets.
* **Scanned or photographed handwriting is not supported yet.**
* **Cursive or joined-up writing is not supported.** The alignment assumes
  each character is made of its own strokes, so print-style handwriting works
  best.
* Maths covers a LaTeX subset, and symbols you have not written come from the
  restyled fallback until you add them with a calibration sheet.
* Layout is simple: no hyphenation, tables or images. Word equation objects
  and images in documents are ignored.

## Privacy and responsible use

Your sample PDFs and `.hwstyle` files contain your actual handwriting, stored
as exact pen outlines. Treat them like a signature. `.gitignore` keeps
`samples/`, `*.hwstyle` and `out/` out of git by default; keep it that way and
do not share style files. Use the tool for your own handwriting, never to
imitate someone else's handwriting or signature. Check whether handwritten
work is expected to be written by hand before submitting generated pages.

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q
```

Most tests run on a synthetic toy style and need no personal data. The tests
that use a real trained style, including the end-to-end pipeline test (which
builds a synthetic iPad-like PDF from your style and retrains on it), are
skipped unless a style is found. Point them at one with
`HANDWRITTEN_TEST_STYLE=/path/to/me.hwstyle`.

| module | role |
| --- | --- |
| `ink.py` | stroke and recognised-text extraction from PDFs |
| `segment.py` | stroke-to-character alignment (DP + EM), outlier rejection |
| `style.py` | glyph bank and statistics, `.hwstyle` I/O |
| `synth.py` | glyph choice, variation, spacing, analogues |
| `fallback.py` | restyled Hershey and procedural glyphs |
| `mathlayout.py` | LaTeX maths typesetting |
| `docparse.py` | input formats into blocks |
| `layout.py` | pages, lines, `--onto` |
| `render.py` | vector PDF and PNG output |
| `template.py` | calibration sheets |
| `viz.py` | glyph atlases |
| `cli.py` | `handwritten` command |
