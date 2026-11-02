"""Parse input documents into a simple block model.

Supported inputs: Markdown / plain text (with inline ``$...$`` / ``\\(...\\)``
and display ``$$...$$`` / ``\\[...\\]`` LaTeX maths), PDF (text layer), DOCX,
and a pragmatic subset of LaTeX (``.tex``).

Line breaks in the source are preserved by default (handwritten homework is
usually laid out line by line); pass ``reflow=True`` to join the lines of a
paragraph and let the layout wrap them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Span:
    text: str
    math: bool = False


@dataclass
class Block:
    kind: str  # line | blank | heading | item | displaymath | pagebreak | gotopage | rule
    spans: List[Span] = field(default_factory=list)
    level: int = 0  # heading level / list nesting
    marker: str = ""  # list marker as written ("1)", "-", "(a)")
    hard_break: bool = False  # line ends with an explicit line break (not joined by reflow)

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans)


PAGEBREAK_RE = re.compile(r"^\s*(\\newpage|\\pagebreak|\\clearpage|<!--\s*pagebreak\s*-->|---\s*pagebreak\s*---|\f)\s*$",
                          re.I)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
BULLET_RE = re.compile(r"^(\s*)([-*+•])\s+(.*)$")
NUMBER_RE = re.compile(r"^(\s*)(\(?[0-9]{1,3}[.)]|\(?[a-zA-Z][.)]|\([ivxlcIVXLC]+\)|[ivx]+\))\s+(.*)$")
HARD_BREAK = "\ue200"  # internal marker for an explicit line break
RULE_RE = re.compile(r"^\s*(\*\s*){3,}$|^\s*(-\s*){3,}$|^\s*(_\s*){3,}$")
GOTO_RE = re.compile(r"^\s*(<!--\s*page\s+(\d+)\s*-->|\\page\{(\d+)\})\s*$", re.I)


def split_inline(text: str, markdown: bool = True) -> List[Span]:
    """Split a line into text and maths spans."""
    spans: List[Span] = []
    i = 0
    buf = []
    n = len(text)

    def flush():
        if buf:
            t = "".join(buf)
            if markdown:
                t = strip_markdown(t)
            if t:
                spans.append(Span(t))
            buf.clear()

    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n and text[i + 1] == "$":
            buf.append("$")
            i += 2
            continue
        if c == "\\" and i + 1 < n and text[i + 1] == "(":
            j = text.find("\\)", i + 2)
            if j != -1:
                flush()
                spans.append(Span(text[i + 2:j].strip(), True))
                i = j + 2
                continue
        if c == "$":
            dbl = i + 1 < n and text[i + 1] == "$"
            start = i + (2 if dbl else 1)
            end_tok = "$$" if dbl else "$"
            j = text.find(end_tok, start)
            while j != -1 and text[j - 1] == "\\":
                j = text.find(end_tok, j + 1)
            if j != -1 and j > start:
                flush()
                spans.append(Span(text[start:j].strip(), True))
                i = j + len(end_tok)
                continue
        buf.append(c)
        i += 1
    flush()
    return spans


_MD_PATTERNS = [
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),
    (re.compile(r"__(.+?)__"), r"\1"),
    (re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])"), r"\1"),
    (re.compile(r"(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])"), r"\1"),
    (re.compile(r"~~(.+?)~~"), r"\1"),
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),
]


_ESCAPES = {"\\*": "\ue000", "\\_": "\ue001", "\\#": "\ue002", "\\`": "\ue003", "\\~": "\ue004",
            "\\[": "\ue005", "\\]": "\ue006"}


def strip_markdown(t: str) -> str:
    """Remove Markdown emphasis / code / link markup, honouring backslash escapes."""
    for esc, ph in _ESCAPES.items():
        t = t.replace(esc, ph)
    for pat, rep in _MD_PATTERNS:
        t = pat.sub(rep, t)
    for esc, ph in _ESCAPES.items():
        t = t.replace(ph, esc[1])
    return t


def parse_text(src: str, markdown: bool = True, reflow: bool = False) -> List[Block]:
    blocks: List[Block] = []
    lines = src.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n\\newpage\n").split("\n")
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        if PAGEBREAK_RE.match(line):
            blocks.append(Block("pagebreak"))
            i += 1
            continue
        gm = GOTO_RE.match(line)
        if gm:
            blocks.append(Block("gotopage", level=int(gm.group(2) or gm.group(3))))
            i += 1
            continue
        stripped = line.strip()
        # display maths: $$ ... $$ or \[ ... \] possibly over several lines
        for open_tok, close_tok in (("$$", "$$"), ("\\[", "\\]")):
            if stripped.startswith(open_tok):
                body = stripped[len(open_tok):]
                j = i
                while close_tok not in body and j + 1 < len(lines):
                    j += 1
                    body += " " + lines[j].strip()
                if close_tok in body:
                    k = body.index(close_tok)
                    rest = body[k + len(close_tok):].strip()
                    blocks.append(Block("displaymath", [Span(body[:k].strip(), True)]))
                    if rest:
                        blocks.append(Block("line", split_inline(rest, markdown)))
                    i = j + 1
                    break
        else:
            if not stripped:
                blocks.append(Block("blank"))
                i += 1
                continue
            if markdown:
                m = HEADING_RE.match(stripped)
                if m:
                    blocks.append(Block("heading", split_inline(m.group(2).strip(), markdown), level=len(m.group(1))))
                    i += 1
                    continue
                if RULE_RE.match(stripped):
                    blocks.append(Block("rule"))
                    i += 1
                    continue
                m = BULLET_RE.match(line)
                if m:
                    blocks.append(Block("item", split_inline(m.group(3), markdown), level=len(m.group(1)) // 2,
                                        marker="-"))
                    i += 1
                    continue
                m = NUMBER_RE.match(line)
                if m:
                    blocks.append(Block("item", split_inline(m.group(3), markdown), level=len(m.group(1)) // 2,
                                        marker=m.group(2)))
                    i += 1
                    continue
            hard = raw.endswith(HARD_BREAK) or (markdown and (raw.endswith("  ") or raw.rstrip().endswith("\\")
                                                              and not raw.rstrip().endswith("\\\\")))
            text = line.strip().rstrip(HARD_BREAK).rstrip()
            if markdown and text.endswith("\\") and not text.endswith("\\\\"):
                text = text[:-1].rstrip()
            b = Block("line", split_inline(text, markdown))
            b.hard_break = hard
            blocks.append(b)
            i += 1
            continue
    if reflow:
        blocks = _reflow(blocks)
    # collapse runs of blank lines into one paragraph gap
    blocks = [b for k, b in enumerate(blocks) if not (b.kind == "blank" and k and blocks[k - 1].kind == "blank")]
    # drop leading/trailing blanks
    while blocks and blocks[0].kind == "blank":
        blocks.pop(0)
    while blocks and blocks[-1].kind == "blank":
        blocks.pop()
    return blocks


def _reflow(blocks: List[Block]) -> List[Block]:
    """Join consecutive lines into paragraphs (Markdown/TeX semantics); a
    line ending in a hard break (two spaces, a backslash, TeX ``\\``) ends
    its line."""
    out: List[Block] = []
    for b in blocks:
        if b.kind == "line" and out and out[-1].kind in ("line", "item") and not out[-1].hard_break:
            out[-1].spans = out[-1].spans + [Span(" ")] + b.spans
            out[-1].hard_break = b.hard_break
        else:
            out.append(b)
    return out


# ---------------------------------------------------------------- other formats
def parse_pdf(path: str | Path) -> List[Block]:
    import pymupdf

    blocks: List[Block] = []
    with pymupdf.open(str(path)) as doc:
        for pno, page in enumerate(doc):
            if pno:
                blocks.append(Block("pagebreak"))
            for b in page.get_text("blocks", sort=True):
                if b[6] != 0:
                    continue
                for ln in b[4].splitlines():
                    if ln.strip():
                        blocks.append(Block("line", [Span(ln.strip())]))
                blocks.append(Block("blank"))
    return blocks


def parse_docx(path: str | Path) -> List[Block]:
    import docx  # python-docx

    d = docx.Document(str(path))
    blocks: List[Block] = []
    for p in d.paragraphs:
        txt = p.text.strip()
        style = (p.style.name or "").lower() if p.style is not None else ""
        has_break = any(br.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}type") == "page"
                        for br in p._p.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}br"))
        if has_break:
            blocks.append(Block("pagebreak"))
        if not txt:
            blocks.append(Block("blank"))
            continue
        if style.startswith("heading"):
            lvl = int(re.sub(r"\D", "", style) or 1)
            blocks.append(Block("heading", split_inline(txt, False), level=lvl))
        elif "list" in style:
            blocks.append(Block("item", split_inline(txt, False), marker="-"))
        else:
            for ln in txt.split("\n"):
                blocks.append(Block("line", split_inline(ln, False)))
    return blocks


_TEX_DROP = re.compile(r"\\(documentclass|usepackage|title|author|date|maketitle|label|vspace|hspace|noindent|"
                       r"centering|setlength|newcommand|renewcommand|pagestyle|thispagestyle|bigskip|medskip|"
                       r"smallskip|par)\b\*?(\[[^\]]*\])?(\{[^}]*\})*")
_TEX_MATH = re.compile(
    r"\\begin\{(equation|align|gather|multline|eqnarray|displaymath)(\*?)\}(.*?)\\end\{\1\2\}"
    r"|\$\$(.+?)\$\$|\\\[(.+?)\\\]|(?<!\\)\$(.+?)(?<!\\)\$|\\\((.+?)\\\)", re.S)

def parse_tex(src: str) -> List[Block]:
    """Pragmatic LaTeX subset: sections, paragraphs, itemize/enumerate,
    inline and display maths (multi-line align/gather become stacked rows),
    \\textbf & co. (content kept), \\\\ line breaks, text escapes."""
    m = re.search(r"\\begin\{document\}(.*)\\end\{document\}", src, re.S)
    body = m.group(1) if m else src
    body = re.sub(r"(?<!\\)%.*", "", body)
    # protect maths from the text transformations below
    maths: List[str] = []

    def keep(mm: "re.Match[str]") -> str:
        env, _, env_body, dd, br, d, par = mm.groups()
        if env is not None:
            inner = " ".join(re.sub(r"\\(label|tag)\{[^}]*\}|\\nonumber|\\notag", "", env_body).split())
            if env in ("align", "eqnarray"):
                inner = "\\begin{aligned}" + inner + "\\end{aligned}"
            elif env in ("gather", "multline"):
                inner = "\\begin{gathered}" + inner + "\\end{gathered}"
            maths.append("$$" + inner + "$$")
            return "\n\ue100%d\ue101\n" % (len(maths) - 1)
        if dd is not None or br is not None:
            maths.append("$$" + " ".join((dd or br).split()) + "$$")
            return "\n\ue100%d\ue101\n" % (len(maths) - 1)
        maths.append("$" + " ".join((d or par).split()) + "$")
        return "\ue100%d\ue101" % (len(maths) - 1)

    body = _TEX_MATH.sub(keep, body)
    body = _TEX_DROP.sub("", body)
    body = re.sub(r"\\(section|subsection|subsubsection)\*?\{([^}]*)\}",
                  lambda mm: "\n\n" + "#" * {"section": 1, "subsection": 2, "subsubsection": 3}[mm.group(1)] + " "
                  + mm.group(2) + "\n\n", body)
    body = re.sub(r"\\(newpage|clearpage|pagebreak)\b", "\n\\\\newpage\n", body)
    body = re.sub(r"\\begin\{(itemize|enumerate|description)\}|\\end\{(itemize|enumerate|description)\}",
                  "\n\n", body)
    def item(mm: "re.Match[str]") -> str:
        if mm.group(1):
            return "\n" + mm.group(1)[1:-1] + " "
        return "\n- "

    body = re.sub(r"\s*\\item\s*(\[[^\]]*\])?", item, body)
    for _ in range(3):  # nested formatting commands
        body = re.sub(r"\\(textbf|textit|textrm|textsf|texttt|emph|underline|text|mbox)\{([^{}]*)\}", r"\2", body)
    body = body.replace("\\\\", HARD_BREAK + "\n").replace("~", " ")
    for esc, lit in (("\\%", "%"), ("\\&", "&"), ("\\{", "{"), ("\\}", "}"), ("\\ldots", "..."),
                     ("\\dots", "..."), ("``", "\u201c"), ("''", "\u201d"), ("---", "\u2014"), ("--", "\u2013")):
        body = body.replace(esc, lit)
    body = re.sub("\ue100(\\d+)\ue101", lambda mm: maths[int(mm.group(1))], body)
    return parse_text(body, markdown=True, reflow=True)


def parse_file(path: str | Path, reflow: Optional[bool] = None) -> List[Block]:
    """Parse a document.  ``reflow`` defaults to True for Markdown (single
    newlines join, as in Markdown) and False for plain text (lines kept)."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(p)
    if ext == ".docx":
        return parse_docx(p)
    text = p.read_text(encoding="utf-8")
    if ext == ".tex":
        return parse_tex(text)
    md = ext in (".md", ".markdown", "")
    return parse_text(text, markdown=md, reflow=md if reflow is None else reflow)
