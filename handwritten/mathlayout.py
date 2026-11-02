"""Typeset a practical subset of LaTeX maths in the writer's handwriting.

The pipeline is ``tokenize`` -> ``Parser`` (recursive descent, builds a small
AST) -> ``MathLayout`` (turns the AST into an :class:`InkBox`).

Every letter, digit and symbol is drawn with ``synth.glyph_box`` so it is the
writer's own glyph (or the synthesiser's restyled fallback; the few symbols
the fallback cannot draw get a procedural stroke here).  Ordinary symbols are
spaced with the writer's letter gaps; operators and relations get TeX-like
extra space.  Scripts are placed against the base's actual ink profile (so a
slanted letter's subscript tucks in under it), parentheses are the writer's
real ones stretched to the content, and the structural marks a student draws
by hand -- fraction bars, radicals, brackets, braces, accents -- are synthetic
centre-line strokes with the writer's pen width, slant and a little wobble.

Supported (a pragmatic subset): super/subscripts (nested, combined, primes),
``\\frac \\dfrac \\tfrac \\cfrac \\binom``, ``\\over \\choose``, ``\\sqrt[n]``,
``\\left \\middle \\right`` (and automatic sizing of matched ``( ) [ ] \\{ \\}``
pairs and ``|x|`` bars), ``\\big``-family delimiters, big operators with
limits, operator names, Greek, relations / binary operators / arrows with
TeX-like spacing, spacing commands, ``\\text \\mathrm \\operatorname \\mathbb``,
``\\pmod \\bmod``, accents, ``\\overset \\underset \\stackrel``, ``\\boxed``,
``\\not``, ``\\phantom`` and matrix-like environments (``matrix pmatrix
bmatrix Bmatrix vmatrix Vmatrix cases aligned align gathered array``).  Rows
separated by ``\\\\`` at the top level are stacked (aligned at ``&``).

Unknown commands render their name in handwriting; malformed input never
raises (the worst case is the source drawn as plain handwriting).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .box import InkBox
from .geometry import flatten_subpath

# ---------------------------------------------------------------------------
# Tunables (all lengths in x-heights of the current size)
# ---------------------------------------------------------------------------
SCRIPT_SCALE = 0.70  # superscripts / subscripts / limits
MIN_SCALE = 0.42
FRAC_SCALE_INLINE = 0.80
AXIS = 0.52  # maths axis (fraction bars, operators) above the baseline
EM = 1.8  # one "em" in x-heights for the writer (\quad); TeX's mu = EM / 18
BIN_GAP = 0.32
REL_GAP = 0.50
PUNCT_GAP = 0.30
OP_GAP = 0.32
INNER_GAP = 0.16
DELIM_TOP = 1.45  # normal-size delimiter ink extent above the baseline
DELIM_BOT = 0.33  # ... and below it

# ---------------------------------------------------------------------------
# Symbol tables
# ---------------------------------------------------------------------------
GREEK: Dict[str, str] = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "θ", "iota": "ι", "kappa": "κ", "varkappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "omicron": "ο", "pi": "π", "varpi": "π", "rho": "ρ",
    "varrho": "ρ", "sigma": "σ", "varsigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ",
    "Upsilon": "Y", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    "varGamma": "Γ", "varDelta": "Δ", "varTheta": "Θ", "varLambda": "Λ", "varPi": "Π", "varSigma": "Σ",
    "varPhi": "Φ", "varPsi": "Ψ", "varOmega": "Ω",
}

_REL = {
    "le": "≤", "leq": "≤", "leqslant": "≤", "ge": "≥", "geq": "≥", "geqslant": "≥", "ne": "≠", "neq": "≠",
    "approx": "≈", "equiv": "≡", "sim": "∼", "simeq": "≃", "cong": "≅", "propto": "∝", "to": "→",
    "rightarrow": "→", "leftarrow": "←", "gets": "←", "leftrightarrow": "↔", "Rightarrow": "⇒",
    "Leftarrow": "⇐", "Leftrightarrow": "⇔", "implies": "⟹", "Longrightarrow": "⟹", "impliedby": "⟸",
    "Longleftarrow": "⟸", "iff": "⟺", "Longleftrightarrow": "⟺", "longrightarrow": "⟶",
    "longleftarrow": "⟵", "longleftrightarrow": "⟷", "mapsto": "↦", "longmapsto": "↦",
    "hookrightarrow": "→", "rightharpoonup": "→", "uparrow": "↑", "downarrow": "↓", "Uparrow": "↑",
    "Downarrow": "↓", "in": "∈", "notin": "∉", "ni": "∋", "owns": "∋", "subset": "⊂", "supset": "⊃",
    "subseteq": "⊆", "supseteq": "⊇", "subsetneq": "⊊", "supsetneq": "⊋", "nsubseteq": "⊈",
    "mid": "∣", "nmid": "∤", "parallel": "∥", "perp": "⊥", "ll": "≪", "gg": "≫", "prec": "≺",
    "succ": "≻", "preceq": "⪯", "succeq": "⪰", "models": "⊨", "vdash": "⊢", "dashv": "⊣",
    "asymp": "≍", "doteq": "≐", "coloneqq": "≔", "coloneq": "≔", "lt": "<", "gt": ">",
    "leftrightarrows": "⇄", "nleq": "≰", "ngeq": "≱", "lesssim": "≲", "gtrsim": "≳",
}
_BIN = {
    "pm": "±", "mp": "∓", "cdot": "·", "times": "×", "div": "÷", "cup": "∪", "cap": "∩",
    "setminus": "∖", "smallsetminus": "∖", "circ": "∘", "oplus": "⊕", "otimes": "⊗", "ast": "∗",
    "star": "⋆", "wedge": "∧", "land": "∧", "vee": "∨", "lor": "∨", "bullet": "•", "sqcup": "⊔",
    "sqcap": "⊓", "uplus": "⊎", "odot": "⊙", "ominus": "⊖", "triangle": "△",
}
_ORD = {
    "emptyset": "∅", "varnothing": "∅", "infty": "∞", "partial": "∂", "nabla": "∇", "forall": "∀",
    "exists": "∃", "nexists": "∄", "neg": "¬", "lnot": "¬", "prime": "′", "angle": "∠", "top": "⊤",
    "bot": "⊥", "degree": "°", "ell": "ℓ", "hbar": "ℏ", "hslash": "ℏ", "aleph": "ℵ", "imath": "i",
    "jmath": "j", "Re": "ℜ", "Im": "ℑ", "wp": "℘", "backslash": "\\", "square": "□", "Box": "□",
    "dagger": "†", "ddagger": "‡", "S": "§", "P": "¶", "checkmark": "✓", "flat": "♭", "sharp": "♯",
    "ldots": "…", "dots": "…", "dotsc": "…", "dotso": "…", "cdots": "⋯", "dotsb": "⋯", "dotsm": "⋯",
    "dotsi": "⋯", "vdots": "⋮", "ddots": "⋱", "vert": "|", "Vert": "‖", "surd": "√",
    "mathellipsis": "…", "cdotp": "·", "colon": ":",
}
_OPEN = {"langle": "⟨", "lfloor": "⌊", "lceil": "⌈", "lbrace": "{", "lbrack": "[", "lvert": "|",
         "lVert": "‖", "lparen": "("}
_CLOSE = {"rangle": "⟩", "rfloor": "⌋", "rceil": "⌉", "rbrace": "}", "rbrack": "]", "rvert": "|",
          "rVert": "‖", "rparen": ")"}

SYMBOLS: Dict[str, Tuple[str, str]] = {}
for _tbl, _cls in ((_ORD, "ord"), (_REL, "rel"), (_BIN, "bin"), (_OPEN, "open"), (_CLOSE, "close")):
    for _k, _v in _tbl.items():
        SYMBOLS[_k] = (_v, _cls)
SYMBOLS["colon"] = (":", "punct")
for _k, _v in GREEK.items():
    SYMBOLS[_k] = (_v, "ord")

# single non-letter control symbols: \{ \} \| \% ...
CTRL_SYMBOLS: Dict[str, Tuple[str, str]] = {
    "{": ("{", "open"), "}": ("}", "close"), "|": ("‖", "ord"), "%": ("%", "ord"), "#": ("#", "ord"),
    "$": ("$", "ord"), "&": ("&", "ord"), "_": ("_", "ord"), "/": ("/", "ord"),
}
# spacing in x-heights (TeX mu = EM / 18)
SPACES: Dict[str, float] = {
    ",": 3 * EM / 18, "thinspace": 3 * EM / 18, ":": 4 * EM / 18, ">": 4 * EM / 18, "medspace": 4 * EM / 18,
    ";": 5 * EM / 18, "thickspace": 5 * EM / 18, "!": -3 * EM / 18, "negthinspace": -3 * EM / 18,
    "negmedspace": -4 * EM / 18, "negthickspace": -5 * EM / 18, "quad": EM, "qquad": 2 * EM,
    " ": 0.6, "enspace": EM / 2, "enskip": EM / 2, "space": 0.6, "nobreakspace": 0.6,
}

BIGOPS: Dict[str, str] = {
    "sum": "∑", "prod": "∏", "coprod": "∐", "int": "∫", "iint": "∬", "iiint": "∭", "oint": "∮",
    "bigcup": "⋃", "bigcap": "⋂", "bigoplus": "⨁", "bigotimes": "⨂", "bigvee": "⋁", "bigwedge": "⋀",
    "bigsqcup": "⨆", "biguplus": "⋃", "bigodot": "⨀",
}
INTEGRALS = set("∫∬∭∮")
BIGOP_CHARS = set(BIGOPS.values())

# operator names: name -> takes limits in display style
OPNAMES: Dict[str, bool] = {
    "lim": True, "limsup": True, "liminf": True, "max": True, "min": True, "sup": True, "inf": True,
    "det": True, "gcd": True, "Pr": True, "argmax": True, "argmin": True, "injlim": True, "projlim": True,
    "log": False, "ln": False, "lg": False, "exp": False, "sin": False, "cos": False, "tan": False,
    "sec": False, "csc": False, "cot": False, "arcsin": False, "arccos": False, "arctan": False,
    "sinh": False, "cosh": False, "tanh": False, "coth": False, "deg": False, "dim": False, "ker": False,
    "hom": False, "arg": False, "lcm": False, "sgn": False, "tr": False, "rank": False, "Var": False,
    "Cov": False, "id": False, "mod": False, "erf": False, "Re": False, "Im": False,
}
OPNAME_TEXT = {"limsup": "lim sup", "liminf": "lim inf", "argmax": "arg max", "argmin": "arg min",
               "injlim": "inj lim", "projlim": "proj lim"}

ACCENTS = {"overline", "bar", "hat", "widehat", "tilde", "widetilde", "vec", "overrightarrow",
           "overleftarrow", "dot", "ddot", "dddot", "underline", "check", "widecheck", "breve", "acute",
           "grave", "overbrace", "underbrace", "mathring", "overleftrightarrow"}
FONT_CMDS = {"mathit", "mathbf", "mathsf", "mathtt", "boldsymbol", "bm", "mathnormal", "mathcal",
             "mathscr", "mathfrak", "mathrm", "mathop", "mathord", "pmb", "mathbin", "mathrel", "mathpunct",
             "mathopen", "mathclose", "mathinner", "symbf", "symit", "symrm", "boldmath"}
TEXT_CMDS = {"text", "textrm", "textit", "textbf", "textsf", "texttt", "textnormal", "mbox", "hbox",
             "textup", "textmd", "emph", "fbox"}
IGNORE_CMDS = {"nonumber", "notag", "relax", "allowbreak", "nolinebreak", "linebreak", "displaybreak",
               "hfill", "hfil", "mathstrut", "strut", "noindent", "centering", "limits", "nolimits",
               "nobreak", "protect", "cr", "hline", "middle", "small", "large", "Large", "normalsize",
               "footnotesize", "scriptsize", "tiny", "huge", "Huge", "rm", "it", "bf", "cal", "sf", "tt"}
IGNORE_ARG_CMDS = {"label", "tag", "vspace", "color", "eqref", "ref", "cite", "hypertarget", "pagecolor"}
DOUBLE_STRUCK = {"N": "ℕ", "Z": "ℤ", "Q": "ℚ", "R": "ℝ", "C": "ℂ"}
BIG_SIZES = {"big": 1.25, "Big": 1.6, "bigg": 2.0, "Bigg": 2.45}

# characters typed directly (Unicode input) -> atom class
UNICODE_CLASS: Dict[str, str] = {}
for _tbl, _cls in ((_REL, "rel"), (_BIN, "bin"), (_OPEN, "open"), (_CLOSE, "close")):
    for _v in _tbl.values():
        UNICODE_CLASS.setdefault(_v, _cls)
UNICODE_CLASS.update({"=": "rel", "<": "rel", ">": "rel", ":": "rel", "+": "bin", "-": "bin", "−": "bin",
                      "*": "bin", ",": "punct", ";": "punct", "(": "open", "[": "open", ")": "close",
                      "]": "close", "!": "close", "?": "close", "|": "ord", "‖": "ord"})

DELIM_CHARS = set("()[]{}⟨⟩⌊⌋⌈⌉")
OPEN_CHARS = set("([{⟨⌊⌈")
CLOSE_CHARS = set(")]}⟩⌋⌉")
BAR_CHARS = {"|", "‖", "∣", "∥"}
DELIM_CMD = {"{": "{", "}": "}", "|": "‖", "langle": "⟨", "rangle": "⟩", "lfloor": "⌊", "rfloor": "⌋",
             "lceil": "⌈", "rceil": "⌉", "vert": "|", "Vert": "‖", "lvert": "|", "rvert": "|",
             "lVert": "‖", "rVert": "‖", "lbrace": "{", "rbrace": "}", "lbrack": "[", "rbrack": "]",
             "backslash": "\\", "uparrow": "↑", "downarrow": "↓", "lparen": "(", "rparen": ")",
             "mid": "|"}

# characters substituted before drawing
ALIAS = {"ℓ": "l", "ℏ": "h", "ℵ": "N", "ℜ": "R", "ℑ": "I", "℘": "p", "•": "·", "∗": "*", "⋆": "*",
         "∼": "~", "†": "+", "‡": "+", "♭": "b", "♯": "#", "✓": "v", "§": "S", "¶": "P", "−": "-",
         "ϵ": "ε", "ϑ": "θ", "ϖ": "π", "ϱ": "ρ", "ς": "σ", "ϕ": "φ", "Υ": "Y", "∁": "C", "⊎": "∪",
         "⊔": "∪", "⊓": "∩", "≲": "≤", "≳": "≥", "≍": "≈", "≐": "=", "⇄": "↔", "⟷": "↔"}
WIDE = {"⟹": "⇒", "⟸": "⇐", "⟺": "⇔", "⟶": "→", "⟵": "←"}  # drawn as a widened arrow
RECENTER = {"~", "*", "÷", "±", "∓"}  # glyphs re-centred on the maths axis
UNI_SUP = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ", "0123456789+−=()ni"))
UNI_SUB = dict(zip("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₙᵢⱼₖ", "0123456789+−=()aeoxnijk"))
NOT_OF = {"=": "≠", "∈": "∉", "≤": "≰", "≥": "≱", "⊂": "⊄", "⊃": "⊅", "⊆": "⊈", "∣": "∤", "∃": "∄"}  # \not X
NEGATED = {"∤": "∣", "⊈": "⊆", "∄": "∃", "≰": "≤", "≱": "≥", "⊄": "⊂", "⊅": "⊃"}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
@dataclass
class Tok:
    kind: str  # cmd | char | space | lbrace | rbrace | sup | sub | amp | eof
    value: str = ""
    pos: int = 0


def tokenize(src: str) -> List[Tok]:
    out: List[Tok] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "\\":
            if i + 1 >= n:
                i += 1
                continue
            d = src[i + 1]
            if d.isalpha():
                j = i + 1
                while j < n and src[j].isalpha():
                    j += 1
                out.append(Tok("cmd", src[i + 1:j], i))
                i = j
            else:
                out.append(Tok("cmd", d, i))
                i += 2
            continue
        if c.isspace():
            j = i
            while j < n and src[j].isspace():
                j += 1
            out.append(Tok("space", " ", i))
            i = j
            continue
        kind = {"{": "lbrace", "}": "rbrace", "^": "sup", "_": "sub", "&": "amp"}.get(c, "char")
        if c == "~":
            out.append(Tok("cmd", " ", i))
        elif c == "$":
            pass  # stray maths shift
        else:
            out.append(Tok(kind, c, i))
        i += 1
    out.append(Tok("eof", "", n))
    return out


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------
class Node:
    pass


@dataclass
class Sym(Node):
    ch: str
    cls: str = "ord"  # ord | bin | rel | open | close | punct
    smart: bool = False  # \dots: becomes \cdots between binary operators / relations


@dataclass
class Row(Node):
    items: List[Node] = field(default_factory=list)


@dataclass
class Scripts(Node):
    base: Node
    sup: Optional[Node] = None
    sub: Optional[Node] = None
    primes: int = 0


@dataclass
class Frac(Node):
    num: Node
    den: Node
    style: str = "auto"  # auto | d | t
    bar: bool = True


@dataclass
class Sqrt(Node):
    body: Node
    index: Optional[Node] = None


@dataclass
class Fenced(Node):
    left: str  # delimiter character or "." for none
    body: Row
    right: str
    auto: bool = False  # matched pair found by the parser (not \left..\right)


@dataclass
class Middle(Node):
    ch: str


@dataclass
class BigDelim(Node):
    ch: str
    factor: float
    cls: str = "ord"


@dataclass
class BigOp(Node):
    ch: str
    limits: Optional[bool] = None  # None = default for the operator


@dataclass
class OpName(Node):
    name: str
    limits: bool = False
    cls: str = "op"
    force: Optional[bool] = None  # \limits / \nolimits


@dataclass
class Text(Node):
    text: str


@dataclass
class Space(Node):
    xh: float  # width in x-heights (may be negative)


@dataclass
class Accent(Node):
    kind: str
    body: Node


@dataclass
class Stack(Node):
    base: Node
    over: Optional[Node] = None
    under: Optional[Node] = None


@dataclass
class XArrow(Node):
    over: Optional[Node]
    under: Optional[Node]
    left: bool = False


@dataclass
class Boxed(Node):
    body: Node


@dataclass
class Phantom(Node):
    body: Node
    h: bool = True
    v: bool = True


@dataclass
class Negated(Node):
    body: Node


@dataclass
class StyleSwitch(Node):
    style: str  # display | text | script | scriptscript


@dataclass
class Matrix(Node):
    rows: List[List[Row]]
    kind: str = "matrix"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
MATRIX_ENVS = {"matrix": ("", ""), "pmatrix": ("(", ")"), "bmatrix": ("[", "]"), "Bmatrix": ("{", "}"),
               "vmatrix": ("|", "|"), "Vmatrix": ("‖", "‖"), "smallmatrix": ("", ""),
               "cases": ("{", ""), "dcases": ("{", ""), "rcases": ("", "}"), "array": ("", ""),
               "aligned": ("", ""), "align": ("", ""), "alignat": ("", ""), "alignedat": ("", ""),
               "gathered": ("", ""), "gather": ("", ""), "split": ("", ""), "equation": ("", ""),
               "eqnarray": ("", ""), "multline": ("", ""), "subarray": ("", ""), "tabular": ("", "")}
MAX_DEPTH = 40


class Parser:
    def __init__(self, src: str):
        self.toks = tokenize(src)
        self.i = 0
        self.ctx: List[str] = []  # brace | left | env | cells | opt

    # -------------------------------------------------------------- tokens
    def peek(self, skip_space: bool = True) -> Tok:
        j = self.i
        while skip_space and self.toks[j].kind == "space":
            j += 1
        return self.toks[j]

    def next(self, skip_space: bool = True) -> Tok:
        while skip_space and self.toks[self.i].kind == "space":
            self.i += 1
        t = self.toks[self.i]
        if t.kind != "eof":
            self.i += 1
        return t

    # --------------------------------------------------------------- entry
    def parse(self) -> Node:
        rows = self.parse_cells(top=True)
        if len(rows) == 1 and len(rows[0]) == 1:
            out = rows[0][0]
        else:
            out = Row([Matrix(rows, "aligned" if any(len(r) > 1 for r in rows) else "gathered")])
        # anything left over (unbalanced closers at the top level) is skipped by parse_row
        return out

    def parse_cells(self, top: bool = False) -> List[List[Row]]:
        self.ctx.append("cells")
        rows: List[List[Row]] = [[]]
        try:
            while True:
                cell = self.parse_row()
                rows[-1].append(cell)
                t = self.peek()
                if t.kind == "amp":
                    self.next()
                    continue
                if t.kind == "cmd" and t.value in ("\\", "cr", "newline"):
                    self.next()
                    if self.peek().kind == "char" and self.peek().value == "[":  # \\[2pt]
                        self._skip_until_char("]")
                    rows.append([])
                    continue
                if top and t.kind != "eof":
                    self.next()  # stray closer at the top level
                    continue
                break
        finally:
            self.ctx.pop()
        while len(rows) > 1 and all(not c.items for c in rows[-1]):
            rows.pop()
        return [r for r in rows if r] or [[Row()]]

    def _skip_until_char(self, ch: str) -> None:
        while True:
            t = self.next()
            if t.kind == "eof" or (t.kind == "char" and t.value == ch):
                return

    def _stop(self, t: Tok) -> bool:
        inner = self.ctx[-1] if self.ctx else ""
        if t.kind == "eof":
            return True
        if t.kind == "rbrace":
            return "brace" in self.ctx
        if t.kind == "cmd" and t.value == "right":
            return "left" in self.ctx
        if t.kind == "cmd" and t.value == "end":
            return "env" in self.ctx
        if t.kind == "amp" or (t.kind == "cmd" and t.value in ("\\", "cr", "newline")):
            return inner == "cells"
        if t.kind == "char" and t.value == "]":
            return inner == "opt"
        return False

    def parse_row(self) -> Row:
        items: List[Node] = []
        while True:
            t = self.peek()
            if self._stop(t):
                break
            if t.kind == "rbrace" or t.kind == "amp" or (t.kind == "cmd" and t.value in ("\\", "cr", "newline")):
                self.next()  # stray / out of context: ignore
                continue
            if t.kind == "cmd" and t.value == "end":  # \end without \begin
                self.next()
                self._raw_arg()
                continue
            if t.kind == "cmd" and t.value in ("over", "choose", "atop", "brace", "brack"):
                self.next()
                rest = self.parse_row()
                num, den = Row(_pair_delims(items)), rest
                if t.value == "over":
                    return Row([Frac(num, den)])
                if t.value == "atop":
                    return Row([Frac(num, den, bar=False)])
                l, r = {"choose": ("(", ")"), "brace": ("{", "}"), "brack": ("[", "]")}[t.value]
                return Row([Fenced(l, Row([Frac(num, den, bar=False)]), r)])
            atom = self.parse_atom()
            if atom is None:
                continue
            atom = self.parse_scripts(atom)
            items.append(atom)
        return Row(_pair_delims(items))

    # ------------------------------------------------------------ scripts
    def parse_scripts(self, base: Node) -> Node:
        sup = sub = None
        primes = 0
        while True:
            t = self.peek(skip_space=True)
            if t.kind == "char" and t.value == "'":
                self.next()
                primes += 1
                continue
            if t.kind == "char" and (t.value in UNI_SUP or t.value in UNI_SUB):
                table = UNI_SUP if t.value in UNI_SUP else UNI_SUB
                chars = []
                while self.peek(skip_space=False).kind == "char" and self.peek(skip_space=False).value in table:
                    chars.append(self._char_atom(table[self.next(skip_space=False).value]))
                arg = Row(chars)
                if table is UNI_SUP:
                    if sup is not None:
                        base = Scripts(base, sup, sub, primes)
                        sub, primes = None, 0
                    sup = arg
                else:
                    if sub is not None:
                        base = Scripts(base, sup, sub, primes)
                        sup, primes = None, 0
                    sub = arg
                continue
            if t.kind == "sup" or t.kind == "sub":
                self.next()
                arg = self.parse_arg(script=True)
                if t.kind == "sup":
                    if isinstance(arg, Sym) and arg.ch == "′":
                        primes += 1
                        continue
                    if isinstance(arg, Row) and arg.items and \
                            all(isinstance(x, Sym) and x.ch == "′" for x in arg.items):
                        primes += len(arg.items)
                        continue
                    if sup is not None:  # x^a^b: nest instead of failing
                        base = Scripts(base, sup, sub, primes)
                        sub, primes = None, 0
                    sup = arg
                else:
                    if sub is not None:
                        base = Scripts(base, sup, sub, primes)
                        sup, primes = None, 0
                    sub = arg
                continue
            if t.kind == "cmd" and t.value in ("limits", "nolimits"):
                self.next()
                if isinstance(base, BigOp):
                    base.limits = t.value == "limits"
                elif isinstance(base, OpName):
                    base.force = t.value == "limits"
                continue
            break
        if sup is None and sub is None and primes == 0:
            return base
        return Scripts(base, sup, sub, primes)

    # ------------------------------------------------------------ arguments
    def parse_arg(self, script: bool = False) -> Node:
        t = self.peek()
        if t.kind == "lbrace":
            return self.parse_group()
        if t.kind in ("eof", "rbrace", "amp", "sup", "sub"):
            return Row()
        if t.kind == "cmd":
            if self._stop(t):
                return Row()
            a = self.parse_atom()
            return a if a is not None else Row()
        self.next()
        return self._char_atom(t.value)

    def parse_group(self) -> Row:
        self.next()  # {
        if len(self.ctx) > MAX_DEPTH:
            raise RecursionError("maths nested too deeply")
        self.ctx.append("brace")
        try:
            row = self.parse_row()
        finally:
            self.ctx.pop()
        if self.peek().kind == "rbrace":
            self.next()
        return row

    def _raw_arg(self) -> str:
        """Raw text of a braced argument (for \\text, environment names...)."""
        t = self.peek()
        if t.kind != "lbrace":
            if t.kind in ("char",):
                self.next()
                return t.value
            return ""
        self.next()
        depth = 1
        parts: List[str] = []
        while True:
            t = self.next(skip_space=False)
            if t.kind == "eof":
                break
            if t.kind == "lbrace":
                depth += 1
                continue
            if t.kind == "rbrace":
                depth -= 1
                if depth == 0:
                    break
                continue
            if t.kind == "space":
                parts.append(" ")
            elif t.kind == "cmd":
                v = t.value
                if v in (" ", ",", ";", ":", ">", "quad", "qquad", "\\"):
                    parts.append(" ")
                elif len(v) == 1 and not v.isalpha():
                    parts.append(CTRL_SYMBOLS.get(v, (v, ""))[0] if v != "|" else "|")
                elif v in SYMBOLS:
                    parts.append(SYMBOLS[v][0])
                # other commands (\textbf, \emph...) are dropped, their argument text kept
            elif t.kind == "sup":
                parts.append("^")
            elif t.kind == "sub":
                parts.append("_")
            elif t.kind == "amp":
                parts.append("&")
            else:
                parts.append(t.value)
        return "".join(parts)

    def parse_delim(self) -> str:
        t = self.peek()
        if t.kind == "char":
            self.next()
            return {"<": "⟨", ">": "⟩"}.get(t.value, t.value)
        if t.kind == "cmd" and t.value in DELIM_CMD:
            self.next()
            return DELIM_CMD[t.value]
        if t.kind == "cmd" and t.value in SYMBOLS and len(SYMBOLS[t.value][0]) == 1:
            self.next()
            return SYMBOLS[t.value][0]
        return "."

    # ---------------------------------------------------------------- atoms
    def _char_atom(self, c: str) -> Node:
        c = UNI_SUP.get(c, UNI_SUB.get(c, c))
        if c in BIGOP_CHARS:
            return BigOp(c)
        if c.isalnum() and c not in UNICODE_CLASS:
            return Sym(c, "ord")
        if c == "-":
            return Sym("−", "bin")
        if c == "'":
            return Sym("′", "ord")
        return Sym(c, UNICODE_CLASS.get(c, "ord"))

    def parse_atom(self) -> Optional[Node]:
        t = self.next()
        k = t.kind
        if k == "char":
            return self._char_atom(t.value)
        if k == "lbrace":
            self.i -= 1
            return self.parse_group()
        if k in ("sup", "sub"):
            # script without a base: attach to an empty base
            self.i -= 1
            return Row()
        if k != "cmd":
            return None
        return self.parse_command(t.value)

    def parse_command(self, name: str) -> Optional[Node]:
        if name in SPACES:
            return Space(SPACES[name])
        if len(name) == 1 and name in CTRL_SYMBOLS:
            ch, cls = CTRL_SYMBOLS[name]
            return Sym(ch, cls)
        if name in BIGOPS:
            return BigOp(BIGOPS[name])
        if name in OPNAMES:
            return OpName(OPNAME_TEXT.get(name, name), OPNAMES[name])
        if name in ("dots", "ldots", "dotsc", "dotso"):
            return Sym("…", "ord", smart=(name == "dots"))
        if name in SYMBOLS:
            ch, cls = SYMBOLS[name]
            return Sym(ch, cls)
        if name in ("frac", "dfrac", "tfrac", "cfrac", "sfrac"):
            num = self.parse_arg()
            den = self.parse_arg()
            return Frac(num, den, {"dfrac": "d", "cfrac": "d", "tfrac": "t"}.get(name, "auto"))
        if name in ("binom", "dbinom", "tbinom"):
            n = self.parse_arg()
            k = self.parse_arg()
            return Fenced("(", Row([Frac(n, k, {"dbinom": "d", "tbinom": "t"}.get(name, "auto"), bar=False)]), ")")
        if name == "sqrt":
            index = None
            t = self.peek()
            if t.kind == "char" and t.value == "[":
                self.next()
                self.ctx.append("opt")
                try:
                    index = self.parse_row()
                finally:
                    self.ctx.pop()
                if self.peek().kind == "char" and self.peek().value == "]":
                    self.next()
            return Sqrt(self.parse_arg(), index)
        if name == "left":
            return self.parse_left()
        if name == "right":  # stray \right: draw its delimiter
            d = self.parse_delim()
            return Sym(d, "close") if d != "." else None
        if name == "middle":
            d = self.parse_delim()
            return Middle(d) if d != "." else None
        base = name.rstrip("lrm")
        if base in BIG_SIZES and name in (base, base + "l", base + "r", base + "m"):
            d = self.parse_delim()
            if d == ".":
                return None
            cls = {"l": "open", "r": "close", "m": "rel"}.get(name[len(base):], "ord")
            return BigDelim(d, BIG_SIZES[base], cls)
        if name in TEXT_CMDS:
            return Text(self._raw_arg())
        if name == "operatorname":
            lim = False
            if self.peek().kind == "char" and self.peek().value == "*":
                self.next()
                lim = True
            return OpName(self._raw_arg().strip() or "op", lim)
        if name == "mathbb" or name == "Bbb" or name == "mathbbm":
            arg = self.parse_arg()
            return _map_syms(arg, lambda s: Sym(DOUBLE_STRUCK.get(s.ch, s.ch), s.cls))
        if name in ("mathrm", "mathsf", "mathtt", "mathup", "textup") and self._plain_word_arg():
            return Text(self._raw_arg())  # \mathrm{max}: a handwritten word
        if name in FONT_CMDS:
            return self.parse_arg()
        if name in ACCENTS:
            return Accent(name, self.parse_arg())
        if name in ("pmod", "pod"):
            arg = self.parse_arg()
            inner = [Text("mod"), Space(0.7), arg] if name == "pmod" else [arg]
            return Row([Space(0.55), Fenced("(", Row(inner), ")")])
        if name == "bmod":
            return OpName("mod", False, cls="bin")
        if name == "mod":
            return Row([Space(0.9), OpName("mod", False), Space(0.3)])
        if name in ("overset", "stackrel"):
            over = self.parse_arg(script=True)
            return Stack(self.parse_arg(), over=over)
        if name == "underset":
            under = self.parse_arg(script=True)
            return Stack(self.parse_arg(), under=under)
        if name in ("xrightarrow", "xleftarrow", "xlongrightarrow"):
            under = None
            t = self.peek()
            if t.kind == "char" and t.value == "[":
                self.next()
                self.ctx.append("opt")
                try:
                    under = self.parse_row()
                finally:
                    self.ctx.pop()
                if self.peek().kind == "char" and self.peek().value == "]":
                    self.next()
            return XArrow(self.parse_arg(), under, left=(name == "xleftarrow"))
        if name in ("boxed", "fbox", "framebox"):
            return Boxed(self.parse_arg())
        if name in ("phantom", "hphantom", "vphantom"):
            return Phantom(self.parse_arg(), h=name != "vphantom", v=name != "hphantom")
        if name in ("cancel", "bcancel", "xcancel", "sout"):
            return Negated(self.parse_arg())
        if name == "not":
            t = self.peek()
            if t.kind == "eof":
                return None
            nxt = self.parse_atom()
            if isinstance(nxt, Sym):
                if nxt.ch in NOT_OF:
                    return Sym(NOT_OF[nxt.ch], nxt.cls)
                return Negated(nxt)
            return Negated(nxt) if nxt is not None else None
        if name in ("textcolor", "colorbox"):
            self._raw_arg()
            return self.parse_arg()
        if name in ("hspace", "kern", "mkern", "mskip", "hskip"):
            return Space(_parse_length(self._raw_arg() if self.peek().kind == "lbrace" else self._read_dimen()))
        if name in IGNORE_ARG_CMDS:
            self._raw_arg()
            return None
        if name in ("displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle"):
            return StyleSwitch(name[:-5])
        if name in IGNORE_CMDS or name in ("\\", "cr", "newline"):
            return None
        if name == "begin":
            return self.parse_env()
        if name == "end":
            self._raw_arg()
            return None
        if name in ("sideset",):
            self.parse_arg()
            self.parse_arg()
            return None
        if name in ("substack",):
            arg_rows = self._parse_braced_cells()
            return Matrix(arg_rows, "substack")
        # unknown command: write its name
        return Text(name)

    def _plain_word_arg(self) -> bool:
        """True if the next argument is a braced run of >= 2 letters (and spaces)."""
        j = self.i
        while self.toks[j].kind == "space":
            j += 1
        if self.toks[j].kind != "lbrace":
            return False
        j += 1
        letters = 0
        while self.toks[j].kind in ("char", "space"):
            t = self.toks[j]
            if t.kind == "char":
                if not t.value.isalpha():
                    return False
                letters += 1
            j += 1
        return self.toks[j].kind == "rbrace" and letters >= 2

    def _read_dimen(self) -> str:
        parts = []
        while True:
            t = self.peek(skip_space=False)
            if t.kind == "char" and (t.value.isalnum() or t.value in ".-"):
                parts.append(t.value)
                self.next(skip_space=False)
                if len(parts) > 10:
                    break
            else:
                break
        return "".join(parts)

    def _parse_braced_cells(self) -> List[List[Row]]:
        if self.peek().kind != "lbrace":
            return [[Row()]]
        self.next()
        self.ctx.append("brace")
        try:
            rows = self.parse_cells()
        finally:
            self.ctx.pop()
        if self.peek().kind == "rbrace":
            self.next()
        return rows

    def parse_left(self) -> Node:
        left = self.parse_delim()
        if len(self.ctx) > MAX_DEPTH:
            raise RecursionError("maths nested too deeply")
        self.ctx.append("left")
        try:
            body = self.parse_row()
        finally:
            self.ctx.pop()
        right = "."
        t = self.peek()
        if t.kind == "cmd" and t.value == "right":
            self.next()
            right = self.parse_delim()
        return Fenced(left, body, right)

    def parse_env(self) -> Optional[Node]:
        env = self._raw_arg().strip().rstrip("*")
        if env in ("array", "tabular", "alignat", "alignedat", "subarray"):
            self._raw_arg()  # column spec / number of columns
        self.ctx.append("env")
        try:
            rows = self.parse_cells()
        finally:
            self.ctx.pop()
        t = self.peek()
        if t.kind == "cmd" and t.value == "end":
            self.next()
            self._raw_arg()
        if env not in MATRIX_ENVS:
            env = "matrix" if any(len(r) > 1 for r in rows) else "gathered"
        return Matrix(rows, env)


def _parse_length(s: str) -> float:
    """A TeX length -> x-heights (approximate)."""
    s = s.strip()
    num = ""
    for ch in s:
        if ch.isdigit() or ch in ".-":
            num += ch
        else:
            break
    try:
        v = float(num) if num not in ("", "-", ".") else 1.0
    except ValueError:
        v = 1.0
    unit = s[len(num):].strip()[:2]
    k = {"em": EM, "ex": 1.0, "pt": 0.23, "mm": 0.65, "cm": 6.5, "in": 16.5, "mu": EM / 18, "px": 0.23}.get(unit, 0.23)
    return float(np.clip(v * k, -10.0, 30.0))


def _map_syms(node: Node, f) -> Node:
    if isinstance(node, Sym):
        return f(node)
    if isinstance(node, Row):
        return Row([_map_syms(x, f) for x in node.items])
    return node


def _close_of(node: Node) -> Tuple[Optional[Sym], Optional[Scripts]]:
    if isinstance(node, Sym) and node.ch in CLOSE_CHARS and node.cls == "close":
        return node, None
    if isinstance(node, Scripts) and isinstance(node.base, Sym) and node.base.ch in CLOSE_CHARS \
            and node.base.cls == "close":
        return node.base, node
    return None, None


def _bar_of(node: Node) -> Tuple[Optional[Sym], Optional[Scripts]]:
    if isinstance(node, Sym) and node.ch in ("|", "‖") and node.cls in ("ord", "open", "close"):
        return node, None
    if isinstance(node, Scripts) and isinstance(node.base, Sym) and node.base.ch in ("|", "‖") \
            and node.base.cls in ("ord", "close"):
        return node.base, node
    return None, None


def _pair_delims(items: List[Node]) -> List[Node]:
    """Turn matched opening/closing delimiters (and |x| bars) into ``Fenced``
    groups so they can be sized to their content, as one does by hand."""
    stack_items: List[List[Node]] = [[]]
    opens: List[Sym] = []
    for it in items:
        bar, bar_wrap = _bar_of(it)
        if bar is not None:
            if opens and opens[-1].ch == bar.ch and bar.cls != "open":
                op = opens.pop()
                inner = stack_items.pop()
                f: Node = Fenced(op.ch, Row(inner), bar.ch, auto=True)
                if bar_wrap is not None:
                    f = Scripts(f, bar_wrap.sup, bar_wrap.sub, bar_wrap.primes)
                stack_items[-1].append(f)
                continue
            prev = stack_items[-1][-1] if stack_items[-1] else None
            pc = _cls(prev) if prev is not None else None
            if bar_wrap is None and bar.cls != "close" and (pc in (None, "bin", "rel", "open", "punct", "op")
                                                           or bar.cls == "open"):
                opens.append(bar)
                stack_items.append([])
                continue
            stack_items[-1].append(it)
            continue
        if isinstance(it, Sym) and it.ch in OPEN_CHARS and it.cls == "open":
            opens.append(it)
            stack_items.append([])
            continue
        close, wrap = _close_of(it)
        if close is not None and opens and opens[-1].ch not in ("|", "‖"):
            op = opens.pop()
            inner = stack_items.pop()
            f = Fenced(op.ch, Row(inner), close.ch, auto=True)
            if wrap is not None:
                f = Scripts(f, wrap.sup, wrap.sub, wrap.primes)
            stack_items[-1].append(f)
            continue
        stack_items[-1].append(it)
    while opens:
        op = opens.pop()
        inner = stack_items.pop()
        stack_items[-1].extend([op] + inner)
    out = stack_items[0]
    # a lone bar between two terms ("2 | n") means "divides": relation spacing
    for k in range(1, len(out) - 1):
        it = out[k]
        if isinstance(it, Sym) and it.ch == "|" and it.cls == "ord" \
                and _cls(out[k - 1]) not in ("bin", "rel", "open", "punct", None) \
                and _cls(out[k + 1]) not in ("bin", "rel", "close", "punct", None):
            out[k] = Sym("∣", "rel")
    return out


def parse(src: str) -> Node:
    """Parse LaTeX maths source into an AST (never raises for bad input)."""
    try:
        return Parser(src).parse()
    except RecursionError:
        return Row([Text(src)])


# ---------------------------------------------------------------------------
# Atom classes, edges
# ---------------------------------------------------------------------------
def _cls(node: Optional[Node]) -> Optional[str]:
    if node is None:
        return None
    if isinstance(node, Sym):
        return node.cls
    if isinstance(node, Scripts):
        c = _cls(node.base)
        return c if c is not None else "ord"
    if isinstance(node, (BigOp,)):
        return "op"
    if isinstance(node, OpName):
        return node.cls
    if isinstance(node, (Frac, Matrix)):
        return "inner"
    if isinstance(node, Fenced):
        return "ord" if node.auto else "inner"
    if isinstance(node, BigDelim):
        return node.cls
    if isinstance(node, Middle):
        return "rel"
    if isinstance(node, (Stack,)):
        return _cls(node.base) or "ord"
    if isinstance(node, Negated):
        return _cls(node.body) or "ord"
    if isinstance(node, XArrow):
        return "rel"
    if isinstance(node, (Space, StyleSwitch)):
        return None
    if isinstance(node, Row):
        # a braced group is an ordinary atom; a single-item group keeps its item's class
        # only for spacing helpers like {=} (TeX: ord) -> ord
        return "ord"
    return "ord"


def _opens_with_delim(node: Optional[Node]) -> bool:
    """True for a fenced group such as ``(x)`` / ``\\left( .. \\right)`` (scripts allowed)."""
    if isinstance(node, Scripts):
        node = node.base
    return isinstance(node, Fenced) and node.left != "."


def _is_function_name(node: Optional[Node]) -> bool:
    if isinstance(node, Scripts):
        node = node.base
    return isinstance(node, OpName)


def _edge(node: Optional[Node], right: bool) -> str:
    if node is None:
        return ""
    if isinstance(node, Sym):
        return node.ch
    if isinstance(node, Row):
        items = [x for x in node.items if not isinstance(x, (Space, StyleSwitch))]
        if not items:
            return ""
        return _edge(items[-1] if right else items[0], right)
    if isinstance(node, Scripts):
        if right and (node.sup is not None or node.sub is not None or node.primes):
            return ""
        return _edge(node.base, right)
    if isinstance(node, Fenced):
        d = node.right if right else node.left
        return d if d != "." else _edge(node.body, right)
    if isinstance(node, Text):
        s = node.text.strip()
        return (s[-1] if right else s[0]) if s else ""
    if isinstance(node, OpName):
        return node.name[-1] if right else node.name[0]
    return ""


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _chaikin(p: np.ndarray, n: int = 2) -> np.ndarray:
    for _ in range(n):
        if len(p) < 3:
            return p
        q = 0.75 * p[:-1] + 0.25 * p[1:]
        r = 0.25 * p[:-1] + 0.75 * p[1:]
        mid = np.empty((2 * (len(p) - 1), 2))
        mid[0::2] = q
        mid[1::2] = r
        p = np.vstack([p[:1], mid, p[-1:]])
    return p


def _circle(cx: float, cy: float, r: float, n: int = 28, rx: Optional[float] = None) -> np.ndarray:
    t = np.linspace(0.15, 2 * np.pi + 0.35, n)  # overlap a little, as a pen does
    return np.column_stack([cx + (rx or r) * np.cos(t), cy + r * np.sin(t)])


def _tight(box: InkBox, pad_l: float = 0.0, pad_r: float = 0.0) -> InkBox:
    """Shift so the ink starts at ``pad_l`` and set width = ink width + pads."""
    b = _flat_bounds(box)
    if b is None:
        return box
    out = box.shifted(pad_l - b[0], 0.0)
    out.width = b[2] - b[0] + pad_l + pad_r
    out.refit(keep_width=True)
    return out


def _ink_points(box: InkBox, step: float = 0.6) -> Optional[np.ndarray]:
    """Points on the actual ink (curves flattened, centre lines densified and
    widened by the pen radius) -- tighter than the control-point bounds."""
    pts = []
    for st in box.fills:
        for sp in st:
            if len(sp):
                pts.append(flatten_subpath(sp, 6))
    for p, w in box.lines:
        if not len(p):
            continue
        if len(p) > 1:
            d = np.linalg.norm(np.diff(p, axis=0), axis=1)
            t = np.concatenate([[0.0], np.cumsum(d)])
            n = max(2, int(t[-1] / step) + 2)
            ts = np.linspace(0.0, t[-1], n)
            p = np.column_stack([np.interp(ts, t, p[:, 0]), np.interp(ts, t, p[:, 1])])
        r = w / 2
        pts.extend([p + np.array([r, 0.0]), p - np.array([r, 0.0]), p + np.array([0.0, r]), p - np.array([0.0, r])])
    if not pts:
        return None
    return np.vstack(pts)


def _flat_bounds(box: InkBox) -> Optional[Tuple[float, float, float, float]]:
    a = _ink_points(box)
    if a is None:
        return None
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())


def _ink_right(pts: Optional[np.ndarray], y0: float, y1: float) -> Optional[float]:
    """Rightmost ink x within the band y0 <= y <= y1 (y down)."""
    if pts is None:
        return None
    m = (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    if not m.any():
        return None
    return float(pts[m, 0].max())


A = np.array

# Procedural glyphs for symbols the fallback cannot draw (x-height units,
# y down, baseline 0, axis at -0.5).
def _proc_strokes(ch: str) -> Optional[List[np.ndarray]]:
    m = 0.5
    if ch == "⊥":
        return [A([[0.45, -1.35], [0.45, 0.0]]), A([[0.0, 0.0], [0.9, 0.0]])]
    if ch == "⊤":
        return [A([[0.45, -1.35], [0.45, 0.0]]), A([[0.0, -1.35], [0.9, -1.35]])]
    if ch == "∝":
        return [_chaikin(A([[0.95, -m - 0.3], [0.7, -m - 0.25], [0.5, -m], [0.3, -m + 0.25], [0.1, -m + 0.2],
                            [0.05, -m], [0.1, -m - 0.2], [0.3, -m - 0.25], [0.5, -m], [0.7, -m + 0.25],
                            [0.95, -m + 0.3]]), 2)]
    if ch in "≪≫":
        a, b = (0.45, 0.0) if ch == "≪" else (0.0, 0.45)
        return [A([[a + dx, -m - 0.35], [b + dx, -m], [a + dx, -m + 0.35]]) for dx in (0.0, 0.38)]
    if ch == "∋":
        return [_chaikin(A([[0.1, -m - 0.4], [0.6, -m - 0.4], [0.8, -m], [0.6, -m + 0.4], [0.1, -m + 0.4]]), 2),
                A([[0.2, -m], [0.8, -m]])]
    if ch == "↦":
        return [A([[0.0, -m - 0.22], [0.0, -m + 0.22]]), A([[0.0, -m], [1.2, -m]]),
                A([[0.92, -m - 0.24], [1.2, -m], [0.92, -m + 0.24]])]
    if ch == "∖":
        return [A([[0.0, -1.35], [0.7, 0.15]])]
    if ch in "⊕⊗⊙⊖":
        out = [_circle(0.45, -m, 0.42)]
        if ch == "⊕":
            out += [A([[0.05, -m], [0.85, -m]]), A([[0.45, -m - 0.4], [0.45, -m + 0.4]])]
        elif ch == "⊗":
            out += [A([[0.17, -m - 0.29], [0.73, -m + 0.29]]), A([[0.73, -m - 0.29], [0.17, -m + 0.29]])]
        elif ch == "⊖":
            out += [A([[0.05, -m], [0.85, -m]])]
        else:
            out += [A([[0.44, -m], [0.46, -m + 0.01]])]
        return out
    if ch == "⋮":
        return [A([[0.1, y], [0.11, y + 0.01]]) for y in (-0.05, -0.55, -1.05)]
    if ch == "⋱":
        return [A([[x, y], [x + 0.01, y + 0.01]]) for x, y in ((0.0, -1.05), (0.4, -0.55), (0.8, -0.05))]
    if ch == "≃":
        w = _chaikin(A([[0.05, -m - 0.15], [0.25, -m - 0.3], [0.45, -m - 0.15], [0.65, -m], [0.85, -m - 0.15]]))
        return [w, A([[0.05, -m + 0.2], [0.85, -m + 0.2]])]
    if ch == "≅":
        w = _chaikin(A([[0.05, -m - 0.3], [0.25, -m - 0.45], [0.45, -m - 0.3], [0.65, -m - 0.15], [0.85, -m - 0.3]]))
        return [w, A([[0.05, -m + 0.02], [0.85, -m + 0.02]]), A([[0.05, -m + 0.3], [0.85, -m + 0.3]])]
    if ch in "≺≻":
        p = _chaikin(A([[0.85, -m - 0.42], [0.5, -m - 0.12], [0.08, -m], [0.5, -m + 0.12], [0.85, -m + 0.42]]), 2)
        if ch == "≻":
            p = A([[0.93, 0.0]]) + p * A([-1.0, 1.0])
        return [p]
    if ch in "⪯⪰":
        p = _proc_strokes("≺" if ch == "⪯" else "≻")[0] - A([0.0, 0.15])
        return [p, A([[0.1, -m + 0.5], [0.85, -m + 0.45]])]
    if ch in "⊢⊣⊨":
        if ch == "⊣":
            return [A([[0.8, -1.3], [0.8, 0.0]]), A([[0.0, -0.65], [0.8, -0.65]])]
        out = [A([[0.1, -1.3], [0.1, 0.0]]), A([[0.1, -0.65], [0.85, -0.65]])]
        if ch == "⊨":
            out = [A([[0.1, -1.3], [0.1, 0.0]]), A([[0.1, -0.85], [0.85, -0.85]]), A([[0.1, -0.45], [0.85, -0.45]])]
        return out
    if ch in "⊊⊋":
        base = [_chaikin(A([[0.8, -m - 0.45], [0.3, -m - 0.45], [0.1, -m - 0.05], [0.3, -m + 0.35], [0.8, -m + 0.35]]), 2)]
        if ch == "⊋":
            base = [A([[0.9, 0.0]]) + base[0] * A([-1.0, 1.0])]
        return base + [A([[0.1, -m + 0.6], [0.85, -m + 0.6]]), A([[0.55, -m + 0.45], [0.4, -m + 0.78]])]
    if ch == "∠":
        return [A([[0.9, -1.0], [0.0, 0.0], [0.95, 0.0]])]
    if ch == "△":
        return [A([[0.0, 0.0], [0.45, -1.2], [0.9, 0.0], [0.0, 0.0]])]
    if ch == "□":
        return [A([[0.0, 0.0], [0.0, -1.1], [1.0, -1.1], [1.0, 0.0], [0.0, 0.0]])]
    if ch == "√":
        return [A([[0.0, -0.45], [0.15, -0.55], [0.4, 0.05], [0.75, -1.5]])]
    if ch == "↑":
        return [A([[0.3, 0.3], [0.3, -1.4]]), A([[0.05, -1.1], [0.3, -1.4], [0.55, -1.1]])]
    if ch == "↓":
        return [A([[0.3, -1.4], [0.3, 0.3]]), A([[0.05, 0.0], [0.3, 0.3], [0.55, 0.0]])]
    if ch == "≔":
        return [A([[0.1, -m - 0.18], [0.11, -m - 0.17]]), A([[0.1, -m + 0.18], [0.11, -m + 0.19]]),
                A([[0.35, -m - 0.17], [1.05, -m - 0.17]]), A([[0.35, -m + 0.17], [1.05, -m + 0.17]])]
    return None


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
@dataclass
class Ctx:
    scale: float = 1.0
    display: bool = False
    depth: int = 0  # script depth

    def script(self, k: float = SCRIPT_SCALE) -> "Ctx":
        return Ctx(max(MIN_SCALE, self.scale * k), False, self.depth + 1)

    def with_(self, scale: Optional[float] = None, display: Optional[bool] = None) -> "Ctx":
        return Ctx(self.scale if scale is None else max(MIN_SCALE, scale),
                   self.display if display is None else display, self.depth)


class MathLayout:
    def __init__(self, synth, size: float = 1.0):
        self.s = synth
        self.size = float(size)
        self.st = synth.st
        self.rng = synth.rng
        self.var = float(getattr(synth, "var", 1.0))
        self.slant = float(np.clip(getattr(self.st, "slant", 0.0), -0.4, 0.5))
        self._has_cache: Dict[str, bool] = {}

    # ------------------------------------------------------------ basics
    def u(self, ctx: Ctx) -> float:
        """Current x-height in points."""
        return self.s.xh * self.size * ctx.scale

    def pen(self, ctx: Ctx) -> float:
        return float(self.st.pen_width) * self.u(ctx)

    def jit(self, sd: float) -> float:
        return float(self.rng.normal(0.0, sd * self.var)) if self.var > 0 else 0.0

    def has_glyph(self, ch: str) -> bool:
        r = self._has_cache.get(ch)
        if r is None:
            from .synth import ANALOGS

            bank = self.s.style.bank
            r = bool(bank.get(ch)) or any(bank.get(sub) for sub, _ in ANALOGS.get(ch, [])) \
                or self.s.fallback.strokes(ch) is not None
            self._has_cache[ch] = r
        return r

    def glyph(self, ch: str, ctx: Ctx) -> InkBox:
        return self.s.glyph_box(ch, self.size * ctx.scale)

    def stroke(self, pts: np.ndarray, ctx: Ctx, slant: bool = True, smooth: int = 0, wobble: float = 0.012,
               pen: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """A hand-drawn centre-line stroke (points in pt, y down)."""
        p = np.asarray(pts, float)
        if len(p) == 1:
            p = np.vstack([p, p + np.array([0.01, 0.01])])
        if slant and self.slant:
            p = np.column_stack([p[:, 0] - self.slant * p[:, 1], p[:, 1]])
        if smooth and len(p) >= 3:
            p = _chaikin(p, smooth)
        if wobble and len(p) >= 2 and self.var > 0:
            u = self.u(ctx)
            n = len(p)
            while n < 6:
                # densify (insert segment midpoints: corners are kept) so the wobble can bend the line
                mid = 0.5 * (p[:-1] + p[1:])
                q = np.empty((2 * n - 1, 2))
                q[0::2] = p
                q[1::2] = mid
                p = q
                n = len(p)
            t = np.linspace(0, 1, n)
            ph = self.rng.uniform(0, 2 * np.pi, size=2)
            amp = self.rng.normal(0, wobble * u * self.var, size=2)
            p = p + np.column_stack([amp[0] * np.sin(2 * np.pi * t + ph[0]), amp[1] * np.sin(np.pi * t + ph[1])])
        return p, (self.pen(ctx) if pen is None else pen)

    def proc_box(self, strokes: List[np.ndarray], ctx: Ctx, pen_k: float = 0.95, slant: bool = True) -> InkBox:
        """Procedural glyph (x-height units) styled like the fallback glyphs."""
        u = self.u(ctx) * math.exp(self.jit(self.st.glyph_scale.get("sd", 0.05)))
        lines = []
        for s in strokes:
            p = np.asarray(s, float) * u
            lines.append(self.stroke(p, ctx, slant=slant, smooth=0, wobble=0.01, pen=pen_k * self.pen(ctx)))
        box = InkBox(lines=lines)
        return _tight(box)

    def hline(self, x0: float, x1: float, y: float, ctx: Ctx, pen: Optional[float] = None,
              wob: float = 1.0) -> Tuple[np.ndarray, float]:
        """A slightly wobbly, hand-drawn horizontal line (fraction bar, overline)."""
        u = self.u(ctx)
        L = max(x1 - x0, 0.05 * u)
        n = max(5, int(L / (0.7 * u)) + 3)
        t = np.linspace(0.0, 1.0, n)
        xs = x0 + t * L
        tilt = self.jit(0.012) * L * wob
        arc = self.jit(0.035) * u * wob
        noise = self.rng.normal(0.0, 0.012 * u * self.var * wob, n) if self.var > 0 else np.zeros(n)
        ys = y + tilt * (t - 0.5) + arc * 4 * t * (1 - t) + noise
        p = _chaikin(np.column_stack([xs, ys]), 2)
        return p, (self.pen(ctx) if pen is None else pen)

    # ------------------------------------------------------------ dispatch
    def lay(self, node: Optional[Node], ctx: Ctx) -> InkBox:
        if node is None:
            return InkBox()
        if isinstance(node, Row):
            return self.lay_row(node.items, ctx)
        if isinstance(node, Sym):
            return self.lay_sym(node, ctx)
        if isinstance(node, Scripts):
            return self.lay_scripts(node, ctx)
        if isinstance(node, Frac):
            return self.lay_frac(node, ctx)
        if isinstance(node, Sqrt):
            return self.lay_sqrt(node, ctx)
        if isinstance(node, Fenced):
            return self.lay_fenced(node, ctx)
        if isinstance(node, BigOp):
            return self.lay_bigop(node, ctx)
        if isinstance(node, OpName):
            return self.lay_text(node.name, ctx)
        if isinstance(node, Text):
            return self.lay_text(node.text, ctx)
        if isinstance(node, Space):
            return InkBox(width=max(0.0, node.xh * self.u(ctx)))
        if isinstance(node, Accent):
            return self.lay_accent(node, ctx)
        if isinstance(node, Matrix):
            return self.lay_matrix(node, ctx)
        if isinstance(node, Stack):
            return self.lay_limits(self.lay(node.base, ctx), node.over, node.under, ctx, kind="stack")
        if isinstance(node, BigDelim):
            u = self.u(ctx)
            H = (DELIM_TOP + DELIM_BOT) * u * node.factor
            c = 0.5 * (DELIM_TOP - DELIM_BOT) * u  # centre of a normal delimiter
            return self.delim(node.ch, c + H / 2, H / 2 - c, ctx)
        if isinstance(node, Middle):
            return self.delim(node.ch, DELIM_TOP * self.u(ctx), DELIM_BOT * self.u(ctx), ctx)
        if isinstance(node, XArrow):
            return self.lay_xarrow(node, ctx)
        if isinstance(node, Boxed):
            return self.lay_boxed(node, ctx)
        if isinstance(node, Phantom):
            b = self.lay(node.body, ctx)
            return InkBox(width=b.width if node.h else 0.0, ascent=b.ascent if node.v else 0.0,
                          descent=b.descent if node.v else 0.0)
        if isinstance(node, Negated):
            return self.lay_negated(node, ctx)
        return InkBox()

    # ----------------------------------------------------------------- rows
    def gap(self, lc: str, rc: str, ln: Node, rn: Node, ctx: Ctx) -> float:
        """Inter-atom space in points (TeX-like classes, handwriting gaps)."""
        u = self.u(ctx)
        sc = 1.0 if ctx.depth == 0 else 0.55
        extra = 0.0
        if isinstance(ln, Scripts) and (ln.sup is not None or ln.sub is not None):
            extra = 0.13 * u * sc  # small scripts look cramped against the next symbol
        return self._gap(lc, rc, ln, rn, ctx, u, sc) + extra

    def _gap(self, lc: str, rc: str, ln: Node, rn: Node, ctx: Ctx, u: float, sc: float) -> float:
        j = 1.0 + self.jit(0.12)
        if lc == "rel" and rc == "rel":
            return 0.12 * u * j
        if lc == "rel" or rc == "rel":
            return REL_GAP * u * sc * j
        if lc == "bin" or rc == "bin":
            return BIN_GAP * u * sc * j
        if lc == "punct":
            return PUNCT_GAP * u * sc * j
        if rc == "punct":
            return max(0.0, self.s.letter_gap(_edge(ln, True), ",")) * u * 0.8
        if lc == "op" and (rc == "open" or _opens_with_delim(rn)) and _is_function_name(ln):
            return 0.08 * u * j  # sin(x), max(a, b)
        if lc == "op" or rc == "op":
            return OP_GAP * u * j * (0.7 if ctx.depth else 1.0)
        if lc == "inner" or rc == "inner":
            if lc == "open" or rc == "close":
                return 0.1 * u * j
            return INNER_GAP * u * sc * j
        a, b = _edge(ln, True), _edge(rn, False)
        g = self.s.letter_gap(a, b)
        return float(np.clip(g, 0.02, 0.45)) * u

    def lay_row(self, items: Sequence[Node], ctx: Ctx) -> InkBox:
        # effective classes (binary operators become unary at the start / after operators)
        nodes = [n for n in items]
        classes = [_cls(n) for n in nodes]
        prev = None
        for k, (n, c) in enumerate(zip(nodes, classes)):
            if c is None:
                continue
            if c == "bin":
                nxt = next((classes[m] for m in range(k + 1, len(nodes)) if classes[m] is not None), None)
                if prev in (None, "bin", "rel", "open", "punct", "op") or nxt in (None, "rel", "close", "punct"):
                    classes[k] = "ord"
            prev = classes[k]
        out = InkBox()
        x = 0.0
        pending = 0.0
        last_node: Optional[Node] = None
        last_cls: Optional[str] = None
        for k, (n, c) in enumerate(zip(nodes, classes)):
            if isinstance(n, StyleSwitch):
                if n.style == "display":
                    ctx = ctx.with_(display=True)
                elif n.style == "text":
                    ctx = ctx.with_(display=False)
                elif n.style == "script":
                    ctx = Ctx(max(MIN_SCALE, ctx.scale * SCRIPT_SCALE), False, ctx.depth + 1)
                else:
                    ctx = Ctx(max(MIN_SCALE, ctx.scale * SCRIPT_SCALE ** 2), False, ctx.depth + 2)
                continue
            if isinstance(n, Space):
                pending += n.xh * self.u(ctx)
                continue
            if isinstance(n, Sym) and n.smart and n.ch == "…":
                nxt = next((classes[m] for m in range(k + 1, len(nodes)) if classes[m] is not None), None)
                if last_cls in ("bin", "rel") or nxt in ("bin", "rel"):
                    n = Sym("⋯", n.cls)
            box = self.lay(n, ctx)
            if box.width <= 0 and box.is_empty():
                continue  # empty group ({}): no ink, no extra gap
            if last_node is not None:
                x += self.gap(last_cls or "ord", c or "ord", last_node, n, ctx)
            x += pending
            pending = 0.0
            out.add(box, x, 0.0)
            x += box.width
            last_node, last_cls = n, c
        x += max(0.0, pending)
        out.width = max(0.0, x)
        return out

    # -------------------------------------------------------------- symbols
    def lay_sym(self, node: Sym, ctx: Ctx) -> InkBox:
        ch = ALIAS.get(node.ch, node.ch)
        u = self.u(ctx)
        if ch in DELIM_CHARS or ch in BAR_CHARS or ch in ("/", "\\"):
            return self.delim(ch, DELIM_TOP * u * (1 + self.jit(0.03)), DELIM_BOT * u * (1 + self.jit(0.05)), ctx)
        if ch in ("-",):
            return self.minus(ctx)
        if ch in ("…", "⋯"):
            return self.dots(ctx, axis=(ch == "⋯"))
        if ch == "·":
            return self.dots(ctx, axis=True, n=1)
        if ch == ":":
            return self.colon(ctx)
        if ch == ";":
            return self.colon(ctx, semi=True)
        if ch == "′":
            return self.glyph("′", ctx)
        if ch in WIDE:
            g = self.glyph(WIDE[ch], ctx)
            return _tight(g.transformed(np.array([[1.4, 0.0, 0.0], [0.0, 1.0, 0.0]]), width_scale=1.0))
        if ch in NEGATED:
            return self.lay_negated(Negated(Sym(NEGATED[ch], node.cls)), ctx)
        if ch == "×":
            # the letter-x analogue would read as a variable: draw a small cross on the axis
            cross = [A([[0.0, -AXIS - 0.3], [0.6, -AXIS + 0.3]]), A([[0.6, -AXIS - 0.3], [0.0, -AXIS + 0.3]])]
            return self.proc_box(cross, ctx, slant=False)
        if ch in RECENTER:
            g = self.glyph(ch, ctx)
            if ch in ("~", "*") and not g.is_empty():
                return self.center_on_axis(g, ctx, axis=AXIS + (0.12 if ch == "*" else 0.0))
            return g
        if self.has_glyph(ch):
            return self.glyph(ch, ctx)
        p = _proc_strokes(ch)
        if p is not None:
            return self.proc_box(p, ctx)
        return self.glyph(ch, ctx)  # the synthesiser records it as missing

    def center_on_axis(self, g: InkBox, ctx: Ctx, axis: float = AXIS) -> InkBox:
        b = g.ink_bounds()
        if b is None:
            return g
        cy = 0.5 * (b[1] + b[3])
        out = g.shifted(0.0, -axis * self.u(ctx) - cy)
        out.refit(keep_width=True)
        return out

    def minus(self, ctx: Ctx) -> InkBox:
        u = self.u(ctx)
        g = self.glyph("-", ctx)
        b = g.ink_bounds()
        target = 0.72 * u * (1 + self.jit(0.06))
        if b is None or b[2] - b[0] < 0.05 * u:
            pts, w = self.hline(0.0, target, -AXIS * u, ctx, wob=0.5)
            return _tight(InkBox(lines=[(pts, w)]))
        h = b[3] - b[1]
        kx = target / max(b[2] - b[0], 1e-3)
        ky = 1.0 if h < 0.35 * u else 0.35 * u / h  # a hyphen drawn as a long dash stays thin
        cy = 0.5 * (b[1] + b[3])
        ty = -(AXIS + 0.03) * u
        m = np.array([[kx, 0.0, -b[0] * kx], [0.0, ky, ty - cy * ky]])
        return _tight(g.transformed(m, width_scale=1.0))

    def dots(self, ctx: Ctx, axis: bool = False, n: int = 3) -> InkBox:
        u = self.u(ctx)
        out = InkBox()
        x = 0.0
        for k in range(n):
            d = self.glyph(".", ctx) if self.has_glyph(".") else self.proc_box([A([[0.0, 0.0], [0.02, 0.0]])], ctx)
            if axis:
                d = self.center_on_axis(d, ctx)
            d = _tight(d)
            out.add(d, x, self.jit(0.02) * u)
            x += d.width + (0.28 + self.jit(0.04)) * u
        out.width = x - (0.28 * u if n else 0.0)
        out.refit(keep_width=True)
        return out

    def colon(self, ctx: Ctx, semi: bool = False) -> InkBox:
        u = self.u(ctx)
        top = self.glyph(".", ctx) if self.has_glyph(".") else self.proc_box([A([[0.0, 0.0], [0.02, 0.0]])], ctx)
        top = _tight(top)
        if semi:
            low = _tight(self.glyph(",", ctx))
        else:
            low = _tight(self.glyph(".", ctx) if self.has_glyph(".") else top)
        out = InkBox()
        out.add(low, 0.0, 0.0)
        tb = top.ink_bounds()
        cy = 0.5 * (tb[1] + tb[3]) if tb else 0.0
        out.add(top, 0.04 * u + self.slant * 0.6 * u, -0.62 * u - cy)
        out.refit(keep_width=False)
        return _tight(out)

    # ----------------------------------------------------------- delimiters
    def delim(self, ch: str, top: float, bot: float, ctx: Ctx) -> InkBox:
        """A delimiter whose ink spans ``top`` above to ``bot`` below the baseline (pt)."""
        u = self.u(ctx)
        if ch == ".":
            return InkBox(width=0.1 * u)
        H = top + bot
        if ch in ("‖", "∥"):
            a = self.delim("|", top, bot, ctx)
            b = self.delim("|", top, bot, ctx)
            out = InkBox()
            out.add(a, 0.0, 0.0)
            out.add(b, a.width + 0.12 * u, 0.0)
            out.width = a.width + 0.12 * u + b.width
            return out
        real = ch in ("(", ")", "|", "∣") and self.has_glyph(ch if ch != "∣" else "|")
        if real and ch in "()" and not self.s.style.bank.get(ch):
            real = False
        if real:
            g = self.glyph("|" if ch == "∣" else ch, ctx)
            b = _flat_bounds(g)
            if b is not None and b[3] - b[1] > 0.2 * u:
                y0, y1 = b[1], b[3]
                ky = H / (y1 - y0)
                kx = float(np.clip((H / ((DELIM_TOP + DELIM_BOT) * u)) ** 0.3, 0.85, 1.5))
                if ch in "|∣":
                    kx = 1.0
                s = self.slant
                m = np.array([[kx, s * (kx - ky), s * (ky * y0 + top)], [0.0, ky, -ky * y0 - top]])
                return _tight(g.transformed(m, width_scale=1.0))
        pts = self._delim_strokes(ch, top, bot, u)
        if pts is None:
            g = self.glyph(ch, ctx)
            b = g.ink_bounds()
            if b is None:
                return g
            ky = H / max(b[3] - b[1], 1e-3)
            m = np.array([[1.0, 0.0, -b[0]], [0.0, ky, -top - b[1] * ky]])
            return _tight(g.transformed(m, width_scale=1.0))
        lines = [self.stroke(p, ctx, slant=True, smooth=sm, wobble=0.008) for p, sm in pts]
        return _tight(InkBox(lines=lines))

    def _delim_strokes(self, ch: str, top: float, bot: float, u: float):
        H = top + bot
        y0, y1 = -top, bot
        cy = 0.5 * (y0 + y1)
        if ch in "()":
            w = min(0.3 * u + 0.08 * H, 1.0 * u)
            p = A([[w, y0], [0.15 * w, y0 + 0.3 * H], [0.0, cy], [0.15 * w, y1 - 0.3 * H], [w, y1]])
            if ch == ")":
                p = A([w, 0.0]) + p * A([-1.0, 1.0])
            return [(p, 3)]
        if ch in "[]":
            w = min(0.28 * u + 0.04 * H, 0.7 * u)
            p = A([[w, y0], [0.0, y0], [0.0, y1], [w, y1]])
            if ch == "]":
                p = A([w, 0.0]) + p * A([-1.0, 1.0])
            return [(p, 0)]
        if ch in "{}":
            w = min(0.32 * u + 0.05 * H, 0.8 * u)
            hw = 0.5 * w
            r = min(0.22 * H, 0.35 * u)
            upper = A([[w, y0], [hw + 0.05 * w, y0 + 0.25 * r], [hw, y0 + r], [hw, cy - r], [hw - 0.1 * w, cy - 0.3 * r],
                       [0.0, cy]])
            lower = A([[0.0, cy], [hw - 0.1 * w, cy + 0.3 * r], [hw, cy + r], [hw, y1 - r], [hw + 0.05 * w, y1 - 0.25 * r],
                       [w, y1]])
            if ch == "}":
                upper = A([w, 0.0]) + upper * A([-1.0, 1.0])
                lower = A([w, 0.0]) + lower * A([-1.0, 1.0])
            return [(upper, 2), (lower, 2)]
        if ch in "⟨⟩":
            w = min(0.25 * u + 0.12 * H, 0.75 * u)
            p = A([[w, y0], [0.0, cy], [w, y1]])
            if ch == "⟩":
                p = A([w, 0.0]) + p * A([-1.0, 1.0])
            return [(p, 0)]
        if ch in "⌊⌋⌈⌉":
            w = min(0.3 * u + 0.03 * H, 0.6 * u)
            if ch == "⌊":
                p = A([[0.0, y0], [0.0, y1], [w, y1]])
            elif ch == "⌋":
                p = A([[0.0, y1], [w, y1], [w, y0]])
            elif ch == "⌈":
                p = A([[w, y0], [0.0, y0], [0.0, y1]])
            else:
                p = A([[0.0, y0], [w, y0], [w, y1]])
            return [(p, 0)]
        if ch in ("|", "∣"):
            return [(A([[0.0, y0], [0.0, y1]]), 0)]
        if ch in ("/", "\\"):
            # the writer's slant is added by stroke(); aim for ~0.3 overall lean
            w = max(0.05, 0.32 - self.slant) * H if ch == "/" else (0.3 + self.slant) * H
            p = A([[w, y0], [0.0, y1]]) if ch == "/" else A([[0.0, y0], [w, y1]])
            return [(p, 0)]
        if ch in ("↑", "↓"):
            p = A([[0.3 * u, y1], [0.3 * u, y0]])
            head = A([[0.05 * u, y0 + 0.3 * u], [0.3 * u, y0], [0.55 * u, y0 + 0.3 * u]])
            if ch == "↓":
                p = p[::-1]
                head = A([[0.05 * u, y1 - 0.3 * u], [0.3 * u, y1], [0.55 * u, y1 - 0.3 * u]])
            return [(p, 0), (head, 0)]
        return None

    def fence_extent(self, asc: float, desc: float, u: float) -> Tuple[float, float]:
        """Delimiter ink extents covering content with ascent/descent (pt)."""
        pad = 0.12 * u
        top = max(DELIM_TOP * u, asc + pad)
        bot = max(DELIM_BOT * u, desc + pad)
        if top > DELIM_TOP * u * 1.15 or bot > DELIM_BOT * u * 1.6:
            # a big delimiter is drawn roughly symmetric about the maths axis
            ax = AXIS * u
            half = max(top - ax, bot + ax)
            top = max(top, ax + 0.85 * half)
            bot = max(bot, 0.85 * half - ax)
        return top, bot

    def lay_fenced(self, node: Fenced, ctx: Ctx) -> InkBox:
        u = self.u(ctx)
        segs: List[List[Node]] = [[]]
        mids: List[str] = []
        for it in node.body.items:
            if isinstance(it, Middle):
                mids.append(it.ch)
                segs.append([])
            else:
                segs[-1].append(it)
        boxes = [self.lay_row(s, ctx) for s in segs]
        asc = max([b.ascent for b in boxes] + [0.0])
        desc = max([b.descent for b in boxes] + [0.0])
        top, bot = self.fence_extent(asc, desc, u)
        out = InkBox()
        x = 0.0
        if node.left != ".":
            lb = self.delim(node.left, top, bot, ctx)
            out.add(lb, 0.0, 0.0)
            x = lb.width
        for k, b in enumerate(boxes):
            if k > 0:
                x += 0.25 * u
                mb = self.delim(mids[k - 1], top, bot, ctx)
                out.add(mb, x, 0.0)
                x += mb.width + 0.25 * u
            if not b.is_empty() or b.width > 0:
                lead = (0.0 if (k == 0 and node.left == ".") else self._fence_gap(u, top, bot))
                x += lead
                out.add(b, x, 0.0)
                x += b.width
        if node.right != ".":
            x += self._fence_gap(u, top, bot)
            rb = self.delim(node.right, top, bot, ctx)
            out.add(rb, x, 0.0)
            x += rb.width
        out.width = x
        return out

    def _fence_gap(self, u: float, top: float, bot: float) -> float:
        big = (top + bot) / ((DELIM_TOP + DELIM_BOT) * u)
        return (0.07 + 0.05 * min(big - 1.0, 2.0) + self.jit(0.025)) * u

    # ------------------------------------------------------------- scripts
    def _limits_mode(self, base: Node, ctx: Ctx) -> bool:
        if isinstance(base, BigOp):
            if base.limits is not None:
                return base.limits
            return ctx.display and base.ch not in INTEGRALS
        if isinstance(base, OpName):
            if base.force is not None:
                return base.force
            return base.limits and ctx.display
        if isinstance(base, Accent) and base.kind in ("underbrace", "overbrace"):
            return True
        if isinstance(base, Stack):
            return False
        return False

    def lay_scripts(self, node: Scripts, ctx: Ctx) -> InkBox:
        if self._limits_mode(node.base, ctx) and node.primes == 0:
            bb = self.lay(node.base, ctx)
            return self.lay_limits(bb, node.sup, node.sub, ctx,
                                   kind="op" if isinstance(node.base, (BigOp, OpName)) else "brace")
        u = self.u(ctx)
        bb = self.lay(node.base, ctx)
        sctx = ctx.script()
        out = InkBox()
        out.add(bb, 0.0, 0.0)
        empty_base = bb.is_empty()
        b_asc = bb.ascent if not empty_base else 1.0 * u
        b_desc = bb.descent if not empty_base else 0.0
        x = bb.width
        is_int = isinstance(node.base, BigOp) and node.base.ch in INTEGRALS
        is_op = isinstance(node.base, (BigOp,))
        # primes sit just right of the base, near its top
        if node.primes:
            x += 0.05 * u
            for _ in range(node.primes):
                pb = _tight(self.glyph("′", ctx))
                want_top = max(b_asc, 1.05 * u) + 0.05 * u
                pb_shift = want_top - pb.ascent
                out.add(pb, x, -pb_shift)
                x += pb.width + 0.06 * u
        sup_b = self.lay(node.sup, sctx) if node.sup is not None else None
        sub_b = self.lay(node.sub, sctx) if node.sub is not None else None
        up = dn = 0.0
        if sup_b is not None:
            if is_op:
                up = b_asc - 0.75 * sup_b.ascent
            else:
                up = max(0.78 * u, b_asc - 0.52 * u)
            up = max(up, 0.42 * u + sup_b.descent)
        if sub_b is not None:
            if is_op:
                dn = max(0.3 * u, b_desc - 0.35 * sub_b.ascent)
            else:
                dn = max(0.3 * u, b_desc - 0.25 * u)
            dn = max(dn, sub_b.ascent - 0.78 * u)
        if sup_b is not None and sub_b is not None:
            clear = (up - sup_b.descent) - (sub_b.ascent - dn)
            need = 0.2 * u
            if clear < need:
                d = need - clear
                up += 0.45 * d
                dn += 0.55 * d
        up += self.jit(0.03) * u
        dn += self.jit(0.03) * u
        # place each script just right of the base's ink at the script's height
        # (a slanted letter leans right at the top and left at the bottom)
        pts = _ink_points(bb) if not empty_base else None
        m = 0.12 * u  # vertical margin: thick strokes just outside the band still count
        g_op = 0.16 * u if is_int else 0.12 * u
        if sup_b is not None:
            r = _ink_right(pts, -up - sup_b.ascent - m, -up + sup_b.descent + m)
            if node.primes or r is None:
                sx = x + 0.03 * u
            else:
                sx = r + (g_op if is_op else 0.07 * u)
                if not is_op:
                    sx = float(np.clip(sx, bb.width - 0.25 * u, bb.width + 0.12 * u))
            out.add(sup_b, sx + self.jit(0.015) * u, -up)
            x = max(x, sx + sup_b.width)
        if sub_b is not None:
            r = _ink_right(pts, dn - sub_b.ascent - m, dn + sub_b.descent + (m if is_op else 0.0))
            if empty_base:
                sx = 0.0
            elif r is None:
                sx = bb.width + 0.02 * u
            else:
                sx = r + (g_op if is_op else 0.06 * u)
                if is_op and not is_int:
                    sx = max(sx, bb.width - 0.12 * u)  # upright-looking limits for \sum, \prod
                if not is_op:
                    sx = float(np.clip(sx, bb.width - 0.3 * u, bb.width + 0.1 * u))
            sx = max(sx, 0.0)
            out.add(sub_b, sx + self.jit(0.015) * u, dn)
            x = max(x, sx + sub_b.width)
        out.width = max(out.width, x)
        return out

    def lay_limits(self, bb: InkBox, over: Optional[Node], under: Optional[Node], ctx: Ctx,
                   kind: str = "op") -> InkBox:
        u = self.u(ctx)
        sctx = ctx.script()
        ob = self.lay(over, sctx) if over is not None else None
        ub = self.lay(under, sctx) if under is not None else None
        w = max([bb.width] + [b.width for b in (ob, ub) if b is not None])
        out = InkBox()
        bx = 0.5 * (w - bb.width)
        out.add(bb, bx, 0.0)
        gap_o = (0.2 if kind != "stack" else 0.12) * u
        gap_u = (0.22 if kind != "stack" else 0.12) * u
        if ob is not None:
            # slanted writing: the top of the base is further right
            dx = self.slant * (bb.ascent * 0.5) if kind == "op" else 0.0
            out.add(ob, 0.5 * (w - ob.width) + dx + self.jit(0.03) * u, -(bb.ascent + gap_o + ob.descent))
        if ub is not None:
            dx = -self.slant * (bb.descent * 0.25) if kind == "op" else 0.0
            out.add(ub, 0.5 * (w - ub.width) + dx + self.jit(0.03) * u, bb.descent + gap_u + ub.ascent)
        out.width = w
        b = out.ink_bounds()
        if b is not None and b[0] < 0:
            out = out.shifted(-b[0], 0.0)
            out.width = w - b[0]
        return out

    # ------------------------------------------------------------ fractions
    def lay_frac(self, node: Frac, ctx: Ctx) -> InkBox:
        u = self.u(ctx)
        if node.style == "d":
            k = 1.0 if ctx.depth == 0 else FRAC_SCALE_INLINE
        elif node.style == "t":
            k = FRAC_SCALE_INLINE
        else:
            k = 1.0 if ctx.display else FRAC_SCALE_INLINE
        cctx = Ctx(max(MIN_SCALE, ctx.scale * k), False, ctx.depth + (0 if k >= 1.0 else 1))
        nb = self.lay(node.num, cctx)
        db = self.lay(node.den, cctx)
        pen = self.pen(ctx)
        axis = AXIS * u
        if node.bar:
            gn = (0.24 if ctx.display else 0.2) * u
            gd = (0.28 if ctx.display else 0.22) * u
        else:
            gn = gd = 0.12 * u
        over = (0.14 + abs(self.jit(0.04))) * u if node.bar else 0.0
        w = max(nb.width, db.width) + 2 * over
        pad = 0.08 * u if node.bar else 0.0
        out = InkBox()
        nx = pad + 0.5 * (w - nb.width) + self.jit(0.04) * u
        dx = pad + 0.5 * (w - db.width) + self.jit(0.04) * u
        ny = -axis - pen / 2 - gn - nb.descent
        dy = -axis + pen / 2 + gd + db.ascent
        out.add(nb, nx, ny)
        out.add(db, dx, dy)
        if node.bar:
            x0 = pad + self.jit(0.03) * u
            x1 = pad + w + self.jit(0.03) * u
            out.lines.append(self.hline(x0, x1, -axis, ctx, pen=pen))
        out.width = w + 2 * pad
        out.refit(keep_width=True)
        return out

    # ---------------------------------------------------------------- roots
    def lay_sqrt(self, node: Sqrt, ctx: Ctx) -> InkBox:
        u = self.u(ctx)
        pen = self.pen(ctx)
        bb = self.lay(node.body, ctx)
        asc = max(bb.ascent, 1.0 * u)
        desc = bb.descent
        top = -(asc + 0.22 * u + pen / 2)
        bot = max(desc * 0.85, 0.0) + 0.06 * u
        H = bot - top
        hk = min(H, 2.4 * u) / (1.5 * u)  # size of the entry tick relative to a normal root
        hk = float(np.clip(hk, 0.8, 1.35))
        a = (0.0, bot - 0.42 * u * hk)
        b = (0.16 * u * hk, bot - 0.55 * u * hk + self.jit(0.03) * u)
        c = (0.42 * u * hk, bot)
        d = (c[0] + 0.22 * u + 0.12 * H, top)
        body_x = d[0] + 0.14 * u
        e = (body_x + bb.width + 0.12 * u + abs(self.jit(0.04)) * u, top + self.jit(0.03) * u)
        shift = 0.0
        ib = None
        if node.index is not None:
            ictx = ctx.script(0.55)
            ib = self.lay(node.index, ictx)
            want_right = c[0] - 0.08 * u
            shift = max(0.0, ib.width - want_right + 0.05 * u)
        pts = A([a, b, c, d])
        rad, _ = self.stroke(pts + A([shift, 0.0]), ctx, slant=False, smooth=0, wobble=0.006, pen=pen)
        bar, _ = self.hline(d[0] + shift, e[0] + shift, top, ctx, pen=pen, wob=0.6)
        # one continuous pen movement: radical then overbar
        line = np.vstack([rad, bar[1:]])
        out = InkBox(lines=[(line, pen)])
        out.add(bb, body_x + shift, 0.0)
        if ib is not None:
            iy = b[1] - 0.12 * u - ib.descent
            out.add(ib, max(0.0, shift + c[0] - 0.08 * u - ib.width), iy)
        out.width = e[0] + shift + pen / 2
        out.refit(keep_width=True)
        return out

    # ------------------------------------------------------------ big ops
    def lay_bigop(self, node: BigOp, ctx: Ctx) -> InkBox:
        u = self.u(ctx)
        ch = node.ch
        is_int = ch in INTEGRALS
        if is_int:
            k = 1.15 if ctx.display else 0.82
        else:
            k = 1.45 if ctx.display else 0.95
        octx = ctx.with_(scale=ctx.scale * k)
        if ch in ("∑", "∏"):
            g = self.glyph(ch, octx)
        elif ch == "∐":
            g = self.glyph("∏", octx)
            b = g.ink_bounds()
            if b is not None:
                g = g.transformed(np.array([[1.0, 0.0, 0.0], [0.0, -1.0, b[1] + b[3]]]), width_scale=1.0)
        elif is_int:
            n = {"∫": 1, "∬": 2, "∭": 3, "∮": 1}[ch]
            g = InkBox()
            x = 0.0
            for _ in range(n):
                gi = _tight(self.glyph("∫", octx))
                g.add(gi, x, 0.0)
                x += 0.42 * gi.width
            g.width = x + 0.58 * gi.width
            if ch == "∮":
                b = g.ink_bounds()
                if b is not None:
                    cx, cy = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
                    r = 0.28 * self.u(octx)
                    g.lines.append(self.stroke(_circle(cx, cy, r, rx=r * 1.1), ctx, slant=False, wobble=0.004))
        elif ch in ("⋃", "⋂", "⋁", "⋀"):
            small = {"⋃": "∪", "⋂": "∩", "⋁": "∨", "⋀": "∧"}[ch]
            g = self.glyph(small, octx)
            b = g.ink_bounds()
            if b is not None:
                ky = 1.55
                g = g.transformed(np.array([[1.15, 0.0, 0.0], [0.0, ky, 0.0]]), width_scale=1.0)
        elif ch in ("⨁", "⨂", "⨀"):
            g = self.proc_box(_proc_strokes({"⨁": "⊕", "⨂": "⊗", "⨀": "⊙"}[ch]) or [], octx)
            g = g.transformed(np.array([[1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]), width_scale=1.0)
        elif ch == "⨆":
            g = self.glyph("∪", octx).transformed(np.array([[1.15, 0.0, 0.0], [0.0, 1.55, 0.0]]), width_scale=1.0)
        else:
            g = self.glyph(ch, octx)
        # a bigger symbol is drawn with the same pen: undo the pen-width scaling
        g.lines = [(p, w / k) for p, w in g.lines]
        g = self.center_on_axis(_tight(g), ctx, axis=AXIS + (0.0 if not is_int else -0.05))
        return _tight(g)

    # ---------------------------------------------------------------- text
    def lay_text(self, text: str, ctx: Ctx) -> InkBox:
        sz = self.size * ctx.scale
        words = text.split(" ")
        out = InkBox(text=text)
        x = 0.0
        for k, w in enumerate(words):
            if k > 0:
                x += self.s.word_gap(sz) * (0.75 if ctx.depth == 0 else 0.6)
            if not w:
                continue
            wb = self.s.word(w, sz)
            out.add(wb, x, 0.0)
            x += wb.width
        out.width = x
        return out

    # ---------------------------------------------------------------- accents
    def lay_accent(self, node: Accent, ctx: Ctx) -> InkBox:
        u = self.u(ctx)
        bb = self.lay(node.body, ctx)
        kind = node.kind
        out = InkBox()
        out.add(bb, 0.0, 0.0)
        out.width = bb.width
        ink = bb.ink_bounds() or (0.0, -u, max(bb.width, 0.5 * u), 0.0)
        x0, x1 = ink[0], ink[2]
        asc = max(bb.ascent, 1.0 * u)
        wide = kind in ("overline", "widehat", "widetilde", "overrightarrow", "overleftarrow",
                        "overleftrightarrow", "underline", "overbrace", "underbrace", "widecheck")
        # accent centre: slanted letters lean right, so the top centre is right of the box centre
        cx = 0.5 * (x0 + x1) + (self.slant * 0.25 * u if not wide else 0.0)
        y = -(asc + 0.2 * u)
        pen = self.pen(ctx) * 0.95
        if kind in ("overline", "bar"):
            if kind == "bar":
                hw = max(0.28 * u, min(0.5 * (x1 - x0), 0.45 * u))
                a, b = cx - hw, cx + hw
            else:
                a, b = x0 - 0.03 * u, x1 + 0.03 * u
            out.lines.append(self.hline(a, b, y, ctx, pen=pen, wob=0.6))
        elif kind == "underline":
            yy = max(bb.descent, 0.0) + 0.2 * u
            out.lines.append(self.hline(x0 - 0.03 * u, x1 + 0.03 * u, yy, ctx, pen=pen, wob=0.6))
        elif kind in ("hat", "widehat", "check", "widecheck"):
            hw = 0.24 * u if kind in ("hat", "check") else max(0.24 * u, 0.5 * (x1 - x0))
            h = 0.24 * u
            if kind.endswith("check"):
                p = A([[cx - hw, y - h], [cx, y], [cx + hw, y - h]])
            else:
                p = A([[cx - hw, y], [cx, y - h], [cx + hw, y]])
            out.lines.append(self.stroke(p, ctx, slant=False, wobble=0.006, pen=pen))
        elif kind in ("tilde", "widetilde"):
            hw = 0.28 * u if kind == "tilde" else max(0.28 * u, 0.5 * (x1 - x0))
            t = np.linspace(0, 1, 12)
            p = np.column_stack([cx - hw + 2 * hw * t, y - 0.06 * u - 0.09 * u * np.sin(2 * np.pi * t)])
            out.lines.append(self.stroke(p, ctx, slant=False, wobble=0.004, pen=pen))
        elif kind in ("vec", "overrightarrow", "overleftarrow", "overleftrightarrow"):
            if kind == "vec":
                a, b = cx - 0.3 * u, cx + 0.3 * u
            else:
                a, b = x0 - 0.03 * u, x1 + 0.08 * u
            out.lines.append(self.hline(a, b, y, ctx, pen=pen, wob=0.4))
            hs = 0.16 * u
            if kind != "overleftarrow":
                out.lines.append(self.stroke(A([[b - hs, y - hs], [b, y], [b - hs, y + hs]]), ctx, slant=False,
                                             wobble=0.0, pen=pen))
            if kind in ("overleftarrow", "overleftrightarrow"):
                out.lines.append(self.stroke(A([[a + hs, y - hs], [a, y], [a + hs, y + hs]]), ctx, slant=False,
                                             wobble=0.0, pen=pen))
        elif kind in ("dot", "ddot", "dddot"):
            n = {"dot": 1, "ddot": 2, "dddot": 3}[kind]
            for k in range(n):
                xx = cx + (k - (n - 1) / 2) * 0.22 * u
                out.lines.append((A([[xx, y - 0.04 * u], [xx + 0.02 * u, y - 0.03 * u]]), pen * 1.15))
        elif kind == "mathring":
            out.lines.append(self.stroke(_circle(cx, y - 0.08 * u, 0.1 * u), ctx, slant=False, wobble=0.0, pen=pen))
        elif kind == "breve":
            t = np.linspace(0, np.pi, 10)
            p = np.column_stack([cx - 0.2 * u * np.cos(t), y - 0.18 * u + 0.15 * u * np.sin(t)])
            out.lines.append(self.stroke(p, ctx, slant=False, wobble=0.0, pen=pen))
        elif kind in ("acute", "grave"):
            s = 1 if kind == "acute" else -1
            p = A([[cx - 0.08 * u * s, y], [cx + 0.12 * u * s, y - 0.25 * u]])
            out.lines.append(self.stroke(p, ctx, slant=False, wobble=0.0, pen=pen))
        elif kind in ("overbrace", "underbrace"):
            out.lines.extend(self.hbrace(x0, x1, y if kind == "overbrace" else max(bb.descent, 0) + 0.2 * u,
                                         ctx, down=(kind == "underbrace")))
        out.refit(keep_width=True)
        return out

    def hbrace(self, x0: float, x1: float, y: float, ctx: Ctx, down: bool) -> List[Tuple[np.ndarray, float]]:
        u = self.u(ctx)
        s = 1.0 if down else -1.0
        h = 0.25 * u
        xm = 0.5 * (x0 + x1)
        left = A([[x0, y], [x0 + 0.1 * u, y + s * 0.5 * h], [x0 + 0.3 * u, y + s * 0.5 * h],
                  [xm - 0.25 * u, y + s * 0.5 * h], [xm - 0.08 * u, y + s * 0.6 * h], [xm, y + s * h]])
        right = A([[xm, y + s * h], [xm + 0.08 * u, y + s * 0.6 * h], [xm + 0.25 * u, y + s * 0.5 * h],
                   [x1 - 0.3 * u, y + s * 0.5 * h], [x1 - 0.1 * u, y + s * 0.5 * h], [x1, y]])
        return [self.stroke(left, ctx, slant=False, smooth=2, wobble=0.004),
                self.stroke(right, ctx, slant=False, smooth=2, wobble=0.004)]

    # --------------------------------------------------------------- misc
    def lay_negated(self, node: Negated, ctx: Ctx) -> InkBox:
        u = self.u(ctx)
        bb = self.lay(node.body, ctx)
        b = bb.ink_bounds() or (0.0, -u, 0.6 * u, 0.0)
        cx, cy = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
        h = max(0.5 * (b[3] - b[1]) + 0.15 * u, 0.5 * u)
        p = A([[cx + 0.38 * h, cy - h], [cx - 0.38 * h, cy + h]])
        out = InkBox()
        out.add(bb, 0.0, 0.0)
        out.lines.append(self.stroke(p, ctx, slant=False, wobble=0.004, pen=self.pen(ctx) * 0.95))
        out.width = bb.width
        out.refit(keep_width=False)
        return _tight(out)

    def lay_boxed(self, node: Boxed, ctx: Ctx) -> InkBox:
        u = self.u(ctx)
        bb = self.lay(node.body, ctx)
        p = 0.3 * u
        x0, x1 = 0.0, bb.width + 2 * p
        y0, y1 = -(max(bb.ascent, u) + p), max(bb.descent, 0.0) + p
        out = InkBox()
        out.add(bb, p, 0.0)
        pts = A([[x0 + 0.05 * u, y0], [x1, y0 + 0.02 * u], [x1 - 0.02 * u, y1], [x0, y1 - 0.02 * u],
                 [x0 + 0.03 * u, y0 - 0.05 * u]])
        out.lines.append(self.stroke(pts, ctx, slant=False, wobble=0.008))
        out.width = x1 + self.pen(ctx)
        out.refit(keep_width=True)
        return _tight(out)

    def lay_xarrow(self, node: XArrow, ctx: Ctx) -> InkBox:
        u = self.u(ctx)
        sctx = ctx.script()
        ob = self.lay(node.over, sctx) if node.over is not None else None
        ub = self.lay(node.under, sctx) if node.under is not None else None
        L = max([1.3 * u] + [b.width + 0.6 * u for b in (ob, ub) if b is not None])
        y = -AXIS * u
        hs = 0.22 * u
        line = self.hline(0.0, L, y, ctx, wob=0.4)
        if node.left:
            head = A([[hs, y - hs], [0.0, y], [hs, y + hs]])
        else:
            head = A([[L - hs, y - hs], [L, y], [L - hs, y + hs]])
        arrow = InkBox(lines=[line, self.stroke(head, ctx, slant=False, wobble=0.0)], width=L)
        arrow.refit(keep_width=True)
        return self.lay_limits(arrow, node.over, node.under, ctx, kind="stack")

    # ------------------------------------------------------------ matrices
    def lay_matrix(self, node: Matrix, ctx: Ctx) -> InkBox:
        u = self.u(ctx)
        kind = node.kind
        small = kind in ("smallmatrix", "substack")
        cscale = ctx.scale * (0.7 if small else 1.0)
        cell_display = kind in ("aligned", "align", "gathered", "gather", "split", "equation", "multline",
                                "eqnarray", "alignat", "alignedat", "dcases") and ctx.display
        cctx = Ctx(max(MIN_SCALE, cscale), cell_display, ctx.depth + (1 if small else 0))
        cu = self.u(cctx)
        aligned = kind in ("aligned", "align", "alignat", "alignedat", "split", "eqnarray")
        rows_nodes = node.rows
        ncol = max(len(r) for r in rows_nodes) if rows_nodes else 0
        cells: List[List[InkBox]] = []
        for r in rows_nodes:
            rb = []
            for j, c in enumerate(r):
                items = list(c.items)
                if aligned and j % 2 == 1 and items:
                    # "x &= 1": restore the relation spacing lost at the cell boundary
                    fc = _cls(items[0])
                    if fc in ("rel", "bin"):
                        items = [Space(REL_GAP if fc == "rel" else BIN_GAP)] + items
                rb.append(self.lay_row(items, cctx))
            cells.append(rb)
        colw = [0.0] * ncol
        for rb in cells:
            for j, b in enumerate(rb):
                colw[j] = max(colw[j], b.width)
        if kind in ("cases", "dcases", "rcases"):
            colsep = [1.1 * cu] * max(0, ncol - 1)
            align = ["l"] * ncol
        elif aligned:
            colsep = [(0.0 if j % 2 == 0 else 1.6 * cu) for j in range(max(0, ncol - 1))]
            align = [("r" if j % 2 == 0 else "l") for j in range(ncol)]
        elif kind in ("gathered", "gather", "equation", "multline", "substack"):
            colsep = [1.0 * cu] * max(0, ncol - 1)
            align = ["c"] * ncol
        else:
            colsep = [(0.75 if small else 1.0) * cu] * max(0, ncol - 1)
            align = ["c"] * ncol
        colx = [0.0]
        for j in range(1, ncol):
            colx.append(colx[-1] + colw[j - 1] + colsep[j - 1])
        total_w = (colx[-1] + colw[-1]) if ncol else 0.0
        # vertical: baselines
        row_asc = [max([b.ascent for b in rb] + [0.9 * cu]) for rb in cells]
        row_desc = [max([b.descent for b in rb] + [0.2 * cu]) for rb in cells]
        lead = (0.35 if not small else 0.15) * cu
        min_skip = (2.1 if not small else 1.5) * cu if not cell_display else 2.6 * cu
        ys = [0.0]
        for i in range(1, len(cells)):
            ys.append(ys[-1] + max(row_desc[i - 1] + row_asc[i] + lead, min_skip))
        top = ys[0] - row_asc[0] if cells else 0.0
        bottom = ys[-1] + row_desc[-1] if cells else 0.0
        # centre on the maths axis
        off = -AXIS * u - 0.5 * (top + bottom)
        body = InkBox()
        for i, rb in enumerate(cells):
            for j, b in enumerate(rb):
                if align[j] == "l":
                    x = colx[j]
                elif align[j] == "r":
                    x = colx[j] + colw[j] - b.width
                else:
                    x = colx[j] + 0.5 * (colw[j] - b.width)
                body.add(b, x + self.jit(0.03) * cu, ys[i] + off)
        body.width = total_w
        body.refit(keep_width=True)
        l, r = MATRIX_ENVS.get(kind, ("", ""))
        if not l and not r:
            return body
        bt = -(top + off) + 0.1 * u
        bbm = bottom + off + 0.1 * u
        out = InkBox()
        x = 0.0
        if l:
            lb = self.delim(l, max(bt, DELIM_TOP * u), max(bbm, DELIM_BOT * u), ctx)
            out.add(lb, 0.0, 0.0)
            x = lb.width + 0.2 * u
        out.add(body, x, 0.0)
        x += body.width
        if r:
            x += 0.2 * u
            rb_ = self.delim(r, max(bt, DELIM_TOP * u), max(bbm, DELIM_BOT * u), ctx)
            out.add(rb_, x, 0.0)
            x += rb_.width
        out.width = x
        return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _plain(src: str, synth, size: float) -> InkBox:
    """Last-resort rendering: the source as handwritten words."""
    words = [w for w in src.replace("{", " ").replace("}", " ").replace("\\", " ").split() if w]
    out = InkBox(text=src)
    x = 0.0
    for k, w in enumerate(words):
        if k:
            x += synth.word_gap(size) * 0.6
        b = synth.word(w, size)
        out.add(b, x, 0.0)
        x += b.width
    out.width = x
    return out


PEN_FILLS = True  # convert synthetic centre-line strokes to filled pen strokes in render_math


def render_math(src: str, synth, display: bool = False, size: float = 1.0) -> InkBox:
    """Typeset LaTeX maths ``src`` in the writer's hand.

    Returns an :class:`InkBox` in points (origin at the left end of the
    baseline, y down) whose width/ascent/descent cover all of its ink.
    ``display`` selects display style (bigger operators with limits above and
    below, full-size fractions); ``size`` scales the writer's x-height.
    """
    src = (src or "").strip()
    if not src:
        return InkBox(text="")
    try:
        ast = parse(src)
        box = MathLayout(synth, size).lay(ast, Ctx(1.0, bool(display), 0))
    except Exception:  # never let a formula break a document
        try:
            box = _plain(src, synth, size)
        except Exception:
            return InkBox(width=0.0, text=src)
    b = box.ink_bounds()
    if b is not None:
        if b[0] < -0.01:
            w = box.width
            box = box.shifted(-b[0], 0.0)
            box.width = w - b[0]
        box.refit(keep_width=False)
        bb = box.ink_bounds()
        if bb is not None:
            box.width = max(box.width, bb[2])
    if PEN_FILLS and box.lines:
        # draw synthetic strokes (bars, radicals, brackets) as PencilKit-like
        # filled pen strokes so they match the writer's real ink
        from .fallback import pen_stroke

        unit = synth.xh * size
        var = getattr(synth, "var", 1.0)
        box.fills = list(box.fills) + [pen_stroke(p, w, synth.rng, unit=unit, var=var) for p, w in box.lines
                                       if len(p)]
        box.lines = []
    box.text = src
    return box
