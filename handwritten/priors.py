"""Generic priors for print handwriting, used to bootstrap segmentation.

Metrics are in units of x-height relative to the baseline, y pointing *up*:
``top`` is the highest ink point, ``bottom`` the lowest (negative for
descenders), ``width`` the ink width.  Values come from the Hershey
"futural" single-stroke font (a plain print hand) with typical hand-printed
stroke counts layered on top.  They only seed the EM procedure in
``segment.py``; the user's own statistics replace them after one iteration.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional, Tuple

# Typical number of pen strokes for hand-printed characters.
_STROKES = {}
for _c in "abcdeghlmnopqrsuvwyzCGJLMNOSUVWZ0123689()/\\,.'-_~|<>{}[]&@%$#^`":
    _STROKES[_c] = 1
for _c in "ijtfkx45ABDPQRTXY!?;:\"=+7":
    _STROKES[_c] = 2
for _c in "EFHIK*":
    _STROKES[_c] = 3
_STROKES.update({"E": 3, "#": 4, "%": 3, "$": 2, "&": 1, "@": 1})

# Unicode characters that commonly appear in maths homework and the Hershey
# glyph that approximates them: (font, character in that font)
UNICODE_HERSHEY: Dict[str, Tuple[str, str]] = {
    "α": ("greek", "a"), "β": ("greek", "b"), "γ": ("greek", "g"), "δ": ("greek", "d"),
    "ε": ("greek", "e"), "ζ": ("greek", "z"), "η": ("greek", "h"), "θ": ("greek", "q"),
    "ι": ("greek", "i"), "κ": ("greek", "k"), "λ": ("greek", "l"), "μ": ("greek", "m"),
    "ν": ("greek", "n"), "ξ": ("greek", "x"), "ο": ("greek", "o"), "π": ("greek", "p"),
    "ρ": ("greek", "r"), "σ": ("greek", "s"), "τ": ("greek", "t"), "υ": ("greek", "u"),
    "φ": ("greek", "f"), "χ": ("greek", "c"), "ψ": ("greek", "y"), "ω": ("greek", "w"),
    "Γ": ("greek", "G"), "Δ": ("greek", "D"), "Θ": ("greek", "Q"), "Λ": ("greek", "L"),
    "Ξ": ("greek", "X"), "Π": ("greek", "P"), "Σ": ("greek", "S"), "Φ": ("greek", "F"),
    "Ψ": ("greek", "Y"), "Ω": ("greek", "W"),
}


@lru_cache(maxsize=None)
def _hershey(font: str):
    from HersheyFonts import HersheyFonts

    f = HersheyFonts()
    f.load_default_font(font)
    return f


BASE = 9.0  # Hershey futural baseline y (y down)
XH = 14.0  # Hershey futural x-height in font units


@lru_cache(maxsize=None)
def hershey_metrics(ch: str) -> Optional[Tuple[float, float, float, int]]:
    """(top, bottom, width, n_strokes) in x-height units, or None."""
    font, key = "futural", ch
    if ch in UNICODE_HERSHEY:
        font, key = UNICODE_HERSHEY[ch]
    try:
        f = _hershey(font)
        strokes = [s for s in f.strokes_for_text(key) if len(s)]
    except Exception:
        return None
    if not strokes:
        return None
    xs = [p[0] for s in strokes for p in s]
    ys = [p[1] for s in strokes for p in s]
    top = (BASE - min(ys)) / XH
    bottom = (BASE - max(ys)) / XH
    width = (max(xs) - min(xs)) / XH
    return top, bottom, max(width, 0.08), len(strokes)


@lru_cache(maxsize=None)
def char_prior(ch: str) -> Dict[str, float]:
    """Prior mean/std for a character's top/bottom/width and stroke count."""
    m = hershey_metrics(ch)
    if m is None:
        top, bottom, width, hs = 1.0, 0.0, 0.8, 1
    else:
        top, bottom, width, hs = m
    # Hand-printed capitals and ascenders are taller relative to x-height than
    # in futural; descenders similar.
    if top > 1.2:
        top *= 1.12
    n = _STROKES.get(ch, hs if m is not None else 1)
    return {
        "top": top,
        "bottom": bottom,
        "width": width,
        "top_sd": 0.35,
        "bottom_sd": 0.35,
        "width_sd": 0.25 + 0.35 * width,
        "strokes": float(n),
    }


def stroke_count_logprior(ch: str, n: int) -> float:
    e = char_prior(ch)["strokes"]
    import math

    # Laplace-like prior around the expected count, mild preference for fewer
    return -1.1 * abs(n - e) - (0.3 if n > e else 0.0) - math.log(2.0)


DESCENDERS = set("gjpqy,;()[]{}|/\\")  # chars whose ink may go below the baseline
XHEIGHT_CHARS = set("acemnorsuvwxz")  # chars whose top is the x-height line
BASELINE_CHARS = set("abcdefhiklmnorstuvwxzABCDEFGHIKLMNOPRSTUVWXYZ0123456789")
