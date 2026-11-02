"""Typeset a subset of LaTeX maths in the writer's handwriting.

(Placeholder: renders the maths source as plain handwritten text.  Replaced
by a real layout engine supporting scripts, fractions, roots, big operators.)
"""
from __future__ import annotations

from .box import InkBox, hstack


def render_math(src: str, synth, display: bool = False, size: float = 1.0) -> InkBox:
    words = [w for w in src.replace("{", "").replace("}", "").replace("\\", "").split() if w]
    boxes = [synth.word(w, size) for w in words]
    if not boxes:
        return InkBox()
    gaps = [synth.word_gap(size) * 0.6 for _ in boxes[:-1]]
    return hstack(boxes, gaps)
