"""Text normalisation.

Cleaning happens *before* chunking and embedding for two reasons:

1.  Retrieval quality.  Ligatures, soft hyphens and PDF line-wrapping produce
    tokens that never match the user's query ("conﬁdential" != "confidential",
    "employ-\\nment" != "employment").
2.  Security.  Zero-width and bidirectional control characters can hide text
    from a human reviewer while remaining fully visible to the model -- the
    classic vector for smuggling instructions past a manual document review.
    Stripping them means what the reviewer sees is what the model sees.
"""

from __future__ import annotations

import re
import unicodedata

# Zero-width, bidi overrides and other invisible formatting characters.
# Written as explicit escapes: these are precisely the characters that are
# invisible in a source file, so spelling them out is the point.
_INVISIBLE = re.compile(
    "["
    "​-‏"  # zero-width space/non-joiner/joiner, LRM, RLM
    "‪-‮"  # bidi embedding and override
    "⁠-⁤"  # word joiner, invisible operators
    "­"  # soft hyphen
    "﻿"  # BOM / zero-width no-break space
    "￹-￻"  # interlinear annotation
    "]"
)

# C0/C1 control characters except tab and newline.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "…": "...",
    " ": " ",
}

# "employ-\nment" -> "employment": a hyphen at end of line followed by a
# lowercase continuation is PDF line-wrapping, not a real compound word.
_HYPHEN_WRAP = re.compile(r"(\w)-\s*\n\s*([a-z])")
# A newline between two lowercase letters is a soft wrap, not a paragraph break.
_SOFT_WRAP = re.compile(r"([a-z,;:])\n(?=[a-z])")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
# "P a g e   4   o f   1 2" and similar letter-spaced PDF artefacts.
_PAGE_ARTEFACT = re.compile(
    r"^\s*(page\s*\d+\s*(of\s*\d+)?|\d+\s*/\s*\d+)\s*$", re.IGNORECASE
)


def count_invisible_characters(text: str) -> int:
    """Number of invisible/control characters, used as a security signal."""
    return len(_INVISIBLE.findall(text)) + len(_CONTROL.findall(text))


def strip_invisible(text: str) -> str:
    """Delete zero-width and bidirectional formatting characters.

    Deletion rather than substitution: these characters are inserted *between*
    the letters of a word to break pattern matching while leaving the rendered
    text unchanged, so replacing them with a space would preserve the break
    that the attacker wanted.
    """
    return _INVISIBLE.sub("", text)


def clean_text(text: str, *, dehyphenate: bool = True) -> str:
    """Normalise ``text`` for embedding and display."""
    if not text:
        return ""

    # NFKC folds compatibility forms (full-width Latin, ligatures, superscripts)
    # onto their canonical equivalents, which also collapses a family of
    # homoglyph tricks used to evade keyword-based filters.
    text = unicodedata.normalize("NFKC", text)

    for source, replacement in _LIGATURES.items():
        text = text.replace(source, replacement)

    text = _INVISIBLE.sub("", text)
    text = _CONTROL.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if dehyphenate:
        text = _HYPHEN_WRAP.sub(r"\1\2", text)
        text = _SOFT_WRAP.sub(r"\1 ", text)

    lines = [line for line in text.split("\n") if not _PAGE_ARTEFACT.match(line.strip())]
    text = "\n".join(lines)

    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def strip_repeated_lines(pages: list[str], *, min_repeats: int = 3) -> list[str]:
    """Remove running headers and footers.

    A short line that appears verbatim on most pages is chrome, not content.
    Left in place it is embedded into every chunk, adding a constant vector
    component that blurs the distinction between chunks.
    """
    if len(pages) < min_repeats:
        return pages

    counts: dict[str, int] = {}
    for page in pages:
        for line in {ln.strip() for ln in page.split("\n") if 0 < len(ln.strip()) < 90}:
            counts[line] = counts.get(line, 0) + 1

    threshold = max(min_repeats, int(len(pages) * 0.6))
    boilerplate = {line for line, count in counts.items() if count >= threshold}
    if not boilerplate:
        return pages

    return [
        "\n".join(ln for ln in page.split("\n") if ln.strip() not in boilerplate)
        for page in pages
    ]
