"""Text normalisation for clinical entities and dictionary descriptions."""
from __future__ import annotations

import html
import re

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_NONALNUM = re.compile(r"[^a-z0-9]+")
_SEMTAG_SUFFIX = re.compile(r"\s*\([a-z /\-]+\)\s*$")


def basic_norm(s: str) -> str:
    """Lowercase, unescape html, drop markup, collapse whitespace. Keeps punctuation."""
    s = html.unescape(str(s or ""))
    s = _TAG.sub(" ", s)
    s = s.replace("_", " ")
    return _WS.sub(" ", s.lower()).strip()


def strip_semtag(s: str) -> str:
    """basic_norm plus removal of a trailing SNOMED semantic tag, e.g. '(disorder)'."""
    return _SEMTAG_SUFFIX.sub("", basic_norm(s)).strip()


def key_norm(s: str) -> str:
    """Aggressive key for exact lookup: alnum tokens only."""
    return _NONALNUM.sub(" ", basic_norm(s)).strip()


def tokens(s: str) -> list[str]:
    return key_norm(s).split()
