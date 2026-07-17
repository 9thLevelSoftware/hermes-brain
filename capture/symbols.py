"""Code-symbol extraction — ported from Daem0n-MCP similarity.py.

The one keyword-side differentiator for coding-agent recall: exact
identifiers (CamelCase, snake_case, SCREAMING_SNAKE, `backticked`,
.method) that porter-stemmed FTS and embeddings both blur.

Used at index time (episodes.symbols / memories.symbols FTS columns) and
at query time (recall/search.py expands the query with detected symbols).
Symbols are stored space-joined, snake/camel parts split so FTS matches
both the exact identifier and its words.
"""

from __future__ import annotations

import re

_BACKTICK = re.compile(r"`([a-zA-Z_][a-zA-Z0-9_]*(?:\([^)]*\))?)`")
_PARENS = re.compile(r"\([^)]*\)")
_CAMEL = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z0-9]*)+)\b")
_LOWER_CAMEL = re.compile(r"\b([a-z]+(?:[A-Z][a-z0-9]*)+)\b")
_SNAKE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")
_SCREAMING = re.compile(r"\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b")
_METHOD = re.compile(r"\.([a-zA-Z_][a-zA-Z0-9_]*)")
_CAMEL_SPLIT = re.compile(r"([a-z])([A-Z])")

# Words that match the snake/method patterns but are prose, not code.
_PROSE = frozenset({
    "e_g", "i_e", "the", "and", "for", "not", "with", "this", "that",
    "self", "init", "com", "org", "www", "http", "https", "png", "jpg",
    "md", "txt", "py", "js", "ts",
})

_MAX_SYMBOLS = 64


def extract_code_symbols(text: str) -> list[str]:
    """Extract likely code identifiers, preserving original case."""
    if not text:
        return []
    symbols: set = set()

    for match in _BACKTICK.findall(text):
        clean = _PARENS.sub("", match)
        if len(clean) >= 2:
            symbols.add(clean)

    for pattern, min_len in ((_CAMEL, 3), (_LOWER_CAMEL, 3), (_SNAKE, 3), (_SCREAMING, 3)):
        for match in pattern.findall(text):
            if len(match) >= min_len and match.lower() not in _PROSE:
                symbols.add(match)

    for match in _METHOD.findall(text):
        if len(match) >= 2 and match.lower() not in _PROSE:
            symbols.add(match)

    return sorted(symbols)[:_MAX_SYMBOLS]


def symbols_field(*texts: str) -> str:
    """Build the FTS `symbols` column value for one or more texts.

    Contains each symbol verbatim (lowercased — FTS is case-insensitive
    anyway) plus its split words, so `get_user_by_id` matches queries for
    both the identifier and "get user id".
    """
    out: list[str] = []
    seen: set = set()
    for text in texts:
        for sym in extract_code_symbols(text or ""):
            low = sym.lower()
            if low in seen:
                continue
            seen.add(low)
            out.append(low)
            parts = _CAMEL_SPLIT.sub(r"\1 \2", sym).replace("_", " ").lower()
            if parts != low:
                out.append(parts)
    return " ".join(out)


def expand_query(query: str) -> list[str]:
    """Symbols detected in a *query*, for exact-identifier match boosting."""
    return [s.lower() for s in extract_code_symbols(query)]
