"""Real-data retrieval evaluation (docs/design/alignment-audit.md §F3).

The brain ships six retrieval legs — FTS/BM25, vector KNN, RRF fusion, ColBERT
rerank, graph PPR, and the facts leg — and until this module existed there was
no evidence any of them beat plain BM25 on real data. The bundled fixture
(``tests/eval/harness.py`` + ``eval_basic.json``) proves the pipeline runs
hermetically in CI; it says nothing about retrieval quality on YOUR corpus.

Two stages, deliberately separable:

  generate — sample real memories/episodes and have the aux LLM write
             *paraphrase* questions answerable from each. Gold = the source
             uid. Stored under ``$HERMES_HOME/brain/eval/queryset.json``.
  compare  — run the SAME query set across leg configurations and report
             P@k, MRR, and paired win/loss/tie against the FTS baseline.

Why paraphrase, emphatically: the first attempt at measuring this used queries
lifted verbatim from the indexed text, which hands BM25 an exact-token match
and makes the benchmark meaningless. A query that shares no wording with its
gold document is the only kind that can distinguish lexical from semantic
retrieval.

Why paired counts, emphatically: that same first attempt produced a headline
"vectors are 3.7% worse", which sounded like a finding and was two queries out
of 71. The paired view (87% of queries identical, 5 wins, 4 losses) made the
noise obvious. Means alone at this sample size mislead.

This is a subpackage, not a root module: the Hermes loader eagerly imports
every root ``*.py`` on every CLI invocation, and nothing here belongs in that
path.
"""

from __future__ import annotations

from .compare import CONFIGURATIONS, format_report, run_comparison
from .generate import generate_queryset
from .store import (
    baseline_path,
    load_baseline,
    load_queryset,
    queryset_fingerprint,
    queryset_path,
    save_baseline,
    save_queryset,
)

__all__ = [
    "CONFIGURATIONS",
    "baseline_path",
    "load_baseline",
    "queryset_fingerprint",
    "save_baseline",
    "format_report",
    "format_report_or_none",
    "generate_queryset",
    "load_queryset",
    "queryset_path",
    "run_comparison",
    "save_queryset",
]


def format_report_or_none(report, baseline=None) -> str:
    """format_report, but explicit when every configuration was skipped —
    an empty table reads as "no difference" when it means "nothing ran"."""
    if not report or not any(
        not r.get("skipped") for r in report.get("results", [])
    ):
        return ("No configuration could be scored.\n"
                "  Every leg was unavailable — run 'hermes brain doctor' and look at "
                "the 'legs' line.")
    return format_report(report, baseline)
