"""D2 write-time knowledge rewriting (SEAL-style search aids).

Per RETAINED extraction item the batched LLM additionally emits a small list
of "search aids" — 2-4 short paraphrases / synonyms / implied questions a user
might later type to look the fact up. They fold into BOTH retrieval legs so a
paraphrased query recalls the item:

  * FTS   — appended to the `tags` column (bm25 weight 2.0, a JSON array that
            is never shown as prose).
  * Vector — the EMBEDDED text becomes ``content + " " + aids`` (HyDE/QA
            augmentation), so the vector sits nearer question-phrased queries.

The DISPLAYED `content` stays clean — aids are retrieval-only.

Hermetic: the LLM is a fake installed via ``llm.set_llm_for_tests``; the stub
embedder + sqlite-vec give real cosine behavior with no downloads (mirrors
tests/test_hybrid.py). Skipped wholesale if sqlite-vec is not importable.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("sqlite_vec")

from brain import llm  # noqa: E402
from brain.capture import extract  # noqa: E402
from brain.capture.turns import (  # noqa: E402
    TurnContext,
    capture_session_end,
    capture_turn,
)
from brain.config import DEFAULTS  # noqa: E402
from brain.recall.embed import StubEmbedder  # noqa: E402
from brain.recall.search import search  # noqa: E402
from brain.store import vec as vec_store  # noqa: E402
from conftest import seed_memory  # noqa: E402

# content and aids share NO tokens (not even stopwords) with the paraphrase
# query "door entry pin", so any recall of it via that query is attributable
# ONLY to the aids — the A/B control below relies on this.
_CONTENT = ("The building access keypad combination was rotated to 5731 "
            "during the winter maintenance window.")
_AIDS = ["door code", "entry pin number", "how do I get inside"]
_PARAPHRASE = "door entry pin"

_ITEM = {
    "content": _CONTENT, "kind": "fact", "about_user": False,
    "time_sensitive": False, "instruction_shaped": False,
    "source_uids": [], "search_aids": _AIDS,
}


@pytest.fixture(autouse=True)
def _clear_llm_override():
    yield
    llm.set_llm_for_tests(None)


class _FakeLLM:
    """One canned reply; ignores system/tier (D2 only cares about the items)."""

    def __init__(self, reply: str):
        self.reply = reply

    def __call__(self, prompt, *, system=None, max_tokens=0):
        return self.reply


def _cfg(**overrides):
    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    return cfg


def _ctx(sid):
    return TurnContext(session_id=sid, platform="cli",
                       principal_id="owner-p", trust_tier="owner")


def _extract_one(conn, *, sid="s-d2", item=None, embedder=None) -> int:
    """Drive the REAL extraction path (capture -> sweep) once; return the id
    of the single memory it writes."""
    capture_turn(conn, _ctx(sid),
                 "Facilities rotated the north keypad combination to 5731 "
                 "for the season.",
                 "Logged it, thanks.")
    capture_session_end(conn, sid)
    llm.set_llm_for_tests(_FakeLLM(json.dumps([item or _ITEM])))
    counts = extract.sweep(conn, _cfg(), embedder=embedder)
    assert counts["inserted"] == 1, counts
    return conn.execute("SELECT id FROM memories ORDER BY id").fetchone()["id"]


# ---------------------------------------------------------------------------
# (1) aids land in the FTS field (tags); content stays clean
# ---------------------------------------------------------------------------

def test_aids_go_to_tags_and_content_stays_clean(conn):
    mem_id = _extract_one(conn)
    row = conn.execute("SELECT content, tags FROM memories WHERE id=?",
                       (mem_id,)).fetchone()

    # aids stored verbatim (and in order) in the FTS-indexed tags column
    assert json.loads(row["tags"]) == _AIDS
    # displayed content is untouched — no aid text leaked into it
    assert row["content"] == _CONTENT
    for aid in _AIDS:
        assert aid not in row["content"]


# ---------------------------------------------------------------------------
# (2) a paraphrased query recalls the item — FTS leg, isolated (no embedder)
# ---------------------------------------------------------------------------

def test_paraphrase_recalls_via_fts_tags(conn):
    aug_id = _extract_one(conn)
    # Control: the SAME clean content with NO aids. If the paraphrase matched
    # on content it would surface this too — it must not.
    control_id = seed_memory(conn, _CONTENT)

    mem_ids = [h.id for h in search(conn, _PARAPHRASE, limit=20)
               if h.kind == "memory"]
    assert aug_id in mem_ids, "search aids must make the paraphrase match"
    assert control_id not in mem_ids, (
        "the same content WITHOUT aids must not match the paraphrase — proving "
        "it was the aids, not the content, that were recalled")


# ---------------------------------------------------------------------------
# (3) the aids are folded into the EMBEDDED vector text
# ---------------------------------------------------------------------------

def test_aids_fold_into_embedded_vector(conn):
    embedder = StubEmbedder()
    assert vec_store.ensure_tables(conn, embedder.dim)
    mem_id = _extract_one(conn, embedder=embedder)

    blob = conn.execute("SELECT emb FROM mem_vec WHERE id=?",
                        (mem_id,)).fetchone()["emb"]
    with_aids = embedder.encode_documents([_CONTENT + " " + " ".join(_AIDS)])[0]
    content_only = embedder.encode_documents([_CONTENT])[0]
    cos_aids = extract._int8_cosine(blob, with_aids)
    cos_content = extract._int8_cosine(blob, content_only)

    # The stored vector IS embed(content + aids): ~identical to it, and
    # strictly nearer to it than to embed(content) alone.
    assert cos_aids > 0.999
    assert cos_aids > cos_content


# ---------------------------------------------------------------------------
# (4) end-to-end: paraphrase recall over the fused FTS + vector legs
# ---------------------------------------------------------------------------

def test_paraphrase_recalls_end_to_end_with_embedder(conn):
    embedder = StubEmbedder()
    assert vec_store.ensure_tables(conn, embedder.dim)
    aug_id = _extract_one(conn, embedder=embedder)

    hits = search(conn, _PARAPHRASE, embedder=embedder, limit=20)
    assert any(h.kind == "memory" and h.id == aug_id for h in hits)


# ---------------------------------------------------------------------------
# (5) deterministic guards: aids are capped and sanitized (anti-spam)
# ---------------------------------------------------------------------------

def test_aids_are_capped_and_sanitized(conn):
    messy_item = dict(_ITEM, search_aids=[
        "  wifi login  ",        # whitespace-normalized -> "wifi login"
        "wifi login",            # case/space duplicate -> dropped
        "x",                     # < 2 chars -> dropped
        12345,                   # non-string -> dropped
        "how to connect",
        "the network key",
        "one more phrase",
        "yet another phrase",    # 5th survivor -> over the cap, dropped
        "a" * 200,               # > 80 chars -> dropped
    ])
    mem_id = _extract_one(conn, sid="s-messy", item=messy_item)
    tags = json.loads(conn.execute(
        "SELECT tags FROM memories WHERE id=?", (mem_id,)).fetchone()["tags"])

    assert tags == ["wifi login", "how to connect", "the network key",
                    "one more phrase"]
    assert len(tags) <= extract._MAX_AIDS
    assert "x" not in tags               # too-short dropped
    assert ("a" * 200) not in tags       # too-long dropped
    assert tags.count("wifi login") == 1  # folded duplicate collapsed


# ---------------------------------------------------------------------------
# (6) missing / malformed search_aids degrade to the prior behavior
# ---------------------------------------------------------------------------

def test_missing_aids_keeps_empty_tags(conn):
    no_aids = {k: v for k, v in _ITEM.items() if k != "search_aids"}
    mem_id = _extract_one(conn, sid="s-none", item=no_aids)
    tags = conn.execute("SELECT tags FROM memories WHERE id=?",
                       (mem_id,)).fetchone()["tags"]
    assert tags == "[]"  # unchanged from the pre-D2 default
