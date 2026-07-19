"""Phase C dream: surprisal seeding for consolidate. When `dream_surprisal`
is on, the surprising subset of the candidate window is rendered as an
"anomalies to reconcile" hint appended to the LLM cluster prompt; when off the
prompt is byte-for-byte the plain one; and the computation degrades (no crash)
when no embedder is present. Hermetic: fake LLM via llm.set_llm_for_tests
around EVERY step (CLAUDE.md), StubEmbedder vectors, sqlite-vec required.
"""

from __future__ import annotations

import json
import re

import pytest

pytest.importorskip("sqlite_vec")

from brain import llm  # noqa: E402
from brain.dream import consolidate  # noqa: E402
from brain.dream.shift import Shift  # noqa: E402
from brain.recall.embed import StubEmbedder  # noqa: E402
from brain.store import db  # noqa: E402
from brain.store import vec as vec_store  # noqa: E402
from conftest import seed_memory  # noqa: E402

_ULID_RE = re.compile(r"\[([0-9A-HJKMNP-TV-Z]{26})\]")

# Near-identical extracted observations: pairwise stub cosine well above the
# 0.80 clustering gate so they form ONE cluster and earn one LLM call.
_CLUSTER_TEXTS = [
    "the user always prefers tabs over spaces in python code projects",
    "the user always prefers tabs over spaces in python code repositories",
    "the user always prefers tabs over spaces in python code generally",
]

# A vivid, unique needle we can look for verbatim in the captured prompt.
_ANOMALY_TEXT = "the user always prefers tabs over spaces in python code generally"


@pytest.fixture(autouse=True)
def _clean_llm():
    yield
    llm.set_llm_for_tests(None)


@pytest.fixture
def embedder():
    return StubEmbedder()


def _mk_shift(conn, mode, embedder, *, config=None):
    from brain.dream import lease
    lease.acquire(conn, "dream", "test")
    cfg = {"_forced_mode": mode, "night_budget_usd": 1.0}
    if config:
        cfg.update(config)
    return Shift(
        shift_id="01TESTSHIFTAAAAAAAAAAAAAAA",
        conn=conn,
        config=cfg,
        embedder=embedder,
        started_at=db.iso_now(),
        activity_baseline="",
        holder="test",
    )


def _index(conn, embedder, mem_id, text):
    vec_store.upsert(conn, "mem_vec", mem_id, embedder.encode_documents([text])[0])


def _seed_cluster(conn, embedder):
    assert vec_store.ensure_tables(conn, embedder.dim)
    ids = []
    for text in _CLUSTER_TEXTS:
        mid = seed_memory(conn, text, created_by="extraction")
        _index(conn, embedder, mid, text)
        ids.append(mid)
    conn.commit()
    return ids


def _set_surprise(conn, mem_id, value):
    conn.execute("UPDATE memories SET surprise=? WHERE id=?", (value, mem_id))
    conn.commit()


class CapturingFake:
    """Consolidate fake that records every prompt it is called with and cites
    each uid it sees (so the cluster distills)."""

    def __init__(self, entity="python"):
        self.calls = 0
        self.prompts: list[str] = []
        self.entity = entity

    def __call__(self, prompt, *, system, max_tokens):
        self.calls += 1
        self.prompts.append(prompt)
        return json.dumps({
            "content": "The user consistently prefers tabs over spaces in python code.",
            "cites": _ULID_RE.findall(prompt),
            "entity": self.entity,
            "actionable": True,
        })


# ---------------------------------------------------------------------------
# gate ON: the high-surprise memory shows up in the anomaly hint
# ---------------------------------------------------------------------------

def test_surprisal_on_injects_anomaly_hint(conn, embedder):
    ids = _seed_cluster(conn, embedder)
    # Make the third member conspicuously surprising via the stored column.
    _set_surprise(conn, ids[2], 0.95)
    fake = CapturingFake()
    llm.set_llm_for_tests(fake)

    result = consolidate.run(
        _mk_shift(conn, "active", embedder, config={"dream_surprisal": True}))

    assert result.get("distilled") == 1 and fake.calls == 1
    prompt = fake.prompts[0]
    assert "Anomalies to reconcile" in prompt
    assert _ANOMALY_TEXT in prompt
    # The surprising member's uid is present in the hint block (after the label).
    anomaly_uid = conn.execute(
        "SELECT uid FROM memories WHERE id=?", (ids[2],)).fetchone()["uid"]
    hint_block = prompt.split("Anomalies to reconcile", 1)[1]
    assert anomaly_uid in hint_block


# ---------------------------------------------------------------------------
# gate OFF: no hint section — byte-for-byte the plain prompt
# ---------------------------------------------------------------------------

def test_surprisal_off_no_hint_section(conn, embedder):
    ids = _seed_cluster(conn, embedder)
    _set_surprise(conn, ids[2], 0.95)
    fake = CapturingFake()
    llm.set_llm_for_tests(fake)

    result = consolidate.run(
        _mk_shift(conn, "active", embedder, config={"dream_surprisal": False}))

    assert result.get("distilled") == 1 and fake.calls == 1
    prompt = fake.prompts[0]
    assert "Anomalies to reconcile" not in prompt
    # No trailing hint block: the prompt ends exactly on the plain terminator,
    # i.e. it is byte-for-byte the pre-surprisal prompt (nothing appended).
    assert prompt.endswith("Distill the one lesson these collectively support.")


# ---------------------------------------------------------------------------
# degradation: surprisal computation must not crash without an embedder
# ---------------------------------------------------------------------------

def test_surprisal_degrades_without_embedder(conn, embedder):
    # Seed the rows (need an embedder to build vectors) then run the surprisal
    # computation with the embedder absent: the kNN-density half is skipped and
    # only the stored-surprise decile remains. Must not raise.
    ids = _seed_cluster(conn, embedder)
    _set_surprise(conn, ids[2], 0.9)

    candidates = consolidate._candidates(conn)
    assert candidates

    shift = _mk_shift(conn, "active", embedder=None,
                      config={"dream_surprisal": True})
    rows = consolidate._surprisal_hints(shift, candidates, blobs={})
    # Top-decile by surprise still works with no embedder / no blobs.
    assert any(r["id"] == ids[2] for r in rows)
    hint = consolidate._render_hint(rows)
    assert "Anomalies to reconcile" in hint

    # And the full builder degrades to a plain-but-hinted prompt with no crash.
    prompt = consolidate._cluster_prompt(candidates, hint)
    assert _ANOMALY_TEXT in prompt


def test_surprisal_empty_when_no_signal(conn, embedder):
    # No surprise column set and near-identical vectors: density outliers still
    # return the lowest-density row (a nudge), but a truly empty window yields
    # an empty hint. Verify the empty-window path renders "".
    shift = _mk_shift(conn, "active", embedder, config={"dream_surprisal": True})
    assert consolidate._surprisal_hints(shift, [], blobs={}) == []
    assert consolidate._render_hint([]) == ""
