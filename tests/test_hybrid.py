"""P2 hybrid retrieval: stub embedder + sqlite-vec + RRF fusion.

Uses the config-only 'stub' tier (deterministic hash embeddings) so these
tests are hermetic — no model downloads, no onnxruntime. Skipped wholesale
if sqlite-vec is not importable (the FTS-only floor is covered elsewhere).
"""

from __future__ import annotations

import pytest

pytest.importorskip("sqlite_vec")

from brain.recall.embed import StubEmbedder, get_embedder  # noqa: E402
from brain.recall.fusion import normalized, rrf  # noqa: E402
from brain.recall.search import search  # noqa: E402
from brain.store import db  # noqa: E402
from brain.store import vec as vec_store  # noqa: E402
from conftest import poll_until, seed_memory  # noqa: E402


@pytest.fixture
def embedder():
    return StubEmbedder()


def _index_memory(conn, embedder, mem_id, text):
    vec_store.upsert(conn, "mem_vec", mem_id, embedder.encode_documents([text])[0])
    conn.commit()


def test_vec_roundtrip(conn, embedder):
    assert vec_store.ensure_tables(conn, embedder.dim)
    a = seed_memory(conn, "the deploy pipeline uses github actions")
    b = seed_memory(conn, "cats are excellent debugging companions")
    _index_memory(conn, embedder, a, "the deploy pipeline uses github actions")
    _index_memory(conn, embedder, b, "cats are excellent debugging companions")

    knn = vec_store.knn(conn, "mem_vec", embedder.encode_query("github deploy pipeline"), 2)
    assert knn and knn[0][0] == a  # nearest neighbor is the on-topic row


def test_hybrid_search_fuses_vector_and_fts(conn, embedder):
    assert vec_store.ensure_tables(conn, embedder.dim)
    # FTS can find this by keyword; the stub-vector leg also ranks it first
    # for an overlapping-token query. RRF should keep it on top.
    target = seed_memory(conn, "postgres connection pool size is 20")
    noise = seed_memory(conn, "the office coffee machine is broken again")
    for mid, text in ((target, "postgres connection pool size is 20"),
                      (noise, "the office coffee machine is broken again")):
        _index_memory(conn, embedder, mid, text)

    hits = search(conn, "postgres pool size", embedder=embedder)
    assert hits and hits[0].id == target


def test_vector_leg_respects_scoping(conn, embedder):
    """A vector hit must not bypass scope filters (finding #17 applies to
    EVERY leg, not just FTS)."""
    assert vec_store.ensure_tables(conn, embedder.dim)
    secret = seed_memory(conn, "owner secret: totp backup codes in the safe")
    conn.execute("UPDATE memories SET scope_user='owner' WHERE id=?", (secret,))
    conn.commit()
    _index_memory(conn, embedder, secret, "owner secret: totp backup codes in the safe")

    peer = search(conn, "totp backup codes safe", embedder=embedder,
                  trust_tier="known_user", principal_id="peer-9")
    assert all(h.id != secret for h in peer)
    owner = search(conn, "totp backup codes safe", embedder=embedder, trust_tier="owner")
    assert any(h.id == secret for h in owner)


def test_dim_change_recreates_tables(conn):
    assert vec_store.ensure_tables(conn, 256, "old-embedder:256")
    vec_store.upsert(conn, "mem_vec", 1, [0.1] * 256)
    conn.commit()
    # New contract: a mismatch without allow_rebuild refuses (no-drop live
    # path); with allow_rebuild=True (CLI reindex) it drops and recreates.
    assert not vec_store.ensure_tables(conn, 512, "new-embedder:512")
    assert conn.execute("SELECT count(*) AS n FROM mem_vec").fetchone()["n"] == 1
    assert vec_store.ensure_tables(conn, 512, "new-embedder:512", allow_rebuild=True)
    assert conn.execute("SELECT count(*) AS n FROM mem_vec").fetchone()["n"] == 0
    assert db.get_meta(conn, "vec_dim") == "512"
    assert db.get_meta(conn, "vec_embedder") == "new-embedder:512"


def test_rrf_normalized_band():
    scores = normalized(rrf([["a", "b"], ["b", "c"]]))
    assert scores["b"] == 1.0                      # in both lists
    assert 0.2 <= scores["c"] < scores["a"] < 1.0  # band floor holds


def test_provider_end_to_end_stub_tier(tmp_home):
    """Full loop on the stub tier: capture -> embed -> cross-session recall
    with a query that shares tokens with the stored turn."""
    from brain.config import save_config
    from brain.provider import BrainProvider

    save_config(tmp_home, {"mode": "stub"})

    p1 = BrainProvider()
    p1.initialize("h-one", hermes_home=str(tmp_home), platform="cli")
    p1.sync_turn("the wifi password for the office is hunter2-max",
                 "saved that for you", session_id="h-one")
    p1.shutdown()

    p2 = BrainProvider()
    p2.initialize("h-two", hermes_home=str(tmp_home), platform="cli")
    try:
        p2.queue_prefetch("what is the office wifi password?", session_id="h-two")
        block = poll_until(lambda: p2.prefetch("", session_id="h-two"), timeout=5.0)
        assert "hunter2-max" in block
    finally:
        p2.shutdown()

    conn = db.connect(tmp_home)
    try:
        stats = vec_store.stats(conn)  # loads the extension for this conn
        assert stats and stats["epi_vec"] == 1
    finally:
        conn.close()


def test_get_embedder_never_raises_without_deps(tmp_path, monkeypatch):
    # full tier with no model files and download forbidden -> None, no raise.
    # Point the model cache at an empty tmp dir so the assertion holds even
    # on machines that have the real model files cached.
    import brain.recall.embed as embed_mod

    monkeypatch.setattr(embed_mod, "models_cache_dir", lambda: tmp_path)
    assert get_embedder({"embed_model": "modernbert-embed-base"}, "full",
                        allow_download=False) is None


def test_search_date_window_filters_candidates_in_sql(tmp_home):
    """date_from/date_to restrict candidates IN the query, so an in-window
    memory is found even when out-of-window rows would rank above it (PR #5)."""
    from brain.recall.search import search
    from brain.store import db
    from conftest import iso_days_ago, seed_memory

    conn = db.connect(tmp_home)
    try:
        seed_memory(conn, "deploy runbook alpha edition", valid_from=iso_days_ago(400))
        mid = seed_memory(conn, "deploy runbook beta edition", valid_from=iso_days_ago(100))
        seed_memory(conn, "deploy runbook gamma edition", valid_from=iso_days_ago(3))
        conn.commit()
        hits = search(conn, "deploy runbook edition", trust_tier="owner",
                      date_from=iso_days_ago(150), date_to=iso_days_ago(50))
        texts = " ".join(h.text for h in hits if h.kind == "memory")
        assert "beta edition" in texts        # the only in-window memory
        assert "alpha edition" not in texts    # older than the window
        assert "gamma edition" not in texts    # newer than the window
        assert mid in [h.id for h in hits]
    finally:
        conn.close()
