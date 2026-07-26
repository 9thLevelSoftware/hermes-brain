"""evalkit: paraphrase query generation + leg-configuration comparison.

Hermetic — the aux LLM is a scripted fake installed via set_llm_for_tests
around every step that can reach it (CLAUDE.md gotcha: if hermes-agent happens
to be importable, an unfaked call hits the real aux client).
"""

from __future__ import annotations

import json

import pytest
from brain import llm
from brain.config import DEFAULTS
from brain.evalkit import (
    generate_queryset,
    load_queryset,
    queryset_path,
    run_comparison,
    save_queryset,
)
from brain.evalkit.compare import BASELINE, CONFIGURATIONS, format_report
from brain.evalkit.generate import _too_similar
from conftest import seed_memory


@pytest.fixture(autouse=True)
def _clear_llm():
    yield
    llm.set_llm_for_tests(None)


def _cfg(**over):
    return {**DEFAULTS, **over}


def _seed(conn, n=10):
    uids = []
    for i in range(n):
        seed_memory(
            conn,
            f"The staging database number {i} is PostgreSQL 14 running on a "
            f"rented virtual machine with sixteen gigabytes of memory available.",
        )
    for row in conn.execute("SELECT uid FROM memories ORDER BY id"):
        uids.append(row["uid"])
    return uids


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def _fake_llm_returning_questions(prefix="what powers the test environment"):
    def _fn(prompt, *, system=None, max_tokens=0, **kw):
        n = prompt.count("--- SOURCE ")
        return json.dumps([{"n": i + 1, "q": f"{prefix} {i}?"} for i in range(n)])
    return _fn


def test_generate_builds_query_gold_pairs(conn, tmp_home):
    _seed(conn, 10)
    llm.set_llm_for_tests(_fake_llm_returning_questions())
    queries, meta = generate_queryset(conn, _cfg(), limit=10)

    assert len(queries) == 10
    assert meta["sampled"] == 10 and meta["batches"] >= 1
    for item in queries:
        assert item["query"] and item["gold"]
        assert item["source_uid"] == item["gold"][0]


def test_generate_batches_rather_than_one_call_per_item(conn, tmp_home):
    """150 queries must not mean 150 LLM calls."""
    _seed(conn, 16)
    calls = []

    def _fn(prompt, *, system=None, max_tokens=0, **kw):
        calls.append(prompt)
        n = prompt.count("--- SOURCE ")
        return json.dumps([{"n": i + 1, "q": f"question {i}?"} for i in range(n)])

    llm.set_llm_for_tests(_fn)
    generate_queryset(conn, _cfg(), limit=16)
    assert len(calls) == 2, "16 sources at BATCH=8 should be exactly 2 calls"


def test_generate_rejects_verbatim_questions(conn, tmp_home):
    """A 'paraphrase' that lifts the source's rare words hands BM25 an exact
    match and makes the whole benchmark meaningless."""
    _seed(conn, 4)

    def _echo(prompt, *, system=None, max_tokens=0, **kw):
        n = prompt.count("--- SOURCE ")
        # Echo distinctive source vocabulary straight back.
        return json.dumps([
            {"n": i + 1,
             "q": "staging database PostgreSQL running rented virtual machine gigabytes"}
            for i in range(n)
        ])

    llm.set_llm_for_tests(_echo)
    queries, meta = generate_queryset(conn, _cfg(), limit=4)
    assert queries == []
    assert meta["rejected_verbatim"] == 4


def test_too_similar_flags_copied_identifiers():
    source = "The deployment pipeline uses buildkite with a customized runner image"
    assert _too_similar("buildkite customized deployment pipeline runner image", source)
    assert not _too_similar("how do we ship code to production?", source)


def test_generate_stops_early_when_the_llm_is_unavailable(conn, tmp_home):
    _seed(conn, 16)
    calls = {"n": 0}

    def _fn(prompt, *, system=None, max_tokens=0, **kw):
        calls["n"] += 1
        if calls["n"] > 1:
            raise llm.LLMUnavailable("budget spent")
        n = prompt.count("--- SOURCE ")
        return json.dumps([{"n": i + 1, "q": f"question {i}?"} for i in range(n)])

    llm.set_llm_for_tests(_fn)
    queries, meta = generate_queryset(conn, _cfg(), limit=16)
    assert len(queries) == 8, "the first batch survives"
    assert "stopped_early" in meta


def test_generate_on_an_empty_brain_says_so(conn, tmp_home):
    llm.set_llm_for_tests(_fake_llm_returning_questions())
    queries, meta = generate_queryset(conn, _cfg(), limit=10)
    assert queries == []
    assert "empty" in meta.get("note", "")


def test_generate_never_samples_peer_cards(conn, tmp_home):
    """peer_card is the owner's private theory-of-mind of a person; it must not
    end up quoted in a query set."""
    mem_id = seed_memory(conn, "A long private note about a specific person and "
                              "their communication preferences over many words.")
    conn.execute("UPDATE memories SET kind='peer_card' WHERE id=?", (mem_id,))
    conn.commit()
    llm.set_llm_for_tests(_fake_llm_returning_questions())
    queries, meta = generate_queryset(conn, _cfg(), limit=10)
    assert queries == [] and meta["sampled"] == 0


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------

def test_queryset_round_trips(tmp_home):
    queries = [{"query": "q1", "gold": ["ABC"], "source_kind": "memory",
                "source_uid": "ABC"}]
    path = save_queryset(tmp_home, queries, meta={"sampled": 1})
    assert path == queryset_path(tmp_home)
    data = load_queryset(tmp_home)
    assert data["queries"] == queries and data["meta"]["sampled"] == 1


def test_load_queryset_tolerates_missing_and_corrupt(tmp_home):
    assert load_queryset(tmp_home) is None
    path = queryset_path(tmp_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert load_queryset(tmp_home) is None
    path.write_text('{"schema":1}', encoding="utf-8")
    assert load_queryset(tmp_home) is None


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def test_comparison_scores_the_fts_baseline(conn, tmp_home):
    uids = _seed(conn, 6)
    queries = [{"query": "staging database number 0", "gold": [uids[0]],
                "source_kind": "memory", "source_uid": uids[0]}]
    report = run_comparison(conn, queries, embedder=None, reranker=None)

    base = next(r for r in report["results"] if r["name"] == BASELINE)
    assert base["skipped"] is None and base["n"] == 1
    assert base["mrr"] > 0, "FTS should find a verbatim match"


def test_configurations_needing_absent_legs_are_skipped_not_scored(conn, tmp_home):
    """An absent reranker scoring identically to the baseline is exactly how a
    stage that never executed got reported as 'no improvement'."""
    uids = _seed(conn, 4)
    queries = [{"query": "staging database", "gold": [uids[0]],
                "source_kind": "memory", "source_uid": uids[0]}]
    report = run_comparison(conn, queries, embedder=None, reranker=None)

    by_name = {r["name"]: r for r in report["results"]}
    # Every config needing a leg we do not have is skipped WITH a reason,
    # never silently scored (which would read as "this leg changed nothing").
    for name in ("+vec", "+vec+rerank", "+vec+graph", "+vec+facts", "all"):
        assert by_name[name]["skipped"], f"{name} should be skipped"
    # ...but best-available still RUNS: it is computed from what is present, so
    # you can always see your real stack even when a leg's model is missing.
    assert by_name["best-available"]["skipped"] is None
    assert by_name["+vec"]["skipped"] == "no embedder for this tier"
    # ...and skipped configs never appear in the paired comparison.
    for name in ("+vec", "+vec+rerank", "+vec+graph", "+vec+facts", "all"):
        assert name not in report["paired"]
    # best-available did run, so it is paired — and with no legs available it
    # is the baseline by another name, hence all ties.
    assert report["paired"]["best-available"]["win"] == 0


def test_rerank_skip_reason_names_the_missing_model(conn, tmp_home, monkeypatch):
    """With an embedder AND an index present, the reranker becomes the blocking
    leg — and the message has to say which one, or 'skipped' is useless."""
    from brain.evalkit import compare as compare_mod

    monkeypatch.setattr(compare_mod, "_vector_index_present", lambda _c: True)
    uids = _seed(conn, 3)
    queries = [{"query": "staging database", "gold": [uids[0]],
                "source_kind": "memory", "source_uid": uids[0]}]

    class _FakeEmbedder:
        name, dim = "stub-test", 256

        def encode_queries(self, texts):
            return [[0.0] * self.dim for _ in texts]

        def encode_documents(self, texts):
            return [[0.0] * self.dim for _ in texts]

    report = run_comparison(conn, queries, embedder=_FakeEmbedder(), reranker=None)
    by_name = {r["name"]: r for r in report["results"]}
    assert "rerank model absent" in by_name["+vec+rerank"]["skipped"]
    assert by_name["+vec"]["skipped"] is None, "the vector leg should now run"


def test_report_covers_every_configuration(conn, tmp_home):
    uids = _seed(conn, 4)
    queries = [{"query": "staging database", "gold": [uids[0]],
                "source_kind": "memory", "source_uid": uids[0]}]
    report = run_comparison(conn, queries, embedder=None, reranker=None)
    # Every fixed configuration, PLUS the runtime-computed best-available row.
    assert len(report["results"]) == len(CONFIGURATIONS) + 1
    assert report["results"][-1]["name"] == "best-available"

    text = format_report(report)
    for cfg in CONFIGURATIONS:
        assert cfg.name in text
    assert "best-available" in text
    assert "n=1 is small" in text, "small samples must be called out"


def test_comparison_survives_a_query_with_no_gold(conn, tmp_home):
    _seed(conn, 3)
    queries = [{"query": "", "gold": []}, {"query": "x", "gold": []}]
    report = run_comparison(conn, queries, embedder=None, reranker=None)
    base = next(r for r in report["results"] if r["name"] == BASELINE)
    assert base["skipped"] == "no scorable queries"


def test_vec_is_skipped_when_the_index_is_empty(conn, tmp_home):
    """An embedder with no embedded rows is not a vector leg. Reporting it as
    run — and identical to FTS — reads as 'vectors add nothing' when it means
    'vectors were never consulted'."""
    uids = _seed(conn, 3)
    queries = [{"query": "staging database", "gold": [uids[0]],
                "source_kind": "memory", "source_uid": uids[0]}]

    class _FakeEmbedder:
        name, dim = "stub-test", 256

        def encode_queries(self, texts):
            return [[0.0] * self.dim for _ in texts]

        def encode_documents(self, texts):
            return [[0.0] * self.dim for _ in texts]

    report = run_comparison(conn, queries, embedder=_FakeEmbedder(), reranker=None)
    by_name = {r["name"]: r for r in report["results"]}
    assert "no vector index" in by_name["+vec"]["skipped"]


# ---------------------------------------------------------------------------
# score_weights + best-available (§G1, §G3)
# ---------------------------------------------------------------------------

def test_score_weights_measures_a_candidate_without_applying_it(conn, tmp_home):
    from brain.evalkit.compare import score_weights
    from brain.recall import weights as weights_mod

    uids = _seed(conn, 6)
    queries = [{"query": "staging database number 0", "gold": [uids[0]],
                "source_kind": "memory", "source_uid": uids[0]}]

    result = score_weights(conn, queries, {"fts": 2.0, "vec": 0.5},
                           embedder=None, reranker=None)
    assert result.n == 1
    assert weights_mod.load(conn) == weights_mod.DEFAULT, "measuring must not commit"


def test_score_weights_none_scores_the_active_set(conn, tmp_home):
    from brain.evalkit.compare import score_weights

    uids = _seed(conn, 4)
    queries = [{"query": "staging database number 0", "gold": [uids[0]],
                "source_kind": "memory", "source_uid": uids[0]}]
    assert score_weights(conn, queries, None, embedder=None).n == 1


def test_best_available_reflects_what_is_actually_present(conn):
    """A fixed `all` row is unreachable whenever a leg's model is missing, so
    you cannot see your real stack — the exact blind spot evalkit exists to
    prevent."""
    from brain.evalkit.compare import _best_available

    cfg = _best_available(conn, embedder=None, reranker=None)
    assert cfg.name == "best-available"
    assert cfg.vec is False and cfg.rerank is False
    assert cfg.graph is True and cfg.facts is True


def test_baseline_round_trip_and_delta_column(conn, tmp_home):
    """Approving a tune proposal changes retrieval; without a stored before-run
    there is nothing to compare the after-run against."""
    from brain.evalkit import load_baseline, save_baseline
    from brain.evalkit.compare import format_report

    assert load_baseline(tmp_home) is None

    uids = _seed(conn, 6)
    queries = [{"query": "staging database number 0", "gold": [uids[0]],
                "source_kind": "memory", "source_uid": uids[0]}]
    report = run_comparison(conn, queries, embedder=None, reranker=None)
    save_baseline(tmp_home, report)

    loaded = load_baseline(tmp_home)
    assert loaded is not None
    text = format_report(report, loaded)
    assert "ΔMRR vs saved" in text
    assert "+0.0000" in text, "the same run against itself is a zero delta"


def test_format_report_without_a_baseline_has_no_delta_column(conn, tmp_home):
    from brain.evalkit.compare import format_report

    uids = _seed(conn, 4)
    queries = [{"query": "staging database", "gold": [uids[0]],
                "source_kind": "memory", "source_uid": uids[0]}]
    text = format_report(run_comparison(conn, queries, embedder=None), None)
    assert "ΔMRR" not in text


def test_corrupt_baseline_reads_as_absent(tmp_home):
    from brain.evalkit import baseline_path, load_baseline

    path = baseline_path(tmp_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert load_baseline(tmp_home) is None


def test_baseline_from_a_different_queryset_shows_no_delta(conn, tmp_home, capsys):
    """Subtracting MRRs measured on different datasets produces a number that
    looks like a retrieval delta and is an artifact (review round 2, P2)."""
    from brain.evalkit import save_baseline
    from brain.evalkit.compare import format_report

    uids = _seed(conn, 6)
    first = [{"query": "staging database number 0", "gold": [uids[0]],
              "source_kind": "memory", "source_uid": uids[0]}]
    save_baseline(tmp_home, run_comparison(conn, first, embedder=None))

    # A DIFFERENT query set — e.g. after re-running eval --generate.
    second = [{"query": "staging database number 1", "gold": [uids[1]],
               "source_kind": "memory", "source_uid": uids[1]}]
    report = run_comparison(conn, second, embedder=None)

    from brain.evalkit import load_baseline

    text = format_report(report, load_baseline(tmp_home))
    assert "ΔMRR" not in text
    assert "DIFFERENT query set" in text


def test_baseline_from_the_same_queryset_does_show_a_delta(conn, tmp_home):
    from brain.evalkit import load_baseline, save_baseline
    from brain.evalkit.compare import format_report

    uids = _seed(conn, 6)
    queries = [{"query": "staging database number 0", "gold": [uids[0]],
                "source_kind": "memory", "source_uid": uids[0]}]
    save_baseline(tmp_home, run_comparison(conn, queries, embedder=None))
    report = run_comparison(conn, queries, embedder=None)
    assert "ΔMRR vs saved" in format_report(report, load_baseline(tmp_home))


def test_queryset_fingerprint_is_order_and_content_sensitive():
    from brain.evalkit import queryset_fingerprint

    a = [{"query": "q1", "gold": ["A"]}, {"query": "q2", "gold": ["B"]}]
    b = [{"query": "q1", "gold": ["A"]}, {"query": "q2", "gold": ["C"]}]
    assert queryset_fingerprint(a) == queryset_fingerprint(list(a))
    assert queryset_fingerprint(a) != queryset_fingerprint(b)
    assert queryset_fingerprint([]) == ""
