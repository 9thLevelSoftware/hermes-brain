"""P3 tool surface: schemas, dispatch, scoping, dedup, outcome, manage.

Hermetic: no network, no LLM — the optional embed path uses StubEmbedder
and is skipped when sqlite-vec is not importable.
"""

from __future__ import annotations

import json

import pytest
from brain import config as brain_config  # noqa: E402
from brain import tools  # noqa: E402
from brain.store import db  # noqa: E402
from conftest import seed_memory  # noqa: E402

TOOL_NAMES = ("brain_recall", "brain_remember", "brain_outcome", "brain_manage")


def ctx(**kw):
    defaults = dict(session_id="sess-tools", principal_id="owner",
                    trust_tier="owner", source_author=None, platform="cli",
                    embedder=None, config=dict(brain_config.DEFAULTS))
    defaults.update(kw)
    return tools.ToolContext(**defaults)


def call(conn, tool, args, c=None):
    out = tools.dispatch(conn, tool, args, ctx=c or ctx())
    assert isinstance(out, str)
    return json.loads(out)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

def test_schema_list_shape():
    schemas = tools.get_schemas()
    assert len(schemas) == 4
    names = [s["function"]["name"] for s in schemas]
    assert names == list(TOOL_NAMES)
    for s in schemas:
        assert s["type"] == "function"
        fn = s["function"]
        assert fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert 1 <= len(params["properties"]) <= 6  # never a 28-param union
        for prop in params["properties"].values():
            assert prop.get("description") or prop.get("enum")


def test_schema_documents_query_id_exclusivity_and_enums():
    by_name = {s["function"]["name"]: s["function"] for s in tools.get_schemas()}
    recall = by_name["brain_recall"]["parameters"]["properties"]
    assert "mutually exclusive" in recall["query"]["description"]
    assert "mutually exclusive" in recall["id"]["description"]
    assert recall["depth"]["enum"] == ["quick", "deep"]
    assert recall["limit"]["maximum"] == 25
    assert by_name["brain_remember"]["parameters"]["required"] == ["content"]
    assert by_name["brain_outcome"]["parameters"]["properties"]["outcome"]["enum"] \
        == ["worked", "failed", "mixed"]
    assert set(by_name["brain_manage"]["parameters"]["properties"]["action"]["enum"]) \
        == {"forget", "pin", "unpin", "incognito_on", "incognito_off"}


# ---------------------------------------------------------------------------
# brain_recall
# ---------------------------------------------------------------------------

def _uid8(conn, mem_id):
    return conn.execute("SELECT uid FROM memories WHERE id=?",
                        (mem_id,)).fetchone()["uid"][:8]


def test_recall_by_query_and_scoping(conn):
    open_id = seed_memory(conn, "the deploy pipeline runs on github actions")
    scoped_id = seed_memory(conn, "deploy pipeline secret token lives in vault")
    conn.execute("UPDATE memories SET scope_user='owner' WHERE id=?", (scoped_id,))
    conn.commit()

    owner_out = call(conn, "brain_recall", {"query": "deploy pipeline"})
    owner_text = json.dumps(owner_out["results"])
    assert owner_out["total"] == 2
    assert "github actions" in owner_text
    assert "vault" in owner_text
    assert _uid8(conn, open_id) in owner_text  # index-line grammar carries ids
    assert "hint" in owner_out and "brain_recall(id=" in owner_out["hint"]

    guest = ctx(principal_id="guest-bob", trust_tier="known_user",
                source_author="bob")
    guest_out = call(conn, "brain_recall", {"query": "deploy pipeline"}, guest)
    guest_text = json.dumps(guest_out["results"])
    assert guest_out["total"] == 1
    assert "github actions" in guest_text
    assert "vault" not in guest_text  # owner-scoped row hidden from known_user


def test_recall_depth_deep_full_text_and_note(conn):
    for i in range(4):
        seed_memory(conn, f"gateway retry backoff strategy variant {i} details")
    out = call(conn, "brain_recall", {"query": "gateway retry backoff",
                                      "depth": "deep"})
    assert out["total"] == 4
    with_text = [r for r in out["results"] if "text" in r]
    assert 1 <= len(with_text) <= 3  # full text only for the top 3
    assert all("line" in r for r in out["results"])
    assert "note" in out  # graph placeholder note


def test_recall_by_id_prefix_returns_envelope(conn):
    mem_id = seed_memory(conn, "postgres 16 is the production database",
                         kind="decision", tags=("infra",))
    uid = conn.execute("SELECT uid FROM memories WHERE id=?",
                       (mem_id,)).fetchone()["uid"]
    out = call(conn, "brain_recall", {"id": uid[:8].lower()})  # case-insensitive
    assert out["total"] == 1
    env = out["results"][0]
    assert env["uid"] == uid
    assert env["kind"] == "decision"
    assert env["epistemic"] == "observation"
    assert env["trust_tier"] == "owner"
    assert env["valid_from"] and env["recorded_at"]
    assert env["outcome"] is None
    assert env["counts"]["verification"] == 1
    assert env["content"] == "postgres 16 is the production database"
    assert env["tags"] == ["infra"]


def test_recall_id_scoping_does_not_leak(conn):
    mem_id = seed_memory(conn, "owner private note about credentials")
    conn.execute("UPDATE memories SET scope_user='owner' WHERE id=?", (mem_id,))
    conn.commit()
    uid8 = _uid8(conn, mem_id)
    guest = ctx(principal_id="guest-bob", trust_tier="known_user")
    out = call(conn, "brain_recall", {"id": uid8}, guest)
    assert "error" in out
    assert "no current memory matches" in out["error"]  # same shape as a miss


def test_recall_id_errors_teach(conn):
    # too short
    out = call(conn, "brain_recall", {"id": "01AB"})
    assert "too short" in out["error"]
    assert "brain_recall(id=" in out["recovery_hint"]
    # not found
    out = call(conn, "brain_recall", {"id": "ZZZZZZZZ"})
    assert "no current memory matches" in out["error"]
    assert "brain_recall(query=" in out["recovery_hint"]
    # ambiguous: two crafted uids sharing a 6+ char prefix
    a = seed_memory(conn, "ambiguity target alpha")
    b = seed_memory(conn, "ambiguity target beta")
    conn.execute("UPDATE memories SET uid='TESTAMBIGAAAAAAAAAAAAAAAAA' WHERE id=?", (a,))
    conn.execute("UPDATE memories SET uid='TESTAMBIGBBBBBBBBBBBBBBBBB' WHERE id=?", (b,))
    conn.commit()
    out = call(conn, "brain_recall", {"id": "TESTAMBIG"})
    assert "ambiguous" in out["error"]
    assert "more characters" in out["recovery_hint"]


def test_recall_query_and_id_mutually_exclusive(conn):
    out = call(conn, "brain_recall", {"query": "x y", "id": "01ABCDEF"})
    assert "mutually exclusive" in out["error"]
    out = call(conn, "brain_recall", {})
    assert "needs query or id" in out["error"]
    assert "brain_recall(query=" in out["recovery_hint"]


def test_recall_bad_kind_and_depth_teach(conn):
    out = call(conn, "brain_recall", {"query": "x", "kind": "note"})
    assert "unknown kind" in out["error"]
    assert "fact|decision|preference|warning|insight" in out["recovery_hint"]
    out = call(conn, "brain_recall", {"query": "x", "depth": "graph"})
    assert "unknown depth" in out["error"]


# ---------------------------------------------------------------------------
# brain_remember
# ---------------------------------------------------------------------------

def test_remember_applies_kind_tags_project_ttl_and_caps_trust(conn):
    out = call(conn, "brain_remember", {
        "content": "the user prefers tabs in Makefiles",
        "kind": "preference", "tags": ["style", "make"],
        "project": "hermes", "ttl_days": 7,
    })  # ctx trust is OWNER — must still be capped at 'agent'
    assert out["deduped_against"] is None
    row = conn.execute(
        "SELECT * FROM memories WHERE uid LIKE ?", (out["id"] + "%",)
    ).fetchone()
    assert row["kind"] == "preference"
    assert json.loads(row["tags"]) == ["style", "make"]
    assert row["scope_project"] == "hermes"
    assert row["ttl_at"] is not None and row["ttl_at"] > db.iso_now()
    assert row["half_life_days"] == 7.0
    assert row["trust_tier"] == "agent"  # model writes are 'agent' at most
    assert row["created_by"] == "memory_tool"


def test_remember_ttl_half_life_capped_at_30(conn):
    out = call(conn, "brain_remember", {"content": "quarterly numbers are draft",
                                        "ttl_days": 90})
    row = conn.execute("SELECT half_life_days, ttl_at FROM memories WHERE uid LIKE ?",
                       (out["id"] + "%",)).fetchone()
    assert row["half_life_days"] == 30.0
    assert row["ttl_at"] is not None


def test_remember_lower_trust_caps_lower(conn):
    untrusted = ctx(principal_id=None, trust_tier="untrusted")
    out = call(conn, "brain_remember", {"content": "stranger claims the sky is green"},
               untrusted)
    row = conn.execute("SELECT trust_tier FROM memories WHERE uid LIKE ?",
                       (out["id"] + "%",)).fetchone()
    assert row["trust_tier"] == "untrusted"  # lower than agent stays lower


def test_remember_dedup_reports_merge(conn):
    first = call(conn, "brain_remember", {"content": "the API rate limit is 100 rps",
                                          "kind": "fact", "tags": ["api"]})
    again = call(conn, "brain_remember", {"content": "The API rate limit is 100 RPS",
                                          "kind": "warning"})  # hash-normalized dup
    assert again["deduped_against"] == first["id"]
    assert again["id"] == first["id"]
    row = conn.execute("SELECT kind, verification_count FROM memories WHERE uid LIKE ?",
                       (first["id"] + "%",)).fetchone()
    assert row["verification_count"] == 2  # confirmation, not an edit
    assert row["kind"] == "fact"           # dup did not overwrite kind


def test_remember_errors_teach(conn):
    out = call(conn, "brain_remember", {})
    assert "content is required" in out["error"]
    assert "brain_remember(content=" in out["recovery_hint"]
    out = call(conn, "brain_remember", {"content": "x", "kind": "note"})
    assert "unknown kind" in out["error"]
    out = call(conn, "brain_remember", {"content": "x", "tags": "solo"})
    assert "array of strings" in out["error"]
    assert 'tags=["deploy", "ci"]' in out["recovery_hint"]
    out = call(conn, "brain_remember", {"content": "x", "ttl_days": -3})
    assert "ttl_days" in out["error"]


def test_remember_embeds_when_embedder_present(conn):
    pytest.importorskip("sqlite_vec")
    from brain.recall.embed import StubEmbedder
    from brain.store import vec as vec_store

    embedder = StubEmbedder()
    assert vec_store.ensure_tables(conn, embedder.dim, embedder.name)
    out = call(conn, "brain_remember", {"content": "vector indexed memory"},
               ctx(embedder=embedder))
    row = conn.execute("SELECT id, embedded_with FROM memories WHERE uid LIKE ?",
                       (out["id"] + "%",)).fetchone()
    assert row["embedded_with"] == embedder.name
    n = conn.execute("SELECT count(*) AS n FROM mem_vec WHERE id=?",
                     (row["id"],)).fetchone()["n"]
    assert n == 1


# ---------------------------------------------------------------------------
# brain_outcome
# ---------------------------------------------------------------------------

def test_outcome_mixed_stored_as_partial_with_audit(conn):
    mem_id = seed_memory(conn, "we chose sqlite-vec over faiss", kind="decision")
    uid = conn.execute("SELECT uid FROM memories WHERE id=?",
                       (mem_id,)).fetchone()["uid"]
    out = call(conn, "brain_outcome",
               {"id": uid[:8], "outcome": "mixed", "note": "fast but int8-only"})
    assert out == {"id": uid[:8], "outcome": "partial", "note_saved": True}
    row = conn.execute("SELECT * FROM memories WHERE id=?", (mem_id,)).fetchone()
    assert row["outcome"] == "partial"          # schema stores 'partial'
    assert row["outcome_note"] == "fast but int8-only"
    assert row["outcome_confidence"] is None
    assert row["valid_to"] is None              # NO new version row for outcome
    assert conn.execute("SELECT count(*) AS n FROM memories").fetchone()["n"] == 1
    audit = conn.execute(
        "SELECT * FROM audit_log WHERE action='brain_outcome' AND target=?",
        (uid,)).fetchone()
    assert audit is not None
    assert json.loads(audit["detail"])["outcome"] == "partial"
    # harmful/helpful counters belong to injection feedback — untouched.
    assert row["harmful_count"] == 0 and row["helpful_count"] == 0


def test_outcome_bad_value_teaches(conn):
    seed_memory(conn, "something")
    out = call(conn, "brain_outcome", {"id": "01ABCDEF", "outcome": "great"})
    assert "unknown outcome" in out["error"]
    assert "worked|failed|mixed" in out["recovery_hint"]
    out = call(conn, "brain_outcome", {"outcome": "worked"})
    assert "id is required" in out["error"]


# ---------------------------------------------------------------------------
# brain_manage
# ---------------------------------------------------------------------------

def test_manage_forget_tombstones_and_clears_lane1(conn):
    mem_id = seed_memory(conn, "obsolete server IP is 10.0.0.5", kind="fact")
    uid = conn.execute("SELECT uid FROM memories WHERE id=?",
                       (mem_id,)).fetchone()["uid"]
    conn.execute(
        "INSERT INTO lane1_snapshot (section, rank, memory_id, line, rendered_at)"
        " VALUES ('facts', 0, ?, 'line', ?)", (mem_id, db.iso_now()))
    conn.commit()

    out = call(conn, "brain_manage",
               {"action": "forget", "id": uid[:8], "reason": "server retired"})
    assert out["action"] == "forget"
    assert "tombstoned" in out["note"] and "30" in out["note"]
    row = conn.execute("SELECT status, valid_to FROM memories WHERE id=?",
                       (mem_id,)).fetchone()
    assert row["status"] == "tombstone" and row["valid_to"] is not None
    assert conn.execute("SELECT count(*) AS n FROM lane1_snapshot "
                        "WHERE memory_id=?", (mem_id,)).fetchone()["n"] == 0
    audit = conn.execute("SELECT detail FROM audit_log WHERE action='brain_forget' "
                         "AND target=?", (uid,)).fetchone()
    assert json.loads(audit["detail"])["reason"] == "server retired"
    # excluded from recall immediately
    out = call(conn, "brain_recall", {"query": "obsolete server"})
    assert out["total"] == 0


def test_manage_pin_unpin(conn):
    mem_id = seed_memory(conn, "always run tests before pushing", kind="warning")
    uid8 = _uid8(conn, mem_id)
    out = call(conn, "brain_manage", {"action": "pin", "id": uid8})
    assert out["pinned"] is True
    assert conn.execute("SELECT pinned FROM memories WHERE id=?",
                        (mem_id,)).fetchone()["pinned"] == 1
    out = call(conn, "brain_manage", {"action": "unpin", "id": uid8})
    assert out["pinned"] is False
    assert conn.execute("SELECT pinned FROM memories WHERE id=?",
                        (mem_id,)).fetchone()["pinned"] == 0


def test_manage_incognito_flips_config_file(conn, tmp_home):
    c = ctx(hermes_home=str(tmp_home))
    out = call(conn, "brain_manage", {"action": "incognito_on"}, c)
    assert out["incognito"] is True
    assert "FUTURE sessions" in out["note"] and "next initialize" in out["note"]
    assert brain_config.load_config(tmp_home)["incognito"] is True
    out = call(conn, "brain_manage", {"action": "incognito_off"}, c)
    assert out["incognito"] is False
    assert brain_config.load_config(tmp_home)["incognito"] is False


def test_manage_incognito_without_home_teaches_cli(conn):
    out = call(conn, "brain_manage", {"action": "incognito_on"},
               ctx(hermes_home=None))
    assert "error" in out
    assert "hermes brain incognito on" in out["recovery_hint"]


def test_manage_errors_teach(conn):
    out = call(conn, "brain_manage", {"action": "explode"})
    assert "unknown action" in out["error"]
    assert "brain_manage(action=" in out["recovery_hint"]
    out = call(conn, "brain_manage", {"action": "forget"})
    assert "requires id" in out["error"]
    assert 'action="forget"' in out["recovery_hint"]


# ---------------------------------------------------------------------------
# Dispatch hardening
# ---------------------------------------------------------------------------

def test_unknown_tool_teaches(conn):
    out = call(conn, "brain_zap", {"x": 1})
    assert "unknown tool" in out["error"]
    for name in TOOL_NAMES:
        assert name in out["recovery_hint"]


def test_dispatch_never_raises_on_garbage(conn):
    seed_memory(conn, "fuzz anchor row")
    garbage = [
        None,
        {},
        {"query": {"nested": True}},
        {"query": ["a", "list"]},
        {"query": "ok", "limit": "banana"},
        {"query": "ok", "limit": -5},
        {"id": 12345},
        {"id": ""},
        {"content": 42},
        {"content": "x", "tags": [1, 2]},
        {"content": "x", "ttl_days": "soon"},
        {"content": "x", "ttl_days": True},
        {"id": None, "outcome": "worked"},
        {"action": None},
        {"action": "pin", "id": {"weird": 1}},
        {"outcome": ["worked"]},
    ]
    for tool in TOOL_NAMES + ("", "memories", None):
        for args in garbage:
            raw = tools.dispatch(conn, tool, args, ctx=ctx())
            assert isinstance(raw, str)
            parsed = json.loads(raw)  # always valid JSON
            assert isinstance(parsed, dict)
            if "error" in parsed:
                assert parsed.get("recovery_hint")
    # args of a non-dict type must also come back as a teaching error
    parsed = json.loads(tools.dispatch(conn, "brain_recall", "notadict", ctx=ctx()))
    assert "error" in parsed and parsed["recovery_hint"]
