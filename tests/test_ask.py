"""Phase D dialectic "ask" agent tests. Hermetic: the LLM is always a scripted
fake installed via llm.set_llm_for_tests — no network, no Hermes. The fake
returns a programmed SEQUENCE of action JSONs (one per loop iteration)."""

from __future__ import annotations

import json

import pytest
from brain import llm
from brain.recall.ask import AskResult, ask
from conftest import seed_memory


@pytest.fixture(autouse=True)
def _clear_llm_override():
    yield
    llm.set_llm_for_tests(None)


class ScriptedLLM:
    """Returns a queued sequence of replies (JSON strings), last one sticky.
    Records every prompt so tests can assert what the loop showed the model.
    A reply may be an Exception instance (raised) or a plain string."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.systems = []

    def __call__(self, prompt, *, system=None, max_tokens=0):
        self.prompts.append(prompt)
        self.systems.append(system)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


def _action(name, **kw):
    return json.dumps({"action": name, **kw})


def _uid8(conn, mem_id):
    return conn.execute(
        "SELECT uid FROM memories WHERE id=?", (mem_id,)).fetchone()["uid"][:8]


def _ledger_count(conn):
    return conn.execute("SELECT COUNT(*) FROM llm_ledger").fetchone()[0]


# ---------------------------------------------------------------------------
# 1. Scripted multi-step loop ending in `answer`
# ---------------------------------------------------------------------------

def test_multistep_loop_answers_with_citations(conn):
    mid = seed_memory(conn, "The deploy host for staging is prod-box-7.")
    uid8 = _uid8(conn, mid)
    llm.set_llm_for_tests(ScriptedLLM([
        _action("search_memory", query="deploy host staging"),
        _action("answer", text="Staging deploys to prod-box-7.",
                citations=[uid8]),
    ]))

    res = ask(conn, "Where does staging deploy?", level="deep")

    assert isinstance(res, AskResult)
    assert res.answered is True
    assert res.degraded is False
    assert res.answer == "Staging deploys to prod-box-7."
    assert res.iterations == 2
    assert "search_memory" in res.tools_used
    assert any(c["uid"] == uid8 for c in res.citations)
    # the cited snippet was populated from the scoped search result
    assert any(c["uid"] == uid8 and c["snippet"] for c in res.citations)


# ---------------------------------------------------------------------------
# 2. Iteration-cap termination (fake never answers)
# ---------------------------------------------------------------------------

def test_iteration_cap_terminates_without_infinite_loop(conn):
    seed_memory(conn, "A note about caching and prefetch behavior.")
    # Sticky reply: ALWAYS search, never answer.
    llm.set_llm_for_tests(ScriptedLLM([
        _action("search_memory", query="caching"),
    ]))

    res = ask(conn, "How does caching work?", level="deep", max_iterations=3)

    assert res.iterations == 3           # stopped exactly at the cap
    assert res.answered is False
    assert res.answer is None
    assert res.degraded is False
    assert res.tools_used == ["search_memory"] * 3


def test_fast_level_caps_at_two(conn):
    seed_memory(conn, "Fast-level note.")
    llm.set_llm_for_tests(ScriptedLLM([_action("search_memory", query="note")]))

    res = ask(conn, "Anything?", level="fast", max_iterations=6)

    assert res.level == "fast"
    assert res.iterations == 2           # min(2, 6)
    assert res.answered is False


# ---------------------------------------------------------------------------
# 3. Abstention
# ---------------------------------------------------------------------------

def test_abstention_marks_not_answered(conn):
    seed_memory(conn, "Unrelated content about the weather.")
    llm.set_llm_for_tests(ScriptedLLM([
        _action("search_memory", query="quarterly revenue"),
        _action("answer",
                text="I don't know — there is no evidence of that in memory.",
                citations=[]),
    ]))

    res = ask(conn, "What was Q3 revenue?", level="deep")

    assert res.answered is False
    assert res.answer is None
    assert res.degraded is False


def test_explicit_abstain_flag(conn):
    seed_memory(conn, "Some stored fact.")
    llm.set_llm_for_tests(ScriptedLLM([
        _action("answer", text="Not enough to say confidently.",
                citations=[], abstain=True),
    ]))

    res = ask(conn, "Unanswerable?", level="deep")
    assert res.answered is False
    assert res.iterations == 1


# ---------------------------------------------------------------------------
# 4. Contradiction — both sides surfaced and cited
# ---------------------------------------------------------------------------

def test_contradiction_presents_both_sources(conn):
    a = seed_memory(conn, "The database backend is Postgres.")
    b = seed_memory(conn, "The database backend is now MySQL.")
    ua, ub = _uid8(conn, a), _uid8(conn, b)
    llm.set_llm_for_tests(ScriptedLLM([
        _action("search_memory", query="database backend"),
        _action("answer",
                text="Sources conflict: one says Postgres, another says MySQL.",
                citations=[ua, ub]),
    ]))

    res = ask(conn, "What is the database backend?", level="deep")

    assert res.answered is True
    cited = {c["uid"] for c in res.citations}
    assert ua in cited and ub in cited
    # both citations resolved to real snippets => search_memory surfaced both
    assert all(c["snippet"] for c in res.citations if c["uid"] in (ua, ub))


# ---------------------------------------------------------------------------
# 5. LLMUnavailable degradation (never raises)
# ---------------------------------------------------------------------------

def test_llm_unavailable_degrades_to_recall_only(conn):
    mid = seed_memory(conn, "The API rate limit is 1000 requests per minute.")
    uid8 = _uid8(conn, mid)
    # An empty reply makes llm.call_text raise LLMUnavailable on the first turn.
    llm.set_llm_for_tests(ScriptedLLM([""]))

    res = ask(conn, "What is the API rate limit?", level="deep")

    assert res.degraded is True
    assert res.answered is False
    assert res.answer is None
    assert res.iterations == 0
    # dual-prefetch citations survive the degradation
    assert any(c["uid"] == uid8 for c in res.citations)


def test_raising_llm_degrades(conn):
    seed_memory(conn, "Recoverable seed content about deploys.")
    llm.set_llm_for_tests(ScriptedLLM([llm.LLMUnavailable("budget spent")]))

    res = ask(conn, "deploys?", level="deep")
    assert res.degraded is True
    assert res.answered is False
    # ask must NEVER raise — reaching here is the assertion


# ---------------------------------------------------------------------------
# 6. Per-iteration ledger metering
# ---------------------------------------------------------------------------

def test_ledger_grows_with_iterations(conn):
    seed_memory(conn, "A fact about scaling the worker pool.")
    before = _ledger_count(conn)
    llm.set_llm_for_tests(ScriptedLLM([
        _action("search_memory", query="worker pool"),
        _action("search_memory", query="scaling"),
        _action("answer", text="Scale the worker pool horizontally.",
                citations=[]),
    ]))

    res = ask(conn, "How to scale workers?", level="deep")

    assert res.iterations == 3
    # each successful call_json meters exactly one clean ledger row
    assert _ledger_count(conn) - before == res.iterations


# ---------------------------------------------------------------------------
# 7. SCOPE — a non-owner never reaches a peer_card or a foreign row
# ---------------------------------------------------------------------------

def test_non_owner_scope_never_leaks_peer_card_or_foreign_rows(conn):
    # Seed three matching memories under the shared query term "plan".
    pub = seed_memory(conn, "The public rollout plan ships on Monday.")
    seed_memory(conn, "Peer plan: they distrust the new schema.",
                kind="peer_card")
    # A memory scoped to a DIFFERENT principal.
    foreign = seed_memory(conn, "Rival plan data lives on host zeta.")
    conn.execute("UPDATE memories SET scope_user=? WHERE id=?",
                 ("rival-principal", foreign))
    conn.commit()

    pub8 = _uid8(conn, pub)

    fake = ScriptedLLM([
        _action("search_memory", query="plan"),
        _action("answer", text="The public rollout plan ships Monday.",
                citations=[pub8]),
    ])
    llm.set_llm_for_tests(fake)

    res = ask(conn, "What is the plan?", level="deep",
              trust_tier="known_user", principal_id="peer-9")

    # The loop showed the model its search results inside the prompts; the
    # forbidden rows must appear in NONE of them (covers dual prefetch too).
    # Assert on distinctive content, not uid8 — same-ms ULIDs share a prefix.
    all_prompts = "\n".join(fake.prompts)
    assert "distrust the new schema" not in all_prompts, "peer_card leaked"
    assert "Rival plan data" not in all_prompts, "foreign-principal row leaked"
    assert "public rollout plan" in all_prompts, "unscoped row should be visible"
    # No forbidden content rode out on the citation snippets either.
    snippets = " ".join(c["snippet"] for c in res.citations)
    assert "distrust" not in snippets and "Rival" not in snippets


# ---------------------------------------------------------------------------
# Bonus — grep_episodes + get_reasoning_chain dispatch paths (owner)
# ---------------------------------------------------------------------------

def test_grep_and_reasoning_chain_actions(conn):
    from conftest import seed_episode

    seed_episode(conn, "How do I rotate the signing key?",
                 "Run hermes brain rotate-key.", session_id="s1")
    mid = seed_memory(conn, "The signing key rotates every 90 days.")
    uid8 = _uid8(conn, mid)

    llm.set_llm_for_tests(ScriptedLLM([
        _action("grep_episodes", pattern="signing key"),
        _action("get_reasoning_chain", uid=uid8),
        _action("answer", text="Rotate the signing key every 90 days.",
                citations=[uid8]),
    ]))

    res = ask(conn, "Tell me about the signing key.", level="deep")

    assert res.answered is True
    assert "grep_episodes" in res.tools_used
    assert "get_reasoning_chain" in res.tools_used
    assert res.iterations == 3


def test_grep_evidence_is_citable(conn):
    """An answer built from grep_episodes must be able to CITE the grepped
    episode. grep now registers its hits, so finalization keeps the citation
    instead of discarding it (PR #5 review: grep answers came back citations=[])."""
    from conftest import seed_episode

    ep = seed_episode(conn, "the office wifi password is hunter2-max",
                      "saved that for you")
    conn.execute("UPDATE episodes SET trust_tier='owner' WHERE id=?", (ep,))
    conn.commit()
    ep_uid8 = conn.execute(
        "SELECT uid FROM episodes WHERE id=?", (ep,)).fetchone()["uid"][:8]

    llm.set_llm_for_tests(ScriptedLLM([
        _action("grep_episodes", pattern="wifi password"),
        _action("answer", text="It's hunter2-max.", citations=[ep_uid8]),
    ]))
    res = ask(conn, "what is the office wifi password?", trust_tier="owner")
    assert res.answered
    assert any(c["uid"] == ep_uid8 and c["snippet"] for c in res.citations)
