"""Adversarial: the dialectic ask agent (recall/ask.py) must never leak an
out-of-scope row through ANY action.

A non-owner driving `ask()` through every tool — search_memory, grep_episodes,
get_reasoning_chain, date_range_search — and then citing every uid it can name
must never obtain an owner-scoped memory, a peer_card, or a foreign principal's
row: not in the tool observations the agent reasons over (the prompts), not in
the citations, not in the answer. Scope is enforced inside recall.search for
every leg; ask threads the caller's principal/trust into all of them.

Deterministic + fast: the LLM is a scripted fake (no network, no real model).
"""

from __future__ import annotations

import json

import pytest
from brain import llm
from brain.recall.ask import ask
from conftest import seed_memory

_SECRET = "vaulttoken hunter2 launchcodes"


def _set_scope(conn, mem_id, *, scope_user=None, kind=None):
    if scope_user is not None:
        conn.execute("UPDATE memories SET scope_user=? WHERE id=?", (scope_user, mem_id))
    if kind is not None:
        conn.execute("UPDATE memories SET kind=? WHERE id=?", (kind, mem_id))
    conn.commit()


class ScriptedLLM:
    """Emits a fixed action sequence; records every prompt it is shown so the
    test can assert no out-of-scope content ever reached the model."""

    def __init__(self, actions):
        self._actions = list(actions)
        self.prompts: list[str] = []

    def __call__(self, prompt, *, system=None, max_tokens=1600, **kw):
        self.prompts.append((system or "") + "\n" + prompt)
        if self._actions:
            return json.dumps(self._actions.pop(0))
        return json.dumps({"action": "answer", "text": "done", "citations": [],
                           "abstain": False})

    def seen(self) -> str:
        return "\n".join(self.prompts)


@pytest.fixture(autouse=True)
def _clear_llm():
    yield
    llm.set_llm_for_tests(None)


def _seed_secrets(conn):
    owner_id = seed_memory(conn, f"owner secret: {_SECRET} widget vault", kind="fact")
    _set_scope(conn, owner_id, scope_user="owner")
    peer_id = seed_memory(conn, "mallory keeps probing the widget internals privately",
                          kind="peer_card", trust_tier="owner")
    _set_scope(conn, peer_id, scope_user="mallory")
    alice_id = seed_memory(conn, "alice private widget deploykey abc123", kind="fact")
    _set_scope(conn, alice_id, scope_user="alice")
    glob_id = seed_memory(conn, "the public widget documentation url is on the wiki",
                          kind="fact")
    ids = {"owner": owner_id, "peer": peer_id, "alice": alice_id, "global": glob_id}
    uids = {k: conn.execute("SELECT uid FROM memories WHERE id=?", (v,)).fetchone()["uid"]
            for k, v in ids.items()}
    return uids


def test_nonowner_ask_never_leaks_scoped_rows_through_any_action(conn):
    uids = _seed_secrets(conn)
    fake = ScriptedLLM([
        {"action": "search_memory", "query": "widget secret vault"},
        {"action": "search_memory", "query": "widget", "epistemic": "observation"},
        {"action": "grep_episodes", "pattern": "widget"},
        {"action": "get_reasoning_chain", "uid": uids["owner"]},
        {"action": "get_reasoning_chain", "uid": uids["peer"]},
        {"action": "date_range_search", "query": "widget",
         "from": "2000-01-01", "to": "2100-01-01"},
        {"action": "answer", "text": "here is what I found",
         "citations": [uids["owner"], uids["peer"], uids["alice"], uids["global"]],
         "abstain": False},
    ])
    llm.set_llm_for_tests(fake)

    res = ask(conn, "what is the widget secret vault token?", level="deep",
              principal_id="mallory", trust_tier="known_user",
              max_iterations=8, config={})

    blob = fake.seen() + json.dumps(res.citations) + (res.answer or "")
    # No out-of-scope CONTENT ever reached the agent or the output — this is
    # the real invariant. (uid8 prefixes are NOT asserted: same-millisecond
    # ULIDs collide on the 8-char prefix, so the owner row shares a uid8 with
    # the global row the caller legitimately retrieved — indistinguishable by
    # prefix, but the snippet below proves only the safe content is exposed.)
    assert "vaulttoken hunter2" not in blob        # owner-scoped
    assert "keeps probing the widget internals" not in blob  # peer_card
    assert "private widget deploykey" not in blob  # foreign principal

    # Every cited snippet is real (agent-sourced from what it actually
    # retrieved), never model-supplied text and never out-of-scope content.
    for c in res.citations:
        assert "vaulttoken" not in c["snippet"]
        assert "deploykey" not in c["snippet"]
        assert "probing the widget internals" not in c["snippet"]
    assert uids  # seeds existed (guards against an empty-seed false pass)


def test_owner_ask_does_reach_the_secret(conn):
    """Contrast: the owner is scope-exempt, proving the seeds are real and the
    non-owner exclusion above is a genuine filter, not empty seeds."""
    _seed_secrets(conn)
    fake = ScriptedLLM([
        {"action": "search_memory", "query": "widget secret vault token"},
        {"action": "answer", "text": "found it", "citations": [], "abstain": False},
    ])
    llm.set_llm_for_tests(fake)

    ask(conn, "what is the widget secret vault token?", level="deep",
        principal_id=None, trust_tier="owner", max_iterations=4, config={})

    # The owner's search DID surface the secret into the agent's context.
    assert "vaulttoken hunter2" in fake.seen()


def test_nonowner_reasoning_chain_on_owner_uid_is_not_found(conn):
    """get_reasoning_chain on an owner-only memory must read as 'not found' for
    a non-owner — the same indistinguishable-from-absent contract as recall."""
    uids = _seed_secrets(conn)
    fake = ScriptedLLM([
        {"action": "get_reasoning_chain", "uid": uids["owner"]},
        {"action": "answer", "text": "nothing", "citations": [], "abstain": False},
    ])
    llm.set_llm_for_tests(fake)

    res = ask(conn, "trace the vault token", level="fast",
              principal_id="mallory", trust_tier="known_user",
              max_iterations=4, config={})
    assert "vaulttoken hunter2" not in fake.seen()
    assert "vaulttoken hunter2" not in (res.answer or "")
