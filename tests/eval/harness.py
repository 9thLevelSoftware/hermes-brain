"""Phase F — the BEAM-pattern evaluation harness.

Drives the REAL brain pipeline over a JSON fixture (mirroring ``replay/run.py``)
through five ordered STAGES, each a function below:

  1. ingest   — capture a fixture's sessions (episodes + ingest_buffer) and run
                the REAL extraction sweep with a SCRIPTED fake LLM that emits the
                fixture's gold memory items, so extraction is reproducible in CI.
  2. dream    — run the REAL dream pipeline (``dream/run.py:run_dream``) with the
                same scripted fake, so consolidation/facts/etc. run deterministically
                (strategies that can't act simply no-op).
  3. retrieve — for each labeled query run ``recall/search.py:search`` and score
                P@k (k=5) and MRR against the fixture's relevance labels.
  4. answer   — for each gold QA pair call ``recall/ask.py:ask`` (scripted fake in
                CI) and collect the answers + abstentions.
  5. judge    — a keyword rubric: a non-abstain answer PASSES when it contains the
                gold keywords; an 'unknown' gold expects the answerer to ABSTAIN.

Hermetic by default: a single scripted fake LLM (``ScriptedEvalLLM``) is installed
via ``llm.set_llm_for_tests`` around EVERY stage and dispatches by inspecting the
call (extraction system prompt vs. the ask action menu vs. anything else). Set the
environment variable ``BRAIN_EVAL_REAL=1`` (or pass ``real=True``) to run the SAME
pipeline against the host auxiliary client instead — no fake is installed — so a
human can grade the answerer against a real model.

Module level is stdlib-only; the ``brain`` package (and its heavy siblings) are
imported lazily inside the stage bodies, exactly like the eagerly-loaded root
modules keep their imports light.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

_K = 5              # P@k / top-k for the retrieval metric
_SEARCH_LIMIT = 10  # returned hits per query (gives MRR room below top-k)


# ---------------------------------------------------------------------------
# brain package registration (Hermes-loader parity — see replay/run.py)
# ---------------------------------------------------------------------------

def _ensure_brain() -> None:
    """Register the repo root as package ``brain`` if it is not already (pytest's
    conftest and the Hermes loader both do this; standalone/CLI callers may not)."""
    if "brain" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "brain", REPO_ROOT / "__init__.py", submodule_search_locations=[str(REPO_ROOT)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["brain"] = module
    spec.loader.exec_module(module)


# ---------------------------------------------------------------------------
# The scripted fake LLM — one callable, dispatched by prompt/system
# ---------------------------------------------------------------------------

class ScriptedEvalLLM:
    """A single deterministic fake with the ``llm.set_llm_for_tests`` signature
    ``fn(prompt, *, system, max_tokens) -> str``.

    It discriminates the three brain LLM surfaces by their prompts:

      * extraction  — ``system`` is ``capture/extract.py:_EXTRACT_SYSTEM``
                      ("distill a conversation digest ...").  Routes the digest
                      to a session by its unique ``match`` token and returns that
                      session's gold ``extract`` items as a JSON array.
      * ask         — ``system`` is ``recall/ask.py:_SYSTEM_PROMPT``
                      ("... question-answering agent ...").  Parses ``QUESTION:``
                      out of the user prompt and returns the next scripted action
                      (search_memory, then answer) for that question.
      * everything else (dream strategies) — an empty JSON array, so a strategy
                      that calls the LLM cleanly finds nothing to do.
    """

    def __init__(self, fixture: dict[str, Any]):
        self._sessions = fixture.get("sessions", [])
        self._ask_scripts: dict[str, list[dict]] = {}
        for qa in fixture.get("qa", []):
            actions: list[dict] = [
                {"action": "search_memory", "query": qa.get("search", "")}
            ]
            answer: dict[str, Any] = {
                "action": "answer",
                "text": qa.get("answer_text", ""),
                "citations": [],
            }
            if qa.get("abstain"):
                answer["abstain"] = True
            actions.append(answer)
            self._ask_scripts[qa["question"]] = actions
        self._ask_idx: dict[str, int] = {}
        self.calls: list[dict] = []

    def __call__(self, prompt: str, *, system: str | None = None, max_tokens: int = 0) -> str:
        self.calls.append({"prompt": prompt, "system": system})
        sys_text = system or ""
        if "distill a conversation digest" in sys_text:
            return self._extract_reply(prompt)
        if "question-answering agent" in sys_text:
            return self._ask_reply(prompt)
        # Any other brain-initiated call (dream strategies): no-op empty array.
        return "[]"

    def _extract_reply(self, prompt: str) -> str:
        low = prompt.lower()
        for session in self._sessions:
            match = str(session.get("match", "")).lower()
            if match and match in low:
                return json.dumps(session.get("extract", []))
        return "[]"

    def _ask_reply(self, prompt: str) -> str:
        question = self._parse_question(prompt)
        actions = self._ask_scripts.get(question)
        if not actions:
            # Unknown question: abstain rather than fabricate.
            return json.dumps({"action": "answer", "text": "I don't know.",
                               "abstain": True})
        idx = self._ask_idx.get(question, 0)
        action = actions[min(idx, len(actions) - 1)]
        self._ask_idx[question] = idx + 1
        return json.dumps(action)

    @staticmethod
    def _parse_question(prompt: str) -> str:
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.startswith("QUESTION:"):
                return stripped[len("QUESTION:"):].strip()
        return ""


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def load_fixture(fixture_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(fixture_path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Stage 1 — ingest
# ---------------------------------------------------------------------------

def ingest(conn, config: dict[str, Any], fixture: dict[str, Any], *, embedder=None) -> dict:
    """Capture every fixture session through the REAL capture path (episodes +
    ingest_buffer), then drain the extraction sweep until the buffer is empty.

    The scripted fake LLM must already be installed (run_eval does this) unless
    running against a real model. Returns the accumulated sweep counters plus the
    number of active memories the ingest produced.
    """
    _ensure_brain()
    from brain.capture import extract
    from brain.capture.turns import TurnContext, capture_session_end, capture_turn

    default_trust = str(fixture.get("trust_tier", "owner"))
    default_principal = fixture.get("principal_id")
    platform = str(fixture.get("platform", "cli"))

    for session in fixture.get("sessions", []):
        sid = session["session_id"]
        ctx = TurnContext(
            session_id=sid,
            platform=platform,
            principal_id=session.get("principal_id", default_principal),
            trust_tier=str(session.get("trust_tier", default_trust)),
        )
        for turn in session.get("turns", []):
            capture_turn(conn, ctx, str(turn.get("user", "")),
                         str(turn.get("assistant", "")))
        capture_session_end(conn, sid)

    totals = {"batches": 0, "items": 0, "inserted": 0, "merged": 0,
              "quarantined": 0, "skipped_llm": 0}
    # Drain: one sweep processes a bounded number of session batches; loop until
    # the buffer is empty (or the sweep stops making progress).
    for _ in range(20):
        if extract.pending_count(conn) == 0:
            break
        counts = extract.sweep(conn, config, embedder=embedder,
                               actor="eval", max_rows=100, max_llm_calls=20)
        for key in totals:
            totals[key] += counts.get(key, 0)
        if not any(counts.get(k) for k in
                   ("batches", "inserted", "merged", "quarantined")):
            break  # no forward progress (e.g. LLM unavailable) — stop looping

    totals["active_memories"] = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE valid_to IS NULL AND status='active'"
        " AND live=1"
    ).fetchone()[0]
    totals["pending"] = extract.pending_count(conn)
    return totals


# ---------------------------------------------------------------------------
# Stage 2 — dream
# ---------------------------------------------------------------------------

def dream(conn, config: dict[str, Any], *, embedder=None) -> dict:
    """Run the REAL dream pipeline once. run_dream never raises — a strategy that
    cannot act (or has no LLM path) is logged and the pipeline continues — so this
    stage is robust at the stub tier. Returns a compact summary."""
    _ensure_brain()
    from brain.dream import run_dream

    summary = run_dream(conn, config, embedder=embedder, actor="eval")
    strategies = summary.get("strategies", {})
    return {
        "shift_id": summary.get("shift_id"),
        "ran": list(strategies.keys()),
        "errors": {name: res["error"] for name, res in strategies.items()
                   if isinstance(res, dict) and "error" in res},
    }


# ---------------------------------------------------------------------------
# Stage 3 — retrieve (P@k + MRR)
# ---------------------------------------------------------------------------

def _hit_matches(text: str, labels: list[str]) -> bool:
    low = (text or "").lower()
    return any(str(lbl).lower() in low for lbl in labels)


def retrieve(conn, config: dict[str, Any], fixture: dict[str, Any], *,
             embedder=None, reranker=None) -> dict:
    """Score each labeled query with P@k (k=5) and reciprocal rank.

    Retrieval is over the distilled MEMORY store only (``include_episodes=False``)
    so the metric reflects the semantic memory the pipeline built rather than the
    raw turns that trivially contain the answer — and so the supersession case is
    honest (the retired old row is excluded structurally).

    p_at_k per query = (# labeled-relevant items found within top-k) /
                       min(k, # labeled-relevant items).  The denominator is
    clamped to the label count so a query with fewer than k relevant items can
    still reach 1.0.  The reported ``p_at_5``/``mrr`` are the means over queries.
    """
    _ensure_brain()
    from brain.recall.search import search

    trust_tier = str(fixture.get("trust_tier", "owner"))
    principal_id = fixture.get("principal_id")

    per_query: list[dict] = []
    p_sum = 0.0
    rr_sum = 0.0
    for q in fixture.get("queries", []):
        labels = [str(x) for x in q.get("relevant", [])]
        hits = search(
            conn, str(q.get("query", "")), limit=_SEARCH_LIMIT,
            include_episodes=False, trust_tier=trust_tier,
            principal_id=principal_id, embedder=embedder, reranker=reranker,
        )
        found_labels = {lbl for lbl in labels
                        for h in hits[:_K] if str(lbl).lower() in (h.text or "").lower()}
        first_rank = next((i + 1 for i, h in enumerate(hits)
                           if _hit_matches(h.text, labels)), None)
        denom = min(_K, len(labels)) or 1
        p_at_k = len(found_labels) / denom
        rr = (1.0 / first_rank) if first_rank else 0.0
        p_sum += p_at_k
        rr_sum += rr
        per_query.append({
            "id": q.get("id"),
            "query": q.get("query"),
            "relevant": labels,
            "p_at_k": round(p_at_k, 4),
            "first_relevant_rank": first_rank,
            "reciprocal_rank": round(rr, 4),
            "top_uids": [h.uid[:8] for h in hits[:_K]],
        })

    n = len(per_query) or 1
    return {
        "k": _K,
        "p_at_5": round(p_sum / n, 4),
        "mrr": round(rr_sum / n, 4),
        "queries": per_query,
    }


# ---------------------------------------------------------------------------
# Stage 4 — answer
# ---------------------------------------------------------------------------

def answer(conn, config: dict[str, Any], fixture: dict[str, Any], *,
           embedder=None, reranker=None) -> list[dict]:
    """Run ``recall/ask.py:ask`` for every gold QA pair and collect the results.
    ``ask`` never raises (it degrades to a recall-only result); the scripted fake
    drives the tool loop in CI."""
    _ensure_brain()
    from brain.recall.ask import ask

    trust_tier = str(fixture.get("trust_tier", "owner"))
    principal_id = fixture.get("principal_id")
    max_iter = int(config.get("ask_max_iterations", 6))

    out: list[dict] = []
    for qa in fixture.get("qa", []):
        res = ask(
            conn, str(qa.get("question", "")), level="deep",
            trust_tier=trust_tier, principal_id=principal_id,
            embedder=embedder, reranker=reranker, config=config,
            max_iterations=max_iter,
        )
        out.append({
            "id": qa.get("id"),
            "question": qa.get("question"),
            "answer": res.answer,
            "answered": res.answered,
            "degraded": res.degraded,
            "iterations": res.iterations,
            "citations": res.citations,
        })
    return out


# ---------------------------------------------------------------------------
# Stage 5 — judge (keyword rubric)
# ---------------------------------------------------------------------------

def judge(fixture: dict[str, Any], answers: list[dict]) -> dict:
    """Keyword rubric.  A non-abstain gold PASSES when the produced answer is
    answered and contains every required keyword (case-insensitive).  An 'abstain'
    gold is CORRECT when the answerer abstained (``answered`` is False).  Returns
    the aggregate pass/abstain rates and per-QA detail."""
    by_id = {a.get("id"): a for a in answers}
    detail: list[dict] = []
    ans_total = ans_pass = 0
    abst_total = abst_correct = 0
    for qa in fixture.get("qa", []):
        got = by_id.get(qa.get("id"), {})
        expect_abstain = bool(qa.get("abstain"))
        answered = bool(got.get("answered"))
        text = (got.get("answer") or "")
        low = text.lower()
        if expect_abstain:
            abst_total += 1
            correct = not answered
            abst_correct += 1 if correct else 0
            passed = correct
        else:
            ans_total += 1
            keywords = [str(k) for k in qa.get("keywords", [])]
            has_kw = all(k.lower() in low for k in keywords)
            passed = answered and has_kw
            ans_pass += 1 if passed else 0
        detail.append({
            "id": qa.get("id"),
            "question": qa.get("question"),
            "expected_abstain": expect_abstain,
            "answered": answered,
            "answer": text,
            "pass": passed,
        })
    return {
        "answer_pass_rate": round(ans_pass / ans_total, 4) if ans_total else 1.0,
        "abstain_correct": round(abst_correct / abst_total, 4) if abst_total else 1.0,
        "qa": detail,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_eval(fixture_path: str | Path, *, real: bool = False) -> dict:
    """Run all five stages over ``fixture_path`` and return a metrics dict:

        {
          "p_at_5", "mrr", "k",              # retrieval
          "answer_pass_rate", "abstain_correct",  # answer + judge
          "tier",                            # storage capability tier reached
          "ingest", "dream",                 # per-stage summaries
          "queries", "qa",                   # per-item detail
          "real",                            # whether a real LLM was used
        }

    With ``real`` False (the default) a single ``ScriptedEvalLLM`` is installed
    for the whole run so extraction/dream/ask are deterministic and offline. With
    ``real`` True (or ``BRAIN_EVAL_REAL=1``) NO fake is installed — the brain's
    LLM calls resolve through the host auxiliary client for a human grader.
    """
    _ensure_brain()
    from brain import llm
    from brain.config import load_config
    from brain.store import db

    real = real or os.environ.get("BRAIN_EVAL_REAL") == "1"
    fixture = load_fixture(fixture_path)

    home = Path(tempfile.mkdtemp(prefix="brain-eval-"))
    config = load_config(home)
    config["hermes_home"] = str(home)

    conn = db.connect(home)
    fake = None
    if not real:
        fake = ScriptedEvalLLM(fixture)
        llm.set_llm_for_tests(fake)
    try:
        ingest_counts = ingest(conn, config, fixture)
        dream_summary = dream(conn, config)
        retrieval = retrieve(conn, config, fixture)
        answers = answer(conn, config, fixture)
        rubric = judge(fixture, answers)

        try:
            caps = db.capabilities(conn)
            tier = caps.get("tier") or ("fts5" if caps.get("fts5") else "like")
        except Exception:
            tier = "unknown"

        return {
            "real": real,
            "tier": tier,
            "k": retrieval["k"],
            "p_at_5": retrieval["p_at_5"],
            "mrr": retrieval["mrr"],
            "answer_pass_rate": rubric["answer_pass_rate"],
            "abstain_correct": rubric["abstain_correct"],
            "ingest": ingest_counts,
            "dream": dream_summary,
            "queries": retrieval["queries"],
            "qa": rubric["qa"],
            "llm_calls": len(fake.calls) if fake is not None else None,
        }
    finally:
        if not real:
            llm.set_llm_for_tests(None)
        conn.close()


if __name__ == "__main__":  # pragma: no cover - manual/CLI convenience
    import argparse

    parser = argparse.ArgumentParser(description="Run the brain eval harness.")
    parser.add_argument("--fixture", required=True, help="path to an eval fixture JSON")
    parser.add_argument("--real", action="store_true",
                        help="use the host auxiliary LLM instead of the scripted fake")
    args = parser.parse_args()
    metrics = run_eval(args.fixture, real=args.real)
    print(json.dumps(metrics, indent=2))
