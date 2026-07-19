"""Phase F eval-harness CI entry (stub tier, scripted fake LLM, no network).

Runs the full BEAM pipeline over ``tests/eval/fixtures/eval_basic.json`` and
asserts the floors the fixture is designed to meet: a retrieval P@5 floor and
correct abstention on the 'unknown' gold QA. The scripted fake keeps every
stage — extraction, dream, and the ask loop — deterministic and offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from brain import llm

from tests.eval.harness import run_eval

_FIXTURE = Path(__file__).parent / "eval" / "fixtures" / "eval_basic.json"

# The fixture retrieves all five labeled relevant items within top-5 at stub
# tier; 0.6 is a comfortable floor (>= 3 of 5) that tolerates FTS/LIKE drift.
_P_AT_5_FLOOR = 0.6


@pytest.fixture(autouse=True)
def _clear_llm_override():
    # Belt-and-braces: run_eval clears its own fake in a finally, but never leak
    # an override into sibling tests.
    yield
    llm.set_llm_for_tests(None)


@pytest.fixture(scope="module")
def metrics():
    return run_eval(str(_FIXTURE))


def test_retrieval_meets_p_at_5_floor(metrics):
    assert metrics["k"] == 5
    assert metrics["p_at_5"] >= _P_AT_5_FLOOR, metrics["queries"]
    # MRR should be healthy when the relevant rows rank at (or near) the top.
    assert metrics["mrr"] >= _P_AT_5_FLOOR, metrics["queries"]


def test_answerer_abstains_on_unknown(metrics):
    unknown = next(q for q in metrics["qa"] if q["id"] == "qa_unknown_address")
    assert unknown["expected_abstain"] is True
    assert unknown["answered"] is False, unknown
    assert unknown["pass"] is True
    # Aggregate: every 'unknown' gold was correctly abstained.
    assert metrics["abstain_correct"] == 1.0


def test_supersession_current_truth_retrieved(metrics):
    """Phase B/C end-to-end: the current-truth deploy host (prod-box-9) must be
    retrieved for the supersession query, and the retired host (prod-box-7) must
    be structurally excluded (its memory row was retired by the facts layer)."""
    deploy = next(q for q in metrics["queries"] if q["id"] == "q_deploy")
    assert deploy["first_relevant_rank"] is not None, deploy
    assert deploy["p_at_k"] >= 1.0, deploy


def test_non_abstain_answers_pass_keyword_rubric(metrics):
    # The scripted answers carry the gold keywords, so the pass rate is 1.0.
    assert metrics["answer_pass_rate"] == 1.0, metrics["qa"]
