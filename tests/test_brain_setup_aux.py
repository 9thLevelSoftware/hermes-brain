"""B1 — brain_setup registers auxiliary.brain_extract / brain_consolidate slots
into the Hermes config dict so `hermes model → Configure auxiliary models` and
aux.call_llm("brain_extract"/"brain_consolidate") can pin a per-task model, with
a documented fallback (empty provider/model = inherit the aux default). The
registration is idempotent and never clobbers a user-set block."""

from __future__ import annotations

from brain import brain_setup


def test_register_seeds_both_slots_on_empty_config():
    config: dict = {}
    brain_setup._register_aux_slots(config)
    aux = config["auxiliary"]
    assert aux["brain_extract"] == {"provider": "", "model": ""}
    assert aux["brain_consolidate"] == {"provider": "", "model": ""}


def test_register_creates_only_the_two_brain_slots():
    config: dict = {}
    brain_setup._register_aux_slots(config)
    assert set(config["auxiliary"]) == {"brain_extract", "brain_consolidate"}


def test_register_preserves_existing_auxiliary_tasks():
    config = {"auxiliary": {"vision": {"provider": "openai", "model": "gpt-4o"}}}
    brain_setup._register_aux_slots(config)
    assert config["auxiliary"]["vision"] == {"provider": "openai",
                                             "model": "gpt-4o"}
    assert "brain_extract" in config["auxiliary"]
    assert "brain_consolidate" in config["auxiliary"]


def test_register_does_not_clobber_user_set_block():
    config = {"auxiliary": {"brain_extract": {"provider": "ollama",
                                              "model": "llama3.1:8b"}}}
    brain_setup._register_aux_slots(config)
    # user's pin survives; the other slot is seeded with the default.
    assert config["auxiliary"]["brain_extract"] == {"provider": "ollama",
                                                    "model": "llama3.1:8b"}
    assert config["auxiliary"]["brain_consolidate"] == {"provider": "",
                                                        "model": ""}


def test_register_is_idempotent():
    config: dict = {}
    brain_setup._register_aux_slots(config)
    snapshot = {k: dict(v) for k, v in config["auxiliary"].items()}
    brain_setup._register_aux_slots(config)  # second run
    assert config["auxiliary"] == snapshot


def test_register_reseeds_empty_block():
    # An empty dict carries no user intent -> fill it with the default.
    config = {"auxiliary": {"brain_extract": {}}}
    brain_setup._register_aux_slots(config)
    assert config["auxiliary"]["brain_extract"] == {"provider": "", "model": ""}


def test_register_defensive_on_bad_shapes():
    # Non-dict config: no-op, no raise.
    brain_setup._register_aux_slots(None)  # type: ignore[arg-type]
    # auxiliary present but not a dict: bail without raising or overwriting.
    config = {"auxiliary": "not-a-dict"}
    brain_setup._register_aux_slots(config)
    assert config["auxiliary"] == "not-a-dict"


def test_slots_match_llm_task_map():
    # The seeded slot names must equal the aux task keys llm.py actually calls,
    # or a per-task pin would silently never apply.
    from brain import llm

    assert set(brain_setup._AUX_TASK_SLOTS) == set(llm._TIER_TASK.values())
