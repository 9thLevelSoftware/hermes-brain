"""Pytest bootstrap: load the repo root as package ``brain``, exactly the way
the Hermes plugin loader does (spec_from_file_location with
submodule_search_locations), so every relative import inside the plugin
resolves identically under pytest and under Hermes.

Also home of the small cross-test helpers: ``poll_until`` (worker-thread
assertions without fixed sleeps) and ``seed_memory``/``seed_episode``
(direct-row seeding for recall/render/provider tests — column names from
store/schema.sql, which is law).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# Refuse to execute outside a real pytest run (review finding #1/#20): this
# file lives in tests/ so the Hermes loader's non-recursive *.py glob never
# sees it, but the guard makes side effects impossible even if it is ever
# imported from an agent process. PYTEST_VERSION is set by pytest >= 8.0
# before conftest collection.
if not os.environ.get("PYTEST_VERSION"):
    raise ImportError("tests/conftest.py is pytest-only; refusing to run outside pytest")

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _register_brain() -> None:
    """Mirror the Hermes plugin loader: repo root becomes package 'brain'."""
    if "brain" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "brain", REPO_ROOT / "__init__.py", submodule_search_locations=[str(REPO_ROOT)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["brain"] = module
    spec.loader.exec_module(module)


_register_brain()

# Make `from conftest import ...` resolve to THIS module instance regardless
# of how pytest imported it (tests/ is a package).
sys.modules.setdefault("conftest", sys.modules[__name__])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_home(tmp_path):
    home = tmp_path / "hermes_home"
    home.mkdir()
    return home


@pytest.fixture
def conn(tmp_home):
    from brain.store import db

    connection = db.connect(tmp_home)
    yield connection
    connection.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def poll_until(predicate, timeout=3.0, interval=0.02):
    """Poll ``predicate`` until truthy or timeout; return its last value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return predicate()


def iso_days_ago(days: float) -> str:
    t = time.time() - days * 86400
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + ".000Z"


def seed_memory(conn, content, *, kind="fact", memory_type="semantic", status="active",
                epistemic="observation", outcome=None, pinned=0, half_life_days=None,
                valid_from=None, trust_tier="owner", created_by="test", tags=(),
                summary=None, live=1):
    from brain.capture.symbols import symbols_field
    from brain.store import db

    now = db.iso_now()
    cur = conn.execute(
        "INSERT INTO memories (uid, epistemic, memory_type, kind, status, live, content,"
        " summary, content_hash, symbols, tags, token_len, trust_tier, created_by,"
        " valid_from, recorded_at, pinned, half_life_days, outcome)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (db.new_ulid(), epistemic, memory_type, kind, status, live, content, summary,
         db.content_hash(content), symbols_field(content), json.dumps(list(tags)),
         db.approx_tokens(content), trust_tier, created_by, valid_from or now, now,
         pinned, half_life_days, outcome),
    )
    conn.commit()
    return cur.lastrowid


def seed_episode(conn, user_content, assistant_content, *, session_id="seed", turn_no=1,
                 salience=0.5, platform="cli"):
    from brain.capture.symbols import symbols_field
    from brain.store import db

    cur = conn.execute(
        "INSERT INTO episodes (uid, session_id, turn_no, platform, user_content,"
        " assistant_content, symbols, token_len, salience, ts)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (db.new_ulid(), session_id, turn_no, platform, user_content, assistant_content,
         symbols_field(user_content, assistant_content),
         db.approx_tokens(user_content + assistant_content), salience, db.iso_now()),
    )
    conn.commit()
    return cur.lastrowid
