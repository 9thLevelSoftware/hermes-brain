"""brain.yaml load/save, and the write-only config.json mirror the Hermes
dashboard prefills its provider form from."""

from __future__ import annotations

import json

from brain import config as brain_config


def test_save_config_writes_yaml_and_json_mirror(tmp_home):
    brain_config.save_config(tmp_home, {"lane2_tokens": 321, "recall_mode": "tools"})

    assert brain_config.config_path(tmp_home).exists()
    mirror = brain_config.mirror_path(tmp_home)
    assert mirror.exists(), "the dashboard reads <home>/brain/config.json"

    data = json.loads(mirror.read_text(encoding="utf-8"))
    assert data["lane2_tokens"] == 321
    assert data["recall_mode"] == "tools"
    # The mirror is a FULL snapshot, not a delta: the dashboard prefills every
    # field from it, so a partial file would render defaults for the rest.
    assert set(data) == set(brain_config.DEFAULTS)


def test_mirror_is_write_only_and_never_overrides_yaml(tmp_home):
    """brain.yaml is the single source of truth. If the two ever disagree,
    load_config must follow the YAML — reading the mirror back is exactly how
    the copies would start diverging."""
    brain_config.save_config(tmp_home, {"lane2_tokens": 321})
    mirror = brain_config.mirror_path(tmp_home)
    mirror.write_text(json.dumps({"lane2_tokens": 999}), encoding="utf-8")

    assert brain_config.load_config(tmp_home)["lane2_tokens"] == 321


def test_save_config_survives_an_unwritable_mirror(tmp_home, monkeypatch):
    """A failed mirror must not fail the config save — brain.yaml is what
    matters; the mirror is a dashboard convenience."""
    real_write = brain_config.Path.write_text

    def _fail_only_the_mirror(self, *args, **kwargs):
        if self.name.endswith(".json.tmp"):
            raise OSError("disk full")
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(brain_config.Path, "write_text", _fail_only_the_mirror)
    brain_config.save_config(tmp_home, {"lane2_tokens": 321})

    assert brain_config.load_config(tmp_home)["lane2_tokens"] == 321
    assert not brain_config.mirror_path(tmp_home).exists()


def test_unknown_keys_are_dropped(tmp_home):
    brain_config.save_config(tmp_home, {"lane2_tokens": 321, "not_a_real_key": "x"})
    text = brain_config.config_path(tmp_home).read_text(encoding="utf-8")
    assert "not_a_real_key" not in text


def test_every_default_key_is_actually_read_somewhere():
    """Regression: `dream_schedule` and `dream_time` were prompted by the setup
    wizard and read by NOTHING — the user answered two questions that did not
    exist. A config key the product advertises must do something.

    config.py declares the keys and brain_setup.py only prompts for them, so
    neither counts as a reader; tests don't either.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    declared = re.findall(r'^\s{4}"([a-z0-9_]+)":',
                          (root / "config.py").read_text(encoding="utf-8"), re.M)
    assert declared, "could not parse DEFAULTS"

    sources = [
        p for p in root.rglob("*.py")
        if not {".venv-dev", "tests", "__pycache__"} & set(p.parts)
        and p.name not in ("config.py", "brain_setup.py")
    ]
    corpus = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in sources)

    dead = [k for k in declared if f'"{k}"' not in corpus and f"'{k}'" not in corpus]
    assert not dead, (
        f"config keys declared but never read: {dead} — wire them up or remove "
        f"them (and drop them from brain_setup.config_schema)")
