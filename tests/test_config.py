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


# NOTE: the "every DEFAULTS key is actually read" invariant lives in
# tests/test_audit_fixes.py::test_every_declared_config_key_is_read_somewhere.
# It was duplicated here; one strict, fast version beats two slow near-copies.
