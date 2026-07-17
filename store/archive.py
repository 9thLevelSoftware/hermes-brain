"""Cold-storage content archive — makes tiered forgetting non-destructive.

`forget`'s grace purge nulls a demoted memory's live content down to a short
stub; its audit note has always claimed the raw text "lives on in the episodic
archive" — this module is that archive, so the claim is finally TRUE. `why
<id>` recovers a purged memory from here; compliance `forget --hard` scrubs it.

Layout: one gzipped JSONL file per month under ``$HERMES_HOME/brain/archive/``.
Appends are cheap and crash-safe (concatenated gzip members are standard-
readable); reads scan a month file for the uid (a cold, rare path). Stdlib
only — safe to import from the store subpackage.

An ``archive_ref`` is ``"<YYYY-MM>.jsonl.gz:<uid>"`` (schema.sql memories.
archive_ref). The uid anchor (not a byte offset) keeps appends lock-free and
makes supersede natural: a later line for the same uid wins on read.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# uids are ULIDs (Crockford base32); the guard also blocks path/ref injection.
_UID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def archive_dir(home) -> Path:
    return Path(home) / "brain" / "archive"


def append(home, record: dict) -> str | None:
    """Append one record; return its ``archive_ref`` ('<YYYY-MM>.jsonl.gz:<uid>')
    or None on failure. A None return is load-bearing: the caller MUST NOT
    destroy the live content when archiving did not succeed (never lose raw
    text). Never raises.
    """
    try:
        uid = str(record.get("uid") or "").strip()
        if not uid or not _UID_RE.match(uid):
            logger.warning("archive append: bad/absent uid %r; refusing", uid)
            return None
        from .db import iso_now

        now = str(record.get("archived_at") or iso_now())
        record = {**record, "uid": uid, "archived_at": now}
        yyyymm = now[:7]  # 'YYYY-MM'
        path = archive_dir(home) / f"{yyyymm}.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        with gzip.open(path, "ab") as f:
            f.write(line)
        return f"{yyyymm}.jsonl.gz:{uid}"
    except Exception as e:
        logger.warning("archive append failed for %r: %s", record.get("uid"), e)
        return None


def read(home, ref: str) -> dict | None:
    """Recover the archived record for a ref, or None (malformed/absent).
    Last matching line wins (supersede)."""
    try:
        if not ref or ":" not in ref:
            return None
        fname, uid = ref.rsplit(":", 1)
        if "/" in fname or "\\" in fname or ".." in fname:
            return None
        path = archive_dir(home) / fname
        if not path.exists():
            return None
        found = None
        with gzip.open(path, "rb") as f:
            for raw in f:
                try:
                    rec = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    continue
                if str(rec.get("uid") or "") == uid:
                    found = rec
        return found
    except Exception as e:
        logger.warning("archive read failed for %r: %s", ref, e)
        return None


def recover_content(home, ref: str) -> str | None:
    rec = read(home, ref)
    return None if rec is None else rec.get("content")


def purge_uid(home, uid: str) -> int:
    """Compliance scrub: rewrite every month file dropping all lines for uid.
    Returns lines removed. Best-effort; never raises."""
    removed = 0
    try:
        d = archive_dir(home)
        if not d.is_dir():
            return 0
        for path in sorted(d.glob("*.jsonl.gz")):
            kept: list[bytes] = []
            hit = False
            try:
                with gzip.open(path, "rb") as f:
                    for raw in f:
                        try:
                            rec = json.loads(raw)
                        except (ValueError, UnicodeDecodeError):
                            kept.append(raw)
                            continue
                        if str(rec.get("uid") or "") == uid:
                            hit = True
                            removed += 1
                        else:
                            kept.append(raw)
            except OSError:
                continue
            if hit:
                tmp = path.with_suffix(".gz.tmp")
                with gzip.open(tmp, "wb") as f:
                    for raw in kept:
                        f.write(raw if raw.endswith(b"\n") else raw + b"\n")
                tmp.replace(path)
    except Exception as e:
        logger.warning("archive purge_uid failed for %s: %s", uid, e)
    return removed
