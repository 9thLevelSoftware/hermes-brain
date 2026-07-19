"""Phase G sync engine — surface-only deny-list, push/pull, conflict resolution.

This is the CORE of multi-device encrypted sync and it is SECURITY-CRITICAL.
The load-bearing invariant is the *surface-only deny-list* (:func:`is_syncable`):
a memory may leave the device ONLY if it is global, active and not
instruction-shaped. Scoped/private/quarantined/peer-card/instruction-shaped
rows NEVER serialize, so a synced memory is GLOBAL by construction — the scope
columns are never even placed into an envelope, let alone transmitted.

Data flow:
  * ``push``  walks :func:`store.events.events_since` from an outbox cursor,
    re-fetches each memory, re-checks :func:`is_syncable` at push time, and for
    survivors builds ``{op, uid, ts, origin, memory}`` envelopes that are
    JSON-encoded → ``crypto.encrypt`` → base64 blobs handed to ``client.push``.
  * ``pull`` fetches blobs from ``client.pull``, base64-decodes → decrypts →
    JSON-parses each into an envelope, skips our own writes (origin match), and
    :func:`apply_remote`s the rest with LWW + causal-chain conflict resolution.

Cursors (``sync_outbox_cursor`` / ``sync_pull_cursor``) live in ``meta`` so a
crash mid-sync resumes without loss or double-apply: applying the same envelope
twice is a no-op (keyed on the per-version ``uid``, which is unique), and
re-pushing the same events only re-sends blobs the puller already ignores.

Floor-tier rule: this module lives behind the optional ``[sync]`` extra. The
``cryptography`` dependency is reached ONLY through the passed-in ``crypto``
object (``sync/crypto.py``), never imported here — ``import brain.sync.engine``
succeeds on the stdlib floor tier.
"""

from __future__ import annotations

import base64
import json
import logging
import sqlite3
from typing import Any

from ..store import db, events

# meta keys — the two resumable cursors.
logger = logging.getLogger(__name__)

_OUTBOX_CURSOR = "sync_outbox_cursor"
_PULL_CURSOR = "sync_pull_cursor"


# ---------------------------------------------------------------------------
# Row access — accept either a sqlite3.Row or a plain dict
# ---------------------------------------------------------------------------

def _rowget(row: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a sqlite3.Row or dict, returning ``default`` if absent."""
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[key]
    except (IndexError, KeyError):
        return default


# ---------------------------------------------------------------------------
# 1. Surface-only deny-list — THE load-bearing security rule
# ---------------------------------------------------------------------------

def is_syncable(row: Any) -> bool:
    """A memory may leave the device ONLY if it is NONE of: scope_user set,
    scope_session set, kind == 'peer_card', status == 'quarantined',
    instruction_shaped truthy — and it must be a real *active* row.

    Private/scoped/instruction-shaped rows NEVER serialize. Because a syncable
    row can carry no scope, every memory that leaves the device is GLOBAL by
    construction. The ``status == 'active'`` gate subsumes the quarantined case
    (quarantined rows have ``status='quarantined'``) but the intent is spelled
    out here because this predicate is the whole security boundary.
    """
    if _rowget(row, "status") != "active":
        return False  # excludes quarantined / tombstone / summarized / expired
    return _scope_public(row)


def _scope_public(row: Any) -> bool:
    """The scope half of the deny-list (status-independent): True iff the row
    carries NO privacy scope. Any of scope_user / scope_session / scope_project
    / scope_platform, a peer_card, or instruction-shaped content makes it
    private and un-syncable. Used both by is_syncable (which adds the active
    gate) and by tombstone propagation (which acts on a since-retired row)."""
    private = (
        _rowget(row, "scope_user")            # owner/user-scoped
        or _rowget(row, "scope_session")      # session-ephemeral lane
        or _rowget(row, "scope_project")      # project-private
        or _rowget(row, "scope_platform")     # platform-private
        or _rowget(row, "kind") == "peer_card"  # private theory-of-mind
        or _rowget(row, "instruction_shaped")   # untrusted instruction-shaped
    )
    return not private


# ---------------------------------------------------------------------------
# 2. Serialization — surface-safe fields ONLY (never scope_user/scope_session)
# ---------------------------------------------------------------------------

def serialize_memory(row: Any) -> dict:
    """Surface-safe fields needed to reconstruct the memory on another device.

    NEVER includes ``scope_user`` / ``scope_session`` (the deny-list guarantees
    a serialized row is unscoped anyway, but they are not placed in the envelope
    even by accident). ``supersedes_uid`` is the causal link — the uid of the
    row referenced by ``supersedes_id`` — which ``push`` resolves via a join and
    exposes on the row as the ``supersedes_uid`` column.
    """
    return {
        "uid": _rowget(row, "uid"),
        "content": _rowget(row, "content"),
        "kind": _rowget(row, "kind"),
        "epistemic": _rowget(row, "epistemic"),
        "memory_type": _rowget(row, "memory_type"),
        "status": _rowget(row, "status"),
        "version": _rowget(row, "version"),
        "valid_from": _rowget(row, "valid_from"),
        "valid_to": _rowget(row, "valid_to"),
        "recorded_at": _rowget(row, "recorded_at"),
        "half_life_days": _rowget(row, "half_life_days"),
        "source_platform": _rowget(row, "source_platform"),
        "trust_tier": _rowget(row, "trust_tier"),
        "supersedes_uid": _rowget(row, "supersedes_uid"),
    }


def _fetch_push_row(conn: sqlite3.Connection, uid: str) -> sqlite3.Row | None:
    """Fetch a memory by uid, resolving ``supersedes_id`` → parent uid.

    The LEFT JOIN exposes ``supersedes_uid`` (the causal parent's uid, or NULL)
    as a synthetic column so :func:`serialize_memory` stays a pure row->dict
    function with no DB access of its own.
    """
    return conn.execute(
        "SELECT m.*, parent.uid AS supersedes_uid FROM memories m "
        "LEFT JOIN memories parent ON m.supersedes_id = parent.id "
        "WHERE m.uid = ?",
        (uid,),
    ).fetchone()


# ---------------------------------------------------------------------------
# 3. Push — collect → encrypt → send → advance outbox cursor
# ---------------------------------------------------------------------------

def push(conn: sqlite3.Connection, crypto, client, *, origin: str, limit: int = 500) -> dict:
    """Walk the event log from the outbox cursor and push syncable memories.

    For each event, re-fetch the memory by uid and keep it ONLY if
    :func:`is_syncable` *at push time* — a memory that was created global but has
    since been quarantined (or scoped) must not sync, even though its old create
    event still sits in the log. Survivors become
    ``{op, uid, ts, origin, memory: serialize_memory(row)}`` envelopes, JSON →
    encrypt → base64 → one blob each; ``client.push(blobs)`` ships them.

    On a successful ship the outbox cursor advances to the new events cursor and
    every processed event (pushed AND skipped) is stamped ``synced_at`` — the
    outbox is drained past them either way. Returns
    ``{pushed, skipped_private, cursor}``.
    """
    cursor = db.get_meta(conn, _OUTBOX_CURSOR)
    evs, next_cursor = events.events_since(conn, cursor, limit=limit)

    blobs: list[str] = []
    processed: list[str] = []
    pushed = 0
    skipped_private = 0

    for ev in evs:
        processed.append(ev.event_id)
        row = _fetch_push_row(conn, ev.memory_uid)
        if ev.op == "tombstone":
            # Deletions must propagate so other devices retire the row — but a
            # tombstoned row is no longer 'active', so is_syncable would drop it.
            # Send a CONTENT-FREE tombstone, gated on the scope half of the
            # deny-list (checked on the still-present retired row) so a PRIVATE
            # deletion doesn't even leak a uid. `purge` is intentionally NOT
            # propagated: the tombstone that precedes it in the forget lifecycle
            # already shipped the delete, and the purged row is gone (its scope
            # can no longer be verified).
            if row is not None and _scope_public(row):
                envelope = {"op": "tombstone", "uid": ev.memory_uid, "ts": ev.ts,
                            "origin": origin, "memory": None}
            else:
                skipped_private += 1
                continue
        elif ev.op in ("create", "supersede"):
            if row is None or not is_syncable(row):
                skipped_private += 1
                continue
            envelope = {
                "op": ev.op,
                "uid": ev.memory_uid,
                "ts": ev.ts,
                "origin": origin,
                "memory": serialize_memory(row),
            }
        else:  # purge (and any future op): not propagated
            skipped_private += 1
            continue
        payload = json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode("utf-8")
        token = crypto.encrypt(payload)
        blobs.append(base64.b64encode(token).decode("ascii"))
        pushed += 1

    if blobs:
        # Ship first; only advance the cursor once the relay has the blobs. A
        # crash after this returns but before the commit below simply re-sends
        # on the next run — the puller ignores duplicates (uid idempotency).
        client.push(blobs)

    if next_cursor is not None:
        now = db.iso_now()
        db.set_meta(conn, _OUTBOX_CURSOR, next_cursor)
        conn.executemany(
            "UPDATE memory_events SET synced_at=? WHERE event_id=?",
            [(now, eid) for eid in processed],
        )
        conn.commit()

    return {"pushed": pushed, "skipped_private": skipped_private, "cursor": next_cursor}


# ---------------------------------------------------------------------------
# 4. Pull — fetch → decrypt → apply (conflict resolution) → advance cursor
# ---------------------------------------------------------------------------

def pull(conn: sqlite3.Connection, crypto, client, *, origin: str, limit: int = 500) -> dict:
    """Fetch remote blobs, decrypt, and apply each with conflict resolution.

    Envelopes whose ``origin`` equals ours are skipped (never re-apply our own
    writes — a device that both pushes and pulls against one relay sees them).
    Each remaining envelope goes through :func:`apply_remote`, committed
    per-envelope so a crash leaves the DB consistent and the (not-yet-advanced)
    pull cursor replays only the un-applied tail — every apply is idempotent.
    Returns ``{pulled, applied, conflicts, cursor}``.
    """
    cursor = db.get_meta(conn, _PULL_CURSOR) or "0"
    # RelayClient.pull declares `limit` keyword-only — pass it by keyword or the
    # real client raises before making the request.
    blobs, new_cursor = client.pull(cursor, limit=limit)

    pulled = 0
    applied = 0
    conflicts = 0
    skipped_bad = 0

    for blob in blobs:
        # Decode OUTSIDE-then-inside a guard: a single corrupt / undecryptable
        # blob must NOT wedge the device. Skip it and keep going so the cursor
        # still advances past it and later valid deltas still sync (PR #5 review).
        try:
            token = base64.b64decode(blob)
            plaintext = crypto.decrypt(token)
            envelope = json.loads(
                plaintext.decode("utf-8") if isinstance(plaintext, bytes) else plaintext)
        except Exception as e:
            logger.warning("sync pull: skipping undecodable blob: %s", e)
            skipped_bad += 1
            continue
        pulled += 1
        if envelope.get("origin") == origin:
            continue  # our own write echoed back by the relay
        try:
            action = apply_remote(conn, envelope)
            conn.commit()
        except Exception:
            conn.rollback()
            logger.warning("sync pull: apply_remote failed; skipping one envelope",
                           exc_info=True)
            skipped_bad += 1
            continue  # one bad envelope must not wedge the whole pull
        if action in ("created", "superseded", "tombstoned", "lww_remote"):
            applied += 1
        if action in ("superseded", "tombstoned", "lww_remote", "lww_local"):
            conflicts += 1

    # A pull that changed any row must invalidate generation-keyed caches
    # (QueryCache) on a concurrent provider, or it keeps serving stale hits and
    # never surfaces newly-synced (or hides remotely-deleted) memories.
    if applied:
        db.bump_generation(conn, "mem")
    db.set_meta(conn, _PULL_CURSOR, str(new_cursor))
    conn.commit()

    return {"pulled": pulled, "applied": applied, "conflicts": conflicts,
            "skipped_bad": skipped_bad, "cursor": str(new_cursor)}


# ---------------------------------------------------------------------------
# Conflict resolution helpers
# ---------------------------------------------------------------------------

def _by_uid(conn: sqlite3.Connection, uid: str | None) -> sqlite3.Row | None:
    if not uid:
        return None
    return conn.execute("SELECT * FROM memories WHERE uid = ?", (uid,)).fetchone()


def _newer(a: str | None, b: str | None) -> bool:
    """True iff ISO-8601 timestamp ``a`` is strictly after ``b`` (lexicographic
    order matches chronological order for the fixed-width ``iso_now`` format)."""
    return (a or "") > (b or "")


def _insert_memory(
    conn: sqlite3.Connection,
    mem: dict,
    *,
    valid_to: str | None = None,
    superseded_by: int | None = None,
    supersedes_id: int | None = None,
) -> int:
    """Insert a remote memory as a GLOBAL row (scope_user/scope_session NULL).

    ``created_by='sync'`` marks provenance; ``content_hash`` is recomputed
    locally (never trusted from the wire). Surface fields come from the envelope.
    """
    content = mem.get("content")
    now = db.iso_now()
    cur = conn.execute(
        "INSERT INTO memories (uid, epistemic, memory_type, kind, status, live,"
        " content, content_hash, trust_tier, created_by, instruction_shaped,"
        " version, supersedes_id, superseded_by, valid_from, valid_to,"
        " recorded_at, half_life_days, source_platform)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            mem.get("uid"),
            mem.get("epistemic") or "observation",
            mem.get("memory_type") or "semantic",
            mem.get("kind"),
            mem.get("status") or "active",
            1,
            content,
            db.content_hash(content or ""),
            mem.get("trust_tier") or "untrusted",
            "sync",
            0,  # instruction-shaped rows are never syncable — always 0 on arrival
            mem.get("version") or 1,
            supersedes_id,
            superseded_by,
            mem.get("valid_from") or now,
            valid_to,
            mem.get("recorded_at") or now,
            mem.get("half_life_days"),
            mem.get("source_platform"),
        ),
    )
    return cur.lastrowid


def _resolve_lww(conn: sqlite3.Connection, mem: dict, local: sqlite3.Row, remote_ts: str) -> str:
    """Concurrent versions (both supersede the same parent) — last-writer-wins.

    Newer ``recorded_at`` wins; an exact tie is broken deterministically by uid
    (higher uid wins) so both devices converge on the same winner. When the
    remote wins we close the local current row and insert the remote as the new
    head; when the local wins we do nothing (the losing branch is simply not
    materialized — re-applying it stays a no-op, preserving idempotency).
    """
    local_ts = local["recorded_at"]
    remote_uid = mem.get("uid") or ""
    local_uid = local["uid"] or ""
    remote_wins = _newer(remote_ts, local_ts) or (remote_ts == (local_ts or "") and remote_uid > local_uid)
    if remote_wins:
        # The remote branch shares the local branch's parent — keep the chain
        # rooted there rather than pointing at the branch it displaces.
        new_id = _insert_memory(conn, mem, supersedes_id=local["supersedes_id"])
        conn.execute(
            "UPDATE memories SET valid_to=?, superseded_by=? WHERE id=?",
            (remote_ts or db.iso_now(), new_id, local["id"]),
        )
        return "lww_remote"
    return "lww_local"


def apply_remote(conn: sqlite3.Connection, env: dict) -> str:
    """LWW + causal-chain conflict resolution keyed on the version chain.

    Returns an action string:
      * ``created``     — uid unknown locally; inserted as a GLOBAL active row.
      * ``superseded``  — remote is causally newer (it supersedes a local
                          *current* row); local closed, remote becomes current.
      * ``tombstoned``  — remote op is a newer tombstone of a row we hold.
      * ``lww_remote`` / ``lww_local`` — concurrent branches (both supersede the
                          same parent, which is already closed locally) resolved
                          by last-writer-wins on ``recorded_at`` (tie → uid).
      * ``ignored``     — we already hold this exact version (uid match), or an
                          older/dominated write with nothing to apply.

    Idempotent: applying the same envelope twice is a no-op, because the
    per-version ``uid`` is unique — a second apply finds the row present and
    returns ``ignored`` (or re-loses the same LWW). Never corrupts the version
    chain: the loser always gets ``valid_to`` + ``superseded_by`` set.
    """
    mem = env.get("memory") or {}
    uid = mem.get("uid") or env.get("uid")
    op = env.get("op")
    remote_ts = mem.get("recorded_at") or env.get("ts") or ""

    existing = _by_uid(conn, uid)
    if existing is not None:
        # Idempotency: we already have this exact version row. The only mutation
        # allowed is retiring it via a strictly-newer tombstone.
        if op == "tombstone" and existing["status"] == "active" and _newer(remote_ts, existing["recorded_at"]):
            conn.execute(
                "UPDATE memories SET status='tombstone', content=NULL, valid_to=? WHERE id=?",
                (remote_ts, existing["id"]),
            )
            return "tombstoned"
        return "ignored"

    # A tombstone for a uid we never held is a no-op — it must NEVER fall
    # through to the insert path and create a (content-free) ghost row.
    if op == "tombstone":
        return "ignored"

    supersedes_uid = mem.get("supersedes_uid")
    if supersedes_uid:
        parent = _by_uid(conn, supersedes_uid)
        if parent is not None:
            if parent["valid_to"] is None and parent["status"] == "active":
                # Clean causal supersede: remote directly follows a live local row.
                new_id = _insert_memory(conn, mem, supersedes_id=parent["id"])
                conn.execute(
                    "UPDATE memories SET valid_to=?, superseded_by=? WHERE id=?",
                    (remote_ts or db.iso_now(), new_id, parent["id"]),
                )
                return "superseded"
            # Parent already superseded locally → a concurrent branch exists.
            competitor = conn.execute(
                "SELECT * FROM memories WHERE supersedes_id=? AND valid_to IS NULL "
                "AND status='active'",
                (parent["id"],),
            ).fetchone()
            if competitor is not None:
                return _resolve_lww(conn, mem, competitor, remote_ts)
            # No live competitor (chain moved on / dead-ended): learn the remote.
            _insert_memory(conn, mem)
            return "created"

    # Plain create, or a supersede whose parent this device never saw.
    _insert_memory(conn, mem)
    return "created"
