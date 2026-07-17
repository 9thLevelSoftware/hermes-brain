"""FTS5/BM25 retrieval over memories + episodes, with LIKE fallback.

Design (docs/design/memory-engine.md §3.5): the keyword leg is
recall-oriented — query tokens are OR-joined so partial matches surface,
and BM25 rank is then *modulated* multiplicatively by lifecycle signals
(decay half-life, outcome, kind, feedback counts, pinned) rather than
replaced by them. Episodes are a second leg scored at 0.6x so distilled
memories outrank raw turns; both legs are normalized JOINTLY over one
candidate pool into a floored band, so the 0.6x rule cannot be inverted by
per-leg min-max degeneracies (review finding #11). Quarantined rows never
reach the lanes: the status='active' filter excludes them structurally
(critique item 14).

Access scoping (review finding #17): callers pass their resolved
principal/trust. Owners see everything; non-owner callers see only
unscoped memories or ones scoped to them, and only episodes attributable
to them (principal or author match) — an unenrolled gateway user can never
query-mine another user's turns. The caller's own current session is
excluded from the episode leg (review finding #15) so lane 2 doesn't
recall the turn that was just captured.

Safety rules honored here:
  * Raw user text is NEVER interpolated into a MATCH expression — the
    query is tokenized (Unicode-aware, review finding #9) and each token
    is double-quoted, which makes FTS syntax injection impossible.
  * search()/log_retrieval() are on the prefetch capture path and never
    raise: failures are logged and degrade to [] / no-op.
  * retrieval_log rows are batch-inserted in ONE transaction and the
    recall-count bump is a single UPDATE ... WHERE id IN (critique 32).
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from ..capture.symbols import expand_query
from ..store import db
from ..store import vec as vec_store
from . import fusion

logger = logging.getLogger(__name__)

# Unicode-aware: [^\W_] = "word char minus underscore" under re.UNICODE,
# matching the unicode61 tokenizer's view of the indexed text.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

# bm25() column weights (order = FTS column order in schema.sql).
_MEM_BM25 = "bm25(memory_fts, 1.0, 2.0, 3.0, 2.0)"      # content, summary, symbols, tags
_EPI_BM25 = "bm25(episode_fts, 1.0, 0.8, 3.0)"          # user, assistant, symbols

_EPISODE_SCORE_FACTOR = 0.6
# Normalization band floor: the worst in-set candidate keeps 0.2, so being
# "last of three close matches" is not annihilated to zero (finding #11).
_NORM_FLOOR = 0.2


@dataclass
class Hit:
    kind: str                       # 'memory' | 'episode'
    id: int
    uid: str
    text: str
    summary: str | None
    memory_type: str | None
    mkind: str | None            # memories.kind (fact|decision|warning|...)
    ts: str
    platform: str | None
    score: float
    source: str                     # 'fts' | 'vec' | 'fts+vec' | 'like'


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------

def _tokens(query: str) -> list[str]:
    """Casefolded Unicode word runs from the query plus code-symbol expansions."""
    toks = [t.casefold() for t in _TOKEN_RE.findall(query or "") if t]
    for sym in expand_query(query or ""):
        for part in _TOKEN_RE.findall(sym):
            folded = part.casefold()
            if folded and folded not in toks:
                toks.append(folded)
    return toks


def _match_expr(query: str) -> str:
    """Safe OR-joined MATCH expression; '' when the query has no tokens."""
    return " OR ".join(f'"{t}"' for t in _tokens(query))


# ---------------------------------------------------------------------------
# Score modulation (memory-engine.md §3.5)
# ---------------------------------------------------------------------------

def _age_days(valid_from: str, now: datetime) -> float:
    try:
        then = datetime.fromisoformat(valid_from.replace("Z", "+00:00"))
        return max(0.0, (now - then).total_seconds() / 86400.0)
    except (ValueError, TypeError):
        return 0.0


def _modulate(row: sqlite3.Row, base: float, now: datetime) -> float:
    score = base
    half_life = row["half_life_days"]
    if half_life is not None and half_life > 0:
        score *= max(0.3, 0.5 ** (_age_days(row["valid_from"], now) / half_life))
    if row["outcome"] == "failed":
        score *= 1.5
    if row["kind"] == "warning":
        score *= 1.2
    feedback = 1.0 + 0.1 * min(row["helpful_count"], 5) - 0.15 * min(row["harmful_count"], 5)
    score *= min(2.0, max(0.2, feedback))
    if row["pinned"]:
        score *= 1.3
    return score


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 8,
    kinds: list[str] | None = None,
    exclude_kinds: tuple[str, ...] = (),
    scope_project: str | None = None,
    include_episodes: bool = True,
    episode_limit: int = 4,
    exclude_session: str | None = None,
    principal_id: str | None = None,
    source_author: str | None = None,
    trust_tier: str = "owner",
    embedder=None,
) -> list[Hit]:
    """Ranked, access-scoped recall over memories (+episodes). Never raises.

    Hybrid path (P2): FTS/BM25 and vector-KNN legs each yield a ranked id
    list per table; RRF (k=60, Daem0n's fusion.py — finally wired) combines
    them over ONE keyspace ('m:id'/'e:id'), the fused scores are min-max
    normalized into a floored band once (finding #11 discipline), then the
    usual lifecycle modulation and the 0.6x episode factor apply. With no
    embedder (or no sqlite-vec) the vector legs are simply absent — RRF over
    the FTS lists alone degrades to rank order.
    """
    try:
        match = _match_expr(query)
        if not match:
            return []
        if not db.capabilities(conn).get("fts5"):
            return _like_search(
                conn, query, limit=limit, kinds=kinds, exclude_kinds=exclude_kinds,
                scope_project=scope_project,
                include_episodes=include_episodes, episode_limit=episode_limit,
                exclude_session=exclude_session, principal_id=principal_id,
                source_author=source_author, trust_tier=trust_tier,
            )
        now = datetime.now(UTC)
        want_episodes = include_episodes and episode_limit > 0

        # -- FTS legs (rows arrive filtered and in bm25 order) --
        mem_rows = _memories_rows(conn, match, limit, kinds, scope_project,
                                  principal_id, trust_tier, exclude_kinds)
        epi_rows = _episodes_rows(conn, match, episode_limit * 3, exclude_session,
                                  principal_id, source_author, trust_tier) \
            if want_episodes else []
        rows_by_key = {f"m:{r['id']}": r for r in mem_rows}
        rows_by_key.update({f"e:{r['id']}": r for r in epi_rows})
        rankings: list[list[str]] = [
            [f"m:{r['id']}" for r in mem_rows],
            [f"e:{r['id']}" for r in epi_rows],
        ]
        vec_keys: set = set()

        # -- vector legs (optional; same filters applied at row fetch) --
        if embedder is not None and vec_store.vec_available(conn):
            try:
                qvec = embedder.encode_query(query)
                mem_knn = [i for i, _ in vec_store.knn(conn, "mem_vec", qvec, limit * 3)]
                mvrows = _memories_by_ids(conn, mem_knn, kinds, scope_project,
                                          principal_id, trust_tier, exclude_kinds)
                rankings.append([f"m:{i}" for i in mem_knn if i in mvrows])
                rows_by_key.update({f"m:{i}": r for i, r in mvrows.items()})
                vec_keys.update(f"m:{i}" for i in mvrows)
                if want_episodes:
                    # The exclude_session filter is CORRELATED with the query
                    # (the query IS the current conversation), so the nearest
                    # neighbors are exactly the rows we'll discard — size K to
                    # survive that (finding #5), capped.
                    k = episode_limit * 3
                    if exclude_session:
                        own = conn.execute(
                            "SELECT count(*) FROM episodes WHERE session_id=?",
                            (exclude_session,)).fetchone()[0]
                        k = min(k + own, 512)
                    epi_knn = [i for i, _ in vec_store.knn(conn, "epi_vec", qvec, k)]
                    evrows = _episodes_by_ids(conn, epi_knn, exclude_session,
                                              principal_id, source_author, trust_tier)
                    surviving = [i for i in epi_knn if i in evrows][: episode_limit * 3]
                    rankings.append([f"e:{i}" for i in surviving])
                    rows_by_key.update({f"e:{i}": evrows[i] for i in surviving})
                    vec_keys.update(f"e:{i}" for i in surviving)
            except Exception as e:
                logger.warning("vector leg failed (%s); continuing with FTS only", e)

        fts_keys = {f"m:{r['id']}" for r in mem_rows} | {f"e:{r['id']}" for r in epi_rows}
        bases = fusion.normalized(fusion.rrf(rankings), floor=_NORM_FLOOR)

        def _leg(key: str) -> str:
            """Honest provenance (finding #6): which leg(s) found this row."""
            in_fts, in_vec = key in fts_keys, key in vec_keys
            return "fts+vec" if (in_fts and in_vec) else ("vec" if in_vec else "fts")

        hits: list[Hit] = []
        episode_count = 0
        for key, base in sorted(bases.items(), key=lambda kv: kv[1], reverse=True):
            row = rows_by_key.get(key)
            if row is None:
                continue
            if key.startswith("m:"):
                hits.append(Hit(
                    kind="memory", id=row["id"], uid=row["uid"],
                    text=row["content"] or "", summary=row["summary"],
                    memory_type=row["memory_type"], mkind=row["kind"],
                    ts=row["valid_from"], platform=row["source_platform"],
                    score=_modulate(row, base, now), source=_leg(key),
                ))
            elif episode_count < episode_limit:
                episode_count += 1
                hits.append(Hit(
                    kind="episode", id=row["id"], uid=row["uid"],
                    text=f"{row['user_content']}\n{row['assistant_content']}".strip(),
                    summary=None, memory_type=None, mkind=None,
                    ts=row["ts"], platform=row["platform"],
                    score=base * _EPISODE_SCORE_FACTOR, source=_leg(key),
                ))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]
    except Exception as e:  # capture path — a memory bug must never break the turn
        logger.warning("brain search failed for %r: %s", query, e)
        return []


def _memories_by_ids(conn, ids, kinds, scope_project, principal_id, trust_tier,
                     exclude_kinds=()) -> dict:
    """Fetch vector-candidate memory rows with the SAME access filters as the
    FTS leg — a vector hit must never bypass scoping (finding #17)."""
    if not ids:
        return {}
    sql = (
        f"SELECT m.* FROM memories m WHERE m.id IN ({','.join('?' * len(ids))}) "
        "AND m.valid_to IS NULL AND m.status = 'active' AND m.live = 1"
    )
    params: list = list(ids)
    if kinds:
        sql += f" AND m.kind IN ({','.join('?' * len(kinds))})"
        params.extend(kinds)
    if exclude_kinds:
        sql += f" AND (m.kind IS NULL OR m.kind NOT IN ({','.join('?' * len(exclude_kinds))}))"
        params.extend(exclude_kinds)
    if scope_project is not None:
        sql += " AND (m.scope_project IS NULL OR m.scope_project = ?)"
        params.append(scope_project)
    sql = _scope_memories(sql, params, principal_id, trust_tier)
    return {r["id"]: r for r in conn.execute(sql, params).fetchall()}


def _episodes_by_ids(conn, ids, exclude_session, principal_id, source_author,
                     trust_tier) -> dict:
    if not ids:
        return {}
    sql = f"SELECT e.* FROM episodes e WHERE e.id IN ({','.join('?' * len(ids))})"
    params: list = list(ids)
    if exclude_session:
        sql += " AND e.session_id != ?"
        params.append(exclude_session)
    scoped = _scope_episodes(sql, params, principal_id, source_author, trust_tier)
    if scoped is None:
        return {}
    return {r["id"]: r for r in conn.execute(scoped, params).fetchall()}


def _scope_memories(sql: str, params: list, principal_id: str | None,
                    trust_tier: str) -> str:
    """Non-owner callers see unscoped memories or their own (finding #17)."""
    if trust_tier == "owner":
        return sql
    sql += " AND (m.scope_user IS NULL OR m.scope_user = ?)"
    params.append(principal_id or "")
    return sql


def _scope_episodes(sql: str, params: list, principal_id: str | None,
                    source_author: str | None, trust_tier: str) -> str | None:
    """Non-owner callers see only episodes attributable to THEM; an
    unenrolled caller (no principal, no author) gets no episode leg at all.
    Untrusted-tier episodes are excluded for everyone. Returns None when the
    leg should be skipped entirely."""
    sql += " AND e.trust_tier != 'untrusted'"
    if trust_tier == "owner":
        return sql
    attribution = []
    if principal_id:
        attribution.append("e.principal_id = ?")
        params.append(principal_id)
    if source_author:
        attribution.append("e.source_author = ?")
        params.append(source_author)
    if not attribution:
        return None
    return sql + f" AND ({' OR '.join(attribution)})"


def _memories_rows(
    conn: sqlite3.Connection,
    match: str,
    limit: int,
    kinds: list[str] | None,
    scope_project: str | None,
    principal_id: str | None,
    trust_tier: str,
    exclude_kinds: tuple[str, ...] = (),
) -> list:
    sql = (
        f"SELECT m.*, {_MEM_BM25} AS bm25_score "
        "FROM memory_fts JOIN memories m ON m.id = memory_fts.rowid "
        "WHERE memory_fts MATCH ? "
        "AND m.valid_to IS NULL AND m.status = 'active' AND m.live = 1"
    )
    params: list = [match]
    if kinds:
        sql += f" AND m.kind IN ({','.join('?' * len(kinds))})"
        params.extend(kinds)
    if exclude_kinds:
        sql += f" AND (m.kind IS NULL OR m.kind NOT IN ({','.join('?' * len(exclude_kinds))}))"
        params.extend(exclude_kinds)
    if scope_project is not None:
        sql += " AND (m.scope_project IS NULL OR m.scope_project = ?)"
        params.append(scope_project)
    sql = _scope_memories(sql, params, principal_id, trust_tier)
    sql += " ORDER BY bm25_score LIMIT ?"
    params.append(limit * 3)
    return conn.execute(sql, params).fetchall()


def _episodes_rows(
    conn: sqlite3.Connection,
    match: str,
    episode_limit: int,
    exclude_session: str | None,
    principal_id: str | None,
    source_author: str | None,
    trust_tier: str,
) -> list:
    sql = (
        f"SELECT e.*, {_EPI_BM25} AS bm25_score "
        "FROM episode_fts JOIN episodes e ON e.id = episode_fts.rowid "
        "WHERE episode_fts MATCH ?"
    )
    params: list = [match]
    if exclude_session:
        # Don't recall the turn we just captured (finding #15); cross-session
        # recall is the semantics that matters.
        sql += " AND e.session_id != ?"
        params.append(exclude_session)
    scoped = _scope_episodes(sql, params, principal_id, source_author, trust_tier)
    if scoped is None:
        return []
    scoped += " ORDER BY bm25_score LIMIT ?"
    params.append(episode_limit)
    return conn.execute(scoped, params).fetchall()


# ---------------------------------------------------------------------------
# LIKE fallback (no-FTS5 floor tier)
# ---------------------------------------------------------------------------

def _like_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int,
    kinds: list[str] | None,
    exclude_kinds: tuple[str, ...] = (),
    scope_project: str | None,
    include_episodes: bool,
    episode_limit: int,
    exclude_session: str | None = None,
    principal_id: str | None = None,
    source_author: str | None = None,
    trust_tier: str = "owner",
) -> list[Hit]:
    """Degraded search when capabilities lack fts5: LIKE per token, score by
    match count (fraction of tokens present). Same Hit surface, source='like'.

    exclude_kinds must be honored here too (review): the lane-2 facts path
    passes exclude_kinds=('strategy','guardrail','case') so guidance-type
    memories are never surfaced as recalled facts — dropping it on the
    no-FTS5 floor tier would both leak them into the facts block and
    double-credit them (guidance leg + like leg) in the outcome miner.
    """
    toks = _tokens(query)
    if not toks:
        return []
    hits: list[Hit] = []

    like_clause = " OR ".join(["m.content LIKE ? ESCAPE '\\'"] * len(toks))
    sql = (
        "SELECT m.* FROM memories m "
        f"WHERE ({like_clause}) "
        "AND m.valid_to IS NULL AND m.status = 'active' AND m.live = 1"
    )
    params: list = [f"%{_like_escape(t)}%" for t in toks]
    if kinds:
        sql += f" AND m.kind IN ({','.join('?' * len(kinds))})"
        params.extend(kinds)
    if exclude_kinds:
        sql += f" AND (m.kind IS NULL OR m.kind NOT IN ({','.join('?' * len(exclude_kinds))}))"
        params.extend(exclude_kinds)
    if scope_project is not None:
        sql += " AND (m.scope_project IS NULL OR m.scope_project = ?)"
        params.append(scope_project)
    sql = _scope_memories(sql, params, principal_id, trust_tier)
    sql += " ORDER BY m.valid_from DESC LIMIT ?"
    params.append(limit * 3)

    for row in conn.execute(sql, params).fetchall():
        text = row["content"] or ""
        hits.append(Hit(
            kind="memory", id=row["id"], uid=row["uid"], text=text,
            summary=row["summary"], memory_type=row["memory_type"],
            mkind=row["kind"], ts=row["valid_from"],
            platform=row["source_platform"],
            score=_match_fraction(text, toks), source="like",
        ))

    if include_episodes and episode_limit > 0:
        epi_clause = " OR ".join(
            ["e.user_content LIKE ? ESCAPE '\\'", "e.assistant_content LIKE ? ESCAPE '\\'"] * len(toks)
        )
        epi_sql = f"SELECT e.* FROM episodes e WHERE ({epi_clause})"
        epi_params: list = []
        for t in toks:
            epi_params.extend([f"%{_like_escape(t)}%"] * 2)
        if exclude_session:
            epi_sql += " AND e.session_id != ?"
            epi_params.append(exclude_session)
        scoped = _scope_episodes(epi_sql, epi_params, principal_id, source_author, trust_tier)
        if scoped is not None:
            scoped += " ORDER BY e.ts DESC LIMIT ?"
            epi_params.append(episode_limit)
            for row in conn.execute(scoped, epi_params).fetchall():
                text = f"{row['user_content']}\n{row['assistant_content']}".strip()
                hits.append(Hit(
                    kind="episode", id=row["id"], uid=row["uid"], text=text,
                    summary=None, memory_type=None, mkind=None,
                    ts=row["ts"], platform=row["platform"],
                    score=_match_fraction(text, toks) * _EPISODE_SCORE_FACTOR,
                    source="like",
                ))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:limit]


def _like_escape(token: str) -> str:
    return token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _match_fraction(text: str, toks: list[str]) -> float:
    low = text.casefold()
    return sum(1 for t in toks if t in low) / len(toks)


# ---------------------------------------------------------------------------
# Retrieval bookkeeping
# ---------------------------------------------------------------------------

def log_retrieval(
    conn: sqlite3.Connection,
    session_id: str,
    query: str,
    hits: list[Hit],
    injected_uids: set[str],
    guidance: list[tuple[int, str]] = (),
) -> None:
    """Record candidacy + injection in ONE transaction (critique item 32).

    Only memory hits are logged (retrieval_log.memory_id references
    memories); injected memory hits also get their recall counters bumped
    in a single UPDATE. Never raises — capture path.

    Rows land PENDING (``user_msg_hash``/``user_turn_count`` NULL). They are
    not yet attributable to a turn, because of the host's ordering (verified
    ``memory_manager.py`` docstring + single-worker executor at :376-379):

        turn N:   prefetch_all(msg_N)   -> serves the block cached below
                  ... turn runs ...
                  sync_all(msg_N, resp) -> capture
                  queue_prefetch_all(msg_N) -> WE COMPUTE THE BLOCK HERE

    The block computed from msg_N is injected into turn **N+1**, so the row
    must be stamped with turn N+1's message, not msg_N's. `sync_turn` for
    N+1 arrives after that injection and carries the raw user text, so
    ``stamp_pending_injections`` closes the loop there.

    Why not hash msg_N here: (a) it is the wrong turn, and (b) `query` is
    the host's *stripped* query (``_strip_skill_scaffolding``), which never
    hash-matches the raw text state.db stores — the miner joins on state.db
    `messages`, so only the raw text works.
    """
    try:
        mem_hits = [h for h in hits if h.kind == "memory"]
        # Guidance items (strategy/guardrail/case) are injected via the
        # learned-guidance subsection, not the fact search, but they are
        # memories rows too — logging them is what lets dream/mine credit a
        # strategy item that helped (the flywheel). They arrive already
        # filtered to the ones actually rendered, so all are injected=1.
        if not mem_hits and not guidance:
            return
        now = db.iso_now()
        q_hash = db.content_hash(query)
        rows = [
            (session_id, q_hash, now, h.id, h.source, h.score,
             1 if h.uid in injected_uids else 0)
            for h in mem_hits
        ]
        rows += [(session_id, q_hash, now, gid, "guidance", None, 1)
                 for gid, _uid in guidance]
        injected_ids = [h.id for h in mem_hits if h.uid in injected_uids]
        injected_ids += [gid for gid, _uid in guidance]
        with conn:  # one transaction
            # A newer block supersedes any still-pending one for this session:
            # only the most recent block is ever served by prefetch(), so the
            # older batch was never actually injected. Demote it rather than
            # letting the next stamp credit two batches to one turn.
            conn.execute(
                "UPDATE retrieval_log SET injected=0 WHERE session_id=? "
                "AND injected=1 AND user_msg_hash IS NULL",
                (session_id,),
            )
            conn.executemany(
                "INSERT INTO retrieval_log (session_id, query_hash, ts, "
                "memory_id, leg, rank_score, injected) VALUES (?,?,?,?,?,?,?)",
                rows,
            )
            if injected_ids:
                conn.execute(
                    "UPDATE memories SET recall_count = recall_count + 1, "
                    f"last_recalled_at = ? WHERE id IN ({','.join('?' * len(injected_ids))})",
                    [now, *injected_ids],
                )
    except Exception as e:  # capture path — never break the turn
        logger.warning("brain log_retrieval failed: %s", e)


def stamp_pending_injections(
    conn: sqlite3.Connection,
    session_id: str,
    turn_no: int | None,
    user_msg: str,
) -> int:
    """Attribute this session's pending injections to the turn they landed in.

    Called from the worker's `sync_turn` job — the first moment the brain
    knows the RAW user text of the turn that just consumed the cached block.
    Stamping (user_turn_count, user_msg_hash) is what makes
    `dream/mine_state.py` able to resolve a `turn_id` out of state.db and
    close the injection -> outcome loop.

    Returns the number of rows stamped. Never raises — capture path.
    """
    try:
        if not user_msg or not user_msg.strip():
            return 0  # nothing to hash against state.db messages
        cur = conn.execute(
            "UPDATE retrieval_log SET user_turn_count=?, user_msg_hash=? "
            "WHERE session_id=? AND user_msg_hash IS NULL",
            (turn_no, db.content_hash(user_msg), session_id),
        )
        return cur.rowcount or 0
    except Exception as e:
        logger.warning("brain stamp_pending_injections failed: %s", e)
        return 0
