# Research: daem0n:memory-core

## Summary

Daem0n-MCP's memory core is a per-project SQLite (SQLAlchemy async/aiosqlite) + local embedded Qdrant system with a 4-category memory model (decision/pattern/warning/learning), a hand-rolled TF-IDF index rebuilt in RAM on every process start, ModernBERT (nomic-ai/modernbert-embed-base) embeddings truncated to 256 dims via matryoshka with asymmetric query/document prefixes and ONNX-quantized inference, and a weighted-sum TF-IDF+vector hybrid recall with exponential recency decay and failure/warning boosts. Around this solid core sits a large halo of partially-wired or dead machinery: an 'Auto-Zoom' retrieval router with a semantic exemplar-embedding query classifier that ships permanently in shadow mode (auto_zoom_enabled=False), a BM25+RRF fusion module that is never used for memory retrieval (docs claim BM25+RRF; the real path is TF-IDF weighted combination), and a surprise/importance scoring subsystem whose DB columns and calculator module exist but are never invoked at write time. Pruning/archiving/compaction exist as manual, dry-run-by-default MCP tools with sensible saliency protections (pinned, outcomes, recall_count) but no automatic lifecycle. Concurrency is 'SQLite WAL + NullPool + hope': Qdrant local mode takes an exclusive file lock, so a second concurrent session silently degrades to TF-IDF-only search. The best ideas — episodic-vs-semantic permanence, outcome-driven learning (worked=False boosts as warnings), bi-temporal versioning, active context, prefix-asymmetric truncated embeddings — are worth keeping; the layers of routers/planners/classifiers stacked on a ~hundreds-of-rows corpus are mostly over-engineering.

## Subsystems

### Memory data model & categories

Files: `C:\Users\dasbl\Daem0n-MCP\daem0nmcp\models.py`, `C:\Users\dasbl\Daem0n-MCP\daem0nmcp\migrations\schema.py`

Single `memories` table with category in {decision, pattern, warning, learning}. Patterns/warnings are 'semantic' -> is_permanent=True (auto-set in remember()), never decay; decisions/learnings are 'episodic' and decay. Fields: content, rationale, context JSON, tags JSON, keywords (legacy token string), file_path + file_path_relative (Windows case-folded, dual absolute/relative for portability), vector_embedding BLOB (packed float32), outcome TEXT + worked BOOL (outcome tracking), pinned (prune-protection only — the claimed 'boosted relevance' is never implemented in recall), archived (soft-hide; NULL treated as not-archived for legacy rows), recall_count (incremented on every recall for saliency), surprise_score/importance_score (columns exist, NEVER written — dead), source_client/source_model provenance. Tags auto-inferred at write time via word-boundary regexes (bugfix/tech-debt/perf/warning). Sidecar tables: memory_versions (bi-temporal: changed_at = transaction time, valid_from/valid_to = valid time, invalidated_by_version_id chain; a version row is created on create/outcome/relationship change), memory_relationships (5 typed edges: led_to/supersedes/depends_on/conflicts_with/related_to, confidence float), facts (SHA256 content-hash O(1) lookup, verification_count, Engram-inspired), active_context, memory_communities, extracted_entities/refs, context_triggers.

**Assessment:** The episodic/semantic split with is_permanent, outcome tracking, and pinned/archived flags is genuinely good and simple. Bi-temporal versioning is well-implemented but heavyweight for the actual use (per-memory recall does an N+1 per-memory version query when as_of_time is set). surprise_score/importance_score are pure schema debt — migration 12 added the columns, surprise.py implements the math, nothing ever calls it. 'pinned boosts relevance' is documented but false.

### Embedding pipeline (ModernBERT / ONNX / matryoshka)

Files: `C:\Users\dasbl\Daem0n-MCP\daem0nmcp\vectors.py`, `C:\Users\dasbl\Daem0n-MCP\daem0nmcp\config.py`

Lazy global singleton SentenceTransformer('nomic-ai/modernbert-embed-base', truncate_dim=256, backend='onnx', model_kwargs={'file_name':'onnx/model_quantized.onnx'}) with torch fallback on load failure. Matryoshka truncation to 256 dims (full model is 768). Asymmetric encoding: every query is prefixed 'search_query: ', every document 'search_document: ' (Nomic's training prefixes). Embeddings serialized as struct-packed float32 bytes ('<n>f') and stored twice: SQLite BLOB (memories.vector_embedding) AND Qdrant point vector. No embedding-result caching (only query-vector caching exists in the TF-IDF layer); each recall re-encodes the query. encode text = content + ' ' + rationale.

**Assessment:** Excellent, minimal, correct. ONNX-quantized + matryoshka-256 is a smart local-first choice (small vectors, fast CPU inference). The dual storage (SQLite BLOB + Qdrant) is redundant — SQLite copy is only used to re-upsert into Qdrant on outcome updates; a successor should pick one source of truth. No query-embedding cache is a small miss (classifier + search each encode separately).

### TF-IDF index & legacy similarity engine

Files: `C:\Users\dasbl\Daem0n-MCP\daem0nmcp\similarity.py`

Hand-rolled TFIDFIndex: tokenize() lowercases, splits camelCase/snake_case, filters ~150 stop words, whitelists 2-char tech terms (db/ui/id/io/os/ip/vm/ai/ml), plus extract_code_symbols() regexes for backticked code, CamelCase, snake_case, SCREAMING_SNAKE, .method access. Augmented TF (0.5 + 0.5*tf/max_tf) * smoothed IDF (log((N+1)/(df+1))+1), sparse-dict cosine. Tags injected 3x for weight. LRU query-vector cache (100 entries). Whole index rebuilt from SQLite into RAM on startup and whenever meta-table triggers show external DB change (has_changes_since). Also hosts calculate_memory_decay (exp decay, half-life 30d default, floor 0.3) and detect_conflict (temp TF-IDF over candidates, similarity>=0.5-0.6 + heuristics: similar_failed, existing_warning, potential_duplicate >0.8, polarity_conflict via negation-word mismatch).

**Assessment:** The code-symbol-aware tokenizer is genuinely valuable for a coding-agent memory (exact identifier matching that embeddings miss). O(N) linear scan per query and full in-RAM rebuild are fine at hundreds of memories, unacceptable at global-agent scale. IDF cache is fully invalidated on every single add_document — O(N) recompute churn during batch writes. Conflict detection is a nice cheap idea (esp. polarity/negation check) but threshold-brittle.

### Hybrid retrieval & recall pipeline

Files: `C:\Users\dasbl\Daem0n-MCP\daem0nmcp\memory.py`, `C:\Users\dasbl\Daem0n-MCP\daem0nmcp\qdrant_store.py`

MemoryManager._hybrid_search: TF-IDF top 2k (threshold 0.1, recall uses 0.05) + Qdrant cosine top 2k filtered at vector_threshold 0.3, combined as final = 0.7*tfidf + 0.3*vector (settings.hybrid_vector_weight=0.3). NOT BM25, NOT RRF, despite docstrings. recall(): 5s TTL cache (50 entries, key = all params) -> hybrid search top limit*4 -> SQL fetch with date filters -> post-hoc Python filters (tags, file_path with 4-way endswith matching, bi-temporal as_of_time via per-memory version lookups) -> score = base * decay (permanent=1.0, else exp half-life 30d floor 0.3) * 1.5 if worked==False * 1.2 if category==warning -> sort, paginate, bucket per-category with per-category limit. recall_count incremented on every hit (even cache hits). Qdrant: local file mode, cosine, collection recreated on dimension mismatch (silently discards vectors!), payload {category, tags, file_path, worked, is_permanent} enables metadata filtering (rarely used). Fallback ladder everywhere: Qdrant locked -> TF-IDF only; FTS5 missing -> LIKE. Separate fts_search() uses SQLite FTS5 (content/rationale/tags, triggers keep it synced, bm25() ranking + snippet highlighting) — a parallel keyword path not fused into recall.

**Assessment:** The scoring recipe (semantic match x recency decay x failure boost x warning boost) is the crown jewel — cheap, explainable, and directly encodes 'failures are warnings'. But there are THREE keyword search systems (in-RAM TF-IDF, unused BM25Index, SQLite FTS5) plus vectors, and the actually-shipped fusion is a naive weighted sum of incomparable score scales (TF-IDF cosine vs embedding cosine). Silent collection recreation on dim change is a data-loss footgun. limit*4 pagination-before-category-bucketing gives approximate/unstable pagination.

### BM25 + RRF fusion (dead for memories)

Files: `C:\Users\dasbl\Daem0n-MCP\daem0nmcp\bm25_index.py`, `C:\Users\dasbl\Daem0n-MCP\daem0nmcp\fusion.py`

BM25Index wraps rank_bm25.BM25Okapi (k1=1.5, b=0.75), lazy full rebuild on dirty flag, same tokenizer + 3x tag boost. fusion.reciprocal_rank_fusion: score(d)=sum 1/(k+rank), k=60, over ranked lists. RRFHybridSearch fuses BM25 top-50 + vector top-50 (vector threshold 0.3). CRITICAL: no module imports fusion; BM25Index's only consumer is tool_search.py (MCP tool discovery). Memory retrieval never uses BM25 or RRF — retrieval_router's 'hybrid' delegates back to memory._hybrid_search (TF-IDF weighted sum). The v6 docs/docstrings advertising 'BM25+vector RRF hybrid' are aspirational.

**Assessment:** The code itself is clean and correct — RRF with k=60 is the right rank-based fusion (scale-free, no score-normalization hacks). It was just never wired in. A successor should implement exactly this and delete the weighted-sum path.

### Auto-Zoom retrieval router + query classifiers

Files: `C:\Users\dasbl\Daem0n-MCP\daem0nmcp\retrieval_router.py`, `C:\Users\dasbl\Daem0n-MCP\daem0nmcp\query_classifier.py`, `C:\Users\dasbl\Daem0n-MCP\daem0nmcp\recall_planner.py`

Two classifiers exist. (1) recall_planner.classify_query_complexity: regex patterns (trace/history/why did/when... => COMPLEX; 'what is X'/single word => SIMPLE) + word count (<=3 SIMPLE, <=8 MEDIUM, else COMPLEX); RecallPlanner maps to RecallPlan (max_communities 3/5/10, max_raw 5/10/20, filter thresholds 0.5/0.3/0.2, compression rates). (2) ExemplarQueryClassifier: 6 exemplar phrases per level, embedded lazily with the shared ModernBERT model with query prefix; query classified by max cosine sim to any exemplar; confidence < 0.25 -> fallback MEDIUM. RetrievalRouter.route_search: SIMPLE -> Qdrant vector-only (skips TF-IDF), MEDIUM -> hybrid, COMPLEX -> hybrid seeds + knowledge-graph expansion from top-5 seeds (find_related_memories, depth 2, score = seed*0.8^depth) + up to 5 community summaries. Every path falls back to hybrid on any exception. DEFAULTS: auto_zoom_enabled=False, auto_zoom_shadow=True — so in production it classifies, logs, and then always runs hybrid anyway; only JIT-compression metadata (4K/8K/16K token tiers) is emitted.

**Assessment:** The exemplar-embedding classifier is an elegant, cheap idea (no LLM call, handles semantically-complex short queries) and worth reusing. But the whole router is over-engineered for the deployment reality: shipped disabled, duplicating the regex planner it was meant to replace, and the SIMPLE fast path saves ~milliseconds on a corpus where linear TF-IDF scan is already trivial. Graph expansion with geometric depth discount is a good pattern for COMPLEX queries.

### Surprise / importance scoring

Files: `C:\Users\dasbl\Daem0n-MCP\daem0nmcp\surprise.py`

Titans-inspired: surprise = mean cosine DISTANCE to k=5 nearest existing embeddings, clamped [0,1]; first memory = 1.0. Intended: high surprise = novel = prioritize; importance_score = EWC-style protection weight from recall frequency + positive outcomes. Reality: calculate_surprise/SurpriseCalculator have zero call sites outside the module; memories.surprise_score and importance_score are never populated; nothing gates remember() — every remember() call is stored unconditionally (only conflict WARNINGS are returned, storage still happens).

**Assessment:** Dead code / vaporware. The concept (novelty-gated write, importance-protected decay) is exactly what a continual-learning successor needs, and the k-NN-distance formulation is a reasonable v1 — but in Daem0n it is 100% unwired. The de facto write gate is only the LLM's judgment plus conflict warnings.

### Decay, pruning, archiving, compaction lifecycle

Files: `C:\Users\dasbl\Daem0n-MCP\daem0nmcp\tools\maintenance.py`, `C:\Users\dasbl\Daem0n-MCP\daem0nmcp\workflows\maintain.py`, `C:\Users\dasbl\Daem0n-MCP\daem0nmcp\memory.py`

Decay is read-time only (score multiplier), never destroys data. Destruction is manual via MCP tools, dry_run=True by default: prune_memories deletes episodic memories older than 90d that have NO outcome, are not permanent/pinned/archived, recall_count < 5, and (optionally) not worked=True — i.e. only never-used, never-resolved episodic flotsam. archive is a per-memory soft flag (excluded from index+recall, kept in DB). compact_memories: caller supplies a >=50-char summary; oldest unpinned episodic memories (decisions require recorded outcome) are linked from a new 'learning' summary via 'supersedes' edges and archived, Qdrant vectors deleted, index rebuilt, tags = compacted+checkpoint+majority tags. purge_dream_spam dedupes dream-generated learnings (keep newest per source decision / per day).

**Assessment:** The protection hierarchy (pinned > outcome-recorded > frequently-recalled > successful) is thoughtful and safe. But nothing runs automatically — no scheduler calls prune/compact, so DBs grow until a human invokes maintenance. Compaction requiring the caller (LLM) to write the summary is honest but means it rarely happens. Full index rebuild after every prune/compact is O(N).

### Caching & active context

Files: `C:\Users\dasbl\Daem0n-MCP\daem0nmcp\cache.py`, `C:\Users\dasbl\Daem0n-MCP\daem0nmcp\active_context.py`

TTLCache: thread-safe dict of (timestamp, value), TTL 5s, maxsize 50 (recall) / 50 (rules), lazy expiry + oldest-eviction, hit/miss stats. make_cache_key recursively tuple-izes args. Any write clears the ENTIRE recall cache. ActiveContextManager (MemGPT core-memory style): per-project list of max 10 memory pins in active_context table with priority ordering, reason string, optional expires_at; auto-injected into briefings; failed decisions auto-added at priority 10 by record_outcome. Hard CONTEXT_FULL error at 10 items; condensed mode truncates content to 150 chars.

**Assessment:** Both are appropriately simple. 5s TTL is really a request-dedup cache, not a performance layer — fine. Active context is one of the best ideas in the system: bounded, explainable, auto-populated by failure, cheap. Keep it nearly verbatim.

### Concurrency model & database layer

Files: `C:\Users\dasbl\Daem0n-MCP\daem0nmcp\database.py`, `C:\Users\dasbl\Daem0n-MCP\daem0nmcp\rwlock.py`

Per-project SQLite via create_async_engine(aiosqlite) with NullPool (fresh connection per operation), PRAGMAs: WAL, synchronous=NORMAL, busy_timeout=30000, foreign_keys=ON, temp_store=MEMORY, cache_size=-64000 (64MB). get_session = commit-on-success/rollback-on-error context manager. Cross-process freshness: 'meta' table with memories_last_modified/rules_last_modified updated by SQL triggers; _check_index_freshness compares against _index_built_at and rebuilds the whole in-RAM TF-IDF index when another process wrote. rwlock.py is a clean asyncio Condition-based reader-writer lock, used ONLY by context_manager.py to guard the project-context registry (double-checked locking around per-project singleton creation) — not for memory data. Qdrant local mode holds an exclusive file lock: second concurrent Claude session gets RuntimeError('already accessed by another instance') and that entire process runs TF-IDF-only, silently losing semantic search.

**Assessment:** SQLite WAL + NullPool + trigger-based staleness detection is a pragmatic multi-process story for the DB itself. But the Qdrant single-writer lock means the headline feature (vector search) degrades to keyword search whenever two sessions overlap — logged at INFO and invisible to the user. Full TF-IDF rebuild on any external change is O(N) per detection. The RWLock is well-written but barely used. For a global agent-wide brain, this whole model (per-project embedded stores, exclusive locks) is the main thing to replace with a single served store.

## What Worked

- Episodic vs semantic memory split: patterns/warnings are permanent (no decay), decisions/learnings decay — one boolean encodes a real cognitive distinction
- Outcome-driven learning loop: record_outcome(worked=False) makes the memory a boosted implicit warning (1.5x) and auto-pins it into active context at priority 10 — failure knowledge compounds automatically
- Read-time exponential decay (half-life 30d, floor 0.3) instead of destructive decay — relevance fades but data survives
- Asymmetric ModernBERT embeddings: 'search_query: '/'search_document: ' prefixes + matryoshka truncation to 256d + ONNX-quantized backend = fast, correct, fully local
- Code-symbol-aware tokenizer (backticks, CamelCase, snake_case, SCREAMING_SNAKE, .method) — keyword recall of exact identifiers that pure embeddings miss
- Active context (MemGPT-style): hard cap of 10 pinned memories per project with priority + reason + expiry, auto-injected into briefings
- Layered graceful degradation: Qdrant -> TF-IDF -> FTS5 -> LIKE; classifier failure -> hybrid; nothing ever hard-fails a recall
- Prune protection hierarchy: pinned, has-outcome, recall_count>=5, worked=True, permanent — plus dry_run=True default on all destructive ops
- Write-time conflict detection returning warnings (similar_failed / existing_warning / potential_duplicate / negation-polarity) without blocking the write
- Trigger-maintained meta table for cross-process index staleness detection; SQLite WAL + busy_timeout for multi-session writes
- recall_count saliency tracking incremented even on cache hits — cheap usage signal feeding prune protection
- Exemplar-embedding query classifier: complexity classification by cosine sim to ~6 canned phrases per level, no LLM call needed

## Weaknesses

- Three keyword search engines coexist (in-RAM TF-IDF, unused BM25Index, SQLite FTS5) and the shipped 'hybrid' is a naive 0.7/0.3 weighted sum of incomparable score scales — the clean RRF fusion module (fusion.py) has zero call sites for memory retrieval, while docs claim BM25+RRF
- Surprise/importance scoring is entirely dead: columns migrated, calculator implemented, never called — nothing gates what gets remembered; every remember() stores unconditionally
- Auto-Zoom router ships permanently in shadow mode (auto_zoom_enabled=False): classifies every query with an embedding pass, logs it, then runs hybrid anyway — pure overhead; plus a second, redundant regex classifier in recall_planner
- Qdrant local mode's exclusive file lock silently downgrades any second concurrent session to TF-IDF-only search (INFO log only) — vector search is unavailable exactly when the user runs multiple agents
- Entire TF-IDF index rebuilt in RAM from all rows at startup and on any external change; IDF cache fully invalidated on every add_document; O(N) linear scan per query — none of this scales to an agent-wide global store
- QdrantVectorStore silently deletes and recreates the collection on embedding-dimension mismatch — data loss on a config change
- pinned is documented as 'boosted relevance' but no recall path reads it; bi-temporal as_of_time recall does an N+1 version query per candidate memory
- No automatic lifecycle: prune/compact/cleanup are manual MCP tools; compaction requires the LLM to author the summary; databases grow unboundedly by default
- Embeddings stored redundantly in both SQLite BLOB and Qdrant with no single source of truth; no query-embedding cache (query encoded separately by classifier and search)
- recall() pagination applies offset/limit*4 before category bucketing — page boundaries are approximate and unstable; per-category limit interacts oddly with offset
- Feature halo built for scale the system never reached: communities/Leiden, JIT compression tiers, GraphRAG expansion, recall plans — layered on per-project DBs that typically hold hundreds of rows
- Per-project isolation (storage under each project) is the wrong shape for the successor's goal of one agent-wide global memory; cross-project reads exist only via ad-hoc project_links federation that instantiates whole MemoryManagers per linked repo per recall

## Reusable Ideas

- Keep the 4-field learning core: category + is_permanent (episodic/semantic) + outcome/worked + read-time decay with floor; failed outcomes auto-boost as warnings (1.5x) and auto-enter active context
- Adopt RRF (k=60) over BM25 (rank_bm25, k1=1.5, b=0.75) + vector as the ONE fusion path — the code in fusion.py/bm25_index.py is ready to lift; delete weighted-sum hybrids
- Reuse vectors.py nearly verbatim: nomic-ai/modernbert-embed-base, truncate_dim=256, ONNX quantized, 'search_query: '/'search_document: ' asymmetric prefixes, float32-packed storage — but pick a single vector source of truth (served Qdrant or sqlite-vec), never embedded-local Qdrant with exclusive locks for multi-session use
- Lift the code-symbol tokenizer (similarity.py extract_code_symbols + tokenize) into the successor's keyword indexer — it is the differentiator for coding-agent recall
- Actually wire surprise gating: compute k-NN cosine distance (surprise.py math) at write time from the same Qdrant query used for conflict detection; use it to (a) skip/merge near-duplicates below a threshold, (b) set initial importance; combine with recall_count + worked for an importance score that modulates decay and prune eligibility — this is the continual-learning loop Daem0n designed but never closed
- Keep active context exactly as designed: hard cap ~10, priority + reason + expires_at, auto-add failures, condensed rendering — it is the cheapest high-value feature in the codebase
- Keep write-time conflict detection as non-blocking warnings, especially the negation-polarity heuristic and 'similar approach failed before' surfacing
- Keep the exemplar-embedding query classifier pattern (cosine to canned exemplars, confidence fallback to MEDIUM) if routing is needed — but only add routing tiers after the corpus is big enough that the fast path measurably matters; ship it enabled or not at all
- Keep bi-temporal versioning concept (happened_at/valid_from vs recorded_at) but implement point-in-time filtering as a single SQL join, not per-memory Python loops
- Keep dry_run=True defaults and the prune protection ladder (pinned > outcome > recall_count > successful), but drive prune/compact from a scheduler (the successor's continual-learning daemon) instead of waiting for manual invocation
- Keep SQLite WAL PRAGMAs + trigger-maintained last_modified meta table for cross-process cache/index invalidation — cheap and reliable
- Prefer incremental index updates (FTS5 triggers already do this for keyword search) over full in-RAM rebuilds; SQLite FTS5 with bm25() may be sufficient as the sole keyword leg, eliminating the in-RAM TF-IDF/BM25 duplication entirely
