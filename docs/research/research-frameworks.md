# Research: research:frameworks

## Summary

The open-source agent-memory landscape converged hard in 2025-2026 on a recognizable stack: (1) a three-tier memory taxonomy (episodic/semantic/procedural) plus an in-context 'core/working' tier; (2) LLM-driven fact extraction at write time with ADD/UPDATE/DELETE/NOOP reconciliation against existing memories; (3) multi-signal retrieval (vector + BM25 + entity/graph + temporal) with fusion and reranking — exactly what Daem0n-MCP already has; (4) temporal validity windows on facts (Zep/Graphiti bi-temporal model, now widely copied); (5) asynchronous background memory revision ('sleep-time compute', Letta's term) as the emerging default over inline memory writes; and (6) an explicit consolidation policy layer (importance gating, write-time entity merge, decay, compliance-only eviction) as the recognized frontier problem — 'an agent that remembers everything remembers nothing useful.' Benchmarks are a minefield: LoCoMo/LongMemEval are near-saturated and every vendor self-reports SOTA (the Mem0-vs-Zep public dispute shows scores swing 25+ points based on evaluation configuration); newer benchmarks (BEAM, LongMemEval-V2, OmniMemEval, MemoryArena) exist precisely because the old ones are gamed. Production-grade and maintained: Mem0 (largest ecosystem, ~48K stars), Letta (best conceptual architecture, requires its runtime), Graphiti (Apache-2.0 temporal KG engine, ~24K stars — but Zep's full server went cloud-only), Cognee, MemOS (Apache-2.0, active, strongest self-reported numbers + cross-task skill reuse), Hindsight (newer, strong architecture, vendor-benchmarked). Research-code or thin: A-MEM (NeurIPS 2025 paper, Zettelkasten-style memory evolution — great ideas, not production), LangMem (best procedural-memory API design, but LangGraph-locked and development slowed), MIRIX (six-memory-type multi-agent design, paper-stage). Mixed/commercial: supermemory (effectively closed-source managed service), memori (SQL-native, simple, cheap), Memobase (profile+event schema with buffer/flush batching — good pattern for cheap VPS deployment), Memvid (single-file engine with genuinely useful ideas buried under gimmicky 'video memory' marketing; notably an RFC to adopt it in hermes-agent was closed 'not planned' in May 2026). For Hermes-Brain, which already exceeds most of these systems on retrieval, the genuine beyond-Daem0n gaps the field points to are: sleep-time agents operating on shared editable memory blocks, an explicit four-lever consolidation/forgetting policy, separation of evidence from inference (Hindsight's fact-vs-belief networks), cross-platform memory scoping tags, and — the least-solved, highest-value area — procedural memory that rewrites prompts and skills from experience.

## Findings

### Mem0

The most widely adopted open-source memory layer (~48K GitHub stars, $24M Series A, Apache-2.0 core + paid platform). Two-phase pipeline: extraction (LLM distills salient facts from each message pair) and update (compare new fact against existing memories, then ADD/UPDATE/DELETE/NOOP). Graph variant (Mem0-g) adds entity/relationship extraction; in the hosted product, graph memory is paywalled ($249/mo Pro tier).

**Key techniques:**
- Two-phase extract-then-reconcile pipeline with ADD/UPDATE/DELETE/NOOP operations decided by LLM
- Multi-scope memory tagging: facts tagged with user_id / agent_id / run_id / app_id that compose at retrieval time
- Async (non-blocking) memory writes as default
- 2026 retrieval upgrade: three parallel scoring passes — vector similarity + BM25 + entity matching — fused into a combined score, replacing pure vector search
- Metadata filtering for scoped queries
- 21+ framework integrations (LangGraph, CrewAI, OpenAI Agents SDK, voice stacks like LiveKit/Pipecat)

**Evidence:** Self-reported 66-68% LLM-judge on LoCoMo in the April 2025 arXiv paper (accepted ECAI 2025), ~1.8K tokens/conversation vs 26K full-context, p95 search 200ms. Their 2026 'State of AI Agent Memory' claims 92.5 LoCoMo / 94.4 LongMemEval / 64.1 BEAM-1M — all self-reported vendor numbers. Their evaluation of competitors is contested: Zep published a rebuttal ('Is Mem0 Really SOTA?') claiming Mem0 misconfigured Zep and that corrected Zep scores 75.14% vs Mem0's ~68%. Known failure modes (from HN discussions, Feb 2026): stores explicit facts but does not learn behavioral patterns implicitly (repeated corrections never become a preference); extraction misreads sarcasm/temporary statements as permanent facts; weak temporal reasoning vs KG systems; memory staleness ('confidently wrong' high-relevance outdated facts) admitted as unsolved by Mem0 themselves.

**Applicability:** Hermes-Brain already has hybrid BM25+vector+RRF, so Mem0's headline retrieval is not new. What IS worth stealing: (1) the explicit ADD/UPDATE/DELETE/NOOP reconciliation step at write time — a clean, auditable primitive for memory hygiene that complements Daem0n's contradiction detection; (2) multi-scope tags are directly the right answer for Hermes' cross-platform problem — tag every memory with platform (telegram/discord/slack/cli), user, agent, session, and compose scopes at retrieval so one brain serves all surfaces; (3) async writes so a $5-VPS deployment never blocks a Telegram reply on memory I/O.

**Sources:**
- https://arxiv.org/abs/2504.19413
- https://github.com/mem0ai/mem0
- https://mem0.ai/blog/state-of-ai-agent-memory-2026
- https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/
- https://news.ycombinator.com/item?id=46891715
- https://news.ycombinator.com/item?id=47770220
- https://vectorize.io/articles/mem0-vs-zep

### Letta (MemGPT lineage) + sleep-time agents

The MemGPT successor: a full stateful-agent runtime (not a library) with OS-inspired memory hierarchy. Memory = context engineering: message buffer (recent turns), core memory (editable in-context 'memory blocks' with label/description/value/char-limit), recall memory (searchable full history on disk), archival memory (external vector/graph store). Agents edit their own memory via tools. Sleep-time agents (2025) are background agents sharing memory blocks with the primary agent, revising memory asynchronously during idle time. 2026 'Letta V1' rearchitected the loop away from MemGPT's tool-centric heartbeats toward model-native reasoning, and introduced MemFS — a git-backed memory filesystem projecting memory blocks into markdown files with version history.

**Key techniques:**
- Memory blocks: labeled, size-bounded, in-context, agent-editable, shareable across multiple agents
- Sleep-time compute: dedicated background agent does memory compaction, archive management, and 'learned context' synthesis without blocking the conversation
- Recursive summarization evicting ~70% of buffer on overflow
- Recall vs archival tier separation (raw history vs processed knowledge)
- MemFS: git-backed, human-inspectable/editable memory with version history and conflict resolution

**Evidence:** Backed by $10M seed; actively developed through 2026 (V1 loop rearchitecture). Sleep-time compute has a supporting paper and shipped product docs, not just a blog claim. Original MemGPT scored 93.4% on DMR (a benchmark Zep later beat at 94.8%). Criticisms: requires committing to Letta's whole runtime; DMR is a weak benchmark; earlier MemGPT bundled memory management into the conversation agent, causing latency — which sleep-time agents explicitly fix.

**Applicability:** The single most important pattern for Hermes-Brain. Daem0n's idle-time 'dreaming' already re-evaluates failed decisions; the Letta generalization is a full sleep-time agent that owns ALL memory maintenance — consolidation, block rewriting, archive promotion, skill distillation — over memory blocks shared with the live agent. For a multi-platform Hermes, shared memory blocks are the mechanism by which the Telegram, Discord, and CLI faces read/write one brain. MemFS's git-backed markdown projection is ideal for Hermes' ethos: memory stays greppable and user-auditable on a $5 VPS while the real store stays structured. Do NOT adopt the Letta runtime itself (heavy, framework lock-in); steal the block + sleep-time patterns.

**Sources:**
- https://docs.letta.com/guides/agents/architectures/sleeptime/
- https://www.letta.com/blog/sleep-time-compute/
- https://www.letta.com/blog/agent-memory/
- https://www.letta.com/blog/letta-v1-agent
- https://docs.letta.com/letta-agent/memory

### Zep + Graphiti (temporal knowledge graph)

Zep is a commercial memory service whose open core is Graphiti (Apache-2.0, ~24K stars): a real-time, temporally-aware knowledge graph engine. Three subgraphs — episode (raw messages), semantic entity, community — mirroring episodic/semantic memory. Bi-temporal model: every edge carries both event time and ingestion time plus explicit validity intervals (valid_at/invalid_at); new facts invalidate old ones rather than deleting them. Retrieval is hybrid (cosine + BM25 + graph traversal) with ~300ms P95, no LLM calls at query time. IMPORTANT: Zep Community Edition was deprecated — the full self-hostable Zep server is gone; open-source users get Graphiti (the engine) and must run their own Neo4j/FalkorDB/Kuzu.

**Key techniques:**
- Bi-temporal edges with validity intervals — supersede, don't delete
- Incremental non-lossy graph updates (no batch recompute) for real-time ingestion
- Episode/entity/community three-layer subgraph design
- Community detection for higher-level summaries (label propagation for cheap incremental updates)
- Hybrid retrieval without query-time LLM calls for low latency

**Evidence:** arXiv 2501.13956: 94.8% DMR (vs MemGPT 93.4%) and up to +18.5% accuracy / -90% latency vs full-context baselines on LongMemEval — self-reported but methodologically detailed. Independent 2026 comparisons put Zep at 63.8% LongMemEval with GPT-4o (vs Mem0 49%) — note these come from vectorize.io, itself a memory vendor. Center of the LoCoMo benchmark war with Mem0 (Mem0 claims Zep=58-66%, Zep claims corrected 75.14%). Criticisms: heavy memory footprint (Mem0 measured >600K tokens/conversation ingested into graph); eventual consistency — facts retrievable only after background graph processing completes (Mem0 observed correct answers appearing hours after ingestion); self-hosting now means operating a graph DB yourself.

**Applicability:** Daem0n already has a bi-temporal KG + GraphRAG/Leiden, so Zep validates rather than extends the design. Two refinements worth taking: (1) Graphiti's invalidation-not-deletion policy as the formal contradiction-resolution rule (contradicted fact gets invalid_at set, stays queryable for provenance); (2) their finding that query-time LLM calls are the latency killer — keep Hermes-Brain retrieval LLM-free and push all LLM work to write/sleep time, critical on cheap hardware. The Zep CE deprecation is also a strategic lesson: Hermes-Brain must be fully self-hostable with zero cloud dependency, which is a differentiator the community now actively wants.

**Sources:**
- https://arxiv.org/abs/2501.13956
- https://github.com/getzep/graphiti
- https://blog.getzep.com/announcing-a-new-direction-for-zeps-open-source-strategy/
- https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/
- https://github.com/getzep/zep-papers/issues/5
- https://vectorize.io/articles/zep-alternatives

### LangMem (LangChain)

LangChain's memory SDK (early 2025, MIT). The cleanest articulation of the semantic/episodic/procedural taxonomy in API form. Semantic memory as either 'collections' (unbounded searchable facts) or 'profiles' (single structured document, schema-validated, continuously patched). Episodic memory as distilled few-shot examples from past successful interactions. Procedural memory as learned instructions written back into the agent's own system prompt via 'prompt optimizers' (gradient-style prompt improvement from interaction feedback). Storage-agnostic core API + native LangGraph store integration.

**Key techniques:**
- Profile vs collection duality for semantic memory (bounded structured state vs unbounded fact store)
- Procedural memory = automated system-prompt rewriting from trajectories + feedback (metaprompt/gradient optimizers)
- Episodic memory as auto-distilled few-shot exemplars
- Background 'memory manager' that extracts/consolidates after conversations rather than inline
- Subconscious vs conscious formation distinction (hot path tool calls vs background extraction)

**Evidence:** Official LangChain release with docs and conceptual guide; no benchmark claims of its own (notable honesty). Criticisms from 2026 comparisons: 'severe framework lock-in', flat key-value + vector architecture, personalization-only in practice, slowed development. Mem0's paper reports LangMem-style baselines below its own scores (self-serving, flag accordingly).

**Applicability:** The API shapes to steal, not the library (it's LangGraph-locked and its development cadence has visibly slowed). For Hermes-Brain: (a) the profile/collection split maps perfectly to per-user profiles (one bounded, schema-typed document per user per platform, cheap to load every turn on a VPS) vs the open fact store; (b) the prompt-optimizer pattern is the most concrete existing implementation of 'self-tuning behavior' — Hermes' Reflexion loop output should feed a prompt/skill optimizer that patches system prompt sections and agentskills.io skill files, with diffs logged; (c) episodic few-shot distillation gives skills worked examples for free.

**Sources:**
- https://www.langchain.com/blog/langmem-sdk-launch
- https://langchain-ai.github.io/langmem/concepts/conceptual_guide/
- https://docs.langchain.com/oss/python/concepts/memory
- https://vectorize.io/articles/best-ai-agent-memory-systems

### Cognee

Graph-native open-source 'memory control plane' unifying relational + vector + graph storage. ECL pipeline (Extract-Cognify-Load): ingest from 38+ sources; 'cognify' runs a six-stage pipeline (classify docs, check permissions, chunk, LLM entity/relation extraction, summarize, embed + commit graph edges); optional RDF/OWL ontology validation constrains the graph to a domain schema. A 'memify' layer adds feedback loops: user-rated responses adjust graph edge weights so retrieval sharpens with use.

**Key techniques:**
- ECL pipeline with pluggable tasks (build-your-own memory pipeline as DAG)
- Ontology-constrained knowledge graphs (RDF/OWL validation of extracted triples)
- Feedback-weighted edges: response ratings propagate back into graph edge weights
- Unified query surface over sparse/dense/graph/hybrid/multi-hop retrieval
- Permission-aware ingestion (per-document ACL checks in the pipeline)

**Evidence:** €7.5M seed (2025); claims >1M pipelines/month and 70+ companies. Self-reported benchmarks: beat SOTA on BEAM 100K-token setting by 6.5%, matched at 10M tokens; 0.93 human-level on HotPotQA — all from Cognee's own blog, unverified independently. Criticisms: smaller community than Mem0/Letta; Python-only; much of its comparative content is vendor marketing (their '2026 best memory layers' posts rank themselves first).

**Applicability:** Two ideas beyond Daem0n: (1) feedback-weighted edges is a lightweight continual-learning mechanism — Hermes' existing outcome tracking should write back into retrieval weights (memories that led to successful outcomes rank higher; repeated failure demotes), which is 'improvement over time' without any training; (2) optional ontology validation could keep Hermes' KG from schema drift as it ingests from four different platforms. Python-native, self-hostable, fits the Hermes stack.

**Sources:**
- https://www.cognee.ai/blog/fundamentals/how-cognee-builds-ai-memory
- https://www.cognee.ai/blog/guides/best-open-source-ai-memory-tools-for-llm-agents-and-developers
- https://www.cognee.ai/blog/cognee-news/cognee-raises-seven-million-five-hundred-thousand-dollars-seed

### MemOS (MemTensor)

'Memory operating system' for LLMs (Apache-2.0, ~10K stars, very active — 2026 releases include OpenClaw plugins and memos-local-plugin 2.0). First system to unify three memory substrates: plaintext (facts/documents), activation (KV-cache as reusable memory — precomputed attention states injected instead of re-prefilling), and parametric (LoRA-style weights). MemCube is the unified abstraction wrapping any memory type with metadata, provenance, lifecycle state, and permissions; cubes are composable, shareable, and can TRANSFORM between types (hot plaintext memory gets promoted to KV-cache; stable patterns could distill to parameters). MemScheduler handles async ingestion and hot/cold placement.

**Key techniques:**
- MemCube: one abstraction for plaintext/activation/parametric memory with lifecycle states and access control
- Memory-type transformation as a lifecycle policy (plaintext -> KV-cache -> parametric as stability increases)
- KV-cache memory reuse for latency (claimed 35.24% token savings; large TTFT gains)
- MemScheduler: async ingestion + scheduling of which memories are 'next-scene' prefetched
- Cross-task skill reuse (procedural memories reused across tasks)
- Natural-language memory feedback/correction API

**Evidence:** arXiv 2507.03724; self-reported: 88.83 LoCoMo, 89.20 LongMemEval, first place on their own OmniMemEval (14 products, 10 datasets — vendor-run, treat skeptically), +159% temporal reasoning vs OpenAI memory, 36.63%->50.87% task completion in OpenClaw integration. All numbers from the MemTensor team. Criticisms: OmniMemEval is their own benchmark; the parametric-memory layer is largely aspirational in the OSS release; Chinese-company backing raises supply-chain questions for some deployers; complexity is high.

**Applicability:** The substrate-transformation idea is the most 'beyond Daem0n' concept in the field: on Hermes' GPU-box deployments (local models), stable high-frequency memories can be promoted to precomputed KV-cache for near-zero-latency recall, and genuinely stable procedures could eventually distill to LoRA — while the same MemCube-style abstraction degrades gracefully to plaintext-only on a $5 VPS with API models. Even ignoring activation memory, MemCube's lifecycle-state + provenance + permissions envelope is the right schema for Hermes memory objects. Caveat: activation/parametric layers only work with self-hosted models, so they must be optional capability tiers, which matches Hermes' model-agnostic design.

**Sources:**
- https://github.com/MemTensor/MemOS
- https://arxiv.org/abs/2507.03724
- https://arxiv.org/html/2505.22101v1

### Memobase

User-profile-centric memory for chat applications (open source; FastAPI + Postgres + Redis + pgvector). Memory = typed user profile (configurable topic/subtopic ontology, ~8 topics/40 slots by default) + timeline of user events. Distinctive cold-path architecture: every user gets a write buffer; chats accumulate until the buffer exceeds a token threshold (~1024) or goes idle (~1 hour), then the whole buffer is flushed and batch-processed into profile updates and events.

**Key techniques:**
- Typed, schema-configurable profile ontology instead of free-form fact soup
- Buffer-and-flush batch processing (idle-triggered) instead of per-message LLM extraction
- Separate event timeline for temporal questions
- Profile served whole (no retrieval step) for personalization context

**Evidence:** Active GitHub project (memodb-io/memobase) with Dify plugin, PyPI package, docs; claims better temporal-question performance than fact-based rivals but publishes little rigorous benchmarking. Smaller community; chat-personalization scope only (no institutional/procedural memory).

**Applicability:** The buffer/flush pattern is exactly right for Hermes on cheap hardware: per-message extraction (Mem0-style) costs an LLM call per turn; Memobase-style idle-triggered batch flush cuts that by an order of magnitude and dovetails with Hermes' idle 'dreaming' window — flush buffers, then dream. The typed profile ontology also suits Hermes' cross-platform users: one canonical profile per human, updated from any platform's flushes.

**Sources:**
- https://github.com/memodb-io/memobase
- https://docs.memobase.io/DOC
- https://pypi.org/project/memobase/

### A-MEM (NeurIPS 2025)

Academic 'agentic memory' system (Rutgers/AIOS group) applying Zettelkasten principles: every memory is stored as an atomic note with LLM-generated structured attributes (context description, keywords, tags); on insertion the system retrieves related historical notes, decides link creation agentically, and — the key contribution — triggers 'memory evolution': new memories cause the LLM to UPDATE the contextual descriptions, tags, and links of EXISTING old memories, so the network reorganizes itself continuously.

**Key techniques:**
- Atomic notes with LLM-generated metadata (context/keywords/tags)
- Agentic link generation (LLM decides connections, not just cosine threshold)
- Memory evolution: retroactive rewriting of old memories' metadata when new related memories arrive
- ChromaDB-backed lightweight implementation

**Evidence:** Peer-reviewed (NeurIPS 2025, arXiv 2502.12110); claims superior results over SOTA baselines on LoCoMo across six foundation models with dramatically fewer tokens. Reproduction attempts note reliance on LLM cooperation for structured outputs and cost of per-insertion LLM calls. Maintenance: research code, sporadic commits, no production hardening.

**Applicability:** Memory evolution is the missing piece in most production systems and in Daem0n: contradiction detection updates facts, but A-MEM shows the retrieval scaffolding around old memories (descriptions, tags, links) should also be rewritten as understanding improves. This is a perfect job for the Hermes sleep-time agent: during dreaming, re-describe and re-link old memories in light of new ones. Take the pattern, not the code — A-MEM is research-grade (multiple divergent repos: agiresearch/A-mem, WujiangXu/A-mem-sys), not production infrastructure.

**Sources:**
- https://arxiv.org/abs/2502.12110
- https://github.com/WujiangXu/A-mem-sys
- https://github.com/agiresearch/a-mem

### Hindsight (Vectorize)

Late-2025 open-source memory framework (vectorize-io/hindsight, ~4K stars) built around 'retain, recall, reflect'. Its signature idea: separate memory into four logical networks — world facts (objective claims), agent experiences (what the agent did/observed), entity summaries (synthesized per-entity views), and evolving beliefs (the agent's own inferences, revisable and traceable) — explicitly separating evidence from inference, which Mem0/Zep-style extract-and-store blurs. Retrieval runs four parallel strategies (semantic, BM25, graph, temporal) with cross-encoder reranking; a reflection layer reasons over the bank to answer and update beliefs traceably. Deliberately has no eviction: consolidation replaces deletion.

**Key techniques:**
- Fact/experience/entity-summary/belief network separation (evidence vs inference as a schema property)
- Reflect as a first-class operation producing traceable belief updates
- Four parallel retrieval strategies + cross-encoder rerank
- Write-time entity resolution; temporal validity on claims
- Consolidation-not-eviction philosophy; their 'four levers' essay (importance, merge, decay, eviction) is the best practitioner writeup of forgetting policy

**Evidence:** arXiv 2512.12818: LongMemEval 39.0%->83.6% with a 20B open backbone, 91.4% with larger backbones; LoCoMo 89.61%. Vectorize claims 94.6% LongMemEval 'officially reproduced' and runs a public benchmark site — but Vectorize owns Hindsight, and their many '(2026) X vs Y' comparison articles are vendor content that always favors Hindsight; treat all of it as self-reported. Young project, smaller community, less production mileage than Mem0/Letta.

**Applicability:** Two direct upgrades for Hermes-Brain: (1) split the store by epistemic status — Daem0n tracks contradictions between facts, but Hermes should also keep its own inferences (Reflexion outputs, dream conclusions) in a separate 'beliefs' network that cites supporting facts and gets revised when evidence changes; this makes self-improvement auditable and prevents inference laundering into fact. (2) Adopt the four-lever forgetting policy explicitly: importance gate at write time, entity merge at write time, decay only on temporal claims, eviction only for compliance/user-deletion — 'actively forget' without destroying provenance.

**Sources:**
- https://arxiv.org/abs/2512.12818
- https://github.com/vectorize-io/hindsight
- https://hindsight.vectorize.io/blog/2026/05/21/agent-memory-consolidation
- https://benchmarks.hindsight.vectorize.io/

### supermemory

Commercial 'Memory API' ($3M funding, Oct 2025) with a vector-graph engine: ontology-aware edges, automatic contradiction handling and temporal reasoning, sub-400ms latency claims. Positions memory as a standardized API layer rather than a framework. The GitHub repo exists but the current product is effectively a managed service; true self-hosting requires an enterprise agreement — 'open source' in branding more than practice.

**Key techniques:**
- Memory-as-API abstraction (drop-in via OpenAI-compatible proxy that transparently injects memories)
- Vector graph with ontology-aware edges
- Automatic contradiction + temporal handling at the engine level
- Claimed local mode with local embeddings/Ollama

**Evidence:** 81.6% LongMemEval (GPT-4o) self-reported; marketing-heavy comparison content on their blog. Criticisms in 2026 roundups: closed-source in practice, managed-only, benchmark claims unverified.

**Applicability:** Mostly a cautionary tale plus one good idea. The good idea: the transparent proxy pattern — Hermes-Brain could expose an OpenAI-compatible endpoint that auto-injects/collects memory for ANY tool in the user's stack, extending the brain beyond Hermes itself with zero integration work. The caution: don't build Hermes-Brain against any service whose self-hosting story can be revoked (see also Zep CE).

**Sources:**
- https://github.com/supermemoryai/supermemory
- https://supermemory.ai/blog/best-memory-apis-stateful-ai-agents
- https://vectorize.io/articles/best-ai-agent-memory-systems

### memori (GibsonAI)

SQL-native open-source memory engine (Sept 2025; expanded Dec 2025 to full memory layer across Postgres/MySQL/SQLite/MongoDB). Thesis: most agent memory doesn't need vector infrastructure — structured extraction (entities, relationships, context priority) into plain SQL with full-text search is cheaper, transparent, auditable, and portable. Dual modes: 'conscious' working memory (recent, promoted context) and 'auto' mode (dynamic search per query).

**Key techniques:**
- SQL-only storage/retrieval: entity extraction + relationship tables + FTS instead of embeddings
- Conscious/auto dual retrieval modes (standing working set vs on-demand search)
- One-line enable() wrapper intercepting LLM calls to record/inject memory
- SQLite export = total data portability

**Evidence:** MarkTechPost launch coverage (Sept 2025), InfoQ on the Dec 2025 expansion; no LoCoMo/LongMemEval numbers published — benchmarks absent rather than inflated. Younger and less battle-tested; FTS-only retrieval will miss paraphrase recall that Hermes' hybrid stack already handles.

**Applicability:** Relevant as the floor of Hermes-Brain's hardware range: on a $5 VPS, a SQLite/FTS5-backed mode (Daem0n already has BM25) with memori-style structured tables may be the right default tier, with ONNX embeddings and graph layers activating only when resources allow. Claimed 10-50ms responses and 80-90% infra cost reduction vs vector DBs (vendor numbers). Its 'conscious mode' promotion of essential memories into a standing working set parallels Letta core memory — convergent evidence that an always-loaded working set is a best practice.

**Sources:**
- https://www.marktechpost.com/2025/09/08/gibsonai-releases-memori-an-open-source-sql-native-memory-engine-for-ai-agents/
- https://www.infoq.com/news/2025/12/memori/
- https://neurotechnus.com/2025/09/09/sql-memory-ai-agents/

### Newer 2025-2026 entrants: MIRIX, MemMachine, MemU, EverMemOS, Memvid

The second wave. MIRIX (arXiv 2507.07957): multi-agent memory with SIX types — Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault — where a meta-controller routes updates/retrievals to per-type memory managers; extends to multimodal (screenshots). MemMachine (MemVerge-backed, Apache-2.0): episodic (graph) + profile (SQL) + working memory, positioning as neutral infrastructure. MemU (NevaMind): memory as a sidecar service shared across desktop agent hosts — one store, one embedding space, cross-host learning. EverMemOS (lipps): enterprise memory with construction/perception dual-track 'cognitive loop'. Memvid: Rust single-file '.mv2' engine (BM25+HNSW+WAL+time-travel in one portable file) — v1's 'store memory as QR codes in MP4' framing was widely mocked as marketing gimmickry (HN/Lobsters), v2 dropped most of it; notably, an RFC to adopt Memvid as a pluggable memory backend for Hermes (NousResearch/hermes-agent#23874, May 2026) was closed 'not planned'.

**Key techniques:**
- MIRIX: per-memory-type manager agents + router; Resource memory and Knowledge Vault as distinct types; 99.9% storage reduction on multimodal streams via structured memory instead of raw retention
- MemMachine: clean episodic/profile/working three-way split over graph+SQL
- MemU: cross-host shared memory sidecar (one brain, many agent surfaces)
- Memvid v2: single-file crash-safe memory (embedded WAL, hybrid index, time-travel debugging)

**Evidence:** MIRIX self-reports 85.4% LoCoMo (SOTA claim) and +35% over RAG on ScreenshotVQA — paper-stage, limited production use. MemMachine/MemU/EverMemOS are active repos but young with modest adoption; treat as pattern sources, not dependencies. Memvid criticism documented on HN (item 44134122) and Lobsters.

**Applicability:** MIRIX's six-type taxonomy is the strongest argument that Hermes-Brain should treat Procedural and Resource memory (files, documents the user shared) as first-class stores with their own update policies, not folded into semantic memory. MemU's sidecar model is architecturally what Hermes-Brain should be relative to Telegram/Discord/Slack/CLI surfaces. The closed Hermes Memvid RFC is direct evidence the Nous maintainers prefer an in-house design over adopting a third-party engine — Hermes-Brain should be that in-house design, though Memvid v2's single-portable-file + WAL crash-safety idea is worth borrowing for the low-end deployment tier.

**Sources:**
- https://arxiv.org/abs/2507.07957
- https://github.com/MemMachine/MemMachine
- https://github.com/NevaMind-AI/memU
- https://github.com/lipps/EverMemOS
- https://github.com/memvid/memvid
- https://news.ycombinator.com/item?id=44134122
- https://github.com/NousResearch/hermes-agent/issues/23874

### Benchmark landscape & the credibility problem

Evaluation in this space is broken and every vendor exploits it. LoCoMo (1,540 questions, multi-session conversations) and LongMemEval (500 questions, six ability categories incl. knowledge update, temporal reasoning, abstention) are the de facto standards, and both are near-saturated and configuration-sensitive: the Mem0-vs-Zep war saw Zep's score reported as anywhere from 58.44% to 84% depending on who ran it and which adversarial categories were included; a '100% LoCoMo' claim by one vendor was achieved with top_k=50 (retrieving essentially the whole conversation). Successors: LongMemEval-V2 (arXiv 2605.12493, 'experienced colleague' framing), BEAM (1M-10M token scales, designed so nothing saturates — scores drop to 48-64%), OmniMemEval (MemTensor's own, vendor-run), plus 2026 probes of specific failure modes: STALE (recognizing invalid memories), PersistBench (cross-domain leakage, memory-induced sycophancy), MemSyco-Bench (sycophancy from memory), MemoryArena (interdependent multi-session agentic tasks).

**Key techniques:**
- Cross-vendor scores are incomparable without identical retrieval k, judge model, and question-category inclusion
- BEAM-style scale testing (1M-10M tokens) is where systems actually differentiate
- New failure-mode benchmarks: staleness detection, memory sycophancy, preference persistence
- 'No widely adopted benchmark tests whether agents actually improve over time on real tasks' — the institutional-learning gap

**Evidence:** Mem0 vs Zep dispute fully documented in dueling blog posts and GitHub issues; BEAM gap acknowledged even in Mem0's own 2026 report (92.5 LoCoMo vs 48.6 BEAM-10M); LongMemEval-V2 paper explicitly motivated by V1 saturation.

**Applicability:** Hermes-Brain should not design toward LoCoMo/LongMemEval numbers — they measure conversational QA recall, not what Hermes-Brain targets (cross-platform persistence, procedural learning, self-tuning). Instead: run LongMemEval-V2 + BEAM as sanity checks, add STALE-style staleness probes (Daem0n's contradiction detection should shine there), and build an internal longitudinal eval from Hermes' own outcome tracking: does task success rate on repeated task types actually rise week over week? That metric — improvement over time — is the one nobody's benchmark covers and the one that matters for the project's goals.

**Sources:**
- https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/
- https://github.com/getzep/zep-papers/issues/5
- https://mem0.ai/blog/ai-memory-benchmarks-in-2026
- https://arxiv.org/html/2605.12493v1
- https://arxiv.org/pdf/2607.01071
- https://mem0.ai/blog/state-of-ai-agent-memory-2026

### 2026 architectural convergence (cross-framework synthesis)

Where the field landed. Converged (table stakes): episodic/semantic/procedural taxonomy + an always-in-context core/working tier; LLM extraction at write time with reconcile-against-existing (ADD/UPDATE/DELETE/NOOP); multi-signal retrieval (vector+BM25+entity/graph+temporal) with fusion/rerank; temporal validity windows and supersede-don't-delete; async/background memory writes; write-time entity resolution; memory scoping tags. Emerging (the actual frontier): sleep-time/background consolidation agents (Letta sleep-time, Hermes-style dreaming, 'AutoDream' Feb 2026); explicit four-lever forgetting policy (importance/merge/decay/eviction); evidence-vs-inference separation (Hindsight); memory evolution — retroactively rewriting old memories' metadata (A-MEM); feedback-weighted retrieval (Cognee); substrate promotion plaintext->KV-cache->parameters (MemOS); procedural memory as prompt/skill rewriting (LangMem, MemOS cross-task skill reuse) — universally acknowledged as 'early-stage in tooling' even by Mem0. Maintained/production-grade in mid-2026: Mem0, Letta, Graphiti (engine only), Cognee, MemOS, Memobase, Hindsight (young), memori (young). Hype-decayed or constrained: Zep CE (killed), LangMem (slowed, locked-in), A-MEM/MIRIX (research code), supermemory (closed in practice), Memvid v1 framing (gimmick, rewritten).

**Key techniques:**
- Consensus stack: typed memory tiers + write-time extraction/reconciliation + hybrid retrieval + temporal validity + async consolidation
- The differentiators for 2026+: forgetting policy, belief revision, procedural learning, and longitudinal self-improvement measurement
- 'Institutional knowledge trumps personalization' — systems built for learning-from-experience subsume personalization, not vice versa

**Evidence:** Convergence independently described by Zylos Research, vectorize.io, Atlan, MachineLearningMastery and Mem0's own report; the three-tier taxonomy and multi-signal retrieval appear in every major 2026 framework. Procedural-memory immaturity is admitted across vendors including Mem0 ('early-stage'), and the HN critique that Mem0 'stores memories but doesn't learn user patterns' generalizes to the whole category.

**Applicability:** Daem0n-MCP already implements essentially the entire converged consensus stack (hybrid RRF retrieval, bi-temporal KG, contradiction detection, Reflexion, dreaming, compression). Hermes-Brain's headroom is precisely the emerging list: formalize dreaming into a Letta-style sleep-time agent over shared memory blocks; add the four-lever forgetting policy; split beliefs from evidence; wire outcome tracking into retrieval weights; and build the procedural layer (prompt/skill rewriting into agentskills.io skills) that no shipping framework has nailed — that last one is where Hermes-Brain can lead rather than follow.

**Sources:**
- https://zylos.ai/research/2026-04-05-ai-agent-memory-architectures-persistent-knowledge/
- https://vectorize.io/articles/best-ai-agent-memory-systems
- https://hindsight.vectorize.io/blog/2026/05/21/agent-memory-consolidation
- https://mem0.ai/blog/state-of-ai-agent-memory-2026
- https://machinelearningmastery.com/the-6-best-ai-agent-memory-frameworks-you-should-try-in-2026/
- https://news.ycombinator.com/item?id=46891715

## Top Recommendations

- 1. Formalize 'dreaming' into a Letta-style sleep-time agent operating on shared, size-bounded memory blocks. This is the field's clearest emerging best practice: a background agent that owns ALL memory maintenance (consolidation, block rewriting, archive promotion, A-MEM-style retroactive re-linking/re-describing of old memories, skill distillation) while the live agent stays fast. Shared editable blocks are also the mechanism that unifies Telegram/Discord/Slack/CLI into one brain; project them to git-backed markdown (Letta MemFS pattern) so memory stays user-auditable. (Sources: letta.com/blog/sleep-time-compute, docs.letta.com/guides/agents/architectures/sleeptime, arxiv.org/abs/2502.12110)
- 2. Add an explicit forgetting-policy layer using the four levers — importance gating at write time, entity merge at write time, decay only on temporal claims, eviction only for compliance/user requests — with Graphiti-style supersede-don't-delete (invalid_at timestamps, provenance preserved). Daem0n has retrieval and contradiction detection; what it lacks is a declared policy for what survives. This directly delivers the 'actively forget' goal without silent data loss, the failure mode practitioners fear most. (Sources: hindsight.vectorize.io/blog/2026/05/21/agent-memory-consolidation, arxiv.org/abs/2501.13956)
- 3. Separate evidence from inference in the schema (Hindsight's four networks): world facts, agent experiences, entity summaries, and evolving beliefs, where Reflexion/dream outputs live in the beliefs network, cite supporting facts, and are revised traceably when evidence changes. This makes self-improvement auditable and prevents the agent's own inferences from laundering into 'facts' — a weakness of every extract-and-store system including the current stack. (Source: arxiv.org/abs/2512.12818)
- 4. Build the procedural layer nobody has shipped well: LangMem-style prompt optimizers + MemOS-style cross-task skill reuse, wired into the agentskills.io skill system — outcome tracking feeds a sleep-time optimizer that patches system-prompt sections and writes/edits skill files (with distilled few-shot exemplars from successful episodes), all as logged diffs. Every vendor admits procedural memory is 'early-stage'; this is where Hermes-Brain can lead the field rather than follow it. (Sources: langchain-ai.github.io/langmem/concepts/conceptual_guide/, github.com/MemTensor/MemOS, mem0.ai/blog/state-of-ai-agent-memory-2026)
- 5. Adopt Mem0-style multi-scope tagging (user_id/agent_id/platform/session composing at retrieval) plus Memobase-style buffer-and-flush batch extraction (idle/threshold-triggered) instead of per-message LLM extraction — the correct answer for cross-platform identity and for $5-VPS economics respectively. Keep retrieval LLM-free (Zep's latency lesson) and memory writes async. (Sources: arxiv.org/abs/2504.19413, github.com/memodb-io/memobase, arxiv.org/abs/2501.13956)
- 6. Wire outcome tracking into retrieval as feedback-weighted memory (Cognee's memify pattern): memories that contributed to successful outcomes gain retrieval weight, repeated failure demotes. It is the cheapest genuine continual-learning mechanism available — no training required — and Daem0n already collects the signal. (Source: cognee.ai/blog/fundamentals/how-cognee-builds-ai-memory)
- 7. Design tiered capability, MemOS-style: SQLite+FTS/BM25 floor for the $5 VPS (memori proves this tier is viable), ONNX embeddings + graph mid-tier, and optional KV-cache memory promotion (MemOS activation memory) on GPU boxes running local models. One MemCube-like memory-object envelope (content + provenance + lifecycle state + scope/permissions) across all tiers. (Sources: github.com/MemTensor/MemOS, infoq.com/news/2025/12/memori/)
- 8. Do not chase LoCoMo/LongMemEval scores or adopt a third-party memory runtime. Those benchmarks are saturated and configuration-gamed (the Mem0-Zep dispute swings 25 points); evaluate with LongMemEval-V2/BEAM plus staleness probes, and build the metric that matters and no one else measures: longitudinal task-success improvement from Hermes' own outcome logs. Strategically, remain fully self-hostable with zero cloud dependency — Zep CE's deprecation and supermemory's closed 'open source' show why, and the Nous maintainers' rejection of the Memvid RFC (hermes-agent#23874) confirms in-house is the intended path. (Sources: blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/, blog.getzep.com/announcing-a-new-direction-for-zeps-open-source-strategy/, github.com/NousResearch/hermes-agent/issues/23874)
