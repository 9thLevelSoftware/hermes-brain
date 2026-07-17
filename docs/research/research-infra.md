# Research: research:infra

## Summary

Early-2026 SOTA for local-first memory storage/retrieval strongly favors a single-file SQLite architecture at Hermes-Brain's scale (~100k memories). Key facts: (1) 100k memories at matryoshka-256d float32 is only ~100MB (int8: ~26MB), so brute-force SIMD vector scan in sqlite-vec is fast enough that ANN indexes, vector servers, and disk-native formats are unnecessary complexity — the biggest honest finding is that most 'vector DB' machinery is overkill for a personal agent. (2) EmbeddingGemma-300M (QAT, <200MB RAM, MRL to 256d, #1 MTEB under 500M params) has displaced ModernBERT-era embedders as the quality-per-RAM leader; model2vec/potion static embeddings provide a ~30MB, 500x-faster fallback tier for Termux/$5-VPS. (3) Late-interaction rerankers got tiny: mxbai-edge-colbert-v0 (17M/32M params, beats ColBERTv2 on BEIR) and answerai-colbert-small make CPU reranking of top-50 practical. (4) RRF (k=60) over FTS5+vector remains the robust zero-tuning default; learned convex score combination beats it only when you have tuning data — Hermes's outcome tracking could supply that later. (5) Full GraphRAG-style community summarization is not worth it at this scale; the winning graph evolutions are HippoRAG 2 (Personalized PageRank over an entity graph — cheap, strong on associative/multi-hop memory) and LightRAG-style incremental dual-level graphs; Graphiti's bi-temporal edges largely duplicate what Daem0n-MCP already has. (6) LLMLingua-style compression conflicts with prompt caching (question-aware compression breaks cache prefixes); 2026 practice favors verbatim top-k retrieval injected at a stable position, with caching yielding 41-80% cost cuts in agentic loops — compress for archival storage, not for prompt injection.

## Findings

### Small embedding models by quality-per-RAM (2025-2026)

The current generation of CPU-deployable text embedders. EmbeddingGemma-300M (Google, Sep 2025) is the standout: 308M params, quantization-aware trained, runs in <200MB RAM at int8/int4 with near-lossless quality, Matryoshka truncation to 512/256/128d, #1 on MTEB multilingual/English/code among sub-500M models. Alternatives: nomic-embed-text v1.5 (137M, 274MB, 8192-token context, Apache-2.0, proven ONNX path) and Nomic Embed v2 (MoE, first MoE embedder); Qwen3-Embedding-0.6B (stronger but ~639MB — eats half the RAM budget); snowflake-arctic-embed-xs (33M params, 384d, surprisingly strong for size); bge-m3 (568M — good multilingual but too heavy for a 1-2GB VPS); jina-embeddings-v5-text-small (Feb 2026, 677M, best sub-1B multilingual at 71.7 MTEB-en, robust under binary quantization — an option for GPU-box tier, not the VPS tier). Caveat from a Mar-2026 independent benchmark: sub-335M models degrade badly on long documents (0.40-0.44 accuracy at 8K chars) — chunk memories short.

**Key techniques:**
- Matryoshka Representation Learning: store 256d, keep 768d optional for rerank-by-full-vector
- Quantization-aware training (EmbeddingGemma): int8/int4 ONNX with near-lossless quality
- ONNX Runtime CPU inference with int8 dynamic quantization as the standard serving path
- Instruction-aware embeddings (Qwen3, EmbeddingGemma prompts: 'search result' vs 'question answering' task prefixes)
- Binary quantization tolerance as a 2026 model design goal (jina-v5)

**Evidence:** Google claims #1 MTEB under 500M and <200MB RAM quantized (self-reported but corroborated by HuggingFace's independent blog and MTEB leaderboard placement). Independent Mar-2026 benchmark (Cheney Zhang, 10 models) confirms small models win on quality-per-resource but flags long-document degradation. Ollama size figures: embeddinggemma 622MB unquantized vs nomic-embed 274MB — quantization is what makes EmbeddingGemma fit.

**Applicability:** Direct upgrade path from Daem0n-MCP's ModernBERT-256d: EmbeddingGemma-300M at 256d matryoshka keeps the existing dimension (no reindex of schema, just re-embed), fits the <200MB RAM envelope alongside a Python process on a 1-2GB VPS, and is multilingual (Telegram/Discord users are not English-only). Keep task-prefix prompts per memory type (episodic vs skill vs fact).

**Sources:**
- https://developers.googleblog.com/en/introducing-embeddinggemma/
- https://huggingface.co/blog/embeddinggemma
- https://arxiv.org/pdf/2509.20354
- https://huggingface.co/google/embeddinggemma-300m
- https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models
- https://zc277584121.github.io/rag/2026/03/20/embedding-models-benchmark-2026.html
- https://huggingface.co/Snowflake/snowflake-arctic-embed-xs
- https://huggingface.co/jinaai/jina-embeddings-v5-text-small
- https://www.morphllm.com/ollama-embedding-models

### Static embeddings tier: model2vec / potion-retrieval-32M

Distilled static (non-contextual) embeddings from MinishLab. potion-retrieval-32M (Jan 2025) is the best static retrieval model: ~30MB, CPU-only, ~25,000 sentences/sec (roughly 500x faster than MiniLM), within ~8% of MiniLM's MTEB average; 8MB/4MB variants retain 80-90% of that.

**Key techniques:**
- Token-level distillation from a transformer into a static lookup table
- Zero-inference-cost embedding (lookup + mean pool) — no ONNX runtime needed
- Usable as a coarse first-stage retriever in a two-stage cascade

**Evidence:** MTEB numbers self-reported by MinishLab but independently reproduced in community benchmarks and HN discussion; the 500x speedup claim is consistent across multiple independent write-ups.

**Applicability:** The Termux/$5-VPS floor for Hermes: when EmbeddingGemma's 200MB is too much (Android phone, 512MB VPS), potion-retrieval-32M keeps semantic search alive at ~30MB with graceful quality loss. Also ideal for embedding at write-time on weak hardware with periodic re-embedding by a stronger model during idle 'dreaming'.

**Sources:**
- https://github.com/MinishLab/model2vec
- https://medium.com/kx-systems/model2vec-making-large-scale-embedding-generation-manageable-8cd55b7a288f
- https://news.ycombinator.com/item?id=44023281

### CPU-viable rerankers: edge ColBERT models and small cross-encoders

Late-interaction rerankers shrank dramatically in late 2025. mxbai-edge-colbert-v0 (Mixedbread + answer.ai lineage, Oct 2025) ships 17M and 32M param variants that outperform ColBERTv2 on BEIR and are explicitly designed for local/edge inference. answerai-colbert-small-v1 (33M) remains the proven baseline. Cross-encoder route: bge-reranker-v2-m3 is the self-hosted quality default but is 568M (slow on CPU); mxbai-rerank-v2 (Qwen-based) is open-source SOTA but GPU-class. The AnswerDotAI 'rerankers' library gives a unified API over all of these including PyLate late-interaction models.

**Key techniques:**
- Late interaction (MaxSim over token vectors): score top-50 candidates in ~tens of ms on CPU, ~23ms p50 reported for ColBERT-class scoring
- Two-stage pattern: RRF fuse to top-100, rerank top-25-50, inject top-5-8
- ONNX int8 export of 17-33M rerankers keeps the whole rerank stage under ~100MB RAM
- Late-interaction models can double as retrievers, but at this corpus scale using them rerank-only avoids the large token-level index

**Evidence:** mxbai-edge-colbert benchmarks are self-reported in the tech report (arXiv 2510.14880) but the answer.ai ColBERT-small lineage has extensive independent validation on BEIR. Cross-encoder latency comparisons (Jina v2 ~15x faster than bge-reranker-v2-m3) from vendor blogs — treat exact multipliers skeptically, direction is well corroborated.

**Applicability:** Daem0n-MCP has no reranker — this is the single highest-leverage retrieval-quality upgrade for Hermes-Brain. A 17-32M ColBERT reranking RRF's top-50 memories costs almost nothing on CPU and substantially improves precision of what enters the context window. Make it optional/degradable: skip reranking on Termux-class hardware.

**Sources:**
- https://arxiv.org/pdf/2510.14880
- https://github.com/AnswerDotAI/rerankers
- https://futureagi.com/blog/best-rerankers-for-rag-2026/
- https://localaimaster.com/blog/reranking-cross-encoders-guide

### Embedded vector stores: honest comparison at 100k-memory scale

sqlite-vec: stable v0.1.9 (Mar 2026) is still brute-force KNN, but with metadata columns/partitioning/aux columns (since Nov 2024) and int8/binary quantization; ANN (rescore, IVF, DiskANN) is in active alpha (v0.1.10-alpha.4, May 2026) — promising but not production-stable. Runs everywhere including Android/aarch64 (Termux) and Windows; pure C, zero dependencies, ~few-hundred-KB extension. LanceDB: disk-native IVF-PQ on Arrow/Lance format, can index beyond-RAM datasets, good Python API, but heavier dependency tree (Rust wheels ~50MB+), and IVF-PQ is unnecessary below ~1M vectors. Qdrant embedded/local mode: officially recommended only up to ~20k points, brute-force, single-process, SQLite-backed persistence — NOT a real embedded Qdrant; the server is excellent but that's a separate process with ~100MB+ baseline RAM. DuckDB-VSS: HNSW persistence still experimental (WAL recovery not implemented — documented data-loss risk), non-incremental index serialization; fine for analytics, wrong for an always-on memory store. Chroma 1.0+ (2025 Rust rewrite): 4x faster, legitimate single-node option, but heavier than sqlite-vec and its sweet spot is doc-RAG not agent memory. usearch/hnswlib: excellent raw HNSW libraries (usearch: SIMD, quantized f16/i8, memory-mapped, ~10MB dep) if you want in-process ANN without SQL; vectorlite wraps hnswlib as a SQLite extension. LEANN (Berkeley, MLSys 2026): graph-based selective recomputation gives 97% storage savings by NOT storing embeddings — clever for 60M-chunk laptop corpora, but recomputation needs a fast local embedder and adds latency; overkill at 100k memories where embeddings cost ~26-100MB anyway.

**Key techniques:**
- Do the math first: 100k x 256d f32 = ~100MB; int8 = ~26MB; binary = ~3.2MB — brute-force SIMD scan of 100k int8 vectors is single-digit-to-tens of ms on any CPU
- sqlite-vec metadata columns + partition keys for pre-filtering (user_id, platform, memory_type) before the scan
- Binary quantization first-pass + int8/f32 rescore of top-200 (sqlite-vec supports both representations)
- usearch memory-mapped index as the >1M-vector escape hatch without changing the SQLite source-of-truth
- Avoid: DuckDB-VSS persistence (WAL bug), Qdrant local mode above 20k points, running a vector server on a 1-2GB VPS

**Evidence:** sqlite-vec release cadence verified directly on GitHub releases (v0.1.9 Mar 2026, alphas through May 2026). Qdrant 20k-point local-mode guidance is from Qdrant's own client docs. DuckDB-VSS limitations are from DuckDB's official docs (experimental persistence flag, WAL recovery not implemented). LEANN numbers (97% savings, 50x index reduction) are self-reported but peer-reviewed (MLSys 2026 / ICML workshop).

**Applicability:** Hermes-Brain's source of truth should be one SQLite file: memories table + FTS5 index + sqlite-vec virtual table + graph edge tables, all transactionally consistent (a memory write, its FTS entry, its vector, and its graph edges commit atomically — none of the multi-store sync bugs that plague Chroma+Neo4j+Redis stacks). Single file also makes cross-platform sync (VPS <-> laptop) trivial via file copy/litestream. sqlite-vec's Android build matters for Termux support.

**Sources:**
- https://github.com/asg017/sqlite-vec/releases
- https://github.com/asg017/sqlite-vec
- https://www.marktechpost.com/2024/11/25/sqlite-vec-update-introduces-metadata-columns-partitioning-and-auxiliary-features-for-enhanced-data-retrieval-transforming-vector-search/
- https://deepwiki.com/qdrant/qdrant-client/2.2-local-mode
- https://duckdb.org/docs/current/core_extensions/vss
- https://www.lancedb.com/lp/vs-qdrant
- https://github.com/unum-cloud/usearch
- https://github.com/1yefuwang1/vectorlite
- https://www.trychroma.com/project/1.0.0
- https://github.com/StarTrail-org/LEANN
- https://arxiv.org/abs/2506.08276
- https://shaharia.com/blog/choosing-embeddable-vector-database-go-application/

### Hybrid search and fusion: RRF default, learned fusion as an upgrade

The 200-line pattern — SQLite FTS5 (BM25) + sqlite-vec, two parallel queries, RRF with k=60 — is now well-documented community practice and matches what Daem0n-MCP does. The research nuance (Bruch et al., ACM TOIS 'An Analysis of Fusion Functions for Hybrid Retrieval'): a tuned convex combination of min-max-normalized scores beats RRF in-domain AND out-of-domain and is sample-efficient (one parameter, small tuning set); RRF's rank-only approach is robust zero-shot but non-smooth and ignores score magnitudes. Elastic now ships weighted RRF as a middle ground. FTS5 tricks that matter: porter tokenizer + separate trigram-tokenized index for fuzzy/typo matching, contentless (content='') tables to halve storage against an existing memories table, prefix indexes for autocomplete-style recall, bm25() column weights (title/tags boosted over body), and unicode61 remove_diacritics=2 for multilingual chat text.

**Key techniques:**
- RRF k=60 as zero-config default; weighted RRF (per-source weights) when vector and BM25 quality diverge
- Learned convex combination (alpha * norm_vec + (1-alpha) * norm_bm25) tuned on logged outcomes once enough relevance signal accumulates
- FTS5 contentless tables + external content synced by triggers to the memories table
- Trigram tokenizer side-index for typo-tolerant recall of names/handles (Telegram usernames, etc.)
- RRF as first stage only — feed top-100 into the ColBERT reranker

**Evidence:** The convex-combination-beats-RRF result is peer-reviewed (ACM TOIS 2023) and consistent with 2025-2026 practitioner reports; RRF's robustness without tuning is equally well established. No vendor hype involved.

**Applicability:** Hermes-Brain already has BM25+vector+RRF from Daem0n-MCP; the beyond-that move is closing the loop: use the existing outcome-tracking subsystem to log which retrieved memories were actually useful, then fit the single convex-combination weight (and per-memory-type boosts) from that data — a genuinely self-tuning retriever, cheap to implement, aligned with the continual-learning goal.

**Sources:**
- https://dl.acm.org/doi/10.1145/3596512
- https://media.patentllm.org/blog/database/hybrid-rag-200-lines
- https://www.elastic.co/search-labs/blog/weighted-reciprocal-rank-fusion-rrf
- https://glaforge.dev/posts/2026/02/10/advanced-rag-understanding-reciprocal-rank-fusion-in-hybrid-search/
- https://avchauzov.github.io/blog/2025/hybrid-retrieval-rrf-rank-fusion/
- https://ceaksan.com/en/hybrid-search-fts5-vector-rrf

### GraphRAG evolutions: HippoRAG 2 and LightRAG worth it, full GraphRAG and Graphiti mostly not

HippoRAG 2 (ICML 2025, 'From RAG to Memory'): builds an entity/triple graph plus dense passage nodes, retrieves via Personalized PageRank seeded by query entities — framed explicitly as non-parametric continual learning. Beats SOTA embedding retrieval by ~7% on associative-memory tasks while staying strong on factual QA; the original HippoRAG line claims 10-30x cheaper multi-hop than GraphRAG-style approaches because there is NO community summarization LLM pass. LightRAG (EMNLP 2025): dual-level (entity + theme keyword) graph, incremental set-merge updates (no global rebuild), reported <100 tokens per retrieval vs ~610k for GraphRAG global search, ~$0.15 vs $4-7 per doc-set indexing. LazyGraphRAG (Microsoft): defers all LLM work to query time — conceptually right for a personal agent, but code was never released standalone, so it's a design pattern to borrow, not a dependency. Graphiti/Zep (arXiv 2501.13956): temporal KG with validity intervals on every edge, hybrid semantic+BM25+graph retrieval, sub-200ms — excellent engineering, but it wants a graph DB backend (Neo4j/FalkorDB) and its headline feature (bi-temporal edges, contradiction handling) is something Daem0n-MCP already built. Survey evidence (arXiv 2506.05690): graph value scales with query complexity; for simple lookups graphs add cost without gains.

**Key techniques:**
- Personalized PageRank over an entity graph as the retrieval operator (HippoRAG 2) — runs in milliseconds on 100k-node graphs with scipy/networkx sparse matrices, no LLM at query time
- Query-time-only LLM usage (LazyGraphRAG pattern): never pre-summarize communities; summarize retrieved subgraphs on demand
- Incremental set-merge graph updates on ingest (LightRAG) instead of periodic full rebuilds
- Dense passage nodes linked into the entity graph (HippoRAG 2's 'deeper passage integration') so PPR can land on memories, not just entities
- Edge validity intervals for temporal facts (Graphiti pattern — already present in Daem0n-MCP)

**Evidence:** HippoRAG 2 is peer-reviewed (ICML 2025) with public code; its 7% associativity gain is on standard benchmarks (MuSiQue, 2Wiki, HotpotQA). LightRAG's 6000x token claim is self-reported and compares against GraphRAG's worst-case global search — treat the magnitude skeptically, but the architectural cost difference (no community summarization) is structural and real. Independent survey (arXiv 2506.05690) confirms graphs only pay off on complex/multi-hop queries. LazyGraphRAG results (0.1% of GraphRAG indexing cost) are Microsoft-blog-only, never independently reproduced since code wasn't released.

**Applicability:** Daem0n-MCP's GraphRAG+Leiden+bi-temporal stack is the expensive 2024 recipe. The 2025-2026 move: keep the graph and bi-temporal edges, DROP eager community summarization, and replace community-based retrieval with HippoRAG-2-style PPR seeded from query entities, fused (RRF) with the hybrid dense/BM25 results. At 100k memories the entity graph is maybe 20-50k nodes — PPR is essentially free, needs no graph database (store edges in SQLite, load sparse adjacency into memory), and directly serves associative recall ('what do I know connected to X'), which is the actual value of a graph in a personal agent.

**Sources:**
- https://arxiv.org/abs/2502.14802
- https://proceedings.mlr.press/v267/gutierrez25a.html
- https://github.com/hkuds/lightrag
- https://arxiv.org/html/2410.05779v1
- https://arxiv.org/html/2506.05690v3
- https://arxiv.org/abs/2501.13956
- https://github.com/getzep/graphiti
- https://github.com/microsoft/graphrag/discussions/1490
- https://medium.com/graph-praxis/graphrag-vs-hipporag-vs-pathrag-vs-og-rag-choosing-the-right-architecture-for-your-knowledge-graph-a4745e8b125f

### Context-budget engineering: prompt caching beats prompt compression in 2026

The economics flipped. Provider prompt caching (Anthropic, OpenAI, and OpenAI-compatible local servers) gives 41-80% cost reduction and 13-31% TTFT improvement in long-horizon agentic tasks when the prefix is kept stable ('Don't Break the Cache', arXiv 2601.06007, 500+ agent sessions). LLMLingua/LongLLMLingua-style compression still delivers real savings on one-shot long-context RAG (production 4-10x compression; vendor case studies claim up to 95% cost cuts — self-reported), but question-aware compression re-runs per query and produces different text each time, which destroys cache prefixes — the two techniques are in direct tension for a multi-turn agent. Also, LLMLingua needs a small LM in RAM (even the BERT-class llmlingua-2 is ~500MB+), a real cost on a 1-2GB box. 2026 practitioner consensus for agent memory: retrieve fewer, better memories verbatim (reranker does the budget control), inject them at a stable structural position late in the prompt, and never put timestamps/request IDs in the cached prefix.

**Key techniques:**
- Stable-prefix layout: system prompt + tools + rules (cached) -> conversation -> retrieved memories in a clearly delimited block positioned after the cache breakpoint
- Dynamic content at the END of system content, never interspersed (arXiv 2601.06007 finding #1)
- Exclude dynamic tool results from cached sections; cache-control on the last tool caches all tool definitions as prefix (Anthropic)
- Compression at WRITE time, not read time: LLM-summarize/consolidate old memories during idle 'dreaming' so stored memories are already dense — keeps retrieval verbatim and cache-friendly
- Token budgeting via retrieval depth (top-5-8 post-rerank) instead of post-hoc compression

**Evidence:** Cache-break costs quantified in a controlled study (arXiv 2601.06007: 41-80% cost, 13-31% TTFT, 500+ sessions, 3-50 tool calls). LLMLingua's 95%-savings case study is vendor-marketing (tokenmix.ai) — the underlying LongLLMLingua paper's 21.4% quality-boost-with-4x-compression is peer-reviewed but predates cheap caching. The caching-vs-compression tension is explicitly documented (question-aware compression prevents context caching).

**Applicability:** Daem0n-MCP's LLMLingua stage should be repositioned, not deleted: move compression to the consolidation/dreaming pipeline (compress-once-store-forever) and drop per-query compression. Hermes talks to many providers — Anthropic/OpenAI/DeepSeek all now bill cached tokens at 10-25% of base rate, so a cache-stable memory-injection format is a cross-provider win, and on local llama.cpp/vLLM boxes the same layout wins via KV-cache prefix reuse.

**Sources:**
- https://arxiv.org/pdf/2601.06007
- https://arxiv.org/pdf/2310.06839
- https://tokenmix.ai/blog/llmlingua-prompt-compression-2026
- https://memu.pro/blog/anthropic-claude-prompt-caching-memory
- https://promptbuilder.cc/blog/prompt-caching-token-economics-2025

## Top Recommendations

- RECOMMENDED STACK (primary, 1-2GB VPS, no GPU): ONE SQLite file as source of truth. Embeddings: EmbeddingGemma-300M via ONNX Runtime int8 (<200MB RAM), matryoshka-truncated to 256d — drop-in for the existing ModernBERT 256d pipeline. Vectors: sqlite-vec stable (v0.1.9) brute-force with int8 storage + metadata/partition columns for user/platform/type pre-filtering; at 100k memories a full scan is ~26MB and milliseconds — do NOT add ANN complexity yet, adopt sqlite-vec's DiskANN only when it exits alpha AND corpus passes ~500k. FTS: FTS5 contentless table (porter + a trigram side-index for names/typos, bm25() column weights). Fusion: RRF k=60 initially. Reranker: mxbai-edge-colbert-v0-32M (fallback: answerai-colbert-small-v1) ONNX int8 over RRF top-50. Graph layer: keep the bi-temporal edge tables in the same SQLite, run HippoRAG-2-style Personalized PageRank (scipy sparse) seeded by query entities, RRF-fused with hybrid results.
- Kill eager GraphRAG summarization: drop Leiden community summarization from the hot path (LazyGraphRAG's core insight — defer all LLM work to query time) and replace community retrieval with PPR over the existing entity graph. At personal-agent scale this removes the largest recurring LLM cost in the predecessor while IMPROVING associative recall (HippoRAG 2's peer-reviewed +7% on associativity).
- Reposition LLMLingua from read path to write path: per-query compression breaks provider prompt caches (41-80% cost savings at stake per arXiv 2601.06007) and holds a 500MB model in RAM. Compress/consolidate during idle dreaming instead (compress once, store forever); at query time inject top-5-8 reranked memories VERBATIM in a stable-position block after the cached prefix.
- Add the reranker — it is the single biggest retrieval-quality gap vs the predecessor. A 32M-param late-interaction model reranking 50 candidates is ~tens of ms on CPU and directly controls the context budget. Make it a degradable tier: full stack on VPS/GPU box; on Termux/512MB fall back to potion-retrieval-32M static embeddings (~30MB) + FTS5 + RRF with no reranker — same SQLite schema, swappable encoder.
- Make fusion self-tuning using outcome tracking Hermes already has: start with RRF (zero-config, robust), log which injected memories the agent actually used/credited, then fit a single convex-combination weight over min-max-normalized BM25/vector/PPR scores (ACM TOIS result: tuned convex combination beats RRF in- and out-of-domain, sample-efficient). This turns retrieval itself into a continually-learning component — squarely on-mission for Hermes-Brain.
- Fallback/escape hatches to document in the design: LanceDB if the corpus ever exceeds ~1M vectors or needs beyond-RAM indexing (disk-native IVF-PQ); usearch memory-mapped HNSW if brute-force latency ever bites without wanting to leave Python; LEANN's recompute-don't-store trick only if storage on phones becomes the binding constraint. Explicitly avoid: Qdrant local mode (brute force, ~20k-point guidance, single-process), DuckDB-VSS for persistence (experimental, WAL recovery unimplemented — documented corruption risk for an always-on memory store), and any separate vector/graph server process on the 1-2GB tier.
