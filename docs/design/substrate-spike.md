# Substrate-transformation & scale escape hatches — research spike (D4)

**Status: SPIKE / NOT BUILT.** This is a go/no-go record, not a commitment. Every
item here is gated behind a hard capability check and a concrete trigger; none
ships in the default (stdlib floor / ONNX-full) tiers. Build only when the
stated trigger actually fires. Sources: `docs/research/research-frameworks.md`
(MemOS), `docs/research/research-infra.md`, `docs/design/memory-engine.md §1.6`.

## Why a spike, not a build

The brain's design center is a single portable SQLite file that degrades
gracefully from a full ONNX tier down to an FTS-only floor (Termux/512 MB). The
techniques below either (a) require a GPU box, or (b) only pay off past a corpus
scale a single user will not reach for a long time. Building them now would add
heavy optional deps and a second storage substrate for no present benefit, and
would dilute the "one file, degrades everywhere" invariant. So we record the
decision and the trigger, and move on.

## Tier 1 — parametric / activation memory (GPU-box only)

**MemOS-style substrate transformation** (research-frameworks): promote hot
plaintext memory to **KV-cache (activation) memory**, and stable procedures to
**parametric (LoRA) memory**; a scheduler prefetches the "next scene."

- **Verdict: NO-GO for v1.** Requires model-weight/KV access — impossible through
  the host's `auxiliary_client.call_llm` gateway (the brain never sees weights),
  and meaningless on a cloud API profile.
- **Trigger to revisit:** a dedicated local-GPU Hermes profile exists AND the
  brain is allowed a persistent in-process model handle. At that point a `lora`
  memory tier could compile the top-N most-used `strategy`/`guardrail` items into
  an adapter, and a `kv` tier could pin the lane-1 block as a warm KV prefix.
- **Guardrail:** this must be an *optional profile*, never a default; the
  supersede-don't-delete semantic model (schema.sql) stays the source of truth,
  with parametric memory as a derived, rebuildable cache — never authoritative.

## Tier 2 — vector scale escape hatches (corpus-size triggers)

Current v1: brute-force int8 scan over `mem_vec` (256-d, symmetric int8). Fine to
~100k rows on a laptop. Documented escalation, in order:

| Trigger | Move | Notes |
|---|---|---|
| corpus > ~500k rows AND sqlite-vec DiskANN out of alpha | sqlite-vec **DiskANN** ANN index | stay in the one file; no new dep beyond sqlite-vec |
| corpus > ~1M rows | **LanceDB** (disk-native IVF-PQ) as a sidecar vector store | breaks the single-file invariant — only for a server profile |
| phone storage becomes the binding constraint | **LEANN** (recompute-don't-store embeddings) | trades CPU for disk on the floor tier |
| in-Python ANN wanted without a service | **usearch** (mmap HNSW) | optional, single-process |

**Verdict: NO-GO now** — a single user's store is orders of magnitude below the
500k trigger. The int8 brute-force scan is correct and fast at real sizes.

## Tier 3 — retrieval-quality upgrades that are cheap but deferred

- **Binary-quantization first pass + int8 rescore of top-200** (sqlite-vec
  supports both bit and int8). ~5-10x faster first-stage filtering. **Deferred:**
  only meaningful once the brute-force int8 scan is a measured bottleneck (see
  Tier 2 trigger); wiring it earlier is premature optimization.
- **Matryoshka 768-d kept for rerank-by-full-vector.** v1 stores 256-d only. A
  future high-tier could store 768-d and rerank the ColBERT top-K by full-vector
  cosine. **Deferred:** the ColBERT reranker (now built, A1) already covers the
  rerank stage; adding a second full-vector rerank is redundant until measured.
- **`jina-embeddings-v5-text-small`** as a GPU-tier embedder. Registered idea
  only; the ONNX `modernbert`/`embeddinggemma` tiers cover CPU.

## Cross-cutting: re-embed discipline is already in place

Any tier change that alters the embedding or rerank model is safe because the
`embedded_with` tag + the `hermes brain reindex` rebuild path already exist
(A1 extended the tagging to the reranker). A substrate upgrade re-tags and
rebuilds in the background; it never destroys the live index (the
embedder-identity guard in `store/vec.py`).

## Decision

Ship none of Tier 1–3 now. Keep this doc as the trigger table. The only
near-term item worth a real build is the **learned convex fusion weights**, and
that is already built as **D1** (`recall/fit_weights.py`, shadow-only proposals).
