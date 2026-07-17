# Research: research:academic

## Summary

The 2025-2026 academic frontier splits cleanly into two tracks. Track 1 (weight-level test-time learning — Titans/ATLAS/Nested Learning-HOPE, TTT layers, SEAL self-edits, FOREVER replay) is where the deepest results are, but every one requires weight access and is therefore out of scope for a Python memory server orchestrating frontier APIs; its value to Hermes-Brain is as design principles (surprise-gated writing, momentum, multi-timescale continuum memory, adaptive forgetting gates). Track 2 (memory-as-context continual learning) has converged on a striking consensus: a frozen LLM plus a well-engineered external memory IS a continual learner (HippoRAG 2 'From RAG to Memory', Memento's memory-augmented MDP, ReasoningBank, ACE), and everything in this track is implementable today. The strongest 2025-2026 signals: (1) strategy-level memory distilled from both successes and failures beats raw-trace and success-only memory (ReasoningBank, +34% relative); (2) monolithic memory rewriting causes 'context collapse' — itemized delta updates are mandatory (ACE), independently corroborated by 2026 evidence that self-evolving agents degrade without capability-preservation checks; (3) idle-time compute should be anticipatory, not just reflective (sleep-time compute: 5x token savings, +13-18% accuracy); (4) forgetting decisions must be made at consolidation time with learned multi-factor value scoring — query-similarity is provably the wrong signal, and recency-only retains less than half of what learned weighting keeps; (5) procedural memory with a build/retrieve/update/delete lifecycle (Memp, AWM, Voyager lineage) transfers across models — a perfect fit for Hermes's agentskills.io skill system; (6) constrained Darwin-Gödel-style propose-validate-archive loops over the agent's own skills/prompts (not weights) are now peer-reviewed (ICLR 2026) and practical if gated by statistical acceptance tests and a fixed regression suite. Daem0n's existing stack (bi-temporal graph + contradiction detection, hybrid RRF retrieval, Reflexion, dreaming) is already at or near the published production state of the art for the Storage/Reflection stages; the frontier gap is the 'Experience' stage — cross-trajectory abstraction into strategies, cases, and skills, plus principled value-based forgetting. Vendor benchmark numbers (Mem0/Zep/MemOS on LoCoMo) are actively disputed and should not drive design decisions; use LongMemEval or internal replay evals instead.

## Findings

### ReasoningBank + Memory-aware Test-Time Scaling (Google Cloud AI, Sep 2025)

A strategy-level agent memory framework that distills generalizable reasoning strategies from BOTH self-judged successes and failures (not raw traces, not success-only workflows). At test time the agent retrieves relevant strategy memories, acts, then integrates new distilled learnings back. MaTTS (memory-aware test-time scaling) runs parallel/sequential rollouts whose contrastive signals produce higher-quality memories, creating a memory<->compute synergy.

**Key techniques:**
- Self-judgment of trajectory outcome (no ground-truth labels needed)
- LLM distillation of trajectories into titled strategy items (title + description + actionable insight)
- Failure-derived 'guardrail' memories alongside success memories
- Retrieval of strategy memories into system prompt at task start
- Contrastive memory synthesis across multiple rollouts of the same task

**Evidence:** Up to 34.2% relative task-success improvement over no-memory and beats raw-trace and success-only-workflow memory baselines on WebArena, Mind2Web, and SWE-Bench-Verified. Google-authored, ICLR-submitted; numbers are self-reported but the ablation vs. other memory designs is the useful signal.

**Applicability:** Directly implementable as a Python pipeline over API calls: after each Hermes task, a judge call labels success/failure, a distiller call writes strategy items into the memory store, and retrieval injects top-k strategies at planning time. This is the cleanest published upgrade path over Daem0n's Reflexion loop — it converts Reflexion's episodic self-critiques into a persistent, cross-task, retrievable strategy bank.

**Sources:**
- https://arxiv.org/abs/2509.25140
- https://arxiv.org/html/2509.25140v1
- https://www.marktechpost.com/2025/10/01/google-ai-proposes-reasoningbank-a-strategy-level-i-agent-memory-framework-that-makes-llm-agents-self-evolve-at-test-time/

### ACE — Agentic Context Engineering (Stanford/SambaNova, Oct 2025)

Treats the agent's context/memory as an evolving 'playbook' maintained by a Generator/Reflector/Curator loop. Its key contribution is diagnosing 'context collapse': iteratively re-writing a memory document monolithically degrades and shortens it over time. ACE instead applies structured, itemized DELTA updates (add/edit/deprecate individual bullets with IDs and helpful/harmful counters), preserving detail indefinitely.

**Key techniques:**
- Delta (incremental) context updates instead of monolithic rewrites
- Itemized playbook entries with usage counters (helpful/harmful votes)
- Separate Reflector (extracts lessons from execution feedback) and Curator (merges deltas) roles
- Offline mode (system-prompt optimization) and online mode (agent memory)
- Learning from natural execution feedback, no labels

**Evidence:** +10.6% on agent benchmarks (AppWorld) and +8.6% on finance reasoning vs strong baselines (GEPA, Dynamic Cheatsheet); matched top production agents on the AppWorld leaderboard with a smaller open-source model; 86.9% lower adaptation latency. Self-reported but replicated interest across the community.

**Applicability:** Pure prompt/context method — zero weight access needed. For Hermes-Brain this is the maintenance discipline for every LLM-curated memory artifact: rule files, strategy banks, user profiles, skill descriptions. Never let the dreaming loop 'rewrite the whole notes file'; make it emit typed deltas against itemized entries. Cheap, high leverage, prevents the silent memory-erosion failure mode Daem0n is exposed to.

**Sources:**
- https://arxiv.org/abs/2510.04618
- https://arxiv.org/pdf/2510.04618

### Sleep-time compute (Letta / UC Berkeley, Apr 2025)

Formalizes 'thinking while idle': a model pre-processes persistent context offline, anticipating likely future queries and pre-computing useful inferences before any query arrives, storing results as 'learned context'. Letta ships this as sleep-time agents that share memory blocks with the primary agent and rewrite/compress memory in the background.

**Key techniques:**
- Anticipatory inference: generate likely questions about stored context and pre-answer them
- Rewriting raw context into distilled 'learned context' consumed at test time
- Amortization across related queries on the same context
- Shared memory blocks between a foreground agent and a background consolidation agent

**Evidence:** ~5x reduction in test-time tokens at matched accuracy on Stateful GSM-Symbolic/AIME; +13-18% accuracy when scaling sleep-time compute; 2.5x cost reduction amortized over multi-query settings; effectiveness correlates with query predictability. Production implementation exists in Letta 0.7+.

**Applicability:** Directly implementable and a natural superset of Daem0n's 'dreaming'. Upgrade dreaming from 're-evaluate failed decisions' to a full consolidation shift: summarize the day's episodes, pre-compute answers to predicted user questions (Hermes has highly predictable per-user patterns across Telegram/Discord), refresh user model, distill strategies, decay/promote memories. Best on cheap off-peak API calls; degrades gracefully on a $5 VPS by doing less per cycle.

**Sources:**
- https://arxiv.org/abs/2504.13171
- https://www.letta.com/blog/sleep-time-compute/

### Memp + Agent Workflow Memory + Voyager lineage — procedural memory / skill distillation

Memp (Zhejiang/Alibaba, 2508.06433) builds lifelong procedural memory: trajectories are distilled into both fine-grained step instructions and higher-level script-like procedures, with an explicit build/retrieve/update lifecycle where entries are added, revised, or DELETED based on execution feedback; memories transfer across models. AWM (ICML 2025) induces reusable workflows from past trajectories and selectively injects them. Voyager remains the archetype: verified skills stored as runnable code indexed by NL description. 2026 follow-ups (AutoSkill 2603.01145, SkillOpt 2605.23904, 'From Raw Experience to Skill Consumption' 2605.23899) systematize model-generated skill lifecycles.

**Key techniques:**
- Trajectory -> procedure distillation at two abstraction levels
- Procedural memory update ops: add/modify/delete driven by downstream success
- Workflow induction from recurring successful sub-sequences
- Executable skills as code with NL index; composition of skills
- Transfer of procedural memory from strong to weak models

**Evidence:** Memp shows improved success and efficiency on TravelPlanner/ALFWorld and demonstrates stronger-model-built memory boosting weaker models. AWM published at ICML 2025 with large gains on Mind2Web/WebArena. Voyager's skill library is the most replicated pattern in the field.

**Applicability:** Highest-affinity finding for Hermes: it already has an agentskills.io-compatible skill system. Implement a skill-forge pipeline: detect recurring successful trajectories -> LLM distills into a draft skill (markdown + optional script) -> sandbox-verify -> register -> track per-skill outcome stats -> auto-revise or retire failing skills. This is genuine procedural learning with zero weight access, and skills are portable across the model-agnostic providers Hermes supports.

**Sources:**
- https://arxiv.org/html/2508.06433v2
- https://proceedings.mlr.press/v267/wang25bx.html
- https://openreview.net/forum?id=NTAhi2JEEE
- https://arxiv.org/pdf/2603.01145
- https://arxiv.org/pdf/2605.23899

### Memento — case-based reasoning without fine-tuning (Aug 2025)

'Fine-tuning agents without fine-tuning LLMs': formalizes agent improvement as a Memory-augmented MDP where an episodic case bank of (state, plan, outcome) tuples is read/written online; a case-selection policy (non-parametric or a small trainable retriever) picks past cases to condition the frozen planner LLM.

**Key techniques:**
- Case bank of full episodes with outcomes (successes AND failures)
- CBR-style retrieve-reuse-revise-retain loop wrapped around a frozen LLM
- Optional small neural case-selection policy trained online from reward (soft Q-learning) — trains a tiny retriever, not the LLM
- Memory rewriting from environmental feedback

**Evidence:** 87.88% Pass@3 GAIA validation (top-1 at the time), 79.40% test; 66.6% F1 on DeepResearcher beating training-based methods; ablations attribute 4.7-9.6% absolute gains on OOD tasks to case memory alone. Self-reported but GAIA leaderboard placement was public.

**Applicability:** Implementable today. The non-parametric variant is pure Python + embeddings: store every Hermes task episode with outcome, retrieve top-k similar cases (including failures) into the planning prompt. The learned case-selector is optional and trains a tiny model locally — feasible even on a VPS, and it gives Daem0n's outcome tracking a consumer: outcomes become training signal for retrieval, not just logs.

**Sources:**
- https://arxiv.org/abs/2508.16153
- https://arxiv.org/pdf/2508.16153

### A-MEM — agentic Zettelkasten memory (NeurIPS 2025)

Memory notes are structured atomic units (content, context description, keywords, tags) that the system agentically links to related notes on insertion; crucially, new memories trigger 'memory evolution' — the LLM updates the contextual descriptions and tags of EXISTING neighbor notes so the network's understanding refines continuously rather than only appending.

**Key techniques:**
- LLM-generated structured note construction on ingest
- Dynamic link generation to semantically/causally related notes
- Retroactive memory evolution: neighbors are re-described when new info arrives
- Retrieval over the evolved note graph

**Evidence:** NeurIPS 2025 poster; improvements over MemGPT/MemoryBank-class baselines on LoCoMo across six foundation models, especially multi-hop questions. Open source (agiresearch/A-mem).

**Applicability:** Implementable as an ingest-time enrichment stage on top of Daem0n's existing knowledge graph: on each new memory, one LLM call proposes links + updates to k-nearest existing notes. This complements bi-temporal contradiction detection nicely — memory evolution is the constructive counterpart (refine) to contradiction detection (invalidate). Cost: one extra LLM call per ingested memory; batch it into sleep-time.

**Sources:**
- https://arxiv.org/abs/2502.12110
- https://github.com/agiresearch/a-mem
- https://neurips.cc/virtual/2025/poster/119020

### HippoRAG 2 — non-parametric continual learning via hippocampal indexing (ICML 2025)

Successor to HippoRAG: an OpenIE-built KG where Personalized PageRank spreads activation from query-linked seed nodes, now with passage nodes integrated into the graph, query-to-triple matching, and LLM-based 'recognition memory' filtering of seed triples. Positions graph+PPR retrieval as the practical form of continual learning for frozen LLMs.

**Key techniques:**
- Personalized PageRank over an entity+passage graph (single-step multi-hop, no iterative LLM hops)
- Dense-sparse node integration (phrase nodes + full passages in one graph)
- Recognition-memory triple filtering to pick PPR seeds
- Online incremental graph updates as new documents arrive

**Evidence:** ICML 2025 (PMLR v267). ~7-point improvement over standard RAG on associative multi-hop tasks while, unlike most graph-RAG systems, NOT regressing on simple factual recall. Well-maintained OSU repo.

**Applicability:** Implementable and a direct comparison point to Daem0n's GraphRAG+Leiden. Leiden communities answer 'summarize themes'; PPR answers 'multi-hop association from this query' at millisecond scale with no LLM calls at query time — cheaper than community-report generation and better suited to a memory server on small hardware. Recommend adding PPR retrieval over the existing bi-temporal graph as a third retrieval arm in the RRF fusion.

**Sources:**
- https://arxiv.org/abs/2502.14802
- https://proceedings.mlr.press/v267/gutierrez25a.html
- https://github.com/osu-nlp-group/hipporag

### Consolidation-time forgetting: 7-factor value model, MemoryBank/Ebbinghaus, and rate-distortion compaction

The modern evolution of generative-agents recency×importance×relevance scoring. 'Learning What to Remember' (2606.12945) shows retention decisions must be made BEFORE future queries are known ('blind regime') and scores memories on seven cognitively grounded factors: emotional intensity, goal relevance, value alignment, self/user relevance, task utility, reliability, usage history — with learned weights. MemoryBank (AAAI 2024) pioneered Ebbinghaus exponential decay with reinforcement-on-recall; FOREVER (2601.03938) modernizes forgetting-curve replay; 'What to Keep, What to Forget' (2607.08032) gives a rate-distortion framing of memory compaction.

**Key techniques:**
- Multi-factor value scoring at consolidation time, not query time
- Learned factor weights (simple regression over retention outcomes — learned weights hit 0.770 gold-evidence retention vs 0.368 for recency-only)
- Ebbinghaus decay: strength decays exponentially, recall resets/strengthens it (spaced repetition for agents)
- Tiered demotion instead of deletion: full text -> summary -> tombstone
- Rate-distortion view: pick a compression budget, minimize expected loss on future queries

**Evidence:** 2606.12945 shows single-factor and recency baselines badly underperform learned multi-factor weighting; key negative result: query-similarity scoring is near-useless for forgetting decisions (it only measures retrieval). MemoryBank is widely replicated; newer papers are early-stage preprints — treat exact numbers cautiously.

**Applicability:** Fully implementable: this is arithmetic plus one LLM scoring call per memory at ingest/consolidation, and the weights can be fit locally from Hermes's own outcome-tracking data (did a retained memory ever get used?). This is the principled 'actively forget' engine the spec asks for — far better than TTL or recency pruning.

**Sources:**
- https://arxiv.org/abs/2606.12945
- https://arxiv.org/abs/2305.10250
- https://arxiv.org/html/2601.03938v1
- https://arxiv.org/html/2607.08032

### MemOS + MIRIX — memory-as-OS governance and typed memory partitions

MemOS (2507.03724) treats memory as a first-class OS resource: MemCubes encapsulate content + provenance/versioning/permissions metadata, with scheduling, lifecycle states, and migration between plaintext, activation (KV-cache), and parametric memory tiers. MIRIX (2507.07957) partitions agent memory into six types — Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault — each with its own Memory Manager agent plus a Meta Manager that routes updates/retrievals.

**Key techniques:**
- Uniform memory-unit envelope: provenance, version history, access permissions, lifecycle state
- Memory scheduling (what gets loaded/prefetched when)
- Typed memory partitions with type-specific update policies
- Meta-manager routing of writes to the right store
- KV-cache reuse of hot memory (activation tier)

**Evidence:** MemOS reports large LoCoMo gains (self-reported, benchmark contested territory) and 94% time-to-first-token reduction via KV reuse — the KV part needs local model serving. MIRIX shows gains on ScreenshotVQA and LOCOMO. Both are more valuable as architecture blueprints than as benchmark claims.

**Applicability:** The plaintext-tier governance is implementable now and is exactly what a cross-platform (Telegram/Discord/Slack/CLI) brain needs: every memory carries provenance (which platform/user/session), version chain, and lifecycle state; the typed six-partition scheme gives Hermes-Brain its schema (Daem0n already has episodic+semantic+rules; add procedural, resource, and a credentials-grade vault with strict access). Activation/parametric tiers require local weight/KV access — skip unless running the GPU-box profile with vLLM.

**Sources:**
- https://arxiv.org/abs/2507.03724
- https://arxiv.org/abs/2507.07957
- https://arxiv.org/html/2507.07957v1

### Titans -> ATLAS -> Nested Learning/HOPE — test-time memorization in weights (Google Research lineage)

Titans (2501.00663) adds a neural long-term memory module whose weights are updated during the forward pass via a gradient 'surprise' metric with momentum and an adaptive forgetting gate. Nested Learning (NeurIPS 2025, 2512.24695) generalizes this: a model is a set of nested optimization problems updating at different frequencies; the HOPE architecture adds a self-modifying Continuum Memory System with modules updating at multiple timescales. This is the 2026 frontier of 'memory as learning in weights'.

**Key techniques:**
- Surprise-gated memorization (memorize what violates predictions)
- Momentum on surprise (an event's aftermath stays memorable)
- Learned adaptive forgetting/decay gates
- Multi-timescale continuum memory (fast/slow updating modules)
- Self-referential memory optimization

**Evidence:** Peer-reviewed (Titans; Nested Learning at NeurIPS 2025); strong long-context results vs Transformers/modern RNNs at 2M+ context. Real, but these are architectures — no frontier API exposes them.

**Applicability:** NOT implementable without weight access — classify as watch-list. But the design principles port directly to the orchestration layer: (1) surprise-gated writing — only ingest memories that contradict or extend predictions (Hermes can score surprise via contradiction detection or prediction-error prompts); (2) momentum — temporarily boost write-priority for events following a high-surprise event; (3) multi-timescale stores — session cache (fast), episodic (medium), semantic/rules (slow) with different update frequencies, which is exactly the Continuum Memory System pattern in system form.

**Sources:**
- https://arxiv.org/abs/2501.00663
- https://arxiv.org/abs/2512.24695
- https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/
- https://openreview.net/forum?id=8GjSf9Rh7Z

### SEAL — Self-Adapting Language Models (MIT, NeurIPS 2025)

The model generates its own 'self-edits' — restructured training data plus optimization directives — and is finetuned on them, with an RL loop rewarding self-edits by downstream performance of the updated model. The headline demonstration of an LLM teaching itself via generated study material.

**Key techniques:**
- Self-edit generation (rewrite new info into study-sheet form)
- RL over self-edit policies with post-update performance as reward
- LoRA-based inner-loop updates for cheap adaptation

**Evidence:** ~+15% QA accuracy on knowledge incorporation, >50% boost on ARC-style skill learning vs GPT-4.1-generated data baselines; also documents catastrophic forgetting across sequential self-edits as an open problem. MIT, peer-reviewed venue.

**Applicability:** Requires weight access + finetuning infra — not implementable for Hermes-Brain's frontier-API mode (possible future option for a local-GPU profile with LoRA on an open model). The transferable half is the self-edit concept WITHOUT the gradient step: have Hermes rewrite incoming knowledge into optimal retrievable form (implications, restatements, QA pairs) before storage — i.e., SEAL's data-generation stage feeding the RAG store instead of the optimizer. That is implementable today and pairs with sleep-time compute.

**Sources:**
- https://arxiv.org/abs/2506.10943
- https://jyopari.github.io/posts/seal
- https://github.com/Continual-Intelligence/SEAL

### Darwin Gödel Machine / ADAS / AlphaEvolve — empirically validated self-modification loops

ADAS: a meta-agent programs better downstream agents in code, iterating against a benchmark. Darwin Gödel Machine (Sakana/UBC/Clune, ICLR 2026): the agent modifies ITS OWN code, keeps an archive of variants (open-ended evolution, not just hill-climbing), and accepts changes only when they empirically improve coding-benchmark scores. AlphaEvolve applies evolutionary program search with LLM mutations to real scientific/infra problems. PACE (2606.08106) adds anytime-valid statistical acceptance tests for self-evolving agents; a companion 2026 line documents safety risks of experience-driven self-evolution (2604.16968).

**Key techniques:**
- Propose-validate-archive loop: LLM proposes patch, benchmark suite gates acceptance
- Archive/population of variants to escape local optima
- Empirical validation replacing formal proof of improvement
- Statistical acceptance gates (PACE) to prevent regression-by-noise

**Evidence:** DGM: 20%->50% on SWE-bench, 14.2%->30.7% on Polyglot, autonomously discovered improvements like patch validation and better file viewing; peer-reviewed at ICLR 2026. AlphaEvolve produced verified math/infrastructure results at Google. These loops are compute-hungry: DGM-scale runs cost tens of thousands of dollars.

**Applicability:** Implementable in constrained form without weight access: apply the propose-validate-archive pattern to Hermes's OWN skills, prompts, and memory-policy parameters rather than its core code. During sleep-time, propose a revision to an underperforming skill/rule, replay it against archived task cases (Memento's case bank doubles as the eval set), accept only on statistically significant improvement (PACE), keep the archive. Must ship with guardrails — the 2026 safety literature shows evolving memory is an injection/poisoning surface.

**Sources:**
- https://arxiv.org/abs/2505.22954
- https://arxiv.org/html/2505.22954v2
- https://arxiv.org/pdf/2606.08106
- https://arxiv.org/pdf/2604.16968
- https://arxiv.org/html/2504.15228v2

### Zep/Graphiti temporal KG and the Mem0 benchmark wars — production evidence calibration

Zep's Graphiti (2501.13956): bi-temporal knowledge-graph engine (event time + ingestion time, edge invalidation on contradiction) — academic validation of the exact architecture Daem0n already has. Mem0's ECAI 2025 paper (2504.19413) ran the first ten-way memory-system comparison on LoCoMo; Zep published a rebuttal claiming misconfiguration (75.14% corrected vs 65.99% reported), and Letta has criticized LoCoMo itself.

**Key techniques:**
- Bi-temporal edges with validity intervals and contradiction-driven invalidation
- Hybrid search: semantic + BM25 + graph traversal with RRF/MMR
- Mem0: two-phase extract/update pipeline with ADD/UPDATE/DELETE/NOOP ops per candidate fact

**Evidence:** Zep: 94.8% DMR and up to 18.5% accuracy improvement on LongMemEval with 90% latency reduction (self-reported). Mem0: 26% relative improvement over OpenAI memory, 91% lower p95 latency (self-reported, methodology publicly disputed by Zep and Letta). Treat every vendor LoCoMo number as marketing until reproduced; LongMemEval and the newer RealMem (2601.06966) are less-saturated alternatives.

**Applicability:** Two takeaways for Hermes-Brain: (1) Daem0n's bi-temporal graph + contradiction detection is already at the production state of the art — keep it, add Graphiti-style explicit edge-validity intervals if missing; (2) adopt Mem0's cheap extract-then-update op set (ADD/UPDATE/DELETE/NOOP per incoming fact) as the low-cost fast path for conversational fact memory, and benchmark internally on LongMemEval rather than trusting published LoCoMo scores.

**Sources:**
- https://arxiv.org/abs/2501.13956
- https://arxiv.org/abs/2504.19413
- https://blog.devgenius.io/ai-agent-memory-systems-in-2026-mem0-zep-hindsight-memvid-and-everything-in-between-compared-96e35b818da8
- https://arxiv.org/pdf/2601.06966

### 2025-2026 survey layer: lifelong-learning roadmap, self-evolving agents, storage-to-experience taxonomy, and forgetting-in-agents evidence

The consolidating academic frame: 'Lifelong Learning of LLM-based Agents: A Roadmap' (2501.07278) organizes the field into perception/memory/action modules; 'A Survey of Self-Evolving Agents' (2507.21046) covers what/when/how/where to evolve; 'From Storage to Experience' (2605.06716, 2026) proposes the Storage -> Reflection -> Experience evolution taxonomy and flags active memory perception, working-memory organization, experience benchmarking, and distributed multi-agent memory as open frontiers. 'Do Self-Evolving Agents Forget?' (2605.09315, 2026) empirically documents capability degradation in lifelong prompt-level adaptation — catastrophic forgetting exists even WITHOUT weight updates (accumulated context/rules can suppress older capabilities).

**Key techniques:**
- Storage->Reflection->Experience maturity model for memory subsystems
- Active memory perception (agent decides when to consult memory, not just top-k every turn)
- Capability-preservation checks during prompt/memory evolution
- Distributed memory consensus across multi-platform agent instances

**Evidence:** Surveys are peer-visible and heavily cited; 2605.09315's degradation findings are early but align with ACE's context-collapse diagnosis from an independent group — two independent confirmations that naive memory accumulation/rewriting degrades agents.

**Applicability:** Use the Storage->Reflection->Experience taxonomy as Hermes-Brain's maturity roadmap (Daem0n is at Reflection; the Experience stage — cross-trajectory schemas, skills, policies — is the gap). Implement capability-regression tests: a fixed probe-task suite run after each sleep-time consolidation, rolling back memory/rule changes that degrade it. The distributed-memory frontier maps to Hermes's multi-platform deployment: one canonical store, platform-tagged provenance.

**Sources:**
- https://arxiv.org/abs/2501.07278
- https://arxiv.org/pdf/2507.21046
- https://arxiv.org/html/2605.06716v1
- https://arxiv.org/pdf/2605.09315
- https://github.com/Shichun-Liu/Agent-Memory-Paper-List

## Top Recommendations

- 1. Build a ReasoningBank-style strategy memory as the direct evolution of the Reflexion loop: after every task, self-judge outcome, distill titled strategy/guardrail items from both successes and failures, store them retrievably, and inject top-k at planning time. Highest evidence-to-effort ratio in the 2025 literature (+34.2% relative on agent benchmarks, ablated against raw-trace and success-only alternatives) and it upgrades infrastructure Hermes already has (outcome tracking, Reflexion). [https://arxiv.org/abs/2509.25140]
- 2. Adopt ACE delta-update discipline for ALL LLM-curated memory artifacts (rules, strategies, user profiles, skill docs): itemized entries with IDs and helpful/harmful counters, incremental add/edit/deprecate operations, never monolithic rewrites. This prevents the context-collapse/capability-degradation failure mode independently documented by two 2025-2026 groups, and it costs almost nothing to implement. [https://arxiv.org/abs/2510.04618, https://arxiv.org/pdf/2605.09315]
- 3. Upgrade 'dreaming' into a full sleep-time consolidation shift: anticipatory pre-computation of likely user queries (Hermes usage is highly predictable per user), SEAL-style rewriting of new knowledge into optimally retrievable form (implications, QA pairs), episodic-to-semantic summarization, A-MEM neighbor-note evolution, and decay/promotion passes — all as batched off-peak API calls. Documented 5x test-time token savings and +13-18% accuracy. [https://arxiv.org/abs/2504.13171, https://arxiv.org/abs/2506.10943, https://arxiv.org/abs/2502.12110]
- 4. Implement a skill-forge pipeline over the agentskills.io system (Memp + AWM + Voyager pattern): detect recurring successful trajectories, distill them into draft skills at two abstraction levels, sandbox-verify, register, track per-skill outcomes, and auto-revise or retire on failure. This is genuine procedural learning without weights, and Memp shows the resulting memory transfers across models — critical for a model-agnostic agent. [https://arxiv.org/html/2508.06433v2, https://proceedings.mlr.press/v267/wang25bx.html]
- 5. Replace recency-based pruning with consolidation-time multi-factor value scoring for active forgetting: score each memory on ~7 cognitively grounded factors (reliability, user relevance, emotional intensity, task utility, usage history, etc.) with weights fit from Hermes's own retention/usage outcomes, combined with Ebbinghaus decay-plus-reinforcement and tiered demotion (full -> summary -> tombstone). The 2026 evidence shows recency-only retains 0.368 of gold evidence vs 0.770 for learned weighting, and that query-similarity is the wrong signal for forgetting. [https://arxiv.org/abs/2606.12945, https://arxiv.org/abs/2305.10250]
- 6. Add a Memento-style case bank: store full task episodes with outcomes, retrieve similar past cases (including failures) into planning context; optionally train a tiny local case-selection model on outcome signal. The case bank doubles as the replay/eval set for recommendation 7. [https://arxiv.org/abs/2508.16153]
- 7. Run a constrained Darwin-Gödel loop over skills/prompts/policy-parameters (not core code, not weights) during sleep-time: propose one revision to the worst-performing skill or rule, replay against archived cases, accept only on statistically significant improvement (PACE-style gates), maintain a variant archive, and run a fixed capability-regression probe suite after every consolidation with automatic rollback. Ship with memory-poisoning guardrails per the 2026 safety literature. [https://arxiv.org/abs/2505.22954, https://arxiv.org/pdf/2606.08106, https://arxiv.org/pdf/2604.16968]
- 8. Add HippoRAG 2-style Personalized PageRank retrieval over the existing bi-temporal graph as a third arm in RRF fusion (alongside BM25+vector): single-step multi-hop association with zero query-time LLM calls, cheaper and better-suited to small hardware than Leiden community-report generation, and validated at ICML 2025 to avoid the factual-recall regression that plagues most graph-RAG. [https://arxiv.org/abs/2502.14802]
- 9. Adopt MemOS/MIRIX structural conventions: every memory in a uniform envelope (provenance incl. source platform, version chain, permissions, lifecycle state) and typed partitions (core/episodic/semantic/procedural/resource/vault) with type-specific update policies — this solves cross-platform (Telegram/Discord/Slack/CLI) identity and governance cleanly. Skip activation/parametric tiers unless a local-GPU profile exists. [https://arxiv.org/abs/2507.03724, https://arxiv.org/abs/2507.07957]
- 10. Watch-list, do not build: Titans/HOPE-class neural memory modules, TTT layers, and SEAL's RL-over-self-edits all require weight access; revisit only for an optional GPU-box profile (LoRA on an open model). Port their principles now instead: surprise-gated memory writing (via prediction-error/contradiction scoring), surprise momentum, and explicit fast/medium/slow memory timescales. [https://arxiv.org/abs/2501.00663, https://arxiv.org/abs/2512.24695, https://arxiv.org/abs/2506.10943]
