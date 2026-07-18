"""ColBERT late-interaction reranker — the optional 'full'-tier rerank stage.

docs/design/memory-engine.md §3.4: after RRF fuses the FTS + vector (+ graph)
legs, a late-interaction ColBERT model reranks the fused top-K by token-level
MaxSim relevance, BEFORE the lifecycle modulation and episode factor apply.
The stage is strictly optional and degrades to a no-op:

  * lite / fts-only tier, or ``rerank: off`` in brain.yaml -> get_reranker() is None
  * model files absent (never surprise-downloaded on a turn)  -> None
  * any runtime error, or a blown time budget                 -> fused order kept

Primary model: mixedbread-ai/mxbai-edge-colbert-v0-32m (ONNX int8, ModernBERT
backbone, per-token output). Fallback: answerdotai/answerai-colbert-small-v1.
Both reuse the OnnxEmbedder download/cache machinery in embed.py
(``ensure_files``); the model tag (``self.name``) is versioned exactly like
``embedded_with`` so a model swap is auditable (critique item 34).

The reranker runs on the brain-bg worker (``_do_retrieve`` computes the block
for the NEXT turn's prefetch), so it never adds turn latency — the budget guard
is a worker-throughput backstop, not a turn deadline.

Heavy deps (onnxruntime, tokenizers, numpy) are imported lazily; this module is
import-safe with nothing installed.
"""

from __future__ import annotations

import logging
import re
import time

from .embed import ModelDownloadError, ModelSpec, ensure_files

logger = logging.getLogger(__name__)

DEFAULT_BUDGET_S = 2.0
# Rerank at most the fused top-N candidates: the deeper tail cannot reach the
# final top-`limit` anyway, and this bounds the ONNX cost per retrieval.
_MAX_CANDIDATES = 32

# ColBERT models reuse ModelSpec; only key/repo/prefix/files/gated are read by
# ensure_files + this module (native_dim/truncate_to are ignored — no pooling).
RERANK_REGISTRY: dict[str, ModelSpec] = {
    "mxbai-edge-colbert-v0-32m": ModelSpec(
        key="mxbai-edge-colbert-v0-32m",
        repo="mixedbread-ai/mxbai-edge-colbert-v0-32m",
        query_prefix="[Q] ",
        doc_prefix="[D] ",
        native_dim=96,
        truncate_to=None,
        files={
            "model.onnx": "onnx/model_quantized.onnx",
            "tokenizer.json": "tokenizer.json",
        },
    ),
    "answerai-colbert-small-v1": ModelSpec(
        key="answerai-colbert-small-v1",
        repo="answerdotai/answerai-colbert-small-v1",
        query_prefix="[Q] ",
        doc_prefix="[D] ",
        native_dim=96,
        truncate_to=None,
        files={
            "model.onnx": "onnx/model_quantized.onnx",
            "tokenizer.json": "tokenizer.json",
        },
    ),
}
# Primary first: get_reranker walks this order until one model resolves.
_FALLBACK_ORDER = ("mxbai-edge-colbert-v0-32m", "answerai-colbert-small-v1")


class ColbertReranker:
    """onnxruntime + tokenizers late-interaction scorer.

    ``score(query, docs)`` returns one MaxSim relevance per doc: for each query
    token, the max cosine over the doc's tokens, summed. Query and doc token
    embeddings are L2-normalized per token so the dot product IS cosine.
    """

    def __init__(self, spec: ModelSpec, cache_dir=None, allow_download: bool = False) -> None:
        self.spec = spec
        self.name = f"{spec.key}-q8"
        self._dir = ensure_files(spec, cache_dir, download=allow_download)
        self._session = None
        self._tokenizer = None
        self._input_names: list[str] = []

    def _load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime  # lazy heavy dep
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(self._dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=512)
        opts = onnxruntime.SessionOptions()
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = onnxruntime.InferenceSession(
            str(self._dir / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self._input_names = [i.name for i in self._session.get_inputs()]

    def _encode(self, texts: list[str], prefix: str):
        """List of (n_valid_tokens, dim) L2-normalized token matrices (padding
        removed via the attention mask)."""
        self._load()
        import numpy as np

        results = []
        for start in range(0, len(texts), 8):
            batch = [prefix + (t or "") for t in texts[start:start + 8]]
            encs = self._tokenizer.encode_batch(batch)
            max_len = max((len(e.ids) for e in encs), default=1)
            ids = np.zeros((len(encs), max_len), dtype=np.int64)
            mask = np.zeros((len(encs), max_len), dtype=np.int64)
            for i, e in enumerate(encs):
                ids[i, : len(e.ids)] = e.ids
                mask[i, : len(e.ids)] = e.attention_mask
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self._session.run(None, feeds)[0]  # (B, T, D) per-token
            norms = np.clip(np.linalg.norm(hidden, axis=2, keepdims=True), 1e-9, None)
            hidden = hidden / norms
            for i in range(hidden.shape[0]):
                keep = mask[i].astype(bool)
                results.append(hidden[i][keep])
        return results

    def score(self, query: str, docs) -> list[float]:
        docs = list(docs)
        if not docs:
            return []
        q = self._encode([query], self.spec.query_prefix)[0]  # (nq, d)
        out: list[float] = []
        for d in self._encode(docs, self.spec.doc_prefix):
            if q.size == 0 or d.size == 0:
                out.append(0.0)
                continue
            sim = q @ d.T                     # (nq, nd) cosine
            out.append(float(sim.max(axis=1).sum()))
        return out


class StubReranker:
    """Deterministic token-overlap scorer (config mode: rerank_model=stub).

    Test/CI tier only, never auto-selected. Scores each doc by the count of
    query tokens it contains — enough to exercise the reorder/band/budget path
    without any model download.
    """

    name = "stub-rerank"
    _tok = re.compile(r"[^\W_]+", re.UNICODE)

    def score(self, query: str, docs) -> list[float]:
        qset = {t.casefold() for t in self._tok.findall(query or "")}
        out: list[float] = []
        for d in docs:
            dset = {t.casefold() for t in self._tok.findall(d or "")}
            out.append(float(len(qset & dset)))
        return out


def get_reranker(config: dict, mode: str, *, allow_download: bool = False):
    """Reranker for the resolved mode, or None (stage skipped). Never raises.

    ``rerank: off`` disables it; ``rerank_model: stub`` forces the deterministic
    test scorer; otherwise the ColBERT stage is only built on the ONNX ('full')
    tier and walks the fallback chain until a model resolves.
    """
    setting = str((config or {}).get("rerank", "auto")).strip().lower()
    if setting in ("off", "false", "no", "0", "none"):
        return None
    model_key = str((config or {}).get("rerank_model") or "").strip().lower()
    if model_key == "stub":
        return StubReranker()  # config-only test tier; never auto-selected
    if mode != "full":
        return None  # rerank needs the ONNX tier
    try:
        keys = [model_key] if model_key in RERANK_REGISTRY else list(_FALLBACK_ORDER)
        last: Exception | None = None
        for key in keys:
            try:
                return ColbertReranker(RERANK_REGISTRY[key], allow_download=allow_download)
            except ModelDownloadError as e:
                last = e
        if last is not None:
            logger.info("reranker unavailable: %s; rerank stage skipped", last)
    except ImportError as e:
        logger.info("reranker deps missing (%s); rerank stage skipped", e)
    except Exception:
        logger.warning("reranker init failed; rerank stage skipped", exc_info=True)
    return None


def rerank_scores(reranker, query: str, candidates, *,
                  budget_s: float = DEFAULT_BUDGET_S) -> dict | None:
    """Map candidate key -> raw MaxSim relevance for the reranked slice.

    ``candidates`` is a fused-order list of (key, text). Returns a dict over the
    top ``_MAX_CANDIDATES`` slice, or None to signal 'keep the fused order'
    (nothing to reorder, budget blown, or any failure). Never raises.
    """
    if reranker is None or len(candidates) < 2 or not (query or "").strip():
        return None
    try:
        t0 = time.monotonic()
        slice_ = candidates[:_MAX_CANDIDATES]
        scores = reranker.score(query, [text for _key, text in slice_])
        if time.monotonic() - t0 > budget_s:
            logger.info("rerank exceeded %.1fs budget; keeping fused order", budget_s)
            return None
        if not scores or len(scores) != len(slice_):
            return None
        return {key: float(s) for (key, _text), s in zip(slice_, scores, strict=True)}
    except Exception as e:
        logger.warning("rerank failed (%s); keeping fused order", e)
        return None
