"""Tiered embedders: ONNX (full) / model2vec static (lite) / stub (tests).

Tier selection is store.sysinfo.resolve_mode(); every heavy dependency is
imported lazily so the plugin stays importable (and Hermes stays bootable)
with nothing installed — the floor tier simply gets no embedder.

full  — ONNX-quantized transformer via onnxruntime + tokenizers.
        Default model: nomic-ai/modernbert-embed-base (proven in Daem0n-MCP:
        asymmetric 'search_query: '/'search_document: ' prefixes, matryoshka
        truncation to 256d, ungated download). EmbeddingGemma-300m is
        registered as an opt-in alternative (config embed_model) — it holds
        the quality-per-RAM crown (research:infra) but its HF repo is
        license-gated, so it cannot be the zero-friction default; set
        HF_TOKEN and embed_model: embeddinggemma-300m to use it.
lite  — model2vec static embeddings (potion-retrieval-32M, ~30MB, no ONNX).
stub  — deterministic hash-based vectors: real cosine behavior for
        overlapping token sets, zero deps. Config-only tier for tests/CI.

Model files live in a shared, re-downloadable, NON-backed-up cache
(critique item 36): %LOCALAPPDATA%/hermes-brain/models on Windows,
~/.cache/hermes-brain/models elsewhere — never under HERMES_HOME.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import sys
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

TARGET_DIM = 256  # matryoshka truncation target for onnx tier


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    query_prefix: str
    doc_prefix: str
    native_dim: int
    truncate_to: int | None           # matryoshka-safe truncation, or None
    files: dict[str, str] = field(default_factory=dict)  # local name -> repo path
    gated: bool = False


REGISTRY: dict[str, ModelSpec] = {
    "modernbert-embed-base": ModelSpec(
        key="modernbert-embed-base",
        repo="nomic-ai/modernbert-embed-base",
        query_prefix="search_query: ",
        doc_prefix="search_document: ",
        native_dim=768,
        truncate_to=TARGET_DIM,
        files={
            "model.onnx": "onnx/model_quantized.onnx",
            "tokenizer.json": "tokenizer.json",
        },
    ),
    "embeddinggemma-300m": ModelSpec(
        key="embeddinggemma-300m",
        repo="onnx-community/embeddinggemma-300m-ONNX",
        query_prefix="task: search result | query: ",
        doc_prefix="title: none | text: ",
        native_dim=768,
        truncate_to=TARGET_DIM,
        files={
            "model.onnx": "onnx/model_quantized.onnx",
            "tokenizer.json": "tokenizer.json",
        },
        gated=True,
    ),
}

_POTION_REPO = "minishlab/potion-retrieval-32M"


def models_cache_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "hermes-brain" / "models"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "hermes-brain" / "models"


class ModelDownloadError(RuntimeError):
    pass


def ensure_files(spec: ModelSpec, cache_dir: Path | None = None,
                 progress: bool = True, download: bool = True) -> Path:
    """Resolve (and optionally download) the spec's files. Returns dir.

    download=False is the live-session path: a provider must never surprise
    the user with a 90MB fetch mid-turn — models are fetched explicitly by
    setup / 'hermes brain doctor --fix'.
    """
    target = (cache_dir or models_cache_dir()) / spec.key
    target.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    for local_name, repo_path in spec.files.items():
        dest = target / local_name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        if not download:
            raise ModelDownloadError(
                f"model file missing: {dest}. Run 'hermes brain doctor --fix' "
                f"(or 'hermes memory setup') to download {spec.repo}."
            )
        url = f"https://huggingface.co/{spec.repo}/resolve/main/{repo_path}"
        req = urllib.request.Request(url, headers={"User-Agent": "hermes-brain/0.1"})
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        # Per-process temp name: two concurrent downloads must never write
        # the same .part file (interleaved writes corrupt the model).
        tmp = dest.with_suffix(dest.suffix + f".part.{os.getpid()}")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if progress and total:
                        print(f"\r  {spec.key}/{local_name}: {done * 100 // total}% "
                              f"({done >> 20}MB/{total >> 20}MB)",
                              end="", file=sys.stderr, flush=True)
            if progress:
                print(file=sys.stderr)
            if total and done != total:
                raise ModelDownloadError(
                    f"partial download for {url}: got {done} of {total} bytes — "
                    f"the connection dropped; run the download again to retry."
                )
            tmp.replace(dest)
        except ModelDownloadError:
            tmp.unlink(missing_ok=True)
            raise
        except Exception as e:
            tmp.unlink(missing_ok=True)
            hint = (
                f"model download failed for {url}: {e}."
                + (" This repo is LICENSE-GATED on Hugging Face — accept the license "
                   "and set HF_TOKEN in ~/.hermes/.env, or switch back to the default: "
                   "embed_model: modernbert-embed-base" if spec.gated else
                   " Check network access, or run 'hermes brain doctor' later; "
                   "search degrades to FTS-only until models are present.")
            )
            raise ModelDownloadError(hint) from e
    return target


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------

class OnnxEmbedder:
    """onnxruntime + tokenizers; lazy session; mean-pool, normalize,
    matryoshka-truncate, renormalize (Daem0n vectors.py semantics)."""

    def __init__(self, spec: ModelSpec, cache_dir: Path | None = None,
                 allow_download: bool = False) -> None:
        self.spec = spec
        self.dim = spec.truncate_to or spec.native_dim
        self.name = f"{spec.key}-q8:{self.dim}"
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

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        self._load()
        import numpy as np

        out: list[list[float]] = []
        for start in range(0, len(texts), 16):
            batch = [prefix + t for t in texts[start:start + 16]]
            encs = self._tokenizer.encode_batch(batch)
            max_len = max(len(e.ids) for e in encs)
            ids = np.zeros((len(encs), max_len), dtype=np.int64)
            mask = np.zeros((len(encs), max_len), dtype=np.int64)
            for i, e in enumerate(encs):
                ids[i, : len(e.ids)] = e.ids
                mask[i, : len(e.ids)] = e.attention_mask
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self._session.run(None, feeds)[0]  # (B, T, H)
            m = mask[:, :, None].astype(hidden.dtype)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
            if self.spec.truncate_to:
                pooled = pooled[:, : self.spec.truncate_to]
                pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
            out.extend(pooled.tolist())
        return out

    def encode_query(self, text: str) -> list[float]:
        return self._encode([text], self.spec.query_prefix)[0]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(list(texts), self.spec.doc_prefix)


class PotionEmbedder:
    """model2vec static embeddings — the Termux/512MB floor with semantics.

    Honors the no-surprise-download contract like the ONNX tier: the model
    loads from the local cache; a network fetch happens only when the caller
    explicitly allows it (setup / 'hermes brain models --download').
    """

    def __init__(self, allow_download: bool = False) -> None:
        from model2vec import StaticModel  # lazy optional dep

        local_dir = models_cache_dir() / "potion-retrieval-32m"
        if local_dir.is_dir() and any(local_dir.iterdir()):
            self._model = StaticModel.from_pretrained(str(local_dir))
        elif allow_download:
            self._model = StaticModel.from_pretrained(_POTION_REPO)
            try:  # best-effort local copy so later runs never touch the network
                local_dir.mkdir(parents=True, exist_ok=True)
                self._model.save_pretrained(str(local_dir))
            except Exception as e:
                logger.warning("could not cache %s locally at %s: %s",
                               _POTION_REPO, local_dir, e)
        else:
            raise ModelDownloadError(
                "lite-tier model missing: run hermes brain models --download "
                "(or hermes memory setup)"
            )
        self.dim = int(self._model.dim)
        self.name = f"potion-retrieval-32m:{self.dim}"

    def _norm(self, rows) -> list[list[float]]:
        out = []
        for row in rows:
            norm = math.sqrt(sum(v * v for v in row)) or 1.0
            out.append([v / norm for v in row])
        return out

    def encode_query(self, text: str) -> list[float]:
        return self._norm(self._model.encode([text]).tolist())[0]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self._norm(self._model.encode(list(texts)).tolist())


class StubEmbedder:
    """Deterministic hash-based vectors (config mode: stub — tests/CI only).

    Each token hashes to a pseudo-random unit vector; a text is the
    normalized token-vector sum, so token overlap => cosine similarity.
    """

    dim = TARGET_DIM
    name = f"stub-hash:{TARGET_DIM}"
    _token_re = re.compile(r"[^\W_]+", re.UNICODE)

    def _token_vec(self, token: str) -> list[float]:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        vals = []
        seed = digest
        while len(vals) < self.dim:
            seed = hashlib.sha256(seed).digest()
            vals.extend(b / 127.5 - 1.0 for b in seed)
        return vals[: self.dim]

    def _encode_one(self, text: str) -> list[float]:
        acc = [0.0] * self.dim
        tokens = [t.casefold() for t in self._token_re.findall(text)] or ["∅"]
        for tok in tokens:
            for i, v in enumerate(self._token_vec(tok)):
                acc[i] += v
        norm = math.sqrt(sum(v * v for v in acc)) or 1.0
        return [v / norm for v in acc]

    def encode_query(self, text: str) -> list[float]:
        return self._encode_one(text)

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode_one(t) for t in texts]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedder(config: dict, mode: str, allow_download: bool = False):
    """Embedder for a resolved mode, or None (fts-only). Never raises for
    missing optional deps — degradation is logged, not fatal."""
    try:
        if mode == "full":
            key = str(config.get("embed_model") or "modernbert-embed-base")
            spec = REGISTRY.get(key)
            if spec is None:
                logger.warning("unknown embed_model %r; falling back to default", key)
                spec = REGISTRY["modernbert-embed-base"]
            return OnnxEmbedder(spec, allow_download=allow_download)
        if mode == "lite":
            return PotionEmbedder(allow_download=allow_download)
        if mode == "stub":
            return StubEmbedder()
    except ModelDownloadError as e:
        logger.warning("embedder unavailable: %s", e)
    except ImportError as e:
        logger.warning("embedder deps missing (%s); running fts-only", e)
    except Exception:
        logger.warning("embedder init failed; running fts-only", exc_info=True)
    return None
