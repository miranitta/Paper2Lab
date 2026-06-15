"""
indexer.py — Local FAISS RAG index for Paper2Lab.

Purpose
-------
Build a retrieval index from section-aware PDF extraction output.

Default mode is local and cheap:
    sentence-transformers + FAISS

Optional mode supports NVIDIA/Nemotron-style embedding endpoints through
langchain_nvidia_ai_endpoints when NVIDIA_API_KEY is available.

The public functions are intentionally simple:
    build_rag_index(extracted)
    save_rag_index(index, path)
    load_rag_index(path, embedder_backend="local")

No LLM generation happens here. qa.py handles answer synthesis.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "faiss-cpu is required for Paper2Lab RAG. Install with: pip install faiss-cpu"
    ) from exc


EmbedderBackend = Literal["local", "nvidia"]


@dataclass
class RagChunk:
    chunk_id: str
    text: str
    source_type: str  # section | caption | table | metadata
    title: str
    role: str = "other"
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    label: Optional[str] = None
    score: Optional[float] = None

    def to_evidence(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_type": self.source_type,
            "title": self.title,
            "role": self.role,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "label": self.label,
            "score": self.score,
            "text": self.text,
        }


@dataclass
class RagIndex:
    index: Any
    chunks: List[RagChunk]
    embedder_backend: EmbedderBackend
    embedder_model: str
    normalize_embeddings: bool = True


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def _clean(text: str) -> str:
    text = text or ""
    text = text.replace("\x00", " ").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", "", text)
    return text.strip(" .;:\n\t")


def _bad_chunk(text: str) -> bool:
    low = text.lower()
    bad = [
        "corresponding author",
        "how to cite",
        "access this article online",
        "copyright",
        "all rights reserved",
        "gmail.com",
        "@",
        "provided proper attribution",
        "permission to reproduce",
    ]
    if any(x in low for x in bad):
        return True
    if len(text.split()) < 8:
        return True
    if text.count("|") >= 2:
        return True
    return False


def _split_into_windows(text: str, max_words: int = 170, overlap_words: int = 35) -> List[str]:
    """Chunk text by word windows. Simple and robust for noisy PDF text."""
    text = _clean(text)
    if not text:
        return []
    words = text.split()
    if len(words) <= max_words:
        return [] if _bad_chunk(text) else [text]

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunk = _clean(" ".join(words[start:end]))
        if chunk and not _bad_chunk(chunk):
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap_words)
    return chunks


def _table_to_text(table: Dict[str, Any]) -> str:
    data = table.get("data")
    caption = _clean(table.get("caption") or "")
    if not isinstance(data, list):
        return caption
    rows: List[str] = []
    for row in data[:12]:
        if isinstance(row, list):
            cells = [_clean(str(c)) for c in row if c is not None and _clean(str(c))]
            if cells:
                rows.append(" | ".join(cells[:8]))
    body = "\n".join(rows)
    return _clean(f"{caption}\n{body}")


# ---------------------------------------------------------------------------
# Chunk extraction
# ---------------------------------------------------------------------------


def build_chunks(extracted: Dict[str, Any], include_tables: bool = True, include_captions: bool = True) -> List[RagChunk]:
    chunks: List[RagChunk] = []
    counter = 0

    blocked_roles = {"references", "appendix", "boilerplate"}
    blocked_titles = {"front matter", "keywords", "table of contents"}

    for sec_idx, sec in enumerate(extracted.get("sections", []) or []):
        role = sec.get("role", "other")
        title = _clean(sec.get("title") or "Untitled section")
        if role in blocked_roles or title.lower() in blocked_titles:
            continue
        text = sec.get("text") or ""
        for window in _split_into_windows(text):
            counter += 1
            chunks.append(
                RagChunk(
                    chunk_id=f"section-{sec_idx}-{counter}",
                    text=window,
                    source_type="section",
                    title=title,
                    role=role,
                    page_start=sec.get("page_start"),
                    page_end=sec.get("page_end"),
                )
            )

    if include_captions:
        for cap_idx, cap in enumerate(extracted.get("captions", []) or []):
            label = _clean(cap.get("label") or f"caption-{cap_idx}")
            caption = _clean(cap.get("caption") or "")
            if caption and not _bad_chunk(caption):
                chunks.append(
                    RagChunk(
                        chunk_id=f"caption-{cap_idx}",
                        text=caption,
                        source_type="caption",
                        title=label,
                        role="caption",
                        page_start=cap.get("page_number"),
                        page_end=cap.get("page_number"),
                        label=label,
                    )
                )

    if include_tables:
        for table_idx, table in enumerate(extracted.get("tables", []) or []):
            text = _table_to_text(table)
            if text and not _bad_chunk(text):
                label = f"Table {table_idx + 1}"
                chunks.append(
                    RagChunk(
                        chunk_id=f"table-{table_idx}",
                        text=text[:2500],
                        source_type="table",
                        title=label,
                        role="table",
                        page_start=table.get("page_number"),
                        page_end=table.get("page_number"),
                        label=label,
                    )
                )

    return chunks


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------


class BaseEmbedder:
    def encode_documents(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


class LocalSentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            ) from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        # BGE-style instruction prefix helps retrieval quality.
        docs = [f"passage: {t}" for t in texts]
        arr = self.model.encode(docs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return arr.astype("float32")

    def encode_query(self, text: str) -> np.ndarray:
        arr = self.model.encode([f"query: {text}"], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return arr.astype("float32")


class NvidiaEndpointEmbedder(BaseEmbedder):
    """NVIDIA API/NIM embedding backend.

    Requires:
        pip install langchain-nvidia-ai-endpoints
        set NVIDIA_API_KEY=...

    Default model uses NVIDIA's retrieval QA embedding endpoint.
    """

    def __init__(self, model_name: str = "nvidia/nv-embedqa-e5-v5") -> None:
        try:
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Install NVIDIA embeddings support with: pip install langchain-nvidia-ai-endpoints"
            ) from exc
        self.model_name = model_name
        self.embedder = NVIDIAEmbeddings(model=model_name)

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        # NVIDIA embedding endpoints support document embedding methods through LangChain.
        vecs = self.embedder.embed_documents(texts)
        arr = np.array(vecs, dtype="float32")
        return _l2_normalize(arr)

    def encode_query(self, text: str) -> np.ndarray:
        vec = self.embedder.embed_query(text)
        arr = np.array([vec], dtype="float32")
        return _l2_normalize(arr)


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype("float32")


def make_embedder(backend: EmbedderBackend = "local", model_name: Optional[str] = None) -> BaseEmbedder:
    if backend == "local":
        return LocalSentenceTransformerEmbedder(model_name or "BAAI/bge-small-en-v1.5")
    if backend == "nvidia":
        return NvidiaEndpointEmbedder(model_name or "nvidia/nv-embedqa-e5-v5")
    raise ValueError(f"Unknown embedder backend: {backend}")


# ---------------------------------------------------------------------------
# Index build/search/save/load
# ---------------------------------------------------------------------------


def build_rag_index(
    extracted: Dict[str, Any],
    embedder_backend: EmbedderBackend = "local",
    embedder_model: Optional[str] = None,
    include_tables: bool = True,
    include_captions: bool = True,
) -> RagIndex:
    chunks = build_chunks(extracted, include_tables=include_tables, include_captions=include_captions)
    if not chunks:
        raise ValueError("No usable chunks found for RAG indexing.")

    embedder = make_embedder(embedder_backend, embedder_model)
    texts = [c.text for c in chunks]
    embeddings = embedder.encode_documents(texts)
    embeddings = _l2_normalize(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return RagIndex(
        index=index,
        chunks=chunks,
        embedder_backend=embedder_backend,
        embedder_model=embedder_model or ("BAAI/bge-small-en-v1.5" if embedder_backend == "local" else "nvidia/nv-embedqa-e5-v5"),
        normalize_embeddings=True,
    )


def search_rag_index(rag_index: RagIndex, query: str, top_k: int = 5) -> List[RagChunk]:
    embedder = make_embedder(rag_index.embedder_backend, rag_index.embedder_model)
    q = embedder.encode_query(query)
    q = _l2_normalize(q)
    scores, ids = rag_index.index.search(q, top_k)
    hits: List[RagChunk] = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx < 0 or idx >= len(rag_index.chunks):
            continue
        chunk = rag_index.chunks[idx]
        hits.append(
            RagChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source_type=chunk.source_type,
                title=chunk.title,
                role=chunk.role,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                label=chunk.label,
                score=round(float(score), 4),
            )
        )
    return hits


def save_rag_index(rag_index: RagIndex, path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(rag_index.index, str(path / "index.faiss"))
    metadata = {
        "embedder_backend": rag_index.embedder_backend,
        "embedder_model": rag_index.embedder_model,
        "normalize_embeddings": rag_index.normalize_embeddings,
        "chunks": [asdict(c) for c in rag_index.chunks],
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def load_rag_index(path: str | Path) -> RagIndex:
    path = Path(path)
    index_path = path / "index.faiss"
    meta_path = path / "metadata.json"
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing FAISS index files in {path}")
    index = faiss.read_index(str(index_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    chunks = [RagChunk(**c) for c in metadata.get("chunks", [])]
    return RagIndex(
        index=index,
        chunks=chunks,
        embedder_backend=metadata.get("embedder_backend", "local"),
        embedder_model=metadata.get("embedder_model", "BAAI/bge-small-en-v1.5"),
        normalize_embeddings=bool(metadata.get("normalize_embeddings", True)),
    )
