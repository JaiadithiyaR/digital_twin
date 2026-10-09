"""RAG Knowledge Base (D2) — read-only contextual knowledge for the agents.

    documents -> chunking -> embeddings -> vector store -> retrieval -> LLM context

Read-only is structural, not a convention. Two classes enforce it:
- `RagKnowledgeBase` — the only class any agent receives. Public surface: `retrieve()`,
  `is_available`, `count` — no add/write/update/delete/ingest method exists, and its constructor
  takes no D1/registry reference, so no agent has any object-graph path to DT state.
- `RagIngestor` — admin/build-time only (`scripts/ingest_rag.py`), the only class that writes to
  the vector store. Also takes no D1/registry reference.

Embeddings: `all-MiniLM-L6-v2` via ChromaDB's bundled `DefaultEmbeddingFunction` (local ONNX, no
torch dependency). `config.rag.embedding_model` must equal this exact name — anything else raises
at construction rather than silently being ignored.

RAG content is untrusted input, same as any other externally-sourced prompt fragment — nothing
here executes or specially trusts retrieved text.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chromadb
from chromadb.utils import embedding_functions

if TYPE_CHECKING:
    from src.common.config import RagConfig, Settings

logger = logging.getLogger(__name__)

_SUPPORTED_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_SOURCE_COMMENT_RE = re.compile(r"^<!--\s*source:\s*(.+?)\s*-->\s*\n", re.IGNORECASE)


class RagUnavailableError(Exception):
    """Raised when the vector store can't be reached/queried. A caller must proceed without
    retrieved context — never invent a plausible-looking answer in its place."""


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    category: str
    source: str
    document_id: str
    chunk_index: int
    version: str
    distance: float


def _build_embedding_function(embedding_model: str) -> embedding_functions.DefaultEmbeddingFunction:
    if embedding_model != _SUPPORTED_EMBEDDING_MODEL:
        raise ValueError(
            f"config.rag.embedding_model={embedding_model!r} is not supported — this module always "
            f"uses ChromaDB's DefaultEmbeddingFunction, which runs {_SUPPORTED_EMBEDDING_MODEL!r} "
            "specifically (see this module's docstring for why no other model is wired up)."
        )
    return embedding_functions.DefaultEmbeddingFunction()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedy word-based chunker: packs whole words up to `chunk_size` chars (never splits a
    word), carrying ~`chunk_overlap` trailing words into the next chunk. Empty input -> no
    chunks."""
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})")
    words = text.split()
    if not words:
        return []
    full = " ".join(words)
    if len(full) <= chunk_size:
        return [full]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    i = 0
    while i < len(words):
        word = words[i]
        added_len = len(word) + (1 if current else 0)
        if current and current_len + added_len > chunk_size:
            chunks.append(" ".join(current))
            current, current_len = _tail_overlap_words(current, chunk_overlap)
            continue  # retry placing `word` against the freshly-seeded overlap
        current.append(word)
        current_len += added_len
        i += 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def _tail_overlap_words(words: list[str], chunk_overlap: int) -> tuple[list[str], int]:
    """Whole words from the END of `words` totaling up to `chunk_overlap` characters."""
    overlap: list[str] = []
    overlap_len = 0
    for word in reversed(words):
        added = len(word) + (1 if overlap else 0)
        if overlap and overlap_len + added > chunk_overlap:
            break
        overlap.insert(0, word)
        overlap_len += added
    return overlap, overlap_len


def _read_document(path: Path) -> tuple[str | None, str]:
    """Returns (source, text). `source` is the URL/provenance from a leading
    `<!-- source: ... -->` comment if present (stripped from the returned text), else `None` — a
    caller with more context (`ingest_corpus`, which knows the document's ID) supplies the
    fallback so every ingested document still ends up with SOME real provenance recorded, never
    left blank."""
    raw = path.read_text(encoding="utf-8")
    match = _SOURCE_COMMENT_RE.match(raw)
    if match:
        return match.group(1), raw[match.end() :]
    return None, raw


@dataclass(frozen=True)
class IngestionSummary:
    documents_processed: int
    chunks_upserted: int
    per_category: dict[str, int]


class RagIngestor:
    """ADMIN/build-time-only. Never handed to an agent — see module docstring."""

    def __init__(self, chroma_path: Path, collection_name: str, embedding_model: str, chunk_size: int, chunk_overlap: int) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(chroma_path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name, embedding_function=_build_embedding_function(embedding_model)
        )

    @classmethod
    def from_settings(cls, settings: "Settings") -> "RagIngestor":
        cfg = settings.rag
        return cls(
            chroma_path=settings.resolve_path(cfg.chroma_path),
            collection_name=cfg.collection_name,
            embedding_model=cfg.embedding_model,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )

    def ingest_corpus(self, corpus_dirs: dict[str, Path]) -> IngestionSummary:
        """Idempotent: each chunk's ChromaDB ID is deterministic
        (`f"{document_id}::{chunk_index}"`) and upserted, so re-running on unchanged files
        overwrites rather than duplicates. If a document shrinks, stale trailing chunk IDs from
        the longer version are explicitly deleted."""
        documents_processed = 0
        chunks_upserted = 0
        per_category: dict[str, int] = {}

        for category, directory in corpus_dirs.items():
            if not directory.exists():
                logger.warning("RAG corpus directory does not exist, skipping", extra={"component": "rag", "category": category, "path": str(directory)})
                continue
            category_chunks = 0
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in (".md", ".txt"):
                    continue
                document_id = str(path.relative_to(directory))
                source, text = _read_document(path)
                source = source or document_id
                version = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                chunks = chunk_text(text, self._chunk_size, self._chunk_overlap)
                if not chunks:
                    continue
                timestamp = datetime.now(UTC).isoformat()

                ids = [f"{category}/{document_id}::{i}" for i in range(len(chunks))]
                metadatas: list[dict[str, Any]] = [
                    {
                        "document_id": document_id,
                        "source": source,
                        "category": category,
                        "version": version,
                        "timestamp": timestamp,
                        "chunk_index": i,
                    }
                    for i in range(len(chunks))
                ]
                self._collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
                self._prune_stale_trailing_chunks(category, document_id, kept_chunk_count=len(chunks))

                documents_processed += 1
                chunks_upserted += len(chunks)
                category_chunks += len(chunks)
            per_category[category] = category_chunks

        logger.info(
            "RAG corpus ingestion complete",
            extra={"component": "rag", "documents_processed": documents_processed, "chunks_upserted": chunks_upserted, "per_category": per_category},
        )
        return IngestionSummary(documents_processed=documents_processed, chunks_upserted=chunks_upserted, per_category=per_category)

    def _prune_stale_trailing_chunks(self, category: str, document_id: str, kept_chunk_count: int) -> None:
        existing = self._collection.get(where={"$and": [{"category": category}, {"document_id": document_id}]})
        stale_ids = [
            chunk_id
            for chunk_id, metadata in zip(existing["ids"], existing["metadatas"], strict=True)
            if metadata["chunk_index"] >= kept_chunk_count
        ]
        if stale_ids:
            self._collection.delete(ids=stale_ids)

    def count(self) -> int:
        return self._collection.count()

    def smoke_test(self) -> bool:
        """prompt.md §36 step 3-4: insert then retrieve a small test record to verify the
        embed/store/retrieve path genuinely works, before real corpus ingestion runs. Cleans up
        after itself either way — never leaves the smoke-test record in the store."""
        category, doc_id, text = "_smoke_test", "_smoke_test", "This is a small smoke-test record verifying the RAG store's embed/store/retrieve path works."
        chunk_id = f"{category}/{doc_id}::0"
        self._collection.upsert(
            ids=[chunk_id],
            documents=[text],
            metadatas=[{"document_id": doc_id, "source": "internal:rag_kb.py:smoke_test", "category": category, "version": "n/a", "timestamp": "n/a", "chunk_index": 0}],
        )
        try:
            result = self._collection.query(query_texts=["smoke test record"], n_results=1, where={"category": category})
            documents = result.get("documents") or [[]]
            return bool(documents and documents[0] and "smoke-test record" in documents[0][0])
        finally:
            self._collection.delete(ids=[chunk_id])


class RagKnowledgeBase:
    """Agent-facing. The ONLY thing this class can do is retrieve — see module docstring."""

    def __init__(self, chroma_path: Path, collection_name: str, embedding_model: str, default_top_k: int) -> None:
        self._default_top_k = default_top_k
        # A bad CONFIG value (unsupported model) is a programming/config error — it must raise
        # immediately, never be swallowed into "unavailable" state the way a broken/corrupt
        # on-disk store is below.
        embedding_function = _build_embedding_function(embedding_model)
        try:
            client = chromadb.PersistentClient(path=str(chroma_path))
            self._collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)
            self._unavailable_reason: str | None = None
        except Exception as exc:  # a broken/corrupt store must degrade, not crash the caller —
            # `retrieve()` raises RagUnavailableError instead, per prompt.md §61.
            self._collection = None
            self._unavailable_reason = str(exc)
            logger.warning("RAG knowledge base unavailable at construction", extra={"component": "rag", "error": str(exc)})

    @classmethod
    def from_settings(cls, settings: "Settings") -> "RagKnowledgeBase":
        cfg: RagConfig = settings.rag
        return cls(
            chroma_path=settings.resolve_path(cfg.chroma_path),
            collection_name=cfg.collection_name,
            embedding_model=cfg.embedding_model,
            default_top_k=cfg.top_k,
        )

    @property
    def is_available(self) -> bool:
        return self._collection is not None

    def count(self) -> int:
        if self._collection is None:
            return 0
        return self._collection.count()

    def retrieve(self, query: str, *, category: str | None = None, top_k: int | None = None) -> list[RetrievedChunk]:
        """The ONLY way to get data out of this store. Raises `RagUnavailableError` if the store
        couldn't be opened, or if the query itself fails — never returns a fabricated result."""
        if self._collection is None:
            raise RagUnavailableError(self._unavailable_reason or "RAG knowledge base is unavailable")
        try:
            result = self._collection.query(
                query_texts=[query],
                n_results=top_k if top_k is not None else self._default_top_k,
                where={"category": category} if category is not None else None,
            )
        except Exception as exc:
            raise RagUnavailableError(f"RAG query failed: {exc}") from exc

        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        chunks: list[RetrievedChunk] = []
        for text, metadata, distance in zip(documents[0], metadatas[0], distances[0], strict=True):
            chunks.append(
                RetrievedChunk(
                    text=text,
                    category=metadata["category"],
                    source=metadata["source"],
                    document_id=metadata["document_id"],
                    chunk_index=metadata["chunk_index"],
                    version=metadata["version"],
                    distance=distance,
                )
            )
        return chunks
