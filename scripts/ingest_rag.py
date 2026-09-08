#!/usr/bin/env python3
"""D2 — RAG Knowledge Base initialization + corpus ingestion (prompt.md §36).

On first run: initializes a persistent ChromaDB store, creates the collection, inserts and
retrieves a small smoke-test record to verify the embed/store/retrieve path genuinely works, then
runs real corpus ingestion over `rag_data/{oran,digital_twin,policies,history}/`. Idempotent —
safe to re-run any time (e.g. after editing a corpus document); re-ingesting unchanged files never
duplicates them (see `src/rag/rag_kb.py:RagIngestor.ingest_corpus`'s docstring for exactly why).

Usage:
    python scripts/ingest_rag.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_settings  # noqa: E402
from src.rag.rag_kb import RagIngestor  # noqa: E402


def main() -> int:
    settings = load_settings()
    cfg = settings.rag

    print(f"[ingest_rag] initializing persistent ChromaDB at {settings.resolve_path(cfg.chroma_path)}")
    ingestor = RagIngestor.from_settings(settings)
    print(f"[ingest_rag] collection {cfg.collection_name!r} ready ({ingestor.count()} chunks currently stored)")

    print("[ingest_rag] inserting a small smoke-test record and verifying retrieval...")
    if not ingestor.smoke_test():
        print("[ingest_rag] FAILED: smoke-test insert/retrieve did not round-trip correctly", file=sys.stderr)
        return 1
    print("[ingest_rag] smoke test OK")

    print("[ingest_rag] running real corpus ingestion...")
    corpus_dirs = {category: settings.resolve_path(rel_path) for category, rel_path in cfg.corpus_dirs.items()}
    summary = ingestor.ingest_corpus(corpus_dirs)

    print(f"[ingest_rag] documents processed: {summary.documents_processed}")
    print(f"[ingest_rag] chunks upserted: {summary.chunks_upserted}")
    for category, count in summary.per_category.items():
        thin_note = "  <-- placeholder-thin, see corpus README" if count <= 2 else ""
        print(f"[ingest_rag]   {category}: {count} chunks{thin_note}")

    print(f"[ingest_rag] total chunks now stored: {ingestor.count()}")
    print("[ingest_rag] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
