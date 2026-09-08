"""Integration tests: D2 (RAG Knowledge Base) against the REAL `rag_data/` corpus and genuine
embeddings (ChromaDB's bundled ONNX all-MiniLM-L6-v2 — no mocking). Proves the corpus that ships
with this repository actually ingests and retrieves correctly, and that embeddings are real
(semantically meaningful), not a stub returning constant/random vectors.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.config import load_settings
from src.rag.rag_kb import RagIngestor, RagKnowledgeBase

SETTINGS = load_settings()
REPO_ROOT = Path(__file__).resolve().parents[2]


def _real_ingestor(tmp_path) -> RagIngestor:
    """A REAL ingestor pointed at a temp chroma path (never the repo's own `rag_data/chroma`,
    so tests never depend on or mutate whatever ingestion state happens to be on disk) that
    ingests the REAL `rag_data/` corpus documents."""
    return RagIngestor(
        chroma_path=tmp_path / "chroma",
        collection_name=SETTINGS.rag.collection_name,
        embedding_model=SETTINGS.rag.embedding_model,
        chunk_size=SETTINGS.rag.chunk_size,
        chunk_overlap=SETTINGS.rag.chunk_overlap,
    )


def _real_kb(tmp_path) -> RagKnowledgeBase:
    return RagKnowledgeBase(
        chroma_path=tmp_path / "chroma",
        collection_name=SETTINGS.rag.collection_name,
        embedding_model=SETTINGS.rag.embedding_model,
        default_top_k=SETTINGS.rag.top_k,
    )


@pytest.fixture(scope="module")
def ingested_kb(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("rag_integration")
    ingestor = _real_ingestor(tmp_path)
    corpus_dirs = {category: REPO_ROOT / rel_path for category, rel_path in SETTINGS.rag.corpus_dirs.items()}
    summary = ingestor.ingest_corpus(corpus_dirs)
    assert summary.documents_processed > 0, "the real rag_data/ corpus produced zero documents — check it exists"
    return _real_kb(tmp_path), summary


def test_real_corpus_ingests_all_four_categories(ingested_kb):
    _, summary = ingested_kb
    for category in ("oran", "digital_twin", "policies", "history"):
        assert summary.per_category.get(category, 0) > 0, f"category {category!r} produced zero chunks"


def test_real_oran_content_is_retrievable_and_attributed_to_a_real_source(ingested_kb):
    kb, _ = ingested_kb
    results = kb.retrieve("what is the near-RT RIC", category="oran", top_k=3)
    assert results
    assert any("near-rt ric" in r.text.lower() or "near-real-time" in r.text.lower() for r in results)
    assert all(r.source.startswith("https://docs.o-ran-sc.org/") for r in results)


def test_real_digital_twin_content_describes_this_actual_system(ingested_kb):
    kb, _ = ingested_kb
    results = kb.retrieve("what does PPO decide in this system", category="digital_twin", top_k=3)
    assert results
    assert any("ppo" in r.text.lower() for r in results)


def test_real_policies_content_contains_real_enforced_values(ingested_kb):
    kb, _ = ingested_kb
    results = kb.retrieve("verification delta acceptance gate threshold", category="policies", top_k=3)
    assert results
    assert any("0.01" in r.text for r in results)


def test_real_history_content_is_honest_about_being_development_validation(ingested_kb):
    kb, _ = ingested_kb
    results = kb.retrieve("adaptation history record", category="history", top_k=3)
    assert results
    assert any("development" in r.text.lower() or "not a production" in r.text.lower() or "placeholder" in r.text.lower() for r in results)


def test_embeddings_are_genuinely_semantic_not_a_stub(ingested_kb):
    """A real embedding model places semantically similar text closer together than unrelated
    text. If this failed, it would mean the embedding function were a stub (e.g. constant or
    random vectors) rather than genuinely running all-MiniLM-L6-v2."""
    kb, _ = ingested_kb
    similar_results = kb.retrieve("near-RT RIC near-real-time control", category="oran", top_k=1)
    unrelated_results = kb.retrieve("near-RT RIC near-real-time control", category="policies", top_k=1)
    assert similar_results and unrelated_results
    # The on-topic (oran) match must be a closer (smaller distance) match than an off-topic
    # (policies) collection queried with the exact same oran-specific text.
    assert similar_results[0].distance < unrelated_results[0].distance


def test_reingesting_the_real_corpus_twice_is_idempotent(tmp_path):
    ingestor = _real_ingestor(tmp_path)
    corpus_dirs = {category: REPO_ROOT / rel_path for category, rel_path in SETTINGS.rag.corpus_dirs.items()}
    ingestor.ingest_corpus(corpus_dirs)
    first_count = ingestor.count()
    ingestor.ingest_corpus(corpus_dirs)
    second_count = ingestor.count()
    assert first_count == second_count
