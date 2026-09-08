"""Unit tests for D2 — RAG Knowledge Base (prompt.md §35-36).

Uses temporary chroma paths + hand-written corpus documents (not the real `rag_data/`) so
chunking, idempotency, metadata, and — most importantly — the read-only structural boundary can
be tested in isolation and fast. `tests/integration/test_rag_kb.py` covers the real corpus and
genuine embedding-quality behavior.
"""

from __future__ import annotations

import inspect

import pytest

from src.dt_models.d1_model_store import D1Store
from src.registry.model_registry import ModelRegistry
from src.rag.rag_kb import RagIngestor, RagKnowledgeBase, RagUnavailableError, chunk_text


# --- chunk_text ------------------------------------------------------------------------------


def test_chunk_text_short_input_is_a_single_chunk():
    assert chunk_text("short text", chunk_size=800, chunk_overlap=100) == ["short text"]


def test_chunk_text_empty_input_yields_no_chunks():
    assert chunk_text("", chunk_size=800, chunk_overlap=100) == []
    assert chunk_text("   \n\n  ", chunk_size=800, chunk_overlap=100) == []


def test_chunk_text_splits_long_input_into_multiple_chunks():
    text = "word " * 500  # 2500 chars
    chunks = chunk_text(text, chunk_size=800, chunk_overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 850 for c in chunks)  # small slack for whitespace-boundary snapping


def test_chunk_text_consecutive_chunks_overlap():
    text = "".join(f"{i:04d} " for i in range(400))  # deterministic, greppable content
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    # the tail of chunk[0] and the head of chunk[1] should share some content
    tail = chunks[0][-30:]
    assert any(tail[:10] in chunks[i] for i in range(1, len(chunks)))


def test_chunk_text_never_splits_a_word_when_a_nearby_space_exists():
    text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima"
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
    words = set(text.split())
    for chunk in chunks:
        for token in chunk.split():
            assert token in words  # never a truncated partial word


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, chunk_overlap=100)


# --- ingestion + retrieval --------------------------------------------------------------------


def _ingestor(tmp_path) -> RagIngestor:
    return RagIngestor(
        chroma_path=tmp_path / "chroma",
        collection_name="test_kb",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=200,
        chunk_overlap=20,
    )


def _kb(tmp_path) -> RagKnowledgeBase:
    return RagKnowledgeBase(
        chroma_path=tmp_path / "chroma", collection_name="test_kb", embedding_model="all-MiniLM-L6-v2", default_top_k=5
    )


def _write_corpus(tmp_path, category: str, filename: str, text: str) -> None:
    directory = tmp_path / "corpus" / category
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(text)


def test_ingest_then_retrieve_round_trip(tmp_path):
    _write_corpus(tmp_path, "policies", "doc1.md", "The verification delta is 0.01 for the acceptance gate.")
    ingestor = _ingestor(tmp_path)
    summary = ingestor.ingest_corpus({"policies": tmp_path / "corpus" / "policies"})

    assert summary.documents_processed == 1
    assert summary.chunks_upserted >= 1

    kb = _kb(tmp_path)
    results = kb.retrieve("what is the verification delta", top_k=1)
    assert len(results) == 1
    assert "0.01" in results[0].text


def test_retrieved_chunk_metadata_is_correct(tmp_path):
    _write_corpus(tmp_path, "oran", "spec.md", "Some O-RAN content about the near-RT RIC.")
    ingestor = _ingestor(tmp_path)
    ingestor.ingest_corpus({"oran": tmp_path / "corpus" / "oran"})

    kb = _kb(tmp_path)
    results = kb.retrieve("near-RT RIC", top_k=1)
    assert results[0].category == "oran"
    assert results[0].document_id == "spec.md"
    assert results[0].chunk_index == 0
    assert len(results[0].version) == 16  # sha256 hex, truncated


def test_source_comment_is_extracted_and_stripped_from_indexed_text(tmp_path):
    _write_corpus(tmp_path, "oran", "spec.md", "<!-- source: https://example.com/real-spec -->\nReal spec content here.")
    ingestor = _ingestor(tmp_path)
    ingestor.ingest_corpus({"oran": tmp_path / "corpus" / "oran"})

    kb = _kb(tmp_path)
    results = kb.retrieve("spec content", top_k=1)
    assert results[0].source == "https://example.com/real-spec"
    assert "<!-- source" not in results[0].text


def test_source_defaults_to_file_path_without_a_source_comment(tmp_path):
    _write_corpus(tmp_path, "policies", "plain.md", "No source comment in this one.")
    ingestor = _ingestor(tmp_path)
    ingestor.ingest_corpus({"policies": tmp_path / "corpus" / "policies"})

    kb = _kb(tmp_path)
    results = kb.retrieve("no source comment", top_k=1)
    assert results[0].source == "plain.md"


def test_reingesting_unchanged_corpus_does_not_duplicate(tmp_path):
    _write_corpus(tmp_path, "policies", "doc1.md", "Stable content that never changes across ingestion runs.")
    ingestor = _ingestor(tmp_path)
    corpus_dirs = {"policies": tmp_path / "corpus" / "policies"}

    ingestor.ingest_corpus(corpus_dirs)
    count_after_first = ingestor.count()
    ingestor.ingest_corpus(corpus_dirs)
    count_after_second = ingestor.count()

    assert count_after_first == count_after_second
    assert count_after_first > 0


def test_reingesting_changed_content_updates_rather_than_duplicates(tmp_path):
    _write_corpus(tmp_path, "policies", "doc1.md", "Original content version one.")
    ingestor = _ingestor(tmp_path)
    corpus_dirs = {"policies": tmp_path / "corpus" / "policies"}
    ingestor.ingest_corpus(corpus_dirs)
    count_after_first = ingestor.count()

    _write_corpus(tmp_path, "policies", "doc1.md", "Updated content version two, completely different text.")
    ingestor.ingest_corpus(corpus_dirs)
    count_after_second = ingestor.count()

    assert count_after_second == count_after_first  # same number of chunks, content overwritten
    kb = _kb(tmp_path)
    results = kb.retrieve("updated content version two", top_k=1)
    assert "version two" in results[0].text
    assert "version one" not in results[0].text


def test_shrinking_a_document_prunes_stale_trailing_chunks(tmp_path):
    long_text = "sentence number " + " ".join(f"item{i}" for i in range(200))
    _write_corpus(tmp_path, "policies", "doc1.md", long_text)
    ingestor = _ingestor(tmp_path)
    corpus_dirs = {"policies": tmp_path / "corpus" / "policies"}
    ingestor.ingest_corpus(corpus_dirs)
    count_when_long = ingestor.count()
    assert count_when_long > 1

    _write_corpus(tmp_path, "policies", "doc1.md", "much shorter now")
    ingestor.ingest_corpus(corpus_dirs)
    count_when_short = ingestor.count()

    assert count_when_short == 1  # only the current chunk remains, old trailing chunks pruned


def test_category_filter_only_returns_matching_category(tmp_path):
    _write_corpus(tmp_path, "oran", "a.md", "unique_oran_marker content about RIC")
    _write_corpus(tmp_path, "policies", "b.md", "unique_policy_marker content about delta")
    ingestor = _ingestor(tmp_path)
    ingestor.ingest_corpus({"oran": tmp_path / "corpus" / "oran", "policies": tmp_path / "corpus" / "policies"})

    kb = _kb(tmp_path)
    oran_results = kb.retrieve("marker content", category="oran", top_k=5)
    assert all(r.category == "oran" for r in oran_results)
    policy_results = kb.retrieve("marker content", category="policies", top_k=5)
    assert all(r.category == "policies" for r in policy_results)


def test_top_k_limits_result_count(tmp_path):
    for i in range(10):
        _write_corpus(tmp_path, "policies", f"doc{i}.md", f"document number {i} about various policy topics")
    ingestor = _ingestor(tmp_path)
    ingestor.ingest_corpus({"policies": tmp_path / "corpus" / "policies"})

    kb = _kb(tmp_path)
    assert len(kb.retrieve("policy topics", top_k=3)) == 3
    assert len(kb.retrieve("policy topics", top_k=7)) == 7


def test_nonexistent_corpus_directory_is_skipped_not_an_error(tmp_path):
    ingestor = _ingestor(tmp_path)
    summary = ingestor.ingest_corpus({"missing": tmp_path / "does_not_exist"})
    assert summary.documents_processed == 0


def test_unsupported_embedding_model_raises_clear_error(tmp_path):
    with pytest.raises(ValueError, match="not supported"):
        RagIngestor(
            chroma_path=tmp_path / "chroma", collection_name="test_kb", embedding_model="some-other-model",
            chunk_size=200, chunk_overlap=20,
        )
    with pytest.raises(ValueError, match="not supported"):
        RagKnowledgeBase(
            chroma_path=tmp_path / "chroma", collection_name="test_kb", embedding_model="some-other-model", default_top_k=5
        )


def test_smoke_test_round_trips_and_cleans_up(tmp_path):
    ingestor = _ingestor(tmp_path)
    assert ingestor.smoke_test() is True
    assert ingestor.count() == 0  # smoke-test record removed afterward


# --- read-only structural boundary (the key requirement) --------------------------------------


_PLAUSIBLE_WRITE_METHOD_NAMES = [
    "add", "write", "insert", "update", "delete", "upsert", "put", "set", "save", "store",
    "ingest", "ingest_corpus", "promote", "register", "modify", "remove", "create", "commit",
    "write_to_d1", "update_state", "update_dt", "register_version",
]


def test_agent_cannot_reach_a_write_path_through_the_knowledge_base(tmp_path):
    """The concrete "write-through path from an agent" test: simulates an agent holding a
    `RagKnowledgeBase` (the only class any agent ever receives — see module docstring) and
    attempts every plausible write-shaped method name, confirming none exist."""
    kb = _kb(tmp_path)

    for method_name in _PLAUSIBLE_WRITE_METHOD_NAMES:
        assert not hasattr(kb, method_name), f"RagKnowledgeBase unexpectedly exposes {method_name!r}"
        with pytest.raises(AttributeError):
            getattr(kb, method_name)(anything="should not exist")  # type: ignore[operator]

    # Every public method actually present must be read-only in intent (`from_settings` is a
    # constructor helper, not a data-mutating operation).
    public_methods = [name for name in dir(kb) if not name.startswith("_") and callable(getattr(kb, name))]
    assert set(public_methods) <= {"retrieve", "count", "from_settings"}


def test_knowledge_base_holds_no_reference_to_dt_state_objects(tmp_path):
    """Structurally impossible, not just undocumented: the object graph reachable from a
    `RagKnowledgeBase` instance contains nothing D1/registry-shaped."""
    kb = _kb(tmp_path)
    forbidden_type_names = {"D1Store", "ModelRegistry", "DTModelRegistry", "DTOrchestrator"}
    for attr_name, value in vars(kb).items():
        assert type(value).__name__ not in forbidden_type_names, (
            f"RagKnowledgeBase.{attr_name} unexpectedly references a {type(value).__name__} — "
            "this would be a structural write-through path into DT state"
        )


def test_rag_knowledge_base_constructor_accepts_no_d1_or_registry_argument():
    """A generic-programming-level proof, not just a runtime instance check: `RagKnowledgeBase`
    literally has no parameter through which a caller could hand it D1/registry access."""
    params = set(inspect.signature(RagKnowledgeBase.__init__).parameters)
    assert not (params & {"d1_store", "model_registry", "dt_model_registry", "registry"})


def test_ingestor_constructor_also_accepts_no_d1_or_registry_argument():
    """Even the write-CAPABLE (to the vector store) admin class has no path to DT state."""
    params = set(inspect.signature(RagIngestor.__init__).parameters)
    assert not (params & {"d1_store", "model_registry", "dt_model_registry", "registry"})


def test_ingestion_never_touches_d1_or_model_registry_state(tmp_path):
    """A real D1Store and a real ModelRegistry are constructed at fixed paths; a real RAG
    ingestion run happens; both are proven byte-for-byte untouched by it."""
    d1_current = tmp_path / "d1" / "current.parquet"
    d1_history = tmp_path / "d1" / "history.parquet"
    d1_quarantine = tmp_path / "d1" / "quarantine.parquet"
    D1Store(current_state_path=d1_current, history_path=d1_history, quarantine_path=d1_quarantine, history_retention_rows=1000)

    registry_index = tmp_path / "models" / "index.json"
    ModelRegistry(models_dir=tmp_path / "models", index_path=registry_index)

    d1_files_before = {p: p.read_bytes() if p.exists() else None for p in (d1_current, d1_history, d1_quarantine)}
    registry_before = registry_index.read_bytes() if registry_index.exists() else None

    _write_corpus(tmp_path, "policies", "doc1.md", "some policy content")
    ingestor = _ingestor(tmp_path)
    ingestor.ingest_corpus({"policies": tmp_path / "corpus" / "policies"})
    assert ingestor.count() > 0  # the ingestion itself genuinely happened

    for path, before in d1_files_before.items():
        after = path.read_bytes() if path.exists() else None
        assert after == before, f"{path} was modified by RAG ingestion — a write-through path into D1 exists"
    registry_after = registry_index.read_bytes() if registry_index.exists() else None
    assert registry_after == registry_before, "ModelRegistry's index was modified by RAG ingestion"


# --- graceful degradation --------------------------------------------------------------------


def test_retrieve_raises_rag_unavailable_error_never_fabricates_when_store_is_broken(tmp_path):
    """prompt.md §61: "RAG unavailable: continue only where RAG is non-critical; never fabricate
    retrieved information." Points the KB at a path containing a file (not a directory), which
    ChromaDB cannot open as a store — this must degrade to a clear exception, never silently
    return an empty/fake result that looks like "nothing relevant was found"."""
    broken_path = tmp_path / "not_a_directory"
    broken_path.write_text("this is a file, not a chroma directory")

    kb = RagKnowledgeBase(chroma_path=broken_path, collection_name="test_kb", embedding_model="all-MiniLM-L6-v2", default_top_k=5)

    assert kb.is_available is False
    with pytest.raises(RagUnavailableError):
        kb.retrieve("anything")
