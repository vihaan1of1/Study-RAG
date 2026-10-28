"""Tests for Vector Store Disk and User Persistence.

Validates that when a user uploads documents and signs out, the vector index,
document metadata, and text chunks are persisted and restored upon signing back in.
"""

import pytest
from src.pdf_processor import ProcessedDocument, ProcessedPage
from src.text_chunker import TextChunk
from src.vector_store import VectorStore


def test_vector_store_disk_persistence(tmp_path):
    """Verify that VectorStore saves to disk and restores completely."""
    save_dir = tmp_path / "user_vstore"

    # 1. Create and populate store
    store1 = VectorStore()
    doc = ProcessedDocument(
        doc_id="doc_persistent_1",
        filename="lecture_rag.pdf",
        file_size_bytes=4096,
        total_pages=1,
        pages=[ProcessedPage(1, "RAG systems combine retrieval and generative models.", 49)],
    )
    chunks = [
        TextChunk(
            chunk_id="chunk_1",
            doc_id="doc_persistent_1",
            filename="lecture_rag.pdf",
            page_number=1,
            text="RAG systems combine retrieval and generative models.",
            chunk_index=0,
            char_count=49,
        )
    ]
    store1.add_document(doc, chunks)
    assert store1.total_documents == 1
    assert store1.total_chunks == 1

    # 2. Save to disk
    store1.save_to_disk(save_dir)
    assert (save_dir / "index.faiss").exists()
    assert (save_dir / "metadata.json").exists()

    # 3. Create a fresh, empty VectorStore and restore from disk
    store2 = VectorStore()
    assert store2.total_documents == 0
    assert store2.total_chunks == 0

    success = store2.load_from_disk(save_dir)
    assert success is True
    assert store2.total_documents == 1
    assert store2.total_chunks == 1
    assert "doc_persistent_1" in store2.documents
    assert store2.documents["doc_persistent_1"].filename == "lecture_rag.pdf"

    # 4. Search on restored store
    results = store2.search("What do RAG systems combine?", top_k=1)
    assert len(results) == 1
    assert "retrieval and generative models" in results[0]["text"]
