"""StudyRAG Vector Store Module.

Manages local embeddings generation using Sentence Transformers and high-speed
vector similarity search with FAISS CPU. Supports document filtering, metadata
provenance, and safe document removal with index rebuilding.
"""

from __future__ import annotations

import os
# Prevent OpenMP duplicate runtime initialization conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Prevent network timeouts when running in offline/sandboxed environments
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.text_chunker import TextChunk
from src.pdf_processor import ProcessedDocument


class VectorStore:
    """FAISS-based vector database with metadata mapping and document isolation."""

    _embedder_instance: Optional[SentenceTransformer] = None
    _embedder_model_name: Optional[str] = None

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.embedder = self._get_embedder(embedding_model_name)
        dim_method = getattr(self.embedder, "get_embedding_dimension", None) or getattr(
            self.embedder, "get_sentence_embedding_dimension", None
        )
        self.dimension = dim_method() if dim_method else 384

        # FAISS Index using Inner Product (cosine similarity after L2 normalization)
        self.index = faiss.IndexFlatIP(self.dimension)

        # Chunks and documents tracking
        self.chunks: List[TextChunk] = []
        self.documents: Dict[str, ProcessedDocument] = {}
        self.chunk_metadata: List[Dict[str, Any]] = []

    @classmethod
    def _get_embedder(cls, model_name: str) -> SentenceTransformer:
        """Cache and retrieve SentenceTransformer singleton.

        Attempts loading from local cache first to avoid network latency and offline timeouts.
        """
        if cls._embedder_instance is None or cls._embedder_model_name != model_name:
            try:
                cls._embedder_instance = SentenceTransformer(model_name, local_files_only=True)
            except Exception:
                # Fallback to standard online download if not yet locally cached
                cls._embedder_instance = SentenceTransformer(model_name)
            cls._embedder_model_name = model_name
        return cls._embedder_instance

    @property
    def total_chunks(self) -> int:
        """Total number of indexed chunks."""
        return self.index.ntotal

    @property
    def total_documents(self) -> int:
        """Total number of active documents."""
        return len(self.documents)

    def is_document_indexed(self, doc_id: str) -> bool:
        """Check if a document is already in the vector store."""
        return doc_id in self.documents

    def add_document(self, doc: ProcessedDocument, chunks: List[TextChunk]) -> int:
        """Ingest document chunks and their embeddings into the FAISS index.

        Args:
            doc: ProcessedDocument metadata.
            chunks: List of TextChunks from this document.

        Returns:
            Number of newly indexed chunks.
        """
        if not chunks:
            return 0

        if self.is_document_indexed(doc.doc_id):
            return 0  # Avoid duplicate ingestion

        # Extract texts and compute embeddings
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # Add to FAISS index
        self.index.add(embeddings)

        # Update metadata arrays
        self.documents[doc.doc_id] = doc
        self.chunks.extend(chunks)
        for chunk in chunks:
            self.chunk_metadata.append(chunk.to_dict())

        return len(chunks)

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document and completely rebuild the FAISS index.

        Rebuilding guarantees deleted document chunks never leak into future searches.
        """
        if doc_id not in self.documents:
            return False

        # Remove from documents dict
        del self.documents[doc_id]

        # Retain chunks from remaining documents
        remaining_chunks = [c for c in self.chunks if c.doc_id != doc_id]

        # Re-initialize index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        self.chunk_metadata = []

        if remaining_chunks:
            texts = [c.text for c in remaining_chunks]
            embeddings = self.embedder.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)

            self.index.add(embeddings)
            self.chunks = remaining_chunks
            self.chunk_metadata = [c.to_dict() for c in remaining_chunks]

        return True

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve the most relevant chunks for a user query.

        Args:
            query: The question or query string.
            top_k: Number of relevant chunks to retrieve.
            doc_filter: Optional list of allowed doc_ids.

        Returns:
            List of dictionaries containing chunk text, source metadata, and similarity score.
        """
        if self.total_chunks == 0 or not query.strip():
            return []

        # Encode query and normalize for cosine similarity
        query_vec = self.embedder.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # Retrieve a broader pool if filtering is applied
        search_k = min(self.total_chunks, top_k * 4 if doc_filter else top_k)
        scores, indices = self.index.search(query_vec, search_k)

        results: List[Dict[str, Any]] = []
        seen_chunks = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_metadata):
                continue

            meta = self.chunk_metadata[idx]
            doc_id = meta["doc_id"]
            chunk_id = meta["chunk_id"]

            # Filter by document if specified
            if doc_filter is not None and doc_id not in doc_filter:
                continue

            if chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)

            res_item = dict(meta)
            res_item["score"] = float(score)
            results.append(res_item)

            if len(results) >= top_k:
                break

        return results

    def clear(self) -> None:
        """Clear all indexed documents and vectors."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks.clear()
        self.documents.clear()
        self.chunk_metadata.clear()

    def save_to_disk(self, save_dir: Path | str) -> None:
        """Persist FAISS index and all chunk/document metadata to disk."""
        target_dir = Path(save_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        index_path = target_dir / "index.faiss"
        meta_path = target_dir / "metadata.json"

        # 1. Write FAISS index
        faiss.write_index(self.index, str(index_path))

        # 2. Serialize documents and chunks
        payload = {
            "documents": {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()},
            "chunks": [c.to_dict() for c in self.chunks],
            "chunk_metadata": self.chunk_metadata,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def load_from_disk(self, save_dir: Path | str) -> bool:
        """Restore FAISS index and all chunk/document metadata from disk."""
        target_dir = Path(save_dir)
        index_path = target_dir / "index.faiss"
        meta_path = target_dir / "metadata.json"

        if not index_path.exists() or not meta_path.exists():
            return False

        try:
            # 1. Read FAISS index
            self.index = faiss.read_index(str(index_path))

            # 2. Restore documents and chunks
            with open(meta_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.documents = {
                doc_id: ProcessedDocument.from_dict(doc_data)
                for doc_id, doc_data in payload.get("documents", {}).items()
            }
            self.chunks = [
                TextChunk.from_dict(c_data)
                for c_data in payload.get("chunks", [])
            ]
            self.chunk_metadata = payload.get("chunk_metadata", [c.to_dict() for c in self.chunks])
            return True
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error(f"Failed to load vector store from {save_dir}: {exc}")
            return False

    def save_for_user(self, user_id: str) -> None:
        """Save this vector store in the user's isolated storage directory."""
        from config import USERS_DIR
        user_dir = USERS_DIR / user_id / "vector_store"
        self.save_to_disk(user_dir)

    def load_for_user(self, user_id: str) -> bool:
        """Load vector store from the user's isolated storage directory."""
        from config import USERS_DIR
        user_dir = USERS_DIR / user_id / "vector_store"
        return self.load_from_disk(user_dir)
