"""
FAISS Vector Store with persistence, metadata mapping, and similarity search.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from app.models.schemas import DocumentChunk, RetrievedChunk
from app.retrieval.embeddings import EmbeddingEngine
from app.utils.config import INDEX_DIR, logger

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FaissVectorStore:
    """
    In-memory and persisted FAISS vector store for semantic retrieval of paper chunks.
    Supports IndexFlatIP for exact cosine similarity of normalized vectors.
    """

    def __init__(self, embedding_engine: Optional[EmbeddingEngine] = None, index_name: str = "default"):
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.index_name = index_name
        self.dimension = self.embedding_engine.dimension
        self.chunks: List[DocumentChunk] = []
        self.index = None
        self._numpy_vectors = None  # fallback matrix if FAISS is unavailable
        self._init_index()

    def _init_index(self) -> None:
        """Initializes empty FAISS index or numpy fallback."""
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = None
            self._numpy_vectors = np.zeros((0, self.dimension), dtype=np.float32)

    def count(self) -> int:
        return len(self.chunks)

    def add_chunks(self, new_chunks: List[DocumentChunk]) -> int:
        """Embeds and indexes a list of DocumentChunks."""
        if not new_chunks:
            return 0

        # Filter out chunks already present by chunk_id
        existing_ids = {c.chunk_id for c in self.chunks}
        unique_chunks = [c for c in new_chunks if c.chunk_id not in existing_ids]

        if not unique_chunks:
            return 0

        texts = [c.text for c in unique_chunks]
        embeddings = self.embedding_engine.encode(texts)

        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(embeddings)
        else:
            if self._numpy_vectors is None or len(self._numpy_vectors) == 0:
                self._numpy_vectors = embeddings
            else:
                self._numpy_vectors = np.vstack([self._numpy_vectors, embeddings])

        self.chunks.extend(unique_chunks)
        logger.info(f"Indexed {len(unique_chunks)} new chunks. Total in store: {len(self.chunks)}")
        return len(unique_chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
        filter_doc_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Executes semantic search for a query.
        Returns sorted list of RetrievedChunk objects above min_similarity.
        """
        if not self.chunks or (self.index is None and (self._numpy_vectors is None or len(self._numpy_vectors) == 0)):
            return []

        query_vector = self.embedding_engine.encode([query], is_query=True)
        search_k = min(top_k * 3, len(self.chunks))  # Retrieve larger pool for filtering

        if FAISS_AVAILABLE and self.index is not None:
            scores, indices = self.index.search(query_vector, search_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Fallback exact dot product (cosine similarity since normalized)
            scores = np.dot(self._numpy_vectors, query_vector[0])
            indices = np.argsort(scores)[::-1][:search_k]
            scores = scores[indices]

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.chunks):
                continue
            if score < min_similarity:
                continue

            chunk = self.chunks[idx]
            if filter_doc_ids and chunk.document_id not in filter_doc_ids:
                continue

            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    similarity_score=float(score),
                )
            )
            if len(results) >= top_k:
                break

        return results

    def save(self, directory: Optional[Path] = None) -> None:
        """Persists the FAISS index and chunk metadata to disk."""
        target_dir = directory or INDEX_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        faiss_path = target_dir / f"{self.index_name}.faiss"
        meta_path = target_dir / f"{self.index_name}_meta.json"

        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, str(faiss_path))
        elif self._numpy_vectors is not None:
            np.save(str(target_dir / f"{self.index_name}_vecs.npy"), self._numpy_vectors)

        chunks_data = [c.model_dump(mode="json") for c in self.chunks]
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2)
        logger.info(f"Saved vector index to {target_dir} ({len(self.chunks)} chunks)")

    def load(self, directory: Optional[Path] = None) -> bool:
        """Loads a saved FAISS index and chunk metadata from disk."""
        target_dir = directory or INDEX_DIR
        faiss_path = target_dir / f"{self.index_name}.faiss"
        meta_path = target_dir / f"{self.index_name}_meta.json"
        vecs_path = target_dir / f"{self.index_name}_vecs.npy"

        if not meta_path.exists():
            return False

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
                self.chunks = [DocumentChunk(**c) for c in chunks_data]

            if FAISS_AVAILABLE and faiss_path.exists():
                self.index = faiss.read_index(str(faiss_path))
            elif vecs_path.exists():
                self._numpy_vectors = np.load(str(vecs_path))

            logger.info(f"Loaded vector store with {len(self.chunks)} chunks from {target_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to load vector store from {target_dir}: {e}")
            return False

    def clear(self) -> None:
        """Resets the vector index and chunks list."""
        self.chunks.clear()
        self._init_index()
        logger.info("Vector store reset.")
