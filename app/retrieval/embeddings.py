"""
Modular embedding engine supporting sentence-transformers and lightweight local fallback.
"""

import os
import hashlib
import numpy as np
from typing import List, Optional, Union
from app.utils.config import logger


class EmbeddingEngine:
    """
    Generates normalized dense embeddings for research paper chunks and queries.
    Uses sentence-transformers with automatic fallback to high-speed deterministic TF-IDF embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._st_model = None
        self._dimension = 384
        self._fallback_mode = False
        self._init_model()

    def _init_model(self) -> None:
        """Initializes sentence-transformers model if available, else activates local fallback."""
        try:
            from sentence_transformers import SentenceTransformer
            # Disable noisy symlink warnings on mac
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            logger.info(f"Loading SentenceTransformer: {self.model_name}...")
            self._st_model = SentenceTransformer(self.model_name)
            if hasattr(self._st_model, "get_embedding_dimension"):
                self._dimension = self._st_model.get_embedding_dimension()
            else:
                self._dimension = self._st_model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded. Dimension: {self._dimension}")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ({e}). Falling back to fast local TF-IDF vectorizer.")
            self._fallback_mode = True
            self._dimension = 384

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Union[str, List[str]], is_query: bool = False) -> np.ndarray:
        """
        Encodes a single text or list of texts into L2-normalized float32 numpy embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)

        if not self._fallback_mode and self._st_model is not None:
            try:
                embeddings = self._st_model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=32,
                )
                return embeddings.astype(np.float32)
            except Exception as e:
                logger.warning(f"SentenceTransformer encoding failed: {e}. Using fallback vectorizer.")

        # Deterministic local TF-IDF / N-gram hashing vectorizer (always fast, offline, 0 dependencies)
        return self._fallback_encode(texts)

    def _fallback_encode(self, texts: List[str]) -> np.ndarray:
        """Lightweight high-speed pseudo-dense semantic hashing embedding (fallback)."""
        embeddings = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            words = text.lower().split()
            if not words:
                continue
            vec = np.zeros(self._dimension, dtype=np.float32)
            for w in words:
                # Hash word into dimension slots
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                idx = h % self._dimension
                sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
                vec[idx] += sign
            # Character trigram features for subword sensitivity
            for j in range(len(text) - 2):
                tri = text[j:j+3].lower()
                h_tri = int(hashlib.md5(tri.encode("utf-8")).hexdigest(), 16)
                idx_tri = h_tri % self._dimension
                vec[idx_tri] += 0.5
            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
            embeddings[i] = vec
        return embeddings
