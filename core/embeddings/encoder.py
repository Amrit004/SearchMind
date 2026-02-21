"""
SearchMind Embedding Service
Converts text to dense vector representations using sentence-transformers.
Supports multiple models, batching, and Redis caching for repeated queries.
"""
import asyncio
import hashlib
import json
import logging
import numpy as np
from typing import List, Optional, Union
import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Available embedding models with their dimensions
MODELS = {
    "all-MiniLM-L6-v2":       {"dim": 384,  "speed": "fast",   "quality": "good"},
    "all-mpnet-base-v2":       {"dim": 768,  "speed": "medium", "quality": "better"},
    "multi-qa-mpnet-base-dot-v1": {"dim": 768, "speed": "medium", "quality": "best_retrieval"},
    "BAAI/bge-large-en-v1.5": {"dim": 1024, "speed": "slow",   "quality": "best"},
}


class EmbeddingService:
    """
    Wraps sentence-transformers with:
    - Batched encoding for throughput
    - Redis cache to avoid re-encoding identical strings
    - Async interface
    - Fallback to random vectors if model unavailable (testing)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_url: str = "redis://localhost:6379/3",
        cache_ttl: int = 86400,  # 24h
    ):
        self.model_name = model_name
        self.dim = MODELS.get(model_name, {}).get("dim", 384)
        self.cache_url = cache_url
        self.cache_ttl = cache_ttl
        self._model = None
        self._cache: Optional[redis.Redis] = None

    async def load(self):
        """Lazy-load model and connect cache."""
        try:
            from sentence_transformers import SentenceTransformer
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None, lambda: SentenceTransformer(self.model_name)
            )
            logger.info(f"Embedding model loaded: {self.model_name} (dim={self.dim})")
        except ImportError:
            logger.warning("sentence-transformers not installed. Using random embeddings (testing only).")

        try:
            self._cache = redis.from_url(self.cache_url, decode_responses=False)
            await self._cache.ping()
            logger.info("Embedding cache connected")
        except Exception as e:
            logger.warning(f"Redis cache unavailable: {e}. Running without cache.")
            self._cache = None

    async def encode(self, text: str) -> np.ndarray:
        """Encode a single text string to a vector."""
        vectors = await self.encode_batch([text])
        return vectors[0]

    async def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Encode a batch of texts, using cache where possible."""
        results: dict[int, np.ndarray] = {}
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cached = await self._get_cached(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Encode uncached texts
        if uncached_texts:
            vectors = await self._encode_raw(uncached_texts)
            for idx, (orig_idx, text, vec) in enumerate(
                zip(uncached_indices, uncached_texts, vectors)
            ):
                results[orig_idx] = vec
                await self._set_cached(text, vec)

        return [results[i] for i in range(len(texts))]

    async def _encode_raw(self, texts: List[str]) -> List[np.ndarray]:
        """Actually encode texts using the model."""
        if self._model is None:
            # Fallback: deterministic random vectors for testing
            return [
                np.random.RandomState(
                    int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
                ).randn(self.dim).astype(np.float32)
                for t in texts
            ]

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
            ),
        )
        return [embeddings[i] for i in range(len(texts))]

    async def _get_cached(self, text: str) -> Optional[np.ndarray]:
        if self._cache is None:
            return None
        key = self._cache_key(text)
        try:
            data = await self._cache.get(key)
            if data:
                return np.frombuffer(data, dtype=np.float32)
        except Exception:
            pass
        return None

    async def _set_cached(self, text: str, vector: np.ndarray):
        if self._cache is None:
            return
        key = self._cache_key(text)
        try:
            await self._cache.setex(key, self.cache_ttl, vector.astype(np.float32).tobytes())
        except Exception:
            pass

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(f"{self.model_name}:{text}".encode()).hexdigest()
        return f"embed:{digest}"

    @property
    def embedding_dim(self) -> int:
        return self.dim
