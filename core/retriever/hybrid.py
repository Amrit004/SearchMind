"""
SearchMind Hybrid Retriever
Combines dense vector search (semantic) + sparse BM25 (keyword) with fusion reranking.
This outperforms either approach alone on most retrieval benchmarks.
"""
import math
import logging
import re
from collections import defaultdict
from typing import List, Optional, Dict, Tuple
import numpy as np
from storage.vector_store import Document, VectorStore
from core.embeddings.encoder import EmbeddingService

logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25 (Best Match 25) sparse retrieval.
    Industry standard for keyword search, used by Elasticsearch internally.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: Dict[str, List[str]] = {}      # doc_id -> tokens
        self._df: Dict[str, int] = defaultdict(int)  # term -> doc frequency
        self._avgdl: float = 0.0
        self._N: int = 0

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]

    def add(self, doc_id: str, text: str):
        tokens = self._tokenize(text)
        self._docs[doc_id] = tokens
        for term in set(tokens):
            self._df[term] += 1
        self._N += 1
        total_tokens = sum(len(t) for t in self._docs.values())
        self._avgdl = total_tokens / self._N

    def remove(self, doc_id: str) -> bool:
        if doc_id not in self._docs:
            return False
        tokens = self._docs.pop(doc_id)
        for term in set(tokens):
            self._df[term] = max(0, self._df[term] - 1)
        self._N = max(0, self._N - 1)
        return True

    def score(self, query: str, doc_id: str) -> float:
        if doc_id not in self._docs or self._N == 0:
            return 0.0
        query_terms = self._tokenize(query)
        doc_tokens = self._docs[doc_id]
        doc_len = len(doc_tokens)
        tf_map: Dict[str, int] = defaultdict(int)
        for t in doc_tokens:
            tf_map[t] += 1

        score = 0.0
        for term in query_terms:
            if term not in tf_map:
                continue
            tf = tf_map[term]
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avgdl, 1))
            )
            score += idf * tf_norm
        return score

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Return top-k (doc_id, bm25_score) pairs."""
        scores = {}
        for doc_id in self._docs:
            s = self.score(query, doc_id)
            if s > 0:
                scores[doc_id] = s
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class HybridRetriever:
    """
    Hybrid retrieval with Reciprocal Rank Fusion (RRF).
    Combines semantic (dense) and keyword (sparse) results.
    Optionally reranks with a cross-encoder for highest precision.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingService,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self._bm25_indices: Dict[str, BM25Index] = {}
        self._doc_store: Dict[str, Dict[str, Document]] = {}  # collection -> {id: doc}
        self._reranker = None

    def _get_bm25(self, collection: str) -> BM25Index:
        if collection not in self._bm25_indices:
            self._bm25_indices[collection] = BM25Index()
        return self._bm25_indices[collection]

    async def index_document(self, collection: str, doc: Document):
        """Index document in both BM25 and vector store."""
        if doc.vector is None:
            doc.vector = await self.embedder.encode(doc.text)

        # BM25 index
        bm25 = self._get_bm25(collection)
        bm25.add(doc.id, doc.text)

        # Vector index
        await self.vector_store.upsert(collection, [doc])

        # Document store (for retrieval of full content)
        if collection not in self._doc_store:
            self._doc_store[collection] = {}
        self._doc_store[collection][doc.id] = doc

    async def index_batch(self, collection: str, docs: List[Document]):
        """Efficiently index multiple documents in batch."""
        # Batch encode all texts
        texts_needing_vectors = [d for d in docs if d.vector is None]
        if texts_needing_vectors:
            vectors = await self.embedder.encode_batch([d.text for d in texts_needing_vectors])
            for doc, vec in zip(texts_needing_vectors, vectors):
                doc.vector = vec

        # BM25 index
        bm25 = self._get_bm25(collection)
        for doc in docs:
            bm25.add(doc.id, doc.text)

        # Vector store (batch upsert)
        await self.vector_store.upsert(collection, docs)

        # Document store
        if collection not in self._doc_store:
            self._doc_store[collection] = {}
        for doc in docs:
            self._doc_store[collection][doc.id] = doc

        logger.info(f"Indexed {len(docs)} documents in '{collection}'")

    async def search(
        self,
        collection: str,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,  # 0 = pure BM25, 1 = pure vector
        filter_conditions: Optional[Dict] = None,
        use_reranker: bool = False,
    ) -> List[Document]:
        """
        Hybrid search with RRF fusion.
        alpha controls the balance: 0.5 = equal weight, 0.7 = favour semantic.
        """
        query_vector = await self.embedder.encode(query)
        candidates = min(top_k * 10, 200)

        # Run both retrievers in parallel
        vector_results, bm25_results = await asyncio.gather(
            self.vector_store.search(collection, query_vector, candidates, filter_conditions),
            asyncio.to_thread(self._get_bm25(collection).search, query, candidates),
        ) if False else (  # asyncio.gather fallback for sync BM25
            await self.vector_store.search(collection, query_vector, candidates, filter_conditions),
            self._get_bm25(collection).search(query, candidates),
        )

        # RRF fusion
        rrf_scores: Dict[str, float] = defaultdict(float)

        # Vector results contribute with weight alpha
        for rank, doc in enumerate(vector_results):
            rrf_scores[doc.id] += alpha * (1.0 / (self.rrf_k + rank + 1))

        # BM25 results contribute with weight (1 - alpha)
        for rank, (doc_id, _) in enumerate(bm25_results):
            rrf_scores[doc_id] += (1 - alpha) * (1.0 / (self.rrf_k + rank + 1))

        # Sort by fused score
        ranked_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Hydrate documents
        store = self._doc_store.get(collection, {})
        results = []
        for doc_id, score in ranked_ids:
            doc = store.get(doc_id)
            if doc:
                results.append(Document(
                    id=doc.id, text=doc.text, metadata=doc.metadata, score=score
                ))

        # Optional cross-encoder reranking for highest precision
        if use_reranker and results:
            results = await self._rerank(query, results)

        return results

    async def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """Cross-encoder reranking for final precision boost."""
        try:
            from sentence_transformers import CrossEncoder
            if self._reranker is None:
                loop = __import__("asyncio").get_event_loop()
                self._reranker = await loop.run_in_executor(
                    None, lambda: CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                )
            pairs = [(query, doc.text) for doc in docs]
            loop = __import__("asyncio").get_event_loop()
            scores = await loop.run_in_executor(
                None, lambda: self._reranker.predict(pairs)
            )
            for doc, score in zip(docs, scores):
                doc.score = float(score)
            docs.sort(key=lambda d: d.score, reverse=True)
        except Exception as e:
            logger.warning(f"Reranker failed: {e}")
        return docs

    async def delete_document(self, collection: str, doc_id: str) -> bool:
        bm25 = self._get_bm25(collection)
        bm25.remove(doc_id)
        self._doc_store.get(collection, {}).pop(doc_id, None)
        return await self.vector_store.delete(collection, doc_id)


import asyncio
