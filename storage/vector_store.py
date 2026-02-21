"""
SearchMind Vector Store
Manages vector collections in Qdrant with automatic index creation.
Falls back to in-memory HNSW index if Qdrant is unavailable.
"""
import asyncio
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import uuid

logger = logging.getLogger(__name__)


@dataclass
class Document:
    id: str
    text: str
    vector: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class Collection:
    name: str
    dim: int
    doc_count: int = 0
    created_at: str = ""


class InMemoryVectorIndex:
    """
    Brute-force cosine similarity index for development/testing.
    Replace with Qdrant or pgvector in production.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._ids: List[str] = []
        self._vectors: List[np.ndarray] = []
        self._texts: List[str] = []
        self._metadata: List[Dict] = []

    def add(self, doc_id: str, vector: np.ndarray, text: str, metadata: Dict):
        self._ids.append(doc_id)
        self._vectors.append(vector / (np.linalg.norm(vector) + 1e-10))
        self._texts.append(text)
        self._metadata.append(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float, str, Dict]]:
        if not self._vectors:
            return []
        q = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        matrix = np.stack(self._vectors)
        scores = matrix @ q
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self._ids[i], float(scores[i]), self._texts[i], self._metadata[i])
            for i in top_indices
        ]

    def delete(self, doc_id: str) -> bool:
        try:
            idx = self._ids.index(doc_id)
            self._ids.pop(idx)
            self._vectors.pop(idx)
            self._texts.pop(idx)
            self._metadata.pop(idx)
            return True
        except ValueError:
            return False

    def count(self) -> int:
        return len(self._ids)


class VectorStore:
    """
    Multi-collection vector store backed by Qdrant.
    Falls back to in-memory index for development.
    """

    def __init__(self, host: str = "localhost", port: int = 6333):
        self.host = host
        self.port = port
        self._client = None
        self._in_memory: Dict[str, InMemoryVectorIndex] = {}
        self._use_qdrant = False

    async def connect(self):
        try:
            from qdrant_client import AsyncQdrantClient
            self._client = AsyncQdrantClient(host=self.host, port=self.port)
            await self._client.get_collections()  # Test connection
            self._use_qdrant = True
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
        except Exception as e:
            logger.warning(f"Qdrant unavailable ({e}). Using in-memory vector index.")
            self._use_qdrant = False

    async def close(self):
        if self._client:
            await self._client.close()

    async def create_collection(self, name: str, dim: int) -> bool:
        if self._use_qdrant:
            from qdrant_client.models import Distance, VectorParams
            try:
                await self._client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
                logger.info(f"Qdrant collection created: {name} (dim={dim})")
                return True
            except Exception as e:
                logger.error(f"Failed to create collection {name}: {e}")
                return False
        else:
            self._in_memory[name] = InMemoryVectorIndex(dim)
            logger.info(f"In-memory collection created: {name} (dim={dim})")
            return True

    async def upsert(self, collection: str, documents: List[Document]) -> int:
        if not documents:
            return 0

        if self._use_qdrant:
            from qdrant_client.models import PointStruct
            points = [
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.id)),
                    vector=doc.vector.tolist(),
                    payload={
                        "original_id": doc.id,
                        "text": doc.text,
                        **doc.metadata,
                    },
                )
                for doc in documents
                if doc.vector is not None
            ]
            await self._client.upsert(collection_name=collection, points=points)
            return len(points)
        else:
            if collection not in self._in_memory:
                dim = len(documents[0].vector) if documents[0].vector is not None else 384
                self._in_memory[collection] = InMemoryVectorIndex(dim)
            index = self._in_memory[collection]
            for doc in documents:
                if doc.vector is not None:
                    index.add(doc.id, doc.vector, doc.text, doc.metadata)
            return len(documents)

    async def search(
        self,
        collection: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_conditions: Optional[Dict] = None,
    ) -> List[Document]:
        if self._use_qdrant:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            qfilter = None
            if filter_conditions:
                conditions = [
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filter_conditions.items()
                ]
                qfilter = Filter(must=conditions)

            results = await self._client.search(
                collection_name=collection,
                query_vector=query_vector.tolist(),
                limit=top_k,
                query_filter=qfilter,
                with_payload=True,
            )
            return [
                Document(
                    id=r.payload.get("original_id", str(r.id)),
                    text=r.payload.get("text", ""),
                    metadata={k: v for k, v in r.payload.items() if k not in ("original_id", "text")},
                    score=r.score,
                )
                for r in results
            ]
        else:
            if collection not in self._in_memory:
                return []
            raw = self._in_memory[collection].search(query_vector, top_k)
            return [
                Document(id=doc_id, text=text, metadata=meta, score=score)
                for doc_id, score, text, meta in raw
            ]

    async def delete(self, collection: str, doc_id: str) -> bool:
        if self._use_qdrant:
            from qdrant_client.models import PointIdsList
            await self._client.delete(
                collection_name=collection,
                points_selector=PointIdsList(
                    points=[str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))]
                ),
            )
            return True
        elif collection in self._in_memory:
            return self._in_memory[collection].delete(doc_id)
        return False

    async def count(self, collection: str) -> int:
        if self._use_qdrant:
            info = await self._client.get_collection(collection)
            return info.points_count
        elif collection in self._in_memory:
            return self._in_memory[collection].count()
        return 0

    async def list_collections(self) -> List[str]:
        if self._use_qdrant:
            response = await self._client.get_collections()
            return [c.name for c in response.collections]
        return list(self._in_memory.keys())
