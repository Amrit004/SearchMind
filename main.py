"""
SearchMind — Distributed Semantic Search Engine
Hybrid BM25 + vector search with RAG (Retrieval-Augmented Generation).
Built with FastAPI, Qdrant/pgvector, and sentence-transformers.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SearchMind starting — initialising indices and embedding models...")
    from core.embeddings.encoder import EmbeddingService
    from storage.vector_store import VectorStore

    app.state.embedder = EmbeddingService()
    await app.state.embedder.load()

    app.state.vector_store = VectorStore()
    await app.state.vector_store.connect()

    logger.info("SearchMind ready.")
    yield
    await app.state.vector_store.close()


app = FastAPI(
    title="SearchMind API",
    description="Distributed Semantic Search Engine with RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routes import search, index, collections, rag

app.include_router(search.router,      prefix="/api/v1/search",      tags=["search"])
app.include_router(index.router,       prefix="/api/v1/index",        tags=["indexing"])
app.include_router(collections.router, prefix="/api/v1/collections",  tags=["collections"])
app.include_router(rag.router,         prefix="/api/v1/rag",          tags=["rag"])


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "searchmind"}
