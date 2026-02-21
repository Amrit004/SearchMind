from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from storage.vector_store import Document
from core.retriever.hybrid import HybridRetriever
from core.retriever.rag import RAGEngine, OllamaBackend
import uuid

router = APIRouter()


def get_retriever(request: Request) -> HybridRetriever:
    return HybridRetriever(request.app.state.vector_store, request.app.state.embedder)


class SearchRequest(BaseModel):
    query: str
    collection: str = "default"
    top_k: int = 10
    alpha: float = 0.5  # 0=BM25, 1=vector
    filter: Optional[Dict[str, Any]] = None
    use_reranker: bool = False


class SearchResult(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int
    query: str
    collection: str


@router.post("/", response_model=SearchResponse)
async def hybrid_search(req: SearchRequest, retriever: HybridRetriever = Depends(get_retriever)):
    """
    Hybrid BM25 + semantic search with optional reranking.
    alpha=0 → pure keyword, alpha=1 → pure semantic, alpha=0.5 → balanced
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        docs = await retriever.search(
            collection=req.collection,
            query=req.query,
            top_k=req.top_k,
            alpha=req.alpha,
            filter_conditions=req.filter,
            use_reranker=req.use_reranker,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    return SearchResponse(
        results=[
            SearchResult(id=d.id, text=d.text, score=round(d.score, 4), metadata=d.metadata)
            for d in docs
        ],
        total=len(docs),
        query=req.query,
        collection=req.collection,
    )


router_rag = APIRouter()


class RAGRequest(BaseModel):
    question: str
    collection: str = "default"
    top_k: int = 5
    alpha: float = 0.6
    use_reranker: bool = False
    system_prompt: str = ""
    llm_backend: str = "ollama"  # ollama, openai, anthropic
    filter: Optional[Dict] = None


class RAGResponseModel(BaseModel):
    answer: str
    sources: List[SearchResult]
    query: str
    model: str
    retrieval_ms: float
    generation_ms: float


@router_rag.post("/query", response_model=RAGResponseModel)
async def rag_query(req: RAGRequest, request: Request):
    """Ask a question and get an AI-generated answer grounded in your documents."""
    retriever = get_retriever(request)

    backends = {
        "ollama": OllamaBackend(),
    }
    # OpenAI/Anthropic require API keys — add via env vars
    try:
        import os
        if req.llm_backend == "openai" and os.getenv("OPENAI_API_KEY"):
            from core.retriever.rag import OpenAIBackend
            backends["openai"] = OpenAIBackend(api_key=os.getenv("OPENAI_API_KEY", ""))
        elif req.llm_backend == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            from core.retriever.rag import AnthropicBackend
            backends["anthropic"] = AnthropicBackend(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    except Exception:
        pass

    llm = backends.get(req.llm_backend, backends["ollama"])
    engine = RAGEngine(retriever, llm, top_k=req.top_k, alpha=req.alpha)

    try:
        result = await engine.query(
            collection=req.collection,
            question=req.question,
            use_reranker=req.use_reranker,
            system_prompt=req.system_prompt,
            filter_conditions=req.filter,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return RAGResponseModel(
        answer=result.answer,
        sources=[
            SearchResult(id=d.id, text=d.text[:300] + "..." if len(d.text) > 300 else d.text,
                         score=round(d.score, 4), metadata=d.metadata)
            for d in result.sources
        ],
        query=result.query,
        model=result.model,
        retrieval_ms=result.retrieval_ms,
        generation_ms=result.generation_ms,
    )


@router_rag.post("/stream")
async def rag_stream(req: RAGRequest, request: Request):
    """Stream RAG response via Server-Sent Events."""
    retriever = get_retriever(request)
    engine = RAGEngine(retriever, OllamaBackend(), top_k=req.top_k)

    return StreamingResponse(
        engine.stream_query(req.collection, req.question, req.top_k),
        media_type="text/event-stream",
    )
