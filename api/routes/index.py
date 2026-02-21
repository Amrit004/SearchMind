from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from storage.vector_store import Document
from core.retriever.hybrid import HybridRetriever
import uuid

router = APIRouter()


def get_retriever(request: Request) -> HybridRetriever:
    return HybridRetriever(request.app.state.vector_store, request.app.state.embedder)


class IndexRequest(BaseModel):
    collection: str = "default"
    documents: List[Dict[str, Any]]  # [{id?, text, metadata?}]
    chunk_size: int = 512
    chunk_overlap: int = 64


class IndexResponse(BaseModel):
    indexed: int
    collection: str
    doc_ids: List[str]


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Split text into overlapping chunks for better retrieval."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i: i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    return chunks if chunks else [text]


@router.post("/documents", response_model=IndexResponse)
async def index_documents(req: IndexRequest, retriever: HybridRetriever = Depends(get_retriever)):
    """
    Index documents into a collection.
    Long documents are automatically chunked for better retrieval.
    """
    if not req.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    # Ensure collection exists
    await retriever.vector_store.create_collection("default", retriever.embedder.embedding_dim)
    if req.collection != "default":
        await retriever.vector_store.create_collection(req.collection, retriever.embedder.embedding_dim)

    all_docs = []
    all_ids = []

    for raw_doc in req.documents:
        text = raw_doc.get("text", "")
        if not text.strip():
            continue

        base_id = raw_doc.get("id") or str(uuid.uuid4())
        metadata = raw_doc.get("metadata", {})
        metadata["source"] = raw_doc.get("source", base_id)

        # Chunk long documents
        word_count = len(text.split())
        if word_count > req.chunk_size:
            chunks = chunk_text(text, req.chunk_size, req.chunk_overlap)
            for j, chunk in enumerate(chunks):
                chunk_id = f"{base_id}_chunk_{j}"
                all_docs.append(Document(
                    id=chunk_id,
                    text=chunk,
                    metadata={**metadata, "chunk_index": j, "total_chunks": len(chunks), "parent_id": base_id},
                ))
                all_ids.append(chunk_id)
        else:
            all_docs.append(Document(id=base_id, text=text, metadata=metadata))
            all_ids.append(base_id)

    if not all_docs:
        raise HTTPException(status_code=400, detail="No valid text found in documents")

    await retriever.index_batch(req.collection, all_docs)

    return IndexResponse(indexed=len(all_docs), collection=req.collection, doc_ids=all_ids)


@router.post("/file")
async def index_file(
    collection: str = "default",
    file: UploadFile = File(...),
    request: Request = None,
):
    """Upload and index a text file (PDF, TXT, MD)."""
    retriever = get_retriever(request)
    content = await file.read()

    if file.filename.endswith(".pdf"):
        try:
            import pdfplumber, io
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF parsing failed: {e}")
    else:
        text = content.decode("utf-8", errors="replace")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text content extracted from file")

    doc_id = str(uuid.uuid4())
    doc = Document(id=doc_id, text=text, metadata={"source": file.filename, "type": "file_upload"})

    await retriever.vector_store.create_collection(collection, retriever.embedder.embedding_dim)
    await retriever.index_document(collection, doc)

    return {"indexed": 1, "collection": collection, "filename": file.filename, "doc_id": doc_id}


@router.delete("/documents/{collection}/{doc_id}")
async def delete_document(collection: str, doc_id: str, retriever: HybridRetriever = Depends(get_retriever)):
    success = await retriever.delete_document(collection, doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": doc_id, "collection": collection}
