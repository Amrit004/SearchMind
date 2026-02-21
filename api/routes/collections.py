from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List

router = APIRouter()


class CollectionCreate(BaseModel):
    name: str
    dim: int = 384
    description: str = ""


@router.post("/")
async def create_collection(req: CollectionCreate, request: Request):
    dim = req.dim or request.app.state.embedder.embedding_dim
    success = await request.app.state.vector_store.create_collection(req.name, dim)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to create collection")
    return {"created": req.name, "dim": dim}


@router.get("/")
async def list_collections(request: Request):
    names = await request.app.state.vector_store.list_collections()
    return {"collections": names, "count": len(names)}


@router.get("/{name}/stats")
async def collection_stats(name: str, request: Request):
    count = await request.app.state.vector_store.count(name)
    return {"collection": name, "document_count": count}
