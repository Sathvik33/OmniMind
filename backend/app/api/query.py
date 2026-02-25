from fastapi import APIRouter
from pydantic import BaseModel
from backend.app.pipelines.query_pipeline import QueryPipeline


router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


@router.post("/retrieve")
def retrieve_data(request: QueryRequest):
    pipeline = QueryPipeline()
    results = pipeline.retrieve(request.query, request.top_k)

    return {
        "query": request.query,
        "results": results
    }