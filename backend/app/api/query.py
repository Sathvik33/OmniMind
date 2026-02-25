from fastapi import APIRouter
from pydantic import BaseModel
from backend.app.pipelines.query_pipeline import QueryPipeline


router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


@router.post("/query")
def query_data(request: QueryRequest):
    pipeline = QueryPipeline()
    result = pipeline.answer(request.query, request.top_k)

    return result