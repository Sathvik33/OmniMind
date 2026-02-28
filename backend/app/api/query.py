from fastapi import APIRouter
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from backend.app.pipelines.query_pipeline import QueryPipeline

router = APIRouter()
pipeline = QueryPipeline()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


@router.post("/query")
def query_data(request: QueryRequest):
    return pipeline.answer(request.query, request.top_k)


@router.post("/query-stream")
def query_stream(request: QueryRequest):

    def token_generator():
        for token in pipeline.stream_answer(request.query):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")