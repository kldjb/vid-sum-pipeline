from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel


# Initialise router for server
router = APIRouter()

# Define expected JSON payload format
class QueryRequest(BaseModel):
    query: str

    # Get search mode ('global' or 'video', defaults to 'global')
    search_mode: str = "global"
    n_results: int = 3

    # Required if search_mode is 'video'
    video_id: Optional[str] = None

@router.post("/search")
async def process_query(payload: QueryRequest, request: Request):
    """
    Endpoint to process search query and optionally generate a summary.
    """
    # Validate input
    if payload.search_mode == "video" and not payload.video_id:
        raise HTTPException(
            status_code=400, detail="video_id is required for video mode"
            )

    # Get active instances from running FastAPI server
    embedder = request.app.state.embedder
    rag = request.app.state.rag
    generator = request.app.state.generator

    # Embed query
    query_vector = embedder.embed_query(payload.query)

    # Retrieve context from Graph/ChromaDB
    context = rag.retrieve(
        query_embedding=query_vector,
        search_mode=payload.search_mode,
        target_video_id=payload.video_id,
        n_results=payload.n_results
    )

    if not context:
        return {"message": "No relevant scenes found.", "context": []}

    response_data = {"message": "Success", "context": context}

    # Generate video summary if in 'video' mode
    if payload.search_mode == "video":
        output_path = generator.generate_summary(payload.video_id, payload.query, context)
        if output_path:
            response_data["summary_video_path"] = str(output_path)

    return response_data