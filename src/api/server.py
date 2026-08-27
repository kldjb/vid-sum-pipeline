import logging
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from src.api.query import router as query_router
from src.retrieval.graph_rag import GraphRAGPipeline
from src.retrieval.query_embedder import QueryEmbedder
from src.retrieval.video_summary_generator import VideoSummaryGenerator
from src.config import S3_BUCKET_NAME, S3_VIDEO_PREFIX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    )
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise pipeline components once for repeated use during server run:
     - Everything before `yield` runs just once when the server starts.
     - Everything after `yield` runs when the server shuts down.
    """
    logger.info("Loading GraphRAG Server (this may take a few mins)...")
    
    # Load CLIP Model into RAM
    logger.info("Loading CLIP Query Embedder...")
    app.state.embedder = QueryEmbedder()
    
    # Connect to Neo4j and ChromaDB
    logger.info("Connecting to databases...")
    app.state.rag = GraphRAGPipeline()
    
    # Initialise Video Generator
    logger.info("Initialising Video Summary Generator...")
    app.state.generator = VideoSummaryGenerator(
        s3_bucket=S3_BUCKET_NAME,
        s3_prefix=S3_VIDEO_PREFIX,
        output_dir=Path("output_summaries")
    )

    # Load vector index into RAM for faster queries by submitting dummy query
    logger.info(
        "Loading vector index into RAM for faster querying (this will take ~2 minutes)..."
        )
    dummy_vector = [0.0] * 512 
    app.state.rag.collection.query(query_embeddings=[dummy_vector], n_results=1)
    logger.info("Vector index loaded into RAM.")
    
    # Server running via FastAPI
    logger.info("Server is fully loaded and ready for queries!")
    yield
    
    # Close connections on server shut down
    logger.info("Shutting down, closing database connections...")
    if hasattr(app.state, "rag"):
        app.state.rag.close()


# Initialise FastAPI app
app = FastAPI(title="GraphRAG Video Summarisation API", lifespan=lifespan)
app.include_router(query_router)

if __name__ == "__main__":
    # Start server
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=True)