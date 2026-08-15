import logging
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from src.retrieval.graph_rag import GraphRAGPipeline
from src.retrieval.query_embedder import QueryEmbedder
from src.retrieval.video_summary_generator import VideoSummaryGenerator
from src.config import S3_BUCKET_NAME, S3_VIDEO_PREFIX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    )
logger = logging.getLogger(__name__)

# Global dict to hold pipeline components for reuse
pipeline_components = {}

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
    pipeline_components["embedder"] = QueryEmbedder()
    
    # Connect to Neo4j and ChromaDB
    logger.info("Connecting to databases...")
    pipeline_components["rag"] = GraphRAGPipeline()
    
    # Initialise Video Generator
    pipeline_components["generator"] = VideoSummaryGenerator(
        s3_bucket=S3_BUCKET_NAME,
        s3_prefix=S3_VIDEO_PREFIX,
        output_dir=Path("output_summaries")
    )
    
    logger.info("Server is fully loaded and ready for queries!")
    # Server running via FastAPI
    yield
    
    # Close connections on server shut down
    logger.info("Shutting down, closing database connections...")
    if "rag" in pipeline_components:
        pipeline_components["rag"].close()


# Initialise FastAPI app
app = FastAPI(title="GraphRAG Video Summarisation API", lifespan=lifespan)

if __name__ == "__main__":
    # Start server
    uvicorn.run("src.server:app", host="127.0.0.1", port=8000, reload=True)