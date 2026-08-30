import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load environment vars from .env file
load_dotenv(find_dotenv())

# Get absolute path to src dir (Project/src)
SRC_DIR = Path(__file__).resolve().parent

# Get absolute path to project root (Project/)
PROJECT_ROOT = SRC_DIR.parent

# Configure AWS S3
S3_BUCKET_NAME = "video-sum-pipeline"
S3_VIDEO_PREFIX = "raw-data/SM-MrHiSum and SM-VideoXum/ActivityNet-Data"

# Configure Neo4J
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Configure chromedb storage
CHROMA_DB_DIR = PROJECT_ROOT / "Datasets" / "chroma_db"
CHROMADB_COLLECTION_NAME = "multimodal_video_summarisation"
MODALITY_CONFIG = {
    "frame": {
        "filename": "video_embeddings.npy",
    },
    "script": {
        "filename": "script_embeddings.npy",
    },
    "transcript": {
        "filename": "transcript_embeddings.npy",
    },
    "aligned_transcript": {
        "filename": "aligned_transcripts.npy",
    },
}

# Output directory for the generated summary MP4 files
VIDEO_SUM_OUTPUT_DIR = Path("output_summaries")