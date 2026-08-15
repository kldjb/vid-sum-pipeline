import argparse
import logging

from src.retrieval.graph_rag import GraphRAGPipeline
from src.retrieval.query_embedder import QueryEmbedder
from src.retrieval.video_summary_generator import VideoSummaryGenerator
from src.config import S3_BUCKET_NAME, S3_VIDEO_PREFIX, VIDEO_SUM_OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    )
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GraphRAG retrieval pipeline over multimodal video data."
    )

    # Required query flag
    parser.add_argument(
        "-q", "--query",
        type=str,
        required=True,
        help="Text search query (e.g., 'people talking indoors')"
    )

    # Optional flags
    parser.add_argument(
        "-n", "--n-results",
        type=int,
        default=3,
        help="Number of top results to return (default: 3)"
    )
    parser.add_argument(
        "-m", "--search-mode",
        type=str,
        choices=["global", "video"],
        default="global",
        help="Search mode: 'global' for unique top videos or 'video' for summarising specific video (default: 'global')"
    )
    parser.add_argument(
        "-v", "--video-id",
        type=str,
        default=None,
        help="Target video ID (required when --search-mode is 'video')"
    )

    args = parser.parse_args()

    # Validate dependent args
    if args.search_mode == "video" and not args.video_id:
        parser.error("--video-id (-v) is required when --search-mode is 'video'.")

    return args


if __name__ == "__main__":
    # Load args
    args = parse_args()

    # Instantiate query embedder
    embedder = QueryEmbedder()

    # Vectorise query
    logger.info(f"Embedding query: '{args.query}'")
    query_vector = embedder.embed_query(args.query)

    # Retrieve query-relevant data
    retrieval_pipeline = GraphRAGPipeline()

    try:
        logger.info(
            "Running GraphRAG retrieval (search mode='"
            f"{args.search_mode}', top {args.n_results} results)..."
            )
        context = retrieval_pipeline.retrieve(
            query_vector,
            args.search_mode,
            args.video_id,
            args.n_results,
            )
            
        logger.info(f"\n--- Relevant data to query: {args.query} ---")
        for idx, scene in enumerate(context, 1):
            logger.info(f"\n[Result {idx}]")
            for key, val in scene.items():
                logger.info(f"  {key}: {val}")

        # Generate video summary for video-specific queries
        if args.search_mode == "video" and context:
            logger.info("\n--- Generating Video Summary ---")
            vid_sum_generator = VideoSummaryGenerator(
                s3_bucket=S3_BUCKET_NAME,
                s3_prefix=S3_VIDEO_PREFIX,
                output_dir=VIDEO_SUM_OUTPUT_DIR,
            )
            output_path = vid_sum_generator.generate_summary(
                args.video_id, args.query, context
                )
            
            if output_path:
                logger.info(
                    f"Process complete! You can view the summary video at: {output_path}"
                    )
        
    finally:
        retrieval_pipeline.close()