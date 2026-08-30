import logging
import sys
import numpy as np

from src.retrieval.graph_rag import GraphRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Ablation configs
ABLATION_CONFIGS = {
    "visual_baseline": {"modalities": ["frame"], "use_graph": False},
    "fs_multimodal_vector": {"modalities": ["frame", "script"], "use_graph": False},
    "full_graphrag": {"modalities": ["frame", "transcript", "script"], "use_graph": True},
}

def test_ablation_configurations():
    logger.info("Initialising GraphRAG Pipeline for Ablation Test...")
    try:
        pipeline = GraphRAGPipeline()
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)

    # Generate mock 512-dimensional vector (matches ViT-B/16 output)
    mock_query_embedding = np.random.rand(512).astype(np.float32)
    mock_query_embedding = (
        mock_query_embedding / np.linalg.norm(mock_query_embedding)
        ).tolist()

    # Loop ablation tests
    for config_name, settings in ABLATION_CONFIGS.items():
        logger.info(f"\n{'='*50}\nTesting Configuration: {config_name}")
        logger.info(f"Settings -> Modalities: {settings['modalities']} | Use Graph: {settings['use_graph']}")
        
        try:
            # Use global mode to test generic retrieval
            context = pipeline.retrieve(
                query_embedding=mock_query_embedding,
                search_mode="global",
                n_results=2,
                allowed_modalities=settings["modalities"],
                use_graph=settings["use_graph"]
            )
            
            if not context:
                logger.warning(
                    "No context retrieved! Ensure ChromaDB has data for these modalities."
                    )
                continue
                
            # Verify timestamps
            for i, scene in enumerate(context, start=1):
                logger.info(f"\nResult {i}:")
                logger.info(f"Video ID: {scene.get('video_id')}")
                logger.info(f"Segment ID: {scene.get('segment_id')}")
                logger.info(
                    f"Temporal Boundaries: {scene.get('start_frame')} to {scene.get('end_frame')}"
                    )
                logger.info(f"Matched Modalities: {scene.get('matched_modalities')}")
                
                # Check generic timestamps in baseline modes
                if not settings["use_graph"] and "script" in scene.get('matched_modalities', []):
                    if scene.get("start_frame") == 0.0 and scene.get("end_frame") == 0.0:
                        logger.info(
                            "Verified: Script node defaulted to 0.0 without graph traversal."
                            )
                        
        except Exception as e:
            logger.error(f"Pipeline crashed on {config_name}: {e}")

    pipeline.close()
    logger.info("\nTest complete.")

if __name__ == "__main__":
    test_ablation_configurations()