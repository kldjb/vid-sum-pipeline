import os
import numpy as np

from src.retrieval.query_embedder import QueryEmbedder
from src.config import TEXT_ANNOTATIONS_DIR


def run_query_test() -> None:
    """
    Execute sanity check embedding a full script text and comparing
    against stored pooled embeddings to verify the live querying logic.
    """
    embedder = QueryEmbedder()

    video_dir = (
        r"Datasets\SM-MrHiSum and SM-VideoXum"
        r"\SM-VideoXum-Training-Data\extracted_data\v___c8enCfzqw"
    )
    video_id = "v___c8enCfzqw"
    
    emb_path = os.path.join(video_dir, "script_embeddings.npy")
    script_emb = np.load(emb_path)
    
    test_idx = 0
    pth = os.path.join(
        TEXT_ANNOTATIONS_DIR, f"{video_id}_{test_idx}.txt"
    )
    print(pth)
    
    with open(pth, 'r', encoding='utf-8') as f:
        test_text = f.read().replace('\n', ' ').strip()
        
    # --- AMENDMENT: Use updated QueryEmbedder directly ---
    # The embedder now handles sentence parsing and mean-pooling natively
    query_vec = np.array(embedder.embed_query(test_text))
    # -----------------------------------------------------

    valid_mask = np.linalg.norm(script_emb[test_idx], axis=1) > 0
    valid_rows = script_emb[test_idx][valid_mask]
    
    # --- AMENDMENT: Mimic evaluation script pooling ---
    # Mean-pool the valid dataset rows into a single vector
    dataset_vec = np.mean(valid_rows, axis=0)
    
    # L2 normalise the pooled dataset vector
    dataset_vec = dataset_vec / np.linalg.norm(dataset_vec)
    # --------------------------------------------------
    
    # --- AMENDMENT: Single dot product calculation ---
    # Calculate cosine similarity dot product between the two pooled vectors
    sim = np.dot(dataset_vec, query_vec)
    sim = np.clip(sim, -1.0, 1.0)
    # -------------------------------------------------
    
    print(
        "similarity to that annotator's pooled embeddings: "
        f"{sim}"
    )


if __name__ == "__main__":
    run_query_test()