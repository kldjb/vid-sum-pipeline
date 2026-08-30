from pathlib import Path
import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.retrieval.graph_rag import GraphRAGPipeline

# Initialise logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Setup ablation configs
ABLATION_CONFIGS = {
    "visual_baseline": {
        "modalities": ["frame"],
        "use_graph": False,
    },
    "dialogue_baseline": {
        "modalities": ["transcript"],
        "use_graph": False,
    },
    "semantic_baseline": {
        "modalities": ["script"],
        "use_graph": False,
    },
    "ft_multimodal_vector": {
        "modalities": ["frame", "transcript"],
        "use_graph": False,
    },
    "fs_multimodal_vector": {
        "modalities": ["frame", "script"],
        "use_graph": False,
    },
    "ts_multimodal_vector": {
        "modalities": ["transcript", "script"],
        "use_graph": False,
    },
    "trimodal_vector": {
        "modalities": ["frame", "transcript", "script"],
        "use_graph": False,
    },
    "full_graphrag": {
        "modalities": ["frame", "transcript", "script"],
        "use_graph": True,
    },
}


def knapsack_dp(
        values: np.ndarray,
        weights: np.ndarray,
        capacity: int
    ) -> list[int]:
    """
    Solve 0/1 Knapsack using dynamic programming summary selection.

    Args:
        values: Frame importance scores.
        weights: Integer frame costs.
        capacity: Maximum summary frame budget.

    Returns:
        list[int]: Selected frame indices oracle summary.
    """
    n = len(values)
    dp = np.zeros((n + 1, capacity + 1), dtype=np.float32)

    for i in range(1, n + 1):
        w = int(weights[i - 1])
        v = values[i - 1]
        for c in range(capacity + 1):
            if w <= c:
                dp[i, c] = max(dp[i - 1, c], dp[i - 1, c - w] + v)
            else:
                dp[i, c] = dp[i - 1, c]

    selected = []
    c = capacity
    for i in range(n, 0, -1):
        if dp[i, c] != dp[i - 1, c]:
            selected.append(i - 1)
            c -= int(weights[i - 1])

    return selected[::-1]


def compute_oracle_summary_mask(
        gtscores: np.ndarray,
        budget_ratio: float = 0.15
    ) -> np.ndarray:
    """
    Convert multi-annotator scores into binary oracle mask.

    Args:
        gtscores: Array shape [10, n_frames] or [n_frames].
        budget_ratio: Duration budget relative video length.

    Returns:
        np.ndarray: Binary 1D array shape [n_frames].
    """
    if gtscores.ndim == 2:
        mean_scores = np.mean(gtscores, axis=0)
    else:
        mean_scores = gtscores

    n_frames = len(mean_scores)
    capacity = max(1, int(n_frames * budget_ratio))
    weights = np.ones(n_frames, dtype=int)

    selected_indices = knapsack_dp(mean_scores, weights, capacity)

    gt_mask = np.zeros(n_frames, dtype=int)
    gt_mask[selected_indices] = 1
    
    return gt_mask


def intervals_to_binary_mask(
        intervals: list[tuple[float, float]],
        n_frames: int
    ) -> np.ndarray:
    """
    Project retrieved time intervals into 1-fps discrete frame mask.

    Args:
        intervals: List (start_sec, end_sec) tuples.
        n_frames: Total frames source video.

    Returns:
        np.ndarray: Binary frame mask length n_frames.
    """
    pred_mask = np.zeros(n_frames, dtype=int)
    for start, end in intervals:
        s_idx = max(0, int(np.floor(start)))
        e_idx = min(n_frames, int(np.ceil(end)))
        pred_mask[s_idx:e_idx] = 1
        
    return pred_mask


def calculate_temporal_metrics(
        pred_mask: np.ndarray,
        gt_mask: np.ndarray
    ) -> dict:
    """
    Calculate precision, recall, F1, Temporal Intersection over
    Union (IoU) video segments.

    Args:
        pred_mask: Binary array retrieved video frames.
        gt_mask: Binary array ground-truth summary video frames.

    Returns:
        dict: Computed metrics rounded 4 decimal places.
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    pred_count = pred_mask.sum()
    gt_count = gt_mask.sum()

    if pred_count == 0 or gt_count == 0:
        return {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = intersection / pred_count
    recall = intersection / gt_count
    iou = intersection / union if union > 0 else 0.0
    
    if (precision + recall) > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return {
        "iou": round(float(iou), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }


def extract_predicted_intervals(
        context: list[dict]
    ) -> list[tuple[float, float]]:
    """
    Extract time boundaries ChromaDB/Neo4j context dicts.

    Args:
        context: List retrieval dictionaries.

    Returns:
        list[tuple[float, float]]: List (start_sec, end_sec) bounds.
    """
    intervals = []
    for item in context:
        # Check keys for None and identify alternative
        start = item.get("start_frame")
        if start is None:
            start = item.get("chunk_start")
        if start is None:
            start = item.get("timestamp_sec")
        if start is None:
            start = 0.0

        end = item.get("end_frame")
        if end is None:
            end = item.get("chunk_end")
        if end is None:
            end = float(start) + 5.0

        intervals.append((float(start), float(end)))
        
    return intervals


def load_extracted_video_sample(video_dir: Path) -> dict | None:
    """
    Read ground-truth files and generate query embeddings from
    extracted arrays.

    Args:
        video_dir: Path to target video directory.

    Returns:
        dict | None: Contains n_frames, gt_mask, query_vec.
    """
    gtscores_path = video_dir / "gtscores.npy"
    script_emb_path = video_dir / "script_embeddings.npy"

    if not gtscores_path.exists() or not script_emb_path.exists():
        return None

    script_emb = np.load(script_emb_path).reshape(-1, 512)
    norms = np.linalg.norm(script_emb, axis=1)
    
    valid_embs = script_emb[norms > 0]
    if len(valid_embs) == 0:
        return None
        
    query_vec = np.mean(valid_embs, axis=0)
    query_vec = query_vec / np.linalg.norm(query_vec)

    gtscores = np.load(gtscores_path)
    gt_mask = compute_oracle_summary_mask(gtscores, budget_ratio=0.15)
    n_frames = len(gt_mask)

    return {
        "video_id": video_dir.name,
        "n_frames": n_frames,
        "gt_mask": gt_mask,
        "query_vec": query_vec.tolist(),
    }

def process_single_video(
        video_dir: Path,
        rag_instance: GraphRAGPipeline,
        n_results: int = 3
    ) -> list[dict]:
    """
    Process single video through ablation configs.

    Args:
        video_dir: Path to video data folder.
        rag_instance: Initialised GraphRAG pipeline.
        n_results: Number segment candidates to retrieve.

    Returns:
        list[dict]: Evaluation metrics for each configuration.
    """
    sample = load_extracted_video_sample(video_dir)
    if sample is None or sample["gt_mask"].sum() == 0:
        return []

    video_id = sample["video_id"]
    n_frames = sample["n_frames"]
    gt_mask = sample["gt_mask"]
    query_vec = sample["query_vec"]

    # Fetch video's vectors and reuse across all 8 ablations
    cached_vectors = rag_instance.fetch_video_vectors(video_id)
    
    results = []
    for config_name, settings in ABLATION_CONFIGS.items():
        try:
            context = rag_instance.retrieve(
                query_embedding=query_vec,
                search_mode="video",
                target_video_id=video_id,
                allowed_modalities=settings["modalities"],
                use_graph=settings["use_graph"],
                n_results=n_results,
                overfetch_factor=500,
                cached_video_vectors=cached_vectors,
            )

            predicted_intervals = extract_predicted_intervals(context)
            pred_mask = intervals_to_binary_mask(
                predicted_intervals, n_frames
            )
            metrics = calculate_temporal_metrics(pred_mask, gt_mask)

            results.append({
                "video_id": video_id,
                "configuration": config_name,
                **metrics,
            })
        except Exception as e:
            logger.error(
                f"Config {config_name} failed {video_id}: {e}"
            )
            
    return results

def evaluate_extracted_dataset(
        extracted_data_dir: str,
        split_json_path: str = None,
        output_csv: str = "videoxum_ablation_results.csv",
        n_results: int = 3,
    ):
    """
    Run intra-video evaluation against ground-truth test set.

    Args:
        extracted_data_dir: Path to extracted_data folder.
        split_json_path: Path to sm_videoxum_split.json test split.
        output_csv: Destination CSV path.
        n_results: Number segment candidates to retrieve.
    """
    extracted_path = Path(extracted_data_dir)
    
    if split_json_path and Path(split_json_path).exists():
        with open(split_json_path, "r", encoding="utf-8") as f:
            splits = json.load(f)
        target_ids = set(splits.get("test", []))
        video_dirs = [
            extracted_path / vid for vid in target_ids 
            if (extracted_path / vid).is_dir()
        ]
        logger.info(f"Loaded {len(video_dirs)} test-split videos.")
    else:
        video_dirs = [d for d in extracted_path.iterdir() if d.is_dir()]
        logger.info(f"Loaded {len(video_dirs)} total videos.")

    logger.info("Initialising GraphRAG Pipeline...")
    rag_instance = GraphRAGPipeline()
    all_results = []

    logger.info("Starting evaluation...")
    
    for video_dir in tqdm(video_dirs, desc="Evaluating Videos"):
        try:
            # Process directly on main thread
            res = process_single_video(video_dir, rag_instance, n_results)
            if res:
                all_results.extend(res)
        except Exception as e:
            logger.error(f"Execution failed for {video_dir.name}: {e}")

    rag_instance.close()

    if not all_results:
        logger.error("No valid results generated. Aborting save.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    
    summary = df.groupby("configuration")[
        ["iou", "precision", "recall", "f1"]
    ].mean().reset_index()
    
    logger.info("\nAblation Study Summary:")
    print(summary.to_markdown(index=False))


if __name__ == "__main__":
    evaluate_extracted_dataset(
        extracted_data_dir=(
            "Datasets/SM-MrHiSum and SM-VideoXum/"
            "SM-VideoXum-Training-Data/extracted_data"
        ),
        split_json_path=(
            "Datasets/SM-MrHiSum and SM-VideoXum/"
            "SM-VideoXum-Training-Data/sm_videoxum_split.json"
        ),
        output_csv="videoxum_ablation_results.csv",
        n_results=3,
    )