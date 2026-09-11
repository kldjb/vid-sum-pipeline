from pathlib import Path
import hashlib
import json
import logging
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.retrieval.graph_rag import GraphRAGPipeline
from src.preprocess.utils import (
    select_centroid_representative_annotator,
    select_alternate_query_annotator,
)

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
    "visual_baseline_graph": {
        "modalities": ["frame"],
        "use_graph": True,
    },
    "ft_multimodal_vector_graph": {
        "modalities": ["frame", "transcript"],
        "use_graph": True,
    },
}

GLOBAL_SEARCH_MODALITY_CONFIGS = {
    "visual_baseline": ["frame"],
    "dialogue_baseline": ["transcript"],
    "semantic_baseline": ["script"],
    "ft_multimodal_vector": ["frame", "transcript"],
    "fs_multimodal_vector": ["frame", "script"],
    "ts_multimodal_vector": ["transcript", "script"],
    "trimodal_vector": ["frame", "transcript", "script"],
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

def calculate_vt_clipscore(
        pred_mask: np.ndarray,
        video_embeddings: np.ndarray,
        query_vec: np.ndarray
    ) -> float:
    """
    Calculate VT-CLIPScore for cross-modal consistency.

    NOTE: this is an adaptation of VT-CLIPScore for a retrieval-only
    pipeline stage - it measures query-to-retrieved-content alignment,
    not the original paper's generated-text-vs-generated-video
    coherence (Lin et al., 2024). Formula follows the standard
    CLIPScore convention (Hessel et al., 2021): w * max(cos_sim, 0),
    scaled by 100 to roughly match the paper's reported value ranges.

    Args:
        pred_mask: Binary array of retrieved frames.
        video_embeddings: Full per-frame visual embedding array.
        query_vec: Normalised text query embedding.

    Returns:
        float: Alignment score, scaled 0-100.
    """
    if pred_mask.sum() == 0:
        return 0.0

    n_frames = min(len(pred_mask), len(video_embeddings))
    if n_frames != len(video_embeddings) or n_frames != len(pred_mask):
        logger.warning(
            "pred_mask/video_embeddings length mismatch "
            f"({len(pred_mask)} vs {len(video_embeddings)}) - "
            "truncating to the shorter of the two."
        )

    valid_mask = pred_mask[:n_frames].astype(bool)
    selected_frames = video_embeddings[:n_frames][valid_mask]
    if len(selected_frames) == 0:
        return 0.0

    # Mean-pool visual frames and normalise output
    frame_norms = np.linalg.norm(selected_frames, axis=1, keepdims=True)
    frame_norms[frame_norms == 0] = 1.0
    normalised_frames = selected_frames / frame_norms

    vis_summary_vec = np.mean(normalised_frames, axis=0)
    norm_v = np.linalg.norm(vis_summary_vec)
    if norm_v > 0:
        vis_summary_vec = vis_summary_vec / norm_v

    score = np.dot(vis_summary_vec, query_vec) * 100.0
    return round(max(0.0, float(score)), 4)

def evaluate_narrative_cohesion(
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        dilation_sec: int = 3,
        tiou_threshold: float = 0.3
    ) -> dict:
    """
    Evaluate temporal cohesion using dilated ground truth masks.

    Args:
        pred_mask (np.ndarray): Binary array of predicted frames.
        gt_mask (np.ndarray): Binary array of oracle frames.
        dilation_sec (int): Seconds to expand ground truth.
        tiou_threshold (float): Minimum overlap ratio.

    Returns:
        dict: Cohesion metrics including F1 and IoU.
    """
    kernel = np.ones(dilation_sec * 2 + 1)
    dilated_gt = (
        np.convolve(gt_mask, kernel, mode='same') > 0
    ).astype(int)

    intersection = np.logical_and(pred_mask, dilated_gt).sum()
    union = np.logical_or(pred_mask, dilated_gt).sum()
    pred_count = pred_mask.sum()
    gt_count = dilated_gt.sum()

    # Calculate IoU, Recall, and F1 at frame-level
    precision_dil = intersection / pred_count if pred_count > 0 else 0.0
    recall_dil = intersection / gt_count if gt_count > 0 else 0.0
    iou_dil = intersection / union if union > 0 else 0.0

    if (precision_dil + recall_dil) > 0:
        f1_dil = (2 * precision_dil * recall_dil) / (
            precision_dil + recall_dil
        )
    else:
        f1_dil = 0.0

    def get_segments(mask: np.ndarray) -> list[tuple[int, int]]:
        """Extract contiguous segments from binary mask."""
        padded = np.pad(mask, (1, 1), 'constant')
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        return list(zip(starts, ends))

    gt_segments = get_segments(dilated_gt)
    pred_segments = get_segments(pred_mask)

    matched_events = 0
    for gt_start, gt_end in gt_segments:
        event_matched = False
        for p_start, p_end in pred_segments:
            o_start = max(gt_start, p_start)
            o_end = min(gt_end, p_end)
            overlap = max(0, o_end - o_start)
            
            s_union = (gt_end - gt_start) + (p_end - p_start) - overlap
            tiou = overlap / s_union if s_union > 0 else 0

            if tiou >= tiou_threshold:
                event_matched = True
                break
        if event_matched:
            matched_events += 1

    recall_tiou = (
        matched_events / len(gt_segments) if gt_segments else 0.0
    )

    return {
        "iou_dilated": round(float(iou_dil), 4),
        "precision_dilated": round(float(precision_dil), 4),
        "recall_dilated": round(float(recall_dil), 4),
        "f1_dilated": round(float(f1_dil), 4),
        "recall_at_tiou": round(float(recall_tiou), 4)
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
    vid_emb_path = video_dir / "video_embeddings.npy"

    if not gtscores_path.exists() or not script_emb_path.exists() or not vid_emb_path.exists():
        return None

    gtscores = np.load(gtscores_path)  # [n_annotators, n_frames]
    script_emb = np.load(script_emb_path)  # [n_annotators, M_max, 512]

    if script_emb.ndim != 3:
        logger.warning(
            f"{video_dir.name}: script_embeddings shape {script_emb.shape} "
            f"isn't the expected VideoXum layout [n_annotators, M_max, 512]. "
            f"Skipping video."
        )
        return None

    # Align script embeddings to annotator scores and retrieve script
    # embeddings closest to centroid of the 10 different script embeddings
    annotator_idx = select_centroid_representative_annotator(script_emb)
    script_source = script_emb[annotator_idx]  # [M_max, 512]
    target_scores = gtscores[annotator_idx]  # [n_frames]

    norms = np.linalg.norm(script_source, axis=1)
    valid_embs = script_source[norms > 0]  # drop zero-padded rows
    if len(valid_embs) == 0:
        return None

    query_vec = np.mean(valid_embs, axis=0)
    query_vec = query_vec / np.linalg.norm(query_vec)

    gt_mask = compute_oracle_summary_mask(target_scores, budget_ratio=0.15)
    n_frames = len(gt_mask)

    video_embeddings = np.load(vid_emb_path)
    if len(video_embeddings) != n_frames:
        logger.warning(
            f"{video_dir.name}: video_embeddings has {len(video_embeddings)} "
            f"frames but gtscores implies {n_frames}. Using gtscores' n_frames "
            "as the reference length - VT-CLIPScore truncates to the shorter "
            "of the two if this doesn't resolve itself downstream."
        )

    # Get alternative annotator's script embeddings (not stored in vector store)
    # to use as query vector for fairer evaluation (rather than searching for the
    # exact same script embeddings that were ingested into the vector store).
    seed = int(hashlib.md5(video_dir.name.encode('utf-8')).hexdigest(), 16) % (2**32)
    alt_annotator_idx = select_alternate_query_annotator(
        script_emb, exclude_idx=annotator_idx, seed=seed
    )

    alt_query_vec = None
    if alt_annotator_idx is not None:
        alt_source = script_emb[alt_annotator_idx]
        alt_norms = np.linalg.norm(alt_source, axis=1)
        alt_valid = alt_source[alt_norms > 0]
        if len(alt_valid) > 0:
            alt_vec = np.mean(alt_valid, axis=0)
            alt_query_vec = (alt_vec / np.linalg.norm(alt_vec)).tolist()

    return {
        "video_id": video_dir.name,
        "n_frames": n_frames,
        "gt_mask": gt_mask,
        "query_vec": query_vec.tolist(),
        "selected_annotator_index": annotator_idx,
        "alt_annotator_index": alt_annotator_idx,
        "alt_query_vec": alt_query_vec,
        "video_embeddings": video_embeddings,
    }

def calibrate_modality_similarity_stats(
        rag_instance: GraphRAGPipeline,
        sample_video_dirs: list[Path],
        modalities: tuple = ("frame", "transcript", "script"),
    ) -> dict:
    """
    Estimate each modality's typical query-to-candidate cosine similarity
    (mean, std) distribution from a sample of videos to correct the modality gap.

    Args:
        rag_instance: Initialised GraphRAG pipeline.
        sample_video_dirs: Videos to calibrate against.
        modalities: Which modalities to compute stats for.

    Returns:
        dict: {modality: {"mean": float, "std": float}}, only for
        modalities with enough samples to be meaningful.
    """
    samples = {m: [] for m in modalities}

    for video_dir in tqdm(sample_video_dirs, desc="Calibrating modality stats"):
        sample = load_extracted_video_sample(video_dir)
        if sample is None:
            continue
        query_vec = np.asarray(sample["query_vec"], dtype=np.float32)
        cached = rag_instance.fetch_video_vectors(sample["video_id"])

        for embedding, meta in zip(cached.get("embeddings", []), cached.get("metadatas", [])):
            modality = meta.get("modality")
            if modality not in samples:
                continue
            emb = np.asarray(embedding, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm == 0:
                continue
            score = float(np.dot(query_vec, emb / norm))
            samples[modality].append(score)

    stats = {}
    for modality, scores in samples.items():
        if len(scores) < 30:
            logger.warning(
                f"Only {len(scores)} calibration samples for modality "
                f"'{modality}' - skipping calibration for this modality "
                f"(stats would be unreliable). Consider a larger sample."
            )
            continue
        stats[modality] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}
        logger.info(
            f"Modality '{modality}' calibration: mean={stats[modality]['mean']:.4f}, "
            f"std={stats[modality]['std']:.4f} (n={len(scores)})"
        )

    return stats

def process_single_video(
        video_dir: Path,
        rag_instance: GraphRAGPipeline,
        n_results: int = 3,
        modality_calibration: dict = None,
    ) -> list[dict]:
    """
    Process single video through ablation configs, once for the self-
    consistent query (built from the same annotator whose script is
    actually in the corpus) and once for a cross-annotator query (built
    from a different annotator's script, never ingested - genuinely
    independent of the searchable candidate pool). Ground truth stays
    fixed to the corpus annotator's own gtscores in both cases, since
    the corpus content being searched doesn't change - only the query
    construction does.

    Args:
        video_dir: Path to video data folder.
        rag_instance: Initialised GraphRAG pipeline.
        n_results: Number segment candidates to retrieve.
        modality_calibration: Optional per-modality similarity stats from
            calibrate_modality_similarity_stats.

    Returns:
        list[dict]: Evaluation metrics for each (configuration, query_source) pair.
    """
    sample = load_extracted_video_sample(video_dir)
    if sample is None or sample["gt_mask"].sum() == 0:
        return []

    video_id = sample["video_id"]
    n_frames = sample["n_frames"]
    gt_mask = sample["gt_mask"]
    selected_annotator_index = sample["selected_annotator_index"]
    video_embeddings = sample["video_embeddings"]

    query_sources = [("self", sample["query_vec"], selected_annotator_index)]
    # Make sure video's other 9 annotators aren't all empty/padded
    if sample["alt_query_vec"] is not None:
        query_sources.append(
            ("cross_annotator", sample["alt_query_vec"], sample["alt_annotator_index"])
        )

    # Fetch video's vectors and reuse across all 8 ablations
    cached_vectors = rag_instance.fetch_video_vectors(video_id)

    results = []
    for query_source, query_vec, query_annotator_idx in query_sources:
        query_vec_arr = np.array(query_vec)
        for config_name, settings in ABLATION_CONFIGS.items():
            try:
                start_time = time.perf_counter()

                context = rag_instance.retrieve(
                    query_embedding=query_vec,
                    search_mode="video",
                    target_video_id=video_id,
                    allowed_modalities=settings["modalities"],
                    use_graph=settings["use_graph"],
                    n_results=n_results,
                    cached_video_vectors=cached_vectors,
                    modality_calibration=modality_calibration,
                )

                retrieval_time = time.perf_counter() - start_time

                predicted_intervals = extract_predicted_intervals(context)
                pred_mask = intervals_to_binary_mask(
                    predicted_intervals, n_frames
                )
                metrics = evaluate_narrative_cohesion(pred_mask, gt_mask)
                vt_clipscore = calculate_vt_clipscore(
                    pred_mask, video_embeddings, query_vec_arr
                )

                results.append({
                    "video_id": video_id,
                    "configuration": config_name,
                    "query_source": query_source,
                    "query_annotator_index": query_annotator_idx,
                    "selected_annotator_index": selected_annotator_index,
                    "latency_sec": round(retrieval_time, 4),
                    "vt_clipscore": vt_clipscore,
                    **metrics,
                })
            except Exception as e:
                logger.error(
                    f"Config {config_name} ({query_source}) failed {video_id}: {e}"
                )

    return results

def evaluate_extracted_dataset(
        extracted_data_dir: str,
        split_json_path: str = None,
        output_csv: str = "videoxum_ablation_results.csv",
        n_results: int = 3,
        use_modality_calibration: bool = False,
        calibration_sample_size: int = 150,
    ):
    """
    Run intra-video evaluation against ground-truth test set.

    Args:
        extracted_data_dir: Path to extracted_data folder.
        split_json_path: Path to sm_videoxum_split.json test split.
        output_csv: Destination CSV path.
        n_results: Number segment candidates to retrieve.
        use_modality_calibration: if True, corrects the modality gap
            using z-score normalising. Run as a separate pass with a
            different output_csv from the uncalibrated run.
        calibration_sample_size: How many videos to sample when
            estimating per-modality similarity statistics.
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

    modality_calibration = None
    if use_modality_calibration:
        logger.info(
            f"Calibrating modality similarity stats from a sample of "
            f"{calibration_sample_size} videos..."
        )
        rng = np.random.default_rng(42)
        sample_dirs = list(
            rng.choice(video_dirs, size=min(calibration_sample_size, len(video_dirs)), replace=False)
        )
        modality_calibration = calibrate_modality_similarity_stats(rag_instance, sample_dirs)

    logger.info("Starting evaluation...")
    
    for video_dir in tqdm(video_dirs, desc="Evaluating Videos"):
        try:
            # Process directly on main thread
            res = process_single_video(
                video_dir, rag_instance, n_results,
                modality_calibration=modality_calibration,
            )
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
    
    summary = df.groupby(["configuration", "query_source"])[
        [
            "latency_sec",
            "iou_dilated",
            "precision_dilated",
            "recall_dilated",
            "f1_dilated",
            "recall_at_tiou",
            "vt_clipscore",
        ]
    ].mean().reset_index()
    
    logger.info("\nAblation Study Summary (self-consistent vs cross-annotator query):")
    print(summary.to_markdown(index=False))

def compute_global_search_metrics(ranks, k_values=(1, 5, 10)):
    """
    ranks: list of (1-indexed rank of the correct video, or None if it
    didn't appear in the top n_results at all) - one entry per query.
    """
    metrics = {}
    for k in k_values:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        metrics[f"recall_at_{k}"] = hits / len(ranks) if ranks else 0.0

    reciprocal_ranks = [1.0 / r if r is not None else 0.0 for r in ranks]
    metrics["mrr"] = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    return metrics


def evaluate_global_search_dataset(
        extracted_data_dir: str,
        split_json_path: str = None,
        output_csv: str = "videoxum_global_search_results.csv",
        n_results: int = 10,
        k_values: tuple = (1, 5, 10),
        use_modality_calibration: bool = False,
        calibration_sample_size: int = 150,
    ):
    """
    Cross-video (global) search evaluation to determine if a video is
    surface, and at what rank when searching across the entire dataset
    for the video's own query.

    Args:
        extracted_data_dir: Path to extracted_data folder.
        split_json_path: Path to sm_videoxum_split.json test split.
        output_csv: Destination CSV path.
        n_results: Number of unique videos to retrieve per query.
        k_values: Recall@K cutoffs to report.
        use_modality_calibration: if True, corrects the modality gap
            using z-score normalising. Run as a separate pass with a
            different output_csv from the uncalibrated run.
        calibration_sample_size: How many videos to sample when
            estimating per-modality similarity statistics.
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

    logger.info("Initialising GraphRAG Pipeline for global search evaluation...")
    rag_instance = GraphRAGPipeline()

    # Fetch each base modality's vectors from the whole corpus once
    base_modalities = sorted({m for mods in GLOBAL_SEARCH_MODALITY_CONFIGS.values() for m in mods})
    logger.info(f"Caching corpus-wide vectors for modalities: {base_modalities}...")
    base_caches = {}
    for modality in base_modalities:
        cache = rag_instance.fetch_global_vectors_by_modality(modality)
        n_vectors = len(cache["video_ids"])
        est_gb = cache["embeddings"].nbytes / 1e9
        logger.info(f"  '{modality}': {n_vectors:,} vectors cached (~{est_gb:.2f} GB)")
        base_caches[modality] = cache

    # Pre-combine per-config caches once, not per video
    config_caches = {}
    for config_name, modalities in GLOBAL_SEARCH_MODALITY_CONFIGS.items():
        relevant = [base_caches[m] for m in modalities if len(base_caches[m]["video_ids"]) > 0]
        if not relevant:
            config_caches[config_name] = {
                "embeddings": np.zeros((0, 0), dtype=np.float32),
                "video_ids": [],
                "modalities": [],
            }
            continue
        combined_embeddings = np.concatenate([c["embeddings"] for c in relevant], axis=0)
        combined_video_ids = []
        combined_modalities = []
        for c in relevant:
            combined_video_ids.extend(c["video_ids"])
            combined_modalities.extend(c["modalities"])
        config_caches[config_name] = {
            "embeddings": combined_embeddings,
            "video_ids": combined_video_ids,
            "modalities": combined_modalities,
        }

    if use_modality_calibration:
        logger.info(
            f"Calibrating modality similarity stats from a sample of "
            f"{calibration_sample_size} videos..."
        )
        rng = np.random.default_rng(42)
        sample_dirs = list(
            rng.choice(video_dirs, size=min(calibration_sample_size, len(video_dirs)), replace=False)
        )
        modality_calibration = calibrate_modality_similarity_stats(
            rag_instance, sample_dirs, modalities=tuple(base_modalities)
        )
        # Add each config's per-vector mean/std into its cache once
        for config_name in config_caches:
            rag_instance.calibrate_global_cache(config_caches[config_name], modality_calibration)

    all_results = []
    per_config_ranks = {name: [] for name in GLOBAL_SEARCH_MODALITY_CONFIGS}

    for video_dir in tqdm(video_dirs, desc="Evaluating Global Search"):
        sample = load_extracted_video_sample(video_dir)
        if sample is None:
            continue

        true_video_id = sample["video_id"]
        query_vec = sample["query_vec"]

        for config_name in GLOBAL_SEARCH_MODALITY_CONFIGS:
            try:
                ranked_videos = rag_instance.search_top_videos(
                    query_embedding=query_vec,
                    n_results=n_results,
                    cached_global_vectors=config_caches[config_name],
                )
                rank = (
                    ranked_videos.index(true_video_id) + 1
                    if true_video_id in ranked_videos
                    else None
                )
                per_config_ranks[config_name].append(rank)
                all_results.append({
                    "video_id": true_video_id,
                    "configuration": config_name,
                    "rank": rank,
                })
            except Exception as e:
                logger.error(
                    f"Global search config {config_name} failed {true_video_id}: {e}"
                )

    rag_instance.close()

    if not all_results:
        logger.error("No valid global search results generated. Aborting save.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)

    summary_rows = []
    for config_name, ranks in per_config_ranks.items():
        metrics = compute_global_search_metrics(ranks, k_values=k_values)
        summary_rows.append({"configuration": config_name, **metrics})

    summary = pd.DataFrame(summary_rows)
    logger.info("\nGlobal Search Evaluation Summary:")
    print(summary.to_markdown(index=False))


if __name__ == "__main__":
    extracted_data_dir = (
        "Datasets/SM-MrHiSum and SM-VideoXum/"
        "SM-VideoXum-Training-Data/extracted_data"
    )
    split_json_path = (
        "Datasets/SM-MrHiSum and SM-VideoXum/"
        "SM-VideoXum-Training-Data/sm_videoxum_split.json"
    )

    # evaluate_extracted_dataset(
    #     extracted_data_dir=extracted_data_dir,
    #     split_json_path=split_json_path,
    #     output_csv="videoxum_ablation_results.csv",
    #     n_results=10,
    # )
    # Modality-gap-corrected comparison run - separate output CSV
    # evaluate_extracted_dataset(
    #     extracted_data_dir=extracted_data_dir,
    #     split_json_path=split_json_path,
    #     output_csv="videoxum_ablation_results_calibrated.csv",
    #     n_results=10,
    #     use_modality_calibration=True,
    # )
    # evaluate_global_search_dataset(
    #     extracted_data_dir=extracted_data_dir,
    #     split_json_path=split_json_path,
    #     output_csv="videoxum_global_search_results.csv",
    #     n_results=10,
    #     k_values=(1, 5, 10),
    # )
    # Modality-gap-corrected global search - separate output CSV
    evaluate_global_search_dataset(
        extracted_data_dir=extracted_data_dir,
        split_json_path=split_json_path,
        output_csv="videoxum_global_search_results_calibrated.csv",
        n_results=10,
        k_values=(1, 5, 10),
        use_modality_calibration=True,
    )