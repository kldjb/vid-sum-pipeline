import hashlib
import numpy as np

def get_collection_id(video_id, num_collections=20):
    """
    Create deterministic collection ID between 1 and num_collections 
    based on the video_id string.
    """
    hash_val = int(hashlib.md5(video_id.encode('utf-8')).hexdigest(), 16)
    return f"col_{(hash_val % num_collections) + 1}"


def select_centroid_representative_annotator(script_emb: np.ndarray) -> int:
    """
    Pick the annotator script closest to the centroid of all
    annotators' scripts, using mean-pooled sentence-embeddings.

    Args:
        script_emb: [n_annotators, M_max, 512], zero-padded per annotator.

    Returns:
        int: index of most representative annotator.
    """
    n_annotators, _, dim = script_emb.shape
    pooled = np.zeros((n_annotators, dim), dtype=np.float32)
    has_content = np.zeros(n_annotators, dtype=bool)

    for i in range(n_annotators):
        block = script_emb[i]
        norms = np.linalg.norm(block, axis=1)
        valid = block[norms > 0]
        if len(valid) == 0:
            continue  # annotator has no real sentences

        mean_vec = valid.mean(axis=0)
        vec_norm = np.linalg.norm(mean_vec)
        pooled[i] = mean_vec / vec_norm if vec_norm > 0 else mean_vec
        has_content[i] = True

    if not has_content.any():
        return 0  # nothing usable

    centroid = pooled[has_content].mean(axis=0)
    c_norm = np.linalg.norm(centroid)
    if c_norm > 0:
        centroid = centroid / c_norm

    # Compute dot products between annotators and centroid
    sims = pooled @ centroid
    sims[~has_content] = -np.inf  # avoid annotators with no content
    return int(np.argmax(sims))


def select_alternate_query_annotator(
        script_emb: np.ndarray,
        exclude_idx: int,
        seed=None,
    ) -> int | None:
    """
    Pick a different annotator than exclude_idx, uniformly at random
    among annotators with real (non-padded) content.

    Args:
        script_emb: [n_annotators, M_max, 512], zero-padded per annotator.
        exclude_idx: annotator index to exclude from selection.
        seed: for reproducibility.

    Returns:
        int | None: index of the alternate annotator, or None if no
        other annotator has any real content for this video.
    """
    rng = np.random.default_rng(seed)
    n_annotators = script_emb.shape[0]
    candidates = []
    for i in range(n_annotators):
        if i == exclude_idx:
            continue
        norms = np.linalg.norm(script_emb[i], axis=1)
        if np.any(norms > 0):
            candidates.append(i)
    if not candidates:
        return None
    return int(rng.choice(candidates))