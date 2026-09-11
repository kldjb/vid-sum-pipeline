import logging
import numpy as np
import chromadb
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import spacy
from tqdm import tqdm

from src.preprocess.utils import get_collection_id, select_centroid_representative_annotator
from src.config import CHROMADB_COLLECTION_NAME, MODALITY_CONFIG


# Setup logger to track missing files and prevent log spam
logger = logging.getLogger(__name__)
_missing_files_warned = set()

# Load English NLP model sentence boundary detection
logger.info("Loading spaCy English model for sentence parsing...")
nlp = spacy.load("en_core_web_sm")
logger.info("spaCy model loaded successfully.")

# Batch size to process vectors before flushing to ChromaDB
BATCH_SIZE = 500

# Define absolute path to text annotations base directory
TEXT_ANNOTATIONS_DIR = Path(
    "Datasets/SM-MrHiSum and SM-VideoXum/"
    "SM-VideoXum-Text-Annotations"
)

def _normalise_embeddings(arr):
    # Remove infinite or NaN arrays
    if not np.isfinite(arr).all():
        raise ValueError("Array contains NaN/infinite values")
    
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D embedding array, got shape {arr.shape}")
    return arr

def _get_transcript_timestamps(video_dir, row_idx):
    ts_path = video_dir / "transcript_timestamps.npy"
    if ts_path.exists():
        try:
            ts_data = np.load(ts_path)
            if row_idx < len(ts_data):
                return float(ts_data[row_idx][0]), float(ts_data[row_idx][1])
        except Exception:
            pass
    start = float(row_idx * 5.0)
    return start, start + 5.0

def _build_metadata(modality, video_id, row_idx, video_dir, annotator_idx=None):
    row_idx = int(row_idx)
    collection_id = get_collection_id(video_id)
    
    metadata = {
        "modality": modality,
        "video_id": video_id,
        "collection_id": collection_id,
    }
    
    if modality == "frame":
        metadata.update({"frame_id": row_idx, "timestamp_sec": float(row_idx)})
    elif modality == "script":
        metadata.update({"sentence_index": row_idx})
        if annotator_idx is not None:
            # VideoXum only - which of the 10 annotators this sentence came from
            metadata["annotator_index"] = int(annotator_idx)
    elif modality == "transcript":
        start, end = _get_transcript_timestamps(video_dir, row_idx)
        metadata.update({"chunk_start": start, "chunk_end": end})
    elif modality == "aligned_transcript":
        metadata.update({"frame_id": row_idx, "timestamp_sec": float(row_idx)})
        
    return metadata

def _parse_script_sentences(file_path: Path) -> list[str]:
    """
    Parse script text file into separate sentences using spaCy.

    Args:
        file_path (Path): Path target text file.

    Returns:
        list[str]: Extracted text sentences.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().replace('\n', ' ').strip()
        
    # Process text spaCy pipeline
    doc = nlp(text)
    
    # Extract individual sentences, skip empty strings
    sentences = [
        sent.text.strip() for sent in doc.sents if sent.text.strip()
    ]
    
    return sentences

def _parse_txt_line(line):
    line = line.strip()
    if line.startswith('[') and ']' in line:
        return line.split(']', 1)[1].strip()
    return line

def _get_document(modality, video_dir, row_idx):
    """
    Retrieve textual document for vector.
    
    Args:
        modality (str): Type of data processed.
        video_dir (Path): Extracted video data directory.
        row_idx (int): Current index processed.
        
    Returns:
        str: Extracted text string or empty string.
    """
    if modality == "frame":
        return ""

    video_id = video_dir.name
    
    try:
        if modality == "script":
            script_path = (
                TEXT_ANNOTATIONS_DIR / "Scripts" / f"{video_id}.txt"
            )
            if script_path.exists():
                sentences = _parse_script_sentences(script_path)
                if row_idx < len(sentences):
                    return sentences[row_idx]
            else:
                # Check if warned
                if script_path not in _missing_files_warned:
                    logger.warning(f"Script missing: {script_path}")
                    _missing_files_warned.add(script_path)
            return ""

        elif modality == "transcript":
            ts_path = (
                TEXT_ANNOTATIONS_DIR / "Transcripts" / f"{video_id}.txt"
            )
            if ts_path.exists():
                with open(ts_path, 'r', encoding='utf-8') as f:
                    lines = [
                        _parse_txt_line(l) for l in f if l.strip()
                    ]
                    if row_idx < len(lines):
                        return lines[row_idx]
            else:
                # Check if warned
                if ts_path not in _missing_files_warned:
                    logger.warning(f"Transcript missing: {ts_path}")
                    _missing_files_warned.add(ts_path)
            return ""

        elif modality == "aligned_transcript":
            ts_txt_path = (
                TEXT_ANNOTATIONS_DIR / "Transcripts" / f"{video_id}.txt"
            )
            ts_npy_path = video_dir / "transcript_timestamps.npy"
            
            if ts_txt_path.exists() and ts_npy_path.exists():
                timestamps = np.load(ts_npy_path)
                current_time = float(row_idx)
                
                with open(ts_txt_path, 'r', encoding='utf-8') as f:
                    lines = [
                        _parse_txt_line(l) for l in f if l.strip()
                    ]
                
                for i, (start, end) in enumerate(timestamps):
                    if start <= current_time <= end:
                        if i < len(lines):
                            return lines[i]
            else:
                # Check if warned
                if ts_txt_path not in _missing_files_warned:
                    logger.warning(f"Transcript missing: {ts_txt_path}")
                    _missing_files_warned.add(ts_txt_path)
            return ""

    except Exception as e:
        logger.error(
            f"Error parsing document for {modality} "
            f"at {video_dir}, index {row_idx}: {e}"
        )
        return ""
def _load_annotator_script_sentences(video_id, annotator_idx):
    """
    Load and sentence-split the specific annotator's script text:
    Scripts/{video_id}_{annotator_idx}.txt.
    """
    script_path = TEXT_ANNOTATIONS_DIR / "Scripts" / f"{video_id}_{annotator_idx}.txt"
    if not script_path.exists():
        if script_path not in _missing_files_warned:
            logger.warning(f"Script missing: {script_path}")
            _missing_files_warned.add(script_path)
        return []
    return _parse_script_sentences(script_path)


def _process_single_video(path, modality):
    """
    Isolated worker function to read files and prepare data dump.
    """
    video_dir = path.parent
    video_id = video_dir.name

    try:
        raw = np.load(path)
    except Exception as e:
        return False, f"Skipping corrupt or empty file {path}: {e}"

    selected_annotator_idx = None
    script_sentences = None
    if modality == "script" and raw.ndim == 3:
        # VideoXum: pick annotator script closest to the centroid of
        # all 10, and ingest only that one's per-sentence embeddings
        if not np.isfinite(raw).all():
            return False, f"Skipping corrupt script embeddings for {video_id}: NaN/Inf values"
        selected_annotator_idx = select_centroid_representative_annotator(raw)
        raw = raw[selected_annotator_idx] # [M_max, 512]

        script_sentences = _load_annotator_script_sentences(video_id, selected_annotator_idx)
        valid_row_count = int(np.count_nonzero(np.linalg.norm(raw, axis=1)))
        if script_sentences and len(script_sentences) != valid_row_count:
            logger.warning(
                f"{video_id}: annotator {selected_annotator_idx}'s script file "
                f"has {len(script_sentences)} sentences but {valid_row_count} "
                f"valid (non-padded) embedding rows - per-sentence document "
                f"text may be misaligned for this video."
            )

    try:
        arr = _normalise_embeddings(raw)
    except Exception as e:
        return False, f"Skipping corrupt or empty file {path}: {e}"

    v_ids, v_embeddings, v_metadatas, v_documents = [], [], [], []
    skipped_zero = 0
    for row_idx, vec in enumerate(arr):
        if not np.linalg.norm(vec):
            # Zero-padded placeholder
            skipped_zero += 1
            continue

        if modality == "script" and selected_annotator_idx is not None:
            # VideoXum: use the centroid-selected annotator's own script text
            doc_text = ""
            if script_sentences and row_idx < len(script_sentences):
                doc_text = script_sentences[row_idx]
        else:
            # Every other modality, and MrHiSum's script case (no annotator axis)
            doc_text = _get_document(modality, video_dir, row_idx)

        v_ids.append(f"{modality}:{video_id}:{row_idx}")
        v_embeddings.append(vec.tolist())
        metadata = _build_metadata(
            modality, video_id, row_idx, video_dir,
            annotator_idx=selected_annotator_idx,
        )
        v_metadatas.append(metadata)
        v_documents.append(doc_text)

    if skipped_zero:
        logger.info(f"{video_id}/{modality}: skipped {skipped_zero} zero-padded rows.")

    return True, (v_ids, v_embeddings, v_metadatas, v_documents)


def flush_batch(collection, ids, embeddings, metadatas, documents):
    """Upsert batches into ChromaDB (skipping files with exisitng vectors)."""
    if ids:
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

def ingest_embeddings(
        root_dir,
        db_path="Datasets/chroma_db",
        manifest_path="s3_available_videos.txt"
    ):
    client = chromadb.PersistentClient(path=db_path)
    root = Path(root_dir)

    valid_video_ids = set()
    if Path(manifest_path).exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            valid_video_ids = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(valid_video_ids)} valid video IDs from manifest.")
    else:
        print(f"WARNING: Manifest {manifest_path} not found. Proceeding without S3 filtering.")

    collection = client.get_or_create_collection(
        name=CHROMADB_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    for modality, cfg in MODALITY_CONFIG.items():
        print(f"Processing modality: {modality}...")
        videos = sorted(root.rglob(cfg["filename"]))
        if not videos:
            print(f"No videos found for {modality}. Skipping.")
            continue

        # Sequential skip check to prevents DB locking
        videos_to_process = []
        for path in videos:
            video_id = path.parent.name

            if valid_video_ids and video_id not in valid_video_ids:
                continue

            # Check if first vector for video/modality exists
            test_id = f"{modality}:{video_id}:0"
            
            # ID lookup (instsant primary-key search)
            existing = collection.get(ids=[test_id])
            
            if not existing["ids"]:
                videos_to_process.append(path)
                
        print(f"Identified {len(videos_to_process)} new videos to process.")

        # MultiThreading
        ids, embeddings, metadatas, documents = [], [], [], []
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(
                    _process_single_video, p, modality
                    ): p for p in videos_to_process
                }
            
            for future in tqdm(
                as_completed(futures),
                total=len(videos_to_process),
                desc=f"Ingesting {modality}",
                ):
                success, result = future.result()
                
                if not success:
                    # Prevent progress bar breaking
                    tqdm.write(result) 
                    continue
                
                v_ids, v_emb, v_meta, v_doc = result
                ids.extend(v_ids)
                embeddings.extend(v_emb)
                metadatas.extend(v_meta)
                documents.extend(v_doc)

                # Batch dump main thread of vectors
                while len(ids) >= BATCH_SIZE:
                    flush_batch(
                        collection, 
                        ids[:BATCH_SIZE], 
                        embeddings[:BATCH_SIZE], 
                        metadatas[:BATCH_SIZE], 
                        documents[:BATCH_SIZE]
                    )
                    ids = ids[BATCH_SIZE:]
                    embeddings = embeddings[BATCH_SIZE:]
                    metadatas = metadatas[BATCH_SIZE:]
                    documents = documents[BATCH_SIZE:]

        # Dump remaining vectors for modality
        flush_batch(collection, ids, embeddings, metadatas, documents)

    print("Embedding ingestion complete.")

if __name__ == "__main__":
    ingest_embeddings("Datasets/SM-MrHiSum and SM-VideoXum/SM-VideoXum-Training-Data/extracted_data")
    # ingest_embeddings("Datasets/SM-MrHiSum and SM-VideoXum/SM-MrHiSum-Training-Data/extracted_data")