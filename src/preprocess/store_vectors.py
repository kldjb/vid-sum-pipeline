from pathlib import Path
import numpy as np
import re
import chromadb
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.preprocess.utils import get_collection_id
from src.config import CHROMADB_COLLECTION_NAME, MODALITY_CONFIG


BATCH_SIZE = 500

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

def _build_metadata(modality, video_id, row_idx, video_dir):
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
    elif modality == "transcript":
        start, end = _get_transcript_timestamps(video_dir, row_idx)
        metadata.update({"chunk_start": start, "chunk_end": end})
    elif modality == "aligned_transcript":
        metadata.update({"frame_id": row_idx, "timestamp_sec": float(row_idx)})
        
    return metadata

def _parse_script_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().replace('\n', ' ').strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def _parse_txt_line(line):
    line = line.strip()
    if line.startswith('[') and ']' in line:
        return line.split(']', 1)[1].strip()
    return line

def _get_document(modality, video_dir, row_idx):
    if modality == "frame":
        # ChromaDB requires strings; empty string replaces None
        return ""

    try:
        if modality == "script":
            script_path = video_dir / "script.txt"
            if script_path.exists():
                sentences = _parse_script_sentences(script_path)
                if row_idx < len(sentences):
                    return sentences[row_idx]
            return ""

        elif modality == "transcript":
            transcript_path = video_dir / "transcript.txt"
            if transcript_path.exists():
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    lines = [_parse_txt_line(line) for line in f if line.strip()]
                    if row_idx < len(lines):
                        return lines[row_idx]
            return ""

        elif modality == "aligned_transcript":
            transcript_path = video_dir / "transcript.txt"
            ts_path = video_dir / "transcript_timestamps.npy"
            
            if transcript_path.exists() and ts_path.exists():
                timestamps = np.load(ts_path) 
                current_time = float(row_idx) 
                
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    lines = [_parse_txt_line(line) for line in f if line.strip()]
                
                for chunk_idx, (start, end) in enumerate(timestamps):
                    if start <= current_time <= end and chunk_idx < len(lines):
                        return lines[chunk_idx]
            return ""

    except Exception as e:
        print(
            f"Error parsing document for {modality}"
            f" at {video_dir}, index {row_idx}: {e}"
            )
        return ""

def _process_single_video(path, modality):
    """
    Isolated worker function to read files and prepare data dump.
    """
    video_dir = path.parent
    video_id = video_dir.name
    
    try:
        arr = _normalise_embeddings(np.load(path))
    except Exception as e:
        return False, f"Skipping corrupt or empty file {path}: {e}"

    v_ids, v_embeddings, v_metadatas, v_documents = [], [], [], []
    for row_idx, vec in enumerate(arr):
        doc_text = _get_document(modality, video_dir, row_idx)

        v_ids.append(f"{modality}:{video_id}:{row_idx}")
        v_embeddings.append(vec.tolist())
        v_metadatas.append(_build_metadata(modality, video_id, row_idx, video_dir))
        v_documents.append(doc_text)

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