import numpy as np
from neo4j import GraphDatabase
from pathlib import Path

from src.preprocess.utils import get_collection_id, select_centroid_representative_annotator
from src.config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
)

# Set dataset root path
DATASET_ROOT = Path(
    "Datasets/SM-MrHiSum and SM-VideoXum/SM-VideoXum-Training-Data/extracted_data"
    )

def setup_constraints(driver):
    """Create uniqueness constraints and indexes for high-speed lookups."""
    queries = [
        "CREATE CONSTRAINT FOR (c:Collection) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT FOR (v:Video) REQUIRE v.id IS UNIQUE",
        "CREATE CONSTRAINT FOR (s:Segment) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT FOR (f:Frame) REQUIRE f.id IS UNIQUE",
        "CREATE CONSTRAINT FOR (t:Transcript) REQUIRE t.id IS UNIQUE",
        "CREATE CONSTRAINT FOR (sc:Script) REQUIRE sc.id IS UNIQUE"
    ]
    with driver.session() as session:
        for q in queries:
            try:
                session.run(q)
            except Exception:
                # Constraint likely exists
                pass
    print("Database constraints verified.")

def ingest_video_graph(tx, video_id, collection_id, change_points):
    """
    Cypher transaction to merge Video, Collection, and Segments,
    and link segments chronologically.
    """
    # Merge Collection and Video
    tx.run("""
        MERGE (c:Collection {id: $collection_id})
        MERGE (v:Video {id: $video_id})
        MERGE (v)-[:IN_COLLECTION]->(c)
    """, collection_id=collection_id, video_id=video_id)

    # Iterate change points (shots/scenes) to link segments to videos
    for idx, (start_frame, end_frame) in enumerate(change_points):
        segment_id = f"{video_id}:seg_{idx}"
        
        tx.run(
            """
            MATCH (v:Video {id: $video_id})
            MERGE (s:Segment {id: $segment_id})
            ON CREATE SET s.start_frame = $start_frame, s.end_frame = $end_frame, s.segment_index = $idx
            MERGE (s)-[:PART_OF]->(v)
            """,
            video_id=video_id,
            segment_id=segment_id,
            start_frame=int(start_frame),
            end_frame=int(end_frame),
            idx=idx,
        )

def link_frames_and_transcripts(tx, video_id, transcript_chunks):
    """
    Link frame and transcript nodes to corresponding segments
    using start and end frames.
    """
    # Map frames to segments
    tx.run("""
        MATCH (v:Video {id: $video_id})-[:PART_OF]-(s:Segment)
        UNWIND range(s.start_frame, s.end_frame) AS f_idx
        MERGE (f:Frame {id: 'frame:' + $video_id + ':' + f_idx})
        ON CREATE SET f.modality = 'frame', f.video_id = $video_id, f.frame_id = f_idx
        MERGE (f)-[:BELONGS_TO]->(s)
    """, video_id=video_id)

    # Map transcript chunks to segments (assigns transcript to multiple segments
    # if they overlap)
    for chunk_idx, (start_time, end_time) in enumerate(transcript_chunks):
        tx.run(
            """
            MATCH (v:Video {id: $video_id})-[:PART_OF]-(s:Segment)
            WHERE s.start_frame <= $end_time AND s.end_frame >= $start_time
            MERGE (t:Transcript {id: 'transcript:' + $video_id + ':' + $chunk_idx})
            ON CREATE SET t.modality = 'transcript', t.video_id = $video_id, t.chunk_start = $start_time, t.chunk_end = $end_time
            MERGE (t)-[:BELONGS_TO]->(s)
            """,
            video_id=video_id,
            chunk_idx=chunk_idx,
            start_time=float(start_time),
            end_time=float(end_time),
        )

def link_scripts(tx, video_id, script_row_indices):
    """
    Link sentence-level script nodes to corresponding video.

    script_row_indices is the set of row indices stored in ChromaDB
    for video script vectors.
    """
    tx.run(
        """
        MATCH (v:Video {id: $video_id})
        UNWIND $script_row_indices AS s_idx
        MERGE (sc:Script {id: 'script:' + $video_id + ':' + s_idx})
        ON CREATE SET sc.modality = 'script', sc.video_id = $video_id, sc.sentence_index = s_idx
        MERGE (sc)-[:DESCRIBES]->(v)
    """,
        video_id=video_id,
        script_row_indices=[int(i) for i in script_row_indices],
    )

def run_pipeline():
    # Instantiate Neo4J
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    setup_constraints(driver)

    # Only graph videos available in S3 bucket
    manifest_path = "s3_available_videos.txt"
    valid_video_ids = set()
    if Path(manifest_path).exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            valid_video_ids = {line.strip() for line in f if line.strip()}
        print(
            f"Loaded {len(valid_video_ids)} valid video IDs from manifest."
        )
    else:
        print(
            f"WARNING: Manifest {manifest_path} not found."
            "Proceeding without S3 filtering."
        )

    root = DATASET_ROOT.resolve()
    video_dirs = [d for d in root.iterdir() if d.is_dir()]

    print(f"Found {len(video_dirs)} videos to process into Neo4j.")

    for video_dir in video_dirs:
        video_id = video_dir.name

        # Skip if video isn't in S3 bucket
        if valid_video_ids and video_id not in valid_video_ids:
            continue

        collection_id = get_collection_id(video_id)
        
        # Load change points if available
        cp_path = video_dir / "change_points.npy"
        if not cp_path.exists():
            # Mock segments if change points aren't available
            change_points = np.array([[0, 100], [101, 200]])
        else:
            change_points = np.load(cp_path)

        # Load timestamps for transcripts if available
        ts_path = video_dir / "transcript_timestamps.npy"
        transcript_chunks = np.load(ts_path) if ts_path.exists() else []

        # Count total frames from video embedding shape if available
        vid_emb_path = video_dir / "video_embeddings.npy"
        num_frames = len(np.load(vid_emb_path)) if vid_emb_path.exists() else 0

        # Determine which script rows are in ChromaDB for this video
        script_emb_path = video_dir / "script_embeddings.npy"
        if script_emb_path.exists():
            script_arr = np.load(script_emb_path)

            if script_arr.ndim == 3:
                # VideoXum: centroid-based annotator selection
                selected_idx = select_centroid_representative_annotator(script_arr)
                annotator_block = script_arr[selected_idx]
            elif script_arr.ndim == 2:
                annotator_block = script_arr  # MrHiSum: already one script
            else:
                annotator_block = np.empty((0, script_arr.shape[-1]))

            norms = np.linalg.norm(annotator_block, axis=1)
            script_row_indices = np.where(norms > 0)[0]  # drop zero-padded rows
        else:
            script_row_indices = np.array([], dtype=int)

        with driver.session() as session:
            session.execute_write(
                ingest_video_graph, video_id, collection_id, change_points
            )
            if num_frames > 0:
                session.execute_write(
                    link_frames_and_transcripts,
                    video_id,
                    transcript_chunks,
                )
            # Store to graph
            if len(script_row_indices) > 0:
                session.execute_write(link_scripts, video_id, script_row_indices)

        print(f"Successfully graphed video: {video_id}")

    driver.close()
    print("Graph ingestion complete.")

if __name__ == "__main__":
    run_pipeline()