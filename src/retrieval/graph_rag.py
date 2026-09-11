import logging
import sys
import numpy as np
import chromadb
from neo4j import GraphDatabase

from src.config import (
    CHROMADB_COLLECTION_NAME,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    CHROMA_DB_DIR,
)

logger = logging.getLogger(__name__)

class GraphRAGPipeline:
    def __init__(self):
        # Instantiate Neo4j
        self.neo_driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

        # Verify Neo4j connection (database running)
        try:
            self.neo_driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j.")
        except Exception as e:
            logger.error("\nCRITICAL: Could not connect to Neo4j!")
            logger.error("Please ensure the Neo4j database is started and running.")
            sys.exit(1)
        
        # Instantiate ChromaDB Client
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        self.collection = self.chroma_client.get_collection(name=CHROMADB_COLLECTION_NAME)

    def _vector_search_global(
            self,
            query_embedding,
            n_unique_videos=3,
            overfetch_factor=10,
            allowed_modalities=None
            ):
        """
        Fetch top matches across all data, deduplicating video hits.
        
        Overfetch factor required to ensure enough diversity in results
        to return n_unique_videos number of unique videos from hits.
        """
        # Build ChromaDB modality filter
        where_clause = None
        if allowed_modalities:
            if len(allowed_modalities) == 1:
                where_clause = {"modality": allowed_modalities[0]}
            else:
                where_clause = {"modality": {"$in": allowed_modalities}}
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_unique_videos * overfetch_factor,
            where=where_clause
        )
        
        unique_videos = set()
        selected_ids = []
        
        if not results['ids']:
            return []
            
        for vec_id in results['ids'][0]:
            # Assumes your ID format is modality:video_id:index (e.g., 'frame:v_123:45')
            video_id = vec_id.split(':')[1]
            
            if video_id not in unique_videos:
                unique_videos.add(video_id)
                selected_ids.append(vec_id)
                
            if len(unique_videos) == n_unique_videos:
                break
                
        return selected_ids

    def _vector_search_video(
            self,
            query_embedding,
            target_video_id,
            n_results=3,
            allowed_modalities=None,
        ):
        """
        Vector search scoped to single video using ChromaDB metadata
        filter.
        """
        where_clause = {"video_id": target_video_id}
        if allowed_modalities:
            if len(allowed_modalities) == 1:
                where_clause = {"$and": [where_clause, {"modality": allowed_modalities[0]}]}
            else:
                where_clause = {"$and": [where_clause, {"modality": {"$in": allowed_modalities}}]}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
        )

        if not results['ids'] or not results['ids'][0]:
            return []

        return results['ids'][0]

    def fetch_video_vectors(self, target_video_id):
        """
        Fetch every vector (embedding + metadata) belonging to one video
        in a single ChromaDB call. 
        """
        return self.collection.get(
            where={"video_id": target_video_id},
            include=["embeddings", "metadatas"],
        )

    def fetch_global_vectors_by_modality(self, modality, id_batch_size=500):
        """
        Fetch every vector of a single modality across entire vector store,
        pre-normalised for fast repeated cosine-similarity ranking.

        Uses metadata to retrieve modality matching ids, then embeddings are
        pulled back vusing id lookups (chunked to stay under SQLite's
        variable limit, and delay retried if a chunk hits the limit).
        """
        id_only = self.collection.get(
            where={"modality": modality},
            include=[],
        )
        all_ids = id_only.get("ids", [])
        if not all_ids:
            return {"embeddings": np.zeros((0, 0), dtype=np.float32), "video_ids": [], "modalities": []}

        all_embeddings = []
        all_video_ids = []
        i = 0
        current_batch_size = id_batch_size

        while i < len(all_ids):
            chunk = all_ids[i:i + current_batch_size]
            try:
                raw = self.collection.get(
                    ids=chunk,
                    include=["embeddings", "metadatas"],
                )
            except Exception as e:
                if "too many SQL variables" in str(e) and current_batch_size > 10:
                    new_batch_size = max(10, current_batch_size // 4)
                    logger.warning(
                        f"'{modality}' fetch hit SQLite's variable limit at "
                        f"batch_size={current_batch_size} - retrying this "
                        f"chunk with batch_size={new_batch_size}."
                    )
                    current_batch_size = new_batch_size
                    continue
                raise

            all_embeddings.append(np.asarray(raw["embeddings"], dtype=np.float32))
            all_video_ids.extend(meta.get("video_id") for meta in raw.get("metadatas", []))
            i += len(chunk)

        if not all_embeddings:
            return {"embeddings": np.zeros((0, 0), dtype=np.float32), "video_ids": [], "modalities": []}

        embeddings = np.concatenate(all_embeddings, axis=0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalised = embeddings / norms

        return {
            "embeddings": normalised,
            "video_ids": all_video_ids,
            "modalities": [modality] * len(all_video_ids),
        }

    @staticmethod
    def _rank_cached_vectors(
            query_embedding,
            cached_data,
            allowed_modalities,
            n_results,
            modality_calibration=None,
        ):
        """
        Manually calculate cosine-similarity over pre-fetched vectors.

        If modality_calibration used each candidate's raw cosine score is
        z-score normalised against its own modality's typical similarity
        distribution before ranking, correcting the modality gap.
        """
        query_vec = np.asarray(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        scored = []
        ids = cached_data.get("ids", [])
        embeddings = cached_data.get("embeddings", [])
        metadatas = cached_data.get("metadatas", [])

        for vec_id, embedding, meta in zip(ids, embeddings, metadatas):
            modality = meta.get("modality")
            if allowed_modalities and modality not in allowed_modalities:
                continue
            emb = np.asarray(embedding, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm == 0:
                continue
            score = float(np.dot(query_vec, emb / norm))

            if modality_calibration and modality in modality_calibration:
                stats = modality_calibration[modality]
                std = stats["std"] if stats["std"] > 0 else 1.0
                score = (score - stats["mean"]) / std

            scored.append((score, vec_id))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [vec_id for _, vec_id in scored[:n_results]]

    @staticmethod
    def calibrate_global_cache(cached_global_vectors, modality_calibration):
        """
        Precompute per-vector z-score calibration arrays for a global-
        search candidate cache, so _rank_cached_global_vectors can apply
        modality-gap correction as a single vectorised operation per query
        rather than a per-vector lookup.

        Args:
            cached_global_vectors: a cache dict from
                fetch_global_vectors_by_modality, or several such caches
                concatenated together - must include a "modalities" list
                aligned 1:1 with "embeddings"/"video_ids".
            modality_calibration: {modality: {"mean": float, "std":
                float}} from calibrate_modality_similarity_stats.

        Returns:
            dict: cache dict with "cal_means" and "cal_stds" numpy arrays
            added. A modality absent from modality_calibration is left
            uncalibrated.
        """
        video_ids = cached_global_vectors["video_ids"]
        n = len(video_ids)
        modalities = cached_global_vectors.get("modalities")
        if modalities is None or len(modalities) != n:
            raise ValueError(
                "calibrate_global_cache requires cached_global_vectors to "
                "include a 'modalities' list perfectly aligned with 'embeddings'/"
                "'video_ids' - fetch_global_vectors_by_modality provides "
                "this so make sure it's preserved (concatenated alongside "
                "embeddings/video_ids) when combining caches per config."
            )

        modalities_arr = np.asarray(modalities)
        means = np.zeros(n, dtype=np.float32)
        stds = np.ones(n, dtype=np.float32)
        for modality, stats in modality_calibration.items():
            mask = modalities_arr == modality
            if not mask.any():
                continue
            means[mask] = stats["mean"]
            stds[mask] = stats["std"] if stats["std"] > 0 else 1.0

        cached_global_vectors["cal_means"] = means
        cached_global_vectors["cal_stds"] = stds
        return cached_global_vectors

    @staticmethod
    def _rank_cached_global_vectors(query_embedding, cached_global_vectors, n_unique_videos):
        """
        Cosine-similarity ranking over a pre-fetched, pre-normalised, global
        vector set, deduplicated to unique videos.

        If cached_global_vectors contains "cal_means"/"cal_stds" (added
        by calibrate_global_cache), each score is z-score normalised
        against its modality's typical distribution before ranking.
        Without those, ranking falls back to raw cosine similarity.

        Uses partial top-k selection (np.argpartition) rather than a
        full sort to optimise bulk evaluation.
        """
        embeddings = cached_global_vectors["embeddings"]
        video_ids = cached_global_vectors["video_ids"]
        if len(video_ids) == 0:
            return []

        query_vec = np.asarray(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        scores = embeddings @ query_vec

        cal_means = cached_global_vectors.get("cal_means")
        cal_stds = cached_global_vectors.get("cal_stds")
        if cal_means is not None and cal_stds is not None:
            scores = (scores - cal_means) / cal_stds

        # Large overfetch margin to ensure enough diversity in results
        # to return n_unique_videos
        overfetch = min(len(scores), max(n_unique_videos * 20, 50))
        top_unsorted = np.argpartition(-scores, overfetch - 1)[:overfetch]
        top_sorted = top_unsorted[np.argsort(-scores[top_unsorted])]

        seen = []
        for idx in top_sorted:
            vid = video_ids[idx]
            if vid not in seen:
                seen.append(vid)
            if len(seen) == n_unique_videos:
                break
        return seen

    def _fetch_graph_context(self, node_ids):
        """
        Pass ChromaDB vector IDs to Neo4j to retrieve full segment context.

        node_ids are split by modality (encoded in the ID prefix, e.g.
        'frame:vid:5') and matched against their own label (Frame /
        Transcript / Script) so each branch can use its own index,
        rather than a single label-less MATCH (hit {id: node_id}).
        """
        ids_by_modality = {"frame": [], "transcript": [], "script": []}
        for node_id in node_ids:
            modality = node_id.split(':')[0]
            if modality in ids_by_modality:
                ids_by_modality[modality].append(node_id)

        query = """
        CALL {
            UNWIND $frame_ids AS node_id
            MATCH (hit:Frame {id: node_id})
            RETURN hit
        UNION
            UNWIND $transcript_ids AS node_id
            MATCH (hit:Transcript {id: node_id})
            RETURN hit
        UNION
            UNWIND $script_ids AS node_id
            MATCH (hit:Script {id: node_id})
            RETURN hit
        }

        // Traverse Frame, Transcript or Script to find Segments and Videos
        OPTIONAL MATCH (hit)-[:BELONGS_TO]->(s:Segment)-[:PART_OF]->(v_seg:Video)
        OPTIONAL MATCH (hit)-[:DESCRIBES]->(v_script:Video)
        
        // Consolidate video node (one will be null, one will have data)
        WITH hit, s, coalesce(v_seg, v_script) AS v

        // Group by Video (v) and Segment (s) to aggregate vector matches within the same scene
        WITH v, s,
            collect(DISTINCT hit.id) AS matched_vector_ids,
            collect(DISTINCT hit.modality) AS matched_modalities
        
        // Fetch frames and count for segment
        OPTIONAL MATCH (f:Frame)-[:BELONGS_TO]->(s)
        WITH v, s, matched_vector_ids, matched_modalities, count(DISTINCT f) AS total_frames_in_scene
        
        // Fetch transcripts for segment
        OPTIONAL MATCH (t:Transcript)-[:BELONGS_TO]->(s)

        // Package into structured JSON-like format
        RETURN 
            v.id AS video_id,
            s.id AS segment_id,
            s.start_frame AS start_frame,
            s.end_frame AS end_frame,
            matched_vector_ids,
            matched_modalities,
            total_frames_in_scene,
            collect(DISTINCT toString(t.chunk_start) + '-' + toString(t.chunk_end)) AS transcript_timestamps

        // Rank by relevance (number of matching vectors in segment)
        ORDER BY size(matched_vector_ids) DESC 
        """
        
        with self.neo_driver.session() as session:
            result = session.run(
                query,
                frame_ids=ids_by_modality["frame"],
                transcript_ids=ids_by_modality["transcript"],
                script_ids=ids_by_modality["script"],
            )
            return [record.data() for record in result]

    def _fetch_raw_vector_context(self, node_ids):
        """
        Bypasses Neo4j for ablation baselines - formats ChromaDB metadata 
        to mimic Cypher output structure.
        """
        raw_data = self.collection.get(ids=node_ids, include=["metadatas"])
        mock_context = []
        
        for i, node_id in enumerate(raw_data['ids']):
            meta = raw_data['metadatas'][i]
            modality = meta.get("modality")
            
            # Timestamp extraction for baseline (only available for
            # certain modalities, otherwise generic defaults used)
            start = meta.get("timestamp_sec", meta.get("chunk_start", 0.0))
            end = meta.get("timestamp_sec", meta.get("chunk_end", 0.0))

            # Enforce 5s boundaries for missing temporal bounds
            if start == end:
                end = start + 5.0
            elif modality == "frame" or modality == "aligned_transcript":
                # Add default point-in-time vectors padding
                end += 5.0
                
            mock_context.append({
                "video_id": meta.get("video_id"),
                "segment_id": f"mock_seg_{node_id}",
                "start_frame": start,
                "end_frame": end,
                "matched_vector_ids": [node_id],
                "matched_modalities": [modality],
                "total_frames_in_scene": 1,
                "transcript_timestamps": []
            })
            
        return mock_context

    def retrieve(
            self,
            query_embedding,
            search_mode="global",
            target_video_id=None,
            allowed_modalities=None,
            use_graph=True,
            n_results=3,
            overfetch_factor=1500,
            cached_video_vectors=None,
            modality_calibration=None,
        ):
        """
        Main GraphRAG pipeline.
        
        Args:
            - query_embedding
            - search_mode: 'global' (returns unique videos) or 'video' (returns
                specific scenes from 1 video).
            - target_video_id (optional): ID of video to summarise query-relevant scenes.
            - allowed_modalities: list of modalities to filter (e.g., ["frame", "transcript", "script"]).
            - use_graph: whether to traverse Neo4j graph for context or just return ChromaDB hits.
            - n_results: Top-k query-relevant results to return.
            - overfetch_factor: Factor by which to overfetch results before filtering.
            - cached_video_vectors (optional): pre-fetched result of vectors for single video.
            - modality_calibration (optional): per-modality {mean, std} similarity
                stats from calibrate_modality_similarity_stats. Only applied on the
                cached_video_vectors ranking path.
        """
        if search_mode == "global":
            logger.info(
                f"Searching ChromaDB for top {n_results} vector matches..."
                )
            # Get top matching vector IDs
            top_ids = self._vector_search_global(
                query_embedding,
                n_unique_videos=n_results,
                overfetch_factor=overfetch_factor,
                allowed_modalities=allowed_modalities,
                )
        elif search_mode == "video":
            if not target_video_id:
                raise ValueError(
                    "target_video_id must be provided for video-specific searches."
                    )
            if cached_video_vectors is not None:
                logger.info(
                    f"Ranking cached vectors locally for video {target_video_id}..."
                    )
                # Get cached top matching vector IDs
                top_ids = self._rank_cached_vectors(
                    query_embedding,
                    cached_video_vectors,
                    allowed_modalities,
                    n_results,
                    modality_calibration=modality_calibration,
                    )
            else:
                logger.info(
                    f"Searching ChromaDB for top {n_results} vector matches for video"
                    f" {target_video_id}..."
                    )
                # Get top matching vector IDs
                top_ids = self._vector_search_video(
                    query_embedding,
                    target_video_id,
                    n_results=n_results,
                    allowed_modalities=allowed_modalities,
                    )
        else:
            raise ValueError("Invalid search_mode. Must be 'global' or 'video'.")
        
        if not top_ids:
            logger.info("No results found for query.")
            return []
            
        if use_graph:
            logger.info("Traversing Neo4j for surrounding multimodal context...")
            graph_context = self._fetch_graph_context(top_ids)
        else:
            logger.info("Ablation Mode: Skipping Neo4j graph traversal.")
            graph_context = self._fetch_raw_vector_context(top_ids)
            
        return graph_context[:n_results]

    def search_top_videos(
            self,
            query_embedding,
            n_results=10,
            allowed_modalities=None,
            cached_global_vectors=None,
        ):
        """
        Rank-ordered list of unique video IDs for global search. Used for
        Recall@K / MRR evaluation, where rank of videos returned first matters,
        more than full segment context.

        Args:
            query_embedding: Query embedding vector.
            n_results: Number of unique videos to return.
            allowed_modalities: List of modalities to filter (e.g., ["frame",
            "transcript", "script"]).
            cached_global_vectors (optional): pre-fetched result(s) of
                fetch_global_vectors_by_modality. When provided,
                ranks locally in numpy instead of hitting ChromaDB.

        Returns list of video IDs in rank order.
        """
        if cached_global_vectors is not None:
            return self._rank_cached_global_vectors(
                query_embedding, cached_global_vectors, n_results
            )

        vec_ids = self._vector_search_global(
            query_embedding,
            n_unique_videos=n_results,
            overfetch_factor=max(10, n_results * 2),
            allowed_modalities=allowed_modalities,
        )
        return [vec_id.split(':')[1] for vec_id in vec_ids]

    def close(self):
        self.neo_driver.close()

if __name__ == "__main__":
    # Quick test
    pipeline = GraphRAGPipeline()
    
    mock_query_embedding = np.random.rand(512).astype(np.float32).tolist()
    
    context = pipeline.retrieve(mock_query_embedding)
    
    logger.info("\n--- Final GraphRAG Context ---")
    for scene in context:
        logger.info(scene)
        
    pipeline.close()