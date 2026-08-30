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
            overfetch_factor=1500,
            allowed_modalities=None,
        ):
        """
        Filters global vector search for target video.
        """
        # Build ChromaDB modality filter
        where_clause = None
        if allowed_modalities:
            if len(allowed_modalities) == 1:
                where_clause = {"modality": allowed_modalities[0]}
            else:
                where_clause = {"modality": {"$in": allowed_modalities}}
        
        # Query ChromaDB for global neighborhood of near matches
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results * overfetch_factor,
            where=where_clause
        )
        
        if not results['ids']:
            return []
            
        # Post-Filter to video (IDs are formatted like 'frame:v_sUL9HAplalo:45')
        logger.info(f"Filtering vectors to target video: {target_video_id}")
        video_specific_ids = []
        for vec_id in results['ids'][0]:
            # Check if target video ID is in the ID string
            if f":{target_video_id}:" in vec_id:
                video_specific_ids.append(vec_id)
                
        # Return hits
        return video_specific_ids

    def fetch_video_vectors(self, target_video_id):
        """
        Fetch every vector (embedding + metadata) belonging to one video
        in a single ChromaDB call. 
        """
        return self.collection.get(
            where={"video_id": target_video_id},
            include=["embeddings", "metadatas"],
        )
    
    @staticmethod
    def _rank_cached_vectors(
            query_embedding,
            cached_data,
            allowed_modalities,
            n_results,
        ):
        """
        Manually calculate cosine-similarity over pre-fetched vectors
        matching what ChromaDB would return.
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
            if allowed_modalities and meta.get("modality") not in allowed_modalities:
                continue
            emb = np.asarray(embedding, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm == 0:
                continue
            score = float(np.dot(query_vec, emb / norm))
            scored.append((score, vec_id))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [vec_id for _, vec_id in scored[:n_results]]

    def _fetch_graph_context(self, node_ids):
        """Pass ChromaDB IDs to Neo4j to retrieve full segment context."""
        query = """
        // Take list of IDs from ChromaDB
        UNWIND $node_ids AS node_id
        
        // Match specific node (Frame, Transcript, or Script)
        MATCH (hit {id: node_id})
        
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
            result = session.run(query, node_ids=node_ids)
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
        """
        if search_mode == "global":
            logger.info(
                f"Searching ChromaDB for top {n_results} vector matches..."
                )
            top_ids = self._vector_search_global(
                query_embedding,
                n_unique_videos=n_results,
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
                top_ids = self._rank_cached_vectors(
                    query_embedding,
                    cached_video_vectors,
                    allowed_modalities,
                    n_results,
                    )
            else:
                logger.info(
                    f"Searching ChromaDB for top {n_results} vector matches for video"
                    f" {target_video_id}..."
                    )
                top_ids = self._vector_search_video(
                    query_embedding,
                    target_video_id,
                    n_results=n_results,
                    overfetch_factor=overfetch_factor,
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