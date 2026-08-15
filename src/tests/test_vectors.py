import chromadb
from collections import Counter

from src.config import CHROMA_DB_DIR, CHROMADB_COLLECTION_NAME

def inspect_chromadb(
        db_path=CHROMA_DB_DIR,
        collection_name=CHROMADB_COLLECTION_NAME,
        ):
    # Connect to local vector store
    client = chromadb.PersistentClient(path=db_path)
    
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(
            f"Collection '{collection_name}' not found."
             " Did the ingestion script run successfully?"
             )
        return

    # Check total number of vectors
    total_count = collection.count()
    print(f"=== Database Summary ===")
    print(f"Total vectors stored: {total_count:,}\n")

    if total_count == 0:
        print("The database is empty.")
        return

    # Retrieve metadata to verify distribution across modalities and collections
    print("Fetching metadata for distribution analysis (this may take a moment)...")
    all_data = collection.get(include=["metadatas"])
    metadatas = all_data["metadatas"]

    modality_counts = Counter(
        meta.get("modality", "unknown") for meta in metadatas if meta
        )
    collection_counts = Counter(
        meta.get("collection_id", "unknown") for meta in metadatas if meta
        )

    print("--- Modality Breakdown ---")
    for modality, count in modality_counts.items():
        print(f" - {modality}: {count:,} vectors")

    # Show top 5 largest collections to ensure hashing distribution worked
    print("\n--- Collection ID Breakdown ---")
    for col_id, count in collection_counts.most_common(5):
        print(f" - {col_id}: {count:,} vectors")

    if len(collection_counts) > 5:
        print(f"   ... and {len(collection_counts) - 5} more collections.")

    # Show sample of specific records to verify documents and metadata structure
    print("\n=== Sample Record Inspection ===")
    
    # Query text-based record (e.g., transcript) to ensure documents parsed correctly
    transcript_sample = collection.get(
        where={"modality": "transcript"},
        limit=1,
        include=["metadatas", "documents"]
    )
    
    if transcript_sample["ids"]:
        print("\n[Sample Transcript Node]")
        print(f"ID: {transcript_sample['ids'][0]}")
        print(f"Metadata: {transcript_sample['metadatas'][0]}")
        print(f"Document Text: {transcript_sample['documents'][0]}")

    # Query visual record to ensure documents are correctly set to None/null
    frame_sample = collection.get(
        where={"modality": "frame"},
        limit=1,
        include=["metadatas", "documents"]
    )
    
    if frame_sample["ids"]:
        print("\n[Sample Frame Node]")
        print(f"ID: {frame_sample['ids'][0]}")
        print(f"Metadata: {frame_sample['metadatas'][0]}")
        print(f"Document Text: {frame_sample['documents'][0]} (should be None)")

if __name__ == "__main__":
    inspect_chromadb()