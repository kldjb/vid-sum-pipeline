import hashlib

def get_collection_id(video_id, num_collections=20):
    """
    Create deterministic collection ID between 1 and num_collections 
    based on the video_id string.
    """
    hash_val = int(hashlib.md5(video_id.encode('utf-8')).hexdigest(), 16)
    return f"col_{(hash_val % num_collections) + 1}"