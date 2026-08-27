import boto3
import os

BUCKET_NAME = "video-sum-pipeline"
S3_PREFIX = "raw-data/SM-MrHiSum and SM-VideoXum/ActivityNet-Data/"
MANIFEST_FILE = "s3_available_videos.txt"

def generate_manifest():
    print("Fetching list of available videos from S3...")
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    
    video_ids = set()
    
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=S3_PREFIX):
        if 'Contents' in page:
            for obj in page['Contents']:
                # Extract filename without prefix/extension
                # e.g., 'raw-data/.../v_abc123.mp4' -> 'v_abc123'
                filename = os.path.basename(obj['Key'])
                if filename.endswith(".mp4"):
                    video_id = filename.replace(".mp4", "")
                    video_ids.add(video_id)
                    
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        for vid in sorted(video_ids):
            f.write(f"{vid}\n")
            
    print(f"Success! Saved {len(video_ids)} valid video IDs to {MANIFEST_FILE}")

if __name__ == "__main__":
    generate_manifest()