import logging
import os
import boto3
import yt_dlp
import tempfile
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def download_and_upload_mapped_videos(
        extracted_components_dir,
        metadata_csv_path,
        bucket_name,
        s3_prefix,
    ):
    """
    Reads metadata mapping, identifies required video IDs from local directory, 
    downloads corresponding YouTube videos, and uploads them to S3 with their generic names.
    """
    s3_client = boto3.client('s3')
    
    # Load mapping
    logger.info("Loading metadata mapping...")
    df = pd.read_csv(metadata_csv_path)

    # Create dict of {'video_1': 'JhdjUam0l6A', ...}
    id_mapping = dict(zip(df['video_id'], df['youtube_id']))
    
    # Identify required videos based on extracted folders
    video_folders = [
        f.name for f in Path(extracted_components_dir).iterdir() if f.is_dir()
    ]
    
    for folder_name in video_folders:
        # folder_name expected to be "video_1", "video_2", etc.
        if folder_name not in id_mapping:
            logger.warning(f"Skipping {folder_name}: Not found in metadata.csv")
            continue
            
        youtube_id = id_mapping[folder_name]
        youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"
        s3_target_key = f"{s3_prefix}/{folder_name}.mp4"
        
        # Check if already uploaded to S3
        try:
            s3_client.head_object(Bucket=bucket_name, Key=s3_target_key)
            logger.info(f"Skipping {folder_name}: Already in S3.")
            continue
        except s3_client.exceptions.ClientError:
            # Continue download
            pass

        logger.info(f"Downloading {folder_name} (YouTube ID: {youtube_id})...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            local_video_path = os.path.join(temp_dir, f"{folder_name}.mp4")
            
            ydl_opts = {
                'format': 'best',
                'outtmpl': local_video_path,
                'quiet': True,
                'no_warnings': True,
                'cookiefile': r"C:\Users\kldjb\Downloads\www.youtube.com_cookies.txt",
                'sleep_interval': 10,
                'max_sleep_interval': 30,
                'sleep_interval_requests': 2,
            }
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])

                logger.info(
                    f"Uploading {folder_name} to s3://{bucket_name}/{s3_target_key}"
                    )
                s3_client.upload_file(local_video_path, bucket_name, s3_target_key)

            except Exception as e:
                logger.exception(f"Failed to process {folder_name}")
                with open("failed_videos.txt", "a", encoding="utf-8") as failed_videos:
                    failed_videos.write(f"{folder_name},{youtube_id}\n")
                continue

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    BUCKET = r"video-sum-pipeline"
    EXTRACTED_DIR = r"C:\Users\kldjb\Documents\Birkbeck University\Project\Datasets\SM-MrHiSum and SM-VideoXum\SM-MrHiSum-Training-Data\extracted_data\\"
    METADATA_PATH = r"C:\Users\kldjb\Documents\Birkbeck University\Project\Datasets\SM-MrHiSum and SM-VideoXum\SM-MrHiSum-Training-Data\metadata.csv"
    TARGET_FOLDER = r"raw-data/SM-MrHiSum and SM-VideoXum/SM-MrHiSum-Training-Data/videos"
    
    download_and_upload_mapped_videos(
        EXTRACTED_DIR,
        METADATA_PATH,
        BUCKET,
        s3_prefix=TARGET_FOLDER,
    )