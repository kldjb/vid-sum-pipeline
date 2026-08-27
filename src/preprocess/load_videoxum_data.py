import logging
import os
import boto3
import yt_dlp
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

def download_and_upload_videos(extracted_components_dir, bucket_name, s3_prefix):
    """
    Identifies video IDs from local directory, downloads the 
    corresponding YouTube videos, and uploads them to S3.
    """
    s3_client = boto3.client('s3')
    
    # Identify all unique video IDs from extracted folders (e.g. "v_--0edUL8zmA")
    video_folders = [
        f.name for f in Path(extracted_components_dir).iterdir() if f.is_dir()
        ]
    
    for folder_name in video_folders:
        # Strip "v_" prefix to get YouTube ID
        if folder_name.startswith("v_"):
            youtube_id = folder_name[2:]
        else:
            youtube_id = folder_name
            
        youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"
        s3_target_key = f"{s3_prefix}/{folder_name}.mp4"
        
        # Check if already uploaded to S3
        try:
            s3_client.head_object(Bucket=bucket_name, Key=s3_target_key)
            logger.info(f"Skipping {folder_name}: Already in S3.")
            continue
        except s3_client.exceptions.ClientError:
            # Video not in S3, proceed with download
            pass

        logger.info(f"Downloading {youtube_id}...")
        
        # Use temp dir
        with tempfile.TemporaryDirectory() as temp_dir:
            local_video_path = os.path.join(temp_dir, f"{folder_name}.mp4")
            
            # Configure yt-dlp to download best video/audio and save to temp dir
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
                    failed_videos.write(f"{folder_name}\n")
                continue

if __name__ == "__main__":
    BUCKET = r"video-sum-pipeline"
    EXTRACTED_DIR = r"C:\Users\kldjb\Documents\Birkbeck University\Project\Datasets\SM-MrHiSum and SM-VideoXum\SM-VideoXum-Training-Data\extracted_data\\"
    TARGET_FOLDER = "raw-data/SM-MrHiSum and SM-VideoXum/SM-VideoXum-Training-Data/videos"
    download_and_upload_videos(EXTRACTED_DIR, BUCKET, s3_prefix=TARGET_FOLDER)