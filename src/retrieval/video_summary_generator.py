import logging
import subprocess
import tempfile
from pathlib import Path
import boto3

logger = logging.getLogger(__name__)

class VideoSummaryGenerator:
    def __init__(self, s3_bucket: str, s3_prefix: str, output_dir: Path):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.s3_client = boto3.client('s3')

    def _download_video_from_s3(
            self,
            video_id: str,
            download_path: Path,
            ) -> bool:
        """Fetch raw video from S3."""
        s3_key = f"{self.s3_prefix}/{video_id}.mp4"

        try:
            logger.info(f"Downloading raw video '{video_id}.mp4' from S3...")
            self.s3_client.download_file(
                self.s3_bucket, s3_key, str(download_path)
                )
            return True
        except Exception as e:
            logger.error(
                f"Failed to download {video_id} from S3. Check if it exists"
                f" in bucket {self.s3_bucket}. Error: {e}"
                )
            return False

    def generate_summary(
            self,
            video_id: str,
            query_text: str,
            graph_context: list,
            ) -> Path:
        """
        Download raw video from S3, cut it into query-relevant segments, 
        and concatenate them into a single summary MP4 with audio.
        """
        if not graph_context:
            logger.warning("No context provided. Cannot generate summary.")
            return None

        # Sort segments chronologically
        sorted_segments = sorted(graph_context, key=lambda x: x['start_frame'])
        
        cleaned_query = "".join(
            c if c.isalnum() else "_" for c in query_text
            ).strip("_")
        output_file = self.output_dir / f"{video_id}_{cleaned_query}_summary.mp4"

        # Use tmp dir for download and clips
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            source_video = temp_dir_path / f"{video_id}.mp4"

            # Fetch raw video from S3
            success = self._download_video_from_s3(video_id, source_video)
            if not success:
                return None

            clip_paths = []

            # Extract segments using fast stream copying
            for idx, segment in enumerate(sorted_segments):
                start_sec = segment['start_frame']
                end_sec = segment['end_frame']
                
                clip_path = temp_dir_path / f"clip_{idx}.mp4"
                
                cut_command = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", str(start_sec),
                    "-to", str(end_sec),
                    "-i", str(source_video),
                    "-c", "copy",
                    str(clip_path)
                ]
                logger.info(
                    f"Extracting segment {idx+1}/{len(sorted_segments)}"
                    f" ({start_sec}s to {end_sec}s)..."
                    )
                subprocess.run(cut_command, check=True)
                clip_paths.append(clip_path)

            # Create concat list file required by FFmpeg
            concat_list_path = temp_dir_path / "concat_list.txt"
            with open(concat_list_path, "w") as f:
                for clip in clip_paths:
                    f.write(f"file '{clip.as_posix()}'\n")

            # Concatenate all clips into final summary video
            logger.info("Stitching scenes together into final summary...")
            concat_command = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_path),
                "-c", "copy",
                str(output_file)
            ]
            subprocess.run(concat_command, check=True)

        # Temporary directory and raw S3 video are automatically deleted here
        logger.info(f"Summary video generated successfully: {output_file}")
        return output_file