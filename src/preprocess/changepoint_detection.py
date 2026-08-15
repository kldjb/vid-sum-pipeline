import numpy as np
from pathlib import Path
import ruptures as rpt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def calculate_change_points(video_dir):
    """
    Uses Kernel Temporal Segmentation (KTS) via the Pruned Exact Linear Time
    (PELT) algorithm to automatically find the optimal number of segments
    based on a penalty.
    """
    try:
        video_emb_path = video_dir / "video_embeddings.npy"
        if not video_emb_path.exists():
            return False, f"Missing video_embeddings in {video_dir.name}"
            
        embeddings = np.load(video_emb_path)
        n_frames = embeddings.shape[0]
        
        # If video is short, treat as single continuous shot
        if n_frames < 5:
            change_points = np.array([[0, n_frames - 1]])
            np.save(video_dir / "change_points.npy", change_points)
            return True, video_dir.name
            
        # Use Radial Basis Function (RBF) kernel to measure visual similarity
        # between any two frames. Shots must be minimum 2 frames/seconds (as 
        # frames have been extracted at 1 frame per second).
        algo = rpt.Pelt(model="rbf", min_size=2).fit(embeddings)
        
        # Only declare a new shot if it reduces overall visual variance by at
        # least 15 points (higher = fewer/longer shots; lower = more/shorter
        # shots).
        breakpoints = algo.predict(pen=15) 
        
        # Convert breakpoints (e.g., [30, 70, n_frames]) to inclusive shot
        # ranges: [[0, 29], [30, 69], ...]
        change_points = []
        start = 0
        for bp in breakpoints:
            end = bp - 1  # Inclusive end frame
            if start <= end:
                change_points.append([start, end])
            start = bp
            
        np.save(video_dir / "change_points.npy", np.array(change_points, dtype=int))
        return True, video_dir.name
        
    except Exception as e:
        return False, f"Error processing {video_dir.name}: {e}"

def main():
    # Get directories of video data
    dataset_root = Path("../Datasets/SM-MrHiSum and SM-VideoXum/SM-VideoXum-Training-Data/extracted_data")
    video_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]
    
    print(f"Found {len(video_dirs)} videos. Calculating change points via Kernel Temporal Segmentation (KTS)...")
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(calculate_change_points, d): d for d in video_dirs}
        
        for future in tqdm(as_completed(futures), total=len(video_dirs), desc="Running KTS"):
            success, msg = future.result()
            if not success:
                # Prevent breaking progress bar visuals by tracking errors
                tqdm.write(msg)
                
    print("\nChange points generated successfully!")

if __name__ == "__main__":
    main()