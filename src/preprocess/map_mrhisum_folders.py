import os
import pandas as pd
from pathlib import Path

def rename_mrhisum_folders(extracted_dir, metadata_csv):
    print("Loading metadata mapping...")
    df = pd.read_csv(metadata_csv)
    
    # Create dict mapping generic names to youtube urls
    # (e.g.,'video_1' to 'JhdjUam0l6A')
    mapping = dict(zip(df['video_id'], df['youtube_id']))

    extracted_path = Path(extracted_dir)
    renamed_count = 0
    missing_count = 0

    print("Renaming folders...")
    for folder in extracted_path.iterdir():
        if folder.is_dir():
            if folder.name in mapping:
                youtube_id = mapping[folder.name]
                # Format to match ActivityNet naming convention
                new_name = f"v_{youtube_id}" 
                new_path = extracted_path / new_name
                
                # Rename the folder
                if new_path.exists():
                    print(
                        f"Warning: {new_name} already exists. Skipping rename for {folder.name}."
                        )
                    continue
                os.rename(folder, new_path)
                renamed_count += 1
            elif not folder.name.startswith("v_"):
                missing_count += 1

    print(f"Success! Renamed {renamed_count} folders.")
    if missing_count > 0:
        print(f"Note: {missing_count} folders were not found in the CSV.")

if __name__ == "__main__":
    EXTRACTED_DIR = r"C:\Users\kldjb\Documents\Birkbeck University\Project\Datasets\SM-MrHiSum and SM-VideoXum\SM-MrHiSum-Training-Data\extracted_data\\"
    METADATA_PATH = r"C:\Users\kldjb\Documents\Birkbeck University\Project\Datasets\SM-MrHiSum and SM-VideoXum\SM-MrHiSum-Training-Data\metadata.csv"
    
    rename_mrhisum_folders(EXTRACTED_DIR, METADATA_PATH)