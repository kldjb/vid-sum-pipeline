import logging
import sys
import os
import h5py
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
logger = logging.getLogger(__name__)

def get_all_dataset_paths(input_h5):
    """Scans HDF5 file and returns list of dataset paths."""
    dataset_paths = []
    
    def collect_datasets(name, node):
        if isinstance(node, h5py.Dataset):
            dataset_paths.append(name)
            
    with h5py.File(input_h5, 'r') as h5_file:
        h5_file.visititems(collect_datasets)
        
    return dataset_paths

def worker_extract_chunk(input_h5, output_dir, dataset_paths_chunk):
    """
    Isolated worker function to establish independent, read-only
    connection to .h5 file.
    """
    extracted_count = 0
    with h5py.File(input_h5, 'r') as h5_file:
        for name in dataset_paths_chunk:
            node = h5_file[name]
            out_path = os.path.join(output_dir, f"{name}.npy")
            
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            if node.shape == ():
                data = node[()]
            else:
                data = node[:]

            np.save(out_path, data)
            extracted_count += 1
            
    return extracted_count

def chunk_list(lst, n_chunks):
    """Yields equal-sized chunks from list."""
    chunk_size = max(1, len(lst) // n_chunks)
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

if __name__ == "__main__":
    if len(sys.argv) != 3:
        logger.error(
            "Usage: python extract.py <input_file.h5> <output_directory>"
            )
        sys.exit(1)

    input_h5 = sys.argv[1]
    output_dir = sys.argv[2]

    # Collect paths on main thread
    logger.info("Scanning HDF5 file for datasets...")
    all_paths = get_all_dataset_paths(input_h5)
    total_datasets = len(all_paths)
    logger.info(f"Found {total_datasets} datasets to extract.")

    if total_datasets == 0:
        logger.info("Nothing to extract. Exiting...")
        sys.exit(0)

    # Setup multiprocessing to use 80% of available cores
    max_workers = max(1, int(os.cpu_count() * 0.8)) 
    logger.info(f"Starting extraction across {max_workers} CPU cores...")

    # Split list of paths into a chunk for each core
    chunks = list(chunk_list(all_paths, max_workers))
    
    processed_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit chunks to workers
        futures = [
            executor.submit(
                worker_extract_chunk,
                input_h5,
                output_dir,
                chunk,
                ) 
            for chunk in chunks
        ]
        
        # Track progress as chunks complete
        for future in as_completed(futures):
            try:
                count = future.result()
                processed_count += count
                logger.info(
                    f"Progress: {processed_count}/{total_datasets} datasets extracted..."
                    )
            except Exception as e:
                logger.error(f"A worker encountered an error: {e}")

    logger.info("Extraction complete.")