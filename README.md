# Video Summary Pipeline
A pipeline for video summarisation with 2 modes:
* Global - retrieve top N videos related to a query.
* Video - retrieve a query-relevant summary video for a specific video.

## Usage
To start the UI you need to login to aws (for video summarisation), start up a local instance of the graph store and run the backend server and app separately. The following commands assume you have started your graph store instance in Neo4j Desktop.

Install the aws-cli and then in a terminal, run the command:    
`aws login`    
Follow the login instructions in the browser.

After that, start the server by returning to the terminal and running:    
`python -m src.api.server`

**Once the server has completely setup**, in a separate terminal, launch the frontend app by running:    
`streamlit run src/app.py`

## Dataset
This project uses the VideoXum dataset (made up of pre-processed ActivityNet videos) and their fine-tuned CLIP ViT-B/16 embedding model for embedding user queries. This dataset provides raw videos alongside their extracted information. Each video contains the following information:

| Key                    | Description                                                                                                         | Shape / Type                        |
|------------------------|---------------------------------------------------------------------------------------------------------------------|-------------------------------------|
| `n_frames`             | Number of sampled frames from the video (at 1 fps).                                                                          | Scalar integer                       |
| `gtscores`             | Ground-truth scores from the 10 human annotators.                                                                   | `[10, n_frames]`                    |
| `video_embeddings`     | Frame-level CLIP embeddings for the sampled video frames (at 1 fps)                                                                                 | `[n_frames, 512]`                   |
| `script_embeddings`      | Sentence-level CLIP embeddings for the textual description of each of the 10 available ground-truth video summaries (scripts); zero padding if a description has less than `M_max` sentences.                     | `[10, M_max, 512]`                   |
| `transcript_embeddings`| Chunk-level CLIP embeddings for the extracted audio transcript.                                                                 | `[N, 512]` (N = number of transcript chunks)    |
| `transcript_timestamps`| Start and end time for each chunk of the audio transcript.                                   | `[N, 2]` (N = number of transcript chunks)                           |
| `aligned_transcripts`  | Transcript embeddings that are time-aligned with the frame-level embeddings; zero-padding when transcripts are not available (there is no spoken content in the video). | `[n_frames, 512]`                   |

## Vector Store
ChromaDB was used because...

## Graph Database
Neo4J was used along with Cipher because...

## API
FastAPI and Streamlit were used because...

## Preprocessing
| Script                    | Description                                                                                                         |
|------------------------|---------------------------------------------------------------------------------------------------------------------|
| `changepoint_detection.py`             | Calculate scene changepoints.                                                                          |
| `extract_video_data.py`             | Extract video information from videoxum.h5 file.                                                                          | 
| `generate_s3_manifest.py`             | Load video names available in S3 bucket to ensure we're only graphing data we have raw videos for.                                                                          | 
| `graph_data.py`             | Processes video components into graph using CIPHER queries.                                                                          | 
| `load_mrhisum_data.py`             | Legacy script for processing MrHiSum raw videos.changepoints.                                                                          | 
| `load_videoxum_data.py`             | Extracts videos from Youtube and stores them in   S3 bucket.                                                                          | 
| `map_mrhisum_folders.py`             | Legacy script for mapping MrHiSum video names to Youtube URIs.                                                                          | 
| `store_vectors.py`             | Store extracted embeddings to local ChromaDB vector store.                                                                          | 
| `utils.py`             | Utility script for storing collection ID assignments.                                                                           | 
 