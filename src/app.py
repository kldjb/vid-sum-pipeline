import os
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/search"

# Configure Streamlit page layout
st.set_page_config(page_title="GraphRAG Video Summarisation", layout="wide")

st.title("🎬 GraphRAG Video Summarisation 🎬")

# Configure sidebar for user inputs
with st.sidebar:
    st.header("Search Parameters")
    search_mode = st.selectbox(
        "Search Mode", 
        ["global", "video"],
        help="'global' finds relevant scenes across the whole dataset. 'video' summarises relevant scenes within a specific video."
    )
    
    n_results = st.number_input(
        "Number of Results", 
        min_value=1, max_value=10, value=3,
        help="How many top scenes to retrieve."
    )
    
    video_id = st.text_input(
        "Target Video ID", 
        placeholder="e.g., v_sUL9HAplalo",
        help="Required if search mode is set to 'video'."
    )

# Configure main interface
query = st.text_input("Enter your search query", placeholder="e.g., 'sumo wrestling'")

if st.button("Search & Generate Summary"):
    # Input validation
    if not query:
        st.warning("Please enter a search query.")
    elif search_mode == "video" and not video_id:
        st.warning("Please provide a Target Video ID for video mode searches.")
    else:
        with st.spinner("Querying API and generating summary (this may take a few seconds)..."):
            payload = {
                "query": query,
                "search_mode": search_mode,
                "n_results": n_results,
                "video_id": video_id if search_mode == "video" else None
            }
            
            try:
                # Send POST request to FastAPI backend
                response = requests.post(API_URL, json=payload)

                # Catch HTTP errors
                response.raise_for_status()
                data = response.json()
                
                context = data.get("context", [])
                message = data.get("message", "")

                # Check if any valid context was retrieved
                has_context = bool(context) and any(
                    bool(v) for v in (
                        context.values() if isinstance(context, dict) else context
                    )
                )

                if not has_context:
                    st.warning(f"⚠️ {message or 'No relevant scenes found for this query.'}")
                else:
                    st.success("Retrieval complete!")
                    
                    # Display generated video summary if available
                    video_path = data.get("summary_video_path")
                    if video_path and os.path.exists(video_path):
                        st.subheader("Generated Video Summary")
                        st.video(video_path)
                    elif search_mode == "video":
                        st.error("Summary generation failed or file not found.")
                    
                    # Render raw context data in an expandable section
                    with st.expander("View GraphRAG Context Details"):
                        st.json(context)
                    
            except requests.exceptions.RequestException as e:
                st.error(
                    f"Failed to connect to the API. Is the FastAPI server running? Error: {e}"
                    )