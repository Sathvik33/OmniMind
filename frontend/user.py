import streamlit as st
import requests
import time

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Aegis", layout="wide")

if "job_id" not in st.session_state:
    st.session_state.job_id = None

if "video_status" not in st.session_state:
    st.session_state.video_status = "idle"

st.title("Aegis")

tab1, tab2, tab3 = st.tabs(["Video", "Files & Images", "Query"])

with tab1:

    st.subheader("Upload Video")

    video_file = st.file_uploader("Select video", type=["mp4", "mov", "avi"])

    if st.button("Upload and Parse Video"):
        if video_file is not None:
            response = requests.post(
                f"{BACKEND_URL}/ingest-video",
                files={"file": video_file}
            )

            if response.status_code == 200:
                data = response.json()
                st.session_state.job_id = data["job_id"]
                st.session_state.video_status = "processing"
            else:
                st.error("Upload failed")

    if st.session_state.video_status == "processing":
        st.info("Video is parsing")

        job_id = st.session_state.job_id

        while True:
            status_response = requests.get(f"{BACKEND_URL}/video-status/{job_id}")
            status_data = status_response.json()

            if status_data["status"] == "completed":
                st.session_state.video_status = "completed"
                st.success("Parsing completed. You may ask your queries.")
                break

            if status_data["status"] == "failed":
                st.session_state.video_status = "failed"
                st.error("Video processing failed")
                break

            time.sleep(3)

    elif st.session_state.video_status == "completed":
        st.success("Video ready for querying")


with tab2:

    st.subheader("Upload Document")

    doc_file = st.file_uploader(
        "Select document",
        type=["pdf", "txt", "docx", "pptx", "xlsx"]
    )

    if st.button("Upload Document"):
        if doc_file is not None:
            response = requests.post(
                f"{BACKEND_URL}/upload",
                files={"file": doc_file}
            )

            if response.status_code == 200:
                st.success("Document ingested successfully")
            else:
                st.error("Document upload failed")

    st.subheader("Upload Image")

    image_file = st.file_uploader(
        "Select image",
        type=["png", "jpg", "jpeg"]
    )

    if st.button("Upload Image"):
        if image_file is not None:
            response = requests.post(
                f"{BACKEND_URL}/ingest-image",
                files={"file": image_file}
            )

            if response.status_code == 200:
                data = response.json()
                st.success("Image ingested successfully")
                st.write(data["description"])
            else:
                st.error("Image upload failed")

    st.subheader("Memory Control")

    if st.button("Clear Memory"):
        response = requests.delete(f"{BACKEND_URL}/clear-memory")

        if response.status_code == 200:
            st.success("Memory cleared")
        else:
            st.error("Failed to clear memory")

with tab3:

    st.subheader("Ask a Question")

    query = st.text_input("Query")

    if st.button("Submit Query"):
        if query:
            response = requests.post(
                f"{BACKEND_URL}/query",
                json={"query": query}
            )

            if response.status_code == 200:
                result = response.json()
                st.write(result["answer"])

                with st.expander("Context Used"):
                    for item in result["context_used"]:
                        st.write(item)
            else:
                st.error("Query failed")