import streamlit as st
from streamlit.runtime.scriptrunner import RerunException
import requests
import time

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="OmniMind", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_upload" not in st.session_state:
    st.session_state.show_upload = False

if "processing" not in st.session_state:
    st.session_state.processing = False

if "job_id" not in st.session_state:
    st.session_state.job_id = None

if "job_type" not in st.session_state:
    st.session_state.job_type = None

if "just_completed" not in st.session_state:
    st.session_state.just_completed = False

st.title("OmniMind")
st.caption("Your multi-modal RAG assistant. Upload documents, images, or videos and ask")

top_col1, top_col2 = st.columns([10, 3])

with top_col2:
    if st.button("Clear Memory"):
        requests.delete(f"{BACKEND_URL}/clear-memory")
        st.session_state.messages = []
        st.success("Memory cleared")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

col1, col2 = st.columns([1, 12])

with col1:
    if st.button("+"):
        st.session_state.show_upload = not st.session_state.show_upload

with col2:
    prompt = st.chat_input("Ask something from your documents...",
        disabled=st.session_state.processing
    )

if st.session_state.show_upload:
    upload_file = st.file_uploader(
        "Upload file",
        type=["pdf", "txt", "docx", "pptx", "xlsx", "png", "jpg", "jpeg", "mp4", "mov", "avi"]
    )

    if upload_file is not None:

        endpoint = "/upload"
        st.session_state.job_type = "document"

        if upload_file.type.startswith("image"):
            endpoint = "/ingest-image"
            st.session_state.job_type = "image"

        if upload_file.type.startswith("video"):
            endpoint = "/ingest-video"
            st.session_state.job_type = "video"

        response = requests.post(
            f"{BACKEND_URL}{endpoint}",
            files={"file": upload_file}
        )

        if response.status_code == 200:
            data = response.json()

            if "job_id" in data:
                st.session_state.job_id = data["job_id"]
                st.session_state.processing = True
            else:
                st.session_state.processing = False

        st.session_state.show_upload = False

if "refresh_counter" not in st.session_state:
    st.session_state.refresh_counter = 0

if st.session_state.processing:

    status_endpoint = None

    if st.session_state.job_type == "video":
        status_endpoint = f"/video-status/{st.session_state.job_id}"

    elif st.session_state.job_type == "image":
        status_endpoint = f"/image-status/{st.session_state.job_id}"

    if status_endpoint:
        status_response = requests.get(f"{BACKEND_URL}{status_endpoint}")
        status_data = status_response.json()

        if status_data["status"] == "completed":
            st.session_state.processing = False
            st.session_state.just_completed = True
            st.rerun()
        else:
            st.info("Processing...")
            time.sleep(2)
            st.rerun()
            
if st.session_state.just_completed:
    st.success("Ingestion completed. You can enter your query.")
    st.session_state.just_completed = False

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        response = requests.post(
            f"{BACKEND_URL}/query-stream",
            json={"query": prompt},
            stream=True
        )

        full_text = ""

        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                token = chunk.decode("utf-8")
                full_text += token
                placeholder.markdown(full_text)

    st.session_state.messages.append({"role": "assistant", "content": full_text})