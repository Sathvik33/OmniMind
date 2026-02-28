import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Aegis", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_upload" not in st.session_state:
    st.session_state.show_upload = False

st.title("Aegis")

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
    prompt = st.chat_input("Ask something from your documents...")

if st.session_state.show_upload:
    upload_file = st.file_uploader(
        "Upload file",
        type=["pdf", "txt", "docx", "pptx", "xlsx", "png", "jpg", "jpeg", "mp4", "mov", "avi"]
    )

    if upload_file is not None:
        endpoint = "/upload"

        if upload_file.type.startswith("image"):
            endpoint = "/ingest-image"

        if upload_file.type.startswith("video"):
            endpoint = "/ingest-video"

        requests.post(
            f"{BACKEND_URL}{endpoint}",
            files={"file": upload_file}
        )

        st.session_state.show_upload = False

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