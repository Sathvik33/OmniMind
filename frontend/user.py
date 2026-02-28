import streamlit as st
import requests
import time

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Aegis", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "job_id" not in st.session_state:
    st.session_state.job_id = None

if "video_status" not in st.session_state:
    st.session_state.video_status = "idle"

st.title("Aegis")

with st.expander("Upload Files or Video"):

    video_file = st.file_uploader("Video", type=["mp4", "mov", "avi"])
    doc_file = st.file_uploader("Document", type=["pdf", "txt", "docx", "pptx", "xlsx"])
    image_file = st.file_uploader("Image", type=["png", "jpg", "jpeg"])

    if st.button("Upload"):

        if video_file is not None:
            with st.spinner("Uploading video..."):
                response = requests.post(
                    f"{BACKEND_URL}/ingest-video",
                    files={"file": video_file}
                )

            if response.status_code == 200:
                data = response.json()
                st.session_state.job_id = data["job_id"]
                st.session_state.video_status = "processing"

        if doc_file is not None:
            with st.spinner("Uploading document..."):
                requests.post(
                    f"{BACKEND_URL}/upload",
                    files={"file": doc_file}
                )

        if image_file is not None:
            with st.spinner("Processing image..."):
                requests.post(
                    f"{BACKEND_URL}/ingest-image",
                    files={"file": image_file}
                )

if st.session_state.video_status == "processing":

    with st.spinner("Video is parsing..."):
        while True:
            status_response = requests.get(
                f"{BACKEND_URL}/video-status/{st.session_state.job_id}"
            )
            status_data = status_response.json()

            if status_data["status"] == "completed":
                st.session_state.video_status = "completed"
                break

            if status_data["status"] == "failed":
                st.session_state.video_status = "failed"
                break

            time.sleep(3)

    if st.session_state.video_status == "completed":
        st.success("Video ready")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask something")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{BACKEND_URL}/query",
                json={"query": prompt}
            )

            if response.status_code == 200:
                result = response.json()
                answer = result["answer"]
            else:
                answer = "Error generating response."

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})