import streamlit as st
import requests
import time
import re

import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.set_page_config(
    page_title="AEGIS",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(ellipse at top left, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Header ── */
.aegis-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.5rem 0 0.5rem 0;
    margin-bottom: 0.5rem;
}
.aegis-logo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.aegis-logo-icon {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    box-shadow: 0 0 20px rgba(124,58,237,0.5);
}
.aegis-title {
    font-size: 1.7rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #818cf8, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.aegis-subtitle {
    font-size: 0.78rem;
    color: #64748b;
    font-weight: 400;
    margin-top: 1px;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 16px !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 0.75rem !important;
    backdrop-filter: blur(10px);
    animation: fadeSlide 0.3s ease;
}
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* User message accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(124,58,237,0.12) !important;
    border-color: rgba(124,58,237,0.25) !important;
}

/* Assistant message accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(56,189,248,0.06) !important;
    border-color: rgba(56,189,248,0.15) !important;
}

[data-testid="stChatMessage"] p {
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
}

/* ── Chat input bar ── */
[data-testid="stChatInput"] {
    border-radius: 16px !important;
    border: 1px solid rgba(124,58,237,0.4) !important;
    background: rgba(15,12,41,0.8) !important;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 30px rgba(124,58,237,0.15) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(124,58,237,0.8) !important;
    box-shadow: 0 0 40px rgba(124,58,237,0.3) !important;
}
[data-testid="stChatInput"] textarea {
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #475569 !important;
}

/* ── Clear Memory button ── */
div[data-testid="stButton"] > button {
    background: rgba(239,68,68,0.12) !important;
    color: #f87171 !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    border-radius: 10px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 0.4rem 1rem !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stButton"] > button:hover {
    background: rgba(239,68,68,0.25) !important;
    border-color: rgba(239,68,68,0.6) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(239,68,68,0.2) !important;
}

/* ── Upload panel ── */
.upload-panel {
    background: rgba(255,255,255,0.03);
    border: 1px dashed rgba(124,58,237,0.35);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    animation: fadeSlide 0.25s ease;
}
.upload-title {
    color: #a78bfa;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(124,58,237,0.06) !important;
    border: 1px dashed rgba(124,58,237,0.3) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"] > div {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}

/* ── Success / Info / Warning ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-size: 0.85rem !important;
}

/* ── Divider ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124,58,237,0.3), transparent);
    margin: 0.5rem 0 1rem 0;
}

/* ── Processing badge ── */
.processing-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(245,158,11,0.15);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #fbbf24;
    font-weight: 500;
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.6; }
}

/* ── Plus button container ── */
.plus-btn-container button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    width: 42px !important;
    height: 42px !important;
    font-size: 1.4rem !important;
    font-weight: 300 !important;
    padding: 0 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.4) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.plus-btn-container button:hover {
    transform: rotate(45deg) scale(1.1) !important;
    box-shadow: 0 6px 25px rgba(124,58,237,0.6) !important;
}

/* ── Scrollable chat area ── */
.chat-container {
    max-height: 65vh;
    overflow-y: auto;
    padding-right: 4px;
    scrollbar-width: thin;
    scrollbar-color: rgba(124,58,237,0.3) transparent;
}
.chat-container::-webkit-scrollbar { width: 4px; }
.chat-container::-webkit-scrollbar-track { background: transparent; }
.chat-container::-webkit-scrollbar-thumb {
    background: rgba(124,58,237,0.3);
    border-radius: 4px;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: #475569;
}
.empty-state-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    display: block;
    filter: grayscale(0.3);
}
.empty-state-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #64748b;
    margin-bottom: 0.5rem;
}
.empty-state-text {
    font-size: 0.85rem;
    color: #475569;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────
for key, default in {
    "messages":       [],
    "show_upload":    False,
    "processing":     False,
    "job_id":         None,
    "job_type":       None,
    "just_completed": False,
    "refresh_counter": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helper: strip noise from streamed response ─────────────────────────────────
def clean_response(text: str) -> str:
    """
    Remove retrieval metadata, confidence tags, and warning lines
    so the user only sees the clean LLM answer.
    """
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip retrieval headers
        if re.match(r"^\[Retrieved:.*\]$", stripped):
            continue
        # Skip confidence footer
        if re.match(r"^\[Response confidence:.*\]$", stripped):
            continue
        # Skip warning lines
        if stripped.startswith("⚠️") or stripped.startswith("ℹ️"):
            continue
        clean_lines.append(line)

    # Remove leading/trailing blank lines from result
    result = "\n".join(clean_lines).strip()
    return result


# ── Header ─────────────────────────────────────────────────────────────────────
header_col, btn_col = st.columns([8, 2])
with header_col:
    st.markdown("""
    <div class="aegis-header">
        <div class="aegis-logo">
            <div class="aegis-logo-icon">⚡</div>
            <div>
                <div class="aegis-title">AEGIS</div>
                <div class="aegis-subtitle">Multi-modal RAG · Hybrid Search · Guardrails</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with btn_col:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑 Clear", key="clear_btn"):
        try:
            requests.delete(f"{BACKEND_URL}/clear-memory", timeout=5)
        except Exception:
            pass
        st.session_state.messages = []
        st.rerun()

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Chat History ───────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <span class="empty-state-icon">🧠</span>
        <div class="empty-state-title">AEGIS is ready</div>
        <div class="empty-state-text">
            Upload a document, image, or video using the <strong>＋</strong> button below,<br>
            then ask anything about your content.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ── Processing Status ──────────────────────────────────────────────────────────
if st.session_state.just_completed:
    st.success("✅ File ingested — ask away!", icon="✅")
    st.session_state.just_completed = False

if st.session_state.processing:
    st.markdown(
        '<div class="processing-badge">⏳ &nbsp;Processing your file…</div>',
        unsafe_allow_html=True,
    )
    status_endpoint = None
    if st.session_state.job_type == "video":
        status_endpoint = f"/video-status/{st.session_state.job_id}"
    elif st.session_state.job_type == "image":
        status_endpoint = f"/image-status/{st.session_state.job_id}"

    if status_endpoint:
        try:
            status_resp = requests.get(f"{BACKEND_URL}{status_endpoint}", timeout=5)
            if status_resp.json().get("status") == "completed":
                st.session_state.processing     = False
                st.session_state.just_completed = True
                st.rerun()
            else:
                time.sleep(2)
                st.rerun()
        except Exception:
            st.session_state.processing = False

# ── Upload Panel ───────────────────────────────────────────────────────────────
if st.session_state.show_upload:
    st.markdown('<div class="upload-panel">', unsafe_allow_html=True)
    st.markdown('<div class="upload-title">📎 &nbsp;Attach a file</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        label="",
        type=["pdf", "txt", "docx", "pptx", "xlsx", "png", "jpg", "jpeg", "mp4", "mov", "avi"],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded is not None:
        endpoint   = "/upload"
        job_type   = "document"
        if uploaded.type.startswith("image/"):
            endpoint, job_type = "/ingest-image", "image"
        elif uploaded.type.startswith("video/"):
            endpoint, job_type = "/ingest-video", "video"

        with st.spinner(f"Uploading **{uploaded.name}**…"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}{endpoint}",
                    files={"file": uploaded},
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if "job_id" in data:
                        st.session_state.job_id       = data["job_id"]
                        st.session_state.job_type     = job_type
                        st.session_state.processing   = True
                    else:
                        st.session_state.just_completed = True
                else:
                    st.error(f"Upload failed: {resp.status_code}")
            except Exception as e:
                st.error(f"Upload error: {e}")

        st.session_state.show_upload = False
        st.rerun()

# ── Bottom bar: ＋ button + Chat Input ─────────────────────────────────────────
plus_col, input_col = st.columns([1, 11])

with plus_col:
    st.markdown('<div class="plus-btn-container">', unsafe_allow_html=True)
    if st.button("＋", key="plus_btn", help="Attach document / image / video"):
        st.session_state.show_upload = not st.session_state.show_upload
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with input_col:
    prompt = st.chat_input(
        "Ask anything about your uploaded content…",
        disabled=st.session_state.processing,
    )

# ── Handle Query ───────────────────────────────────────────────────────────────
if prompt:
    # Show user bubble
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed    = ""

        try:
            resp = requests.post(
                f"{BACKEND_URL}/query-stream",
                json={"query": prompt},
                stream=True,
                timeout=120,
            )
            for chunk in resp.iter_content(chunk_size=None):
                if chunk:
                    streamed += chunk.decode("utf-8")
                    # Show cleaned text while streaming
                    display = clean_response(streamed)
                    placeholder.markdown(display + "▌")  # typing cursor

            # Final render without cursor
            clean = clean_response(streamed)
            placeholder.markdown(clean)

        except Exception as e:
            clean = f"❌ Could not reach AEGIS backend: `{e}`"
            placeholder.markdown(clean)

    st.session_state.messages.append({"role": "assistant", "content": clean})
    st.rerun()