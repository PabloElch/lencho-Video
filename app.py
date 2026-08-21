import gc
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import streamlit as st
from kokoro_onnx import Kokoro

# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

TARGET_WORDS_PER_CHUNK = 550
OUTPUT_SAMPLE_RATE = 24000

MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.onnx"
)
VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/voices-v1.0.bin"
)

MODEL_FILENAME = "kokoro-v1.0.onnx"
VOICES_FILENAME = "voices-v1.0.bin"

VOICE_MAP = {
    "🇺🇸 Beza (American Female - Warm)": "af_heart",
    "🇺🇸 Birikti (American Female - Soft)": "af_bella",
    "🇺🇸 Demoze (American Female - Clear)": "af_nicole",
    "🇺🇸 Lalise (American Female - News)": "af_sarah",
    "🇺🇸 Efrata (American Female - Casual)": "af_sky",
    "🇺🇸 Lencho (American Male - Deep)": "am_adam",
    "🇺🇸 Dego (American Male - Crisp)": "am_michael",
    "🇬🇧 Bontu (British Female - Professional)": "bf_emma",
    "🇬🇧 Hawi (British Female - Warm)": "bf_isabella",
    "🇬🇧 Lalisa (British Male - Expressive)": "bm_george",
    "🇬🇧 Lemi (British Male - Narration)": "bm_fable",
}

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Lenchxos AI Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# UNIFIED CUSTOM CSS STYLING
# ============================================================

st.markdown(
    """
    <style>
    /* Global Theme & Background */
    .stApp {
        background-color: #0b0b0d !important;
        color: #f2f2f2 !important;
    }
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0b0b0d !important;
    }
    [data-testid="stSidebar"] {
        background-color: #08080a !important;
    }
    
    header {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px !important;
    }

    /* Input text area styling */
    div.stTextArea textarea {
        background-color: #111114 !important;
        color: #f5f5f5 !important;
        border: 1px solid #303036 !important;
        border-radius: 12px !important;
        padding: 16px !important;
        font-size: 15px !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        resize: none;
    }
    div.stTextArea textarea:focus {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.15) !important;
    }
    div.stTextArea textarea::placeholder {
        color: #88888f !important;
    }

    /* Primary SaaS Generate Button */
    .stButton button {
        background: #4f46e5 !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        height: 50px !important;
        font-size: 15px !important;
        letter-spacing: 0.2px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25);
    }
    .stButton button:hover {
        background: #4338ca !important;
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35);
    }

    /* Secondary / Download Buttons */
    div.stDownloadButton button {
        background: #18181c !important;
        color: #ffffff !important;
        border: 1px solid #35353b !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        height: 48px !important;
        transition: all 0.2s ease;
    }
    div.stDownloadButton button:hover {
        background: #24242a !important;
        border-color: #55555d !important;
    }

    /* Metrics & Expanders */
    [data-testid="stMetric"] {
        background-color: #111114;
        border: 1px solid #29292d;
        border-radius: 12px;
        padding: 12px;
    }
    [data-testid="stMetricLabel"] { color: #aaaaaf !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }

    /* Inputs & Selectboxes */
    [data-baseweb="select"] > div {
        background-color: #111114 !important;
        color: #f5f5f5 !important;
        border-color: #303036 !important;
    }
    [data-baseweb="popover"], [role="option"] {
        background-color: #111114 !important;
        color: #f5f5f5 !important;
    }
    [role="option"]:hover {
        background-color: #222226 !important;
    }

    @keyframes softGlow {
        0% { text-shadow: 0 0 4px rgba(79, 70, 229, 0.3), 0 0 10px rgba(79, 70, 229, 0.1); }
        50% { text-shadow: 0 0 12px rgba(79, 70, 229, 0.6), 0 0 20px rgba(79, 70, 229, 0.3); }
        100% { text-shadow: 0 0 4px rgba(79, 70, 229, 0.3), 0 0 10px rgba(79, 70, 229, 0.1); }
    }
    .glowing-name {
        color: #818cf8;
        font-weight: 700;
        animation: softGlow 3s infinite ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPER FUNCTIONS (AUDIO / KOKORO)
# ============================================================

def get_base_work_dir():
    base_dir = Path(tempfile.gettempdir()) / "lenchos_audio_studio"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir

def download_file(url, destination):
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 0:
        return str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)
    return str(destination)

@st.cache_resource(show_spinner="Loading Kokoro Neural weights model...")
def get_kokoro_engine():
    model_dir = get_base_work_dir() / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_FILENAME
    voices_path = model_dir / VOICES_FILENAME
    download_file(MODEL_URL, model_path)
    download_file(VOICES_URL, voices_path)
    return Kokoro(str(model_path), str(voices_path))

def normalize_script(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        cleaned = re.sub(r"\s+", " ", line.strip())
        if cleaned:
            lines.append(cleaned)
    return "\n\n".join(lines)

def split_long_sentence(sentence, target_words):
    words = sentence.split()
    if len(words) <= target_words:
        return [sentence.strip()]
    pieces = []
    for start in range(0, len(words), target_words):
        piece = " ".join(words[start : start + target_words])
        if piece.strip():
            pieces.append(piece.strip())
    return pieces

def split_script_into_chunks(text, target_words=TARGET_WORDS_PER_CHUNK):
    text = normalize_script(text)
    if not text:
        return []
    paragraphs = re.split(r"\n\s*\n", text)
    sentences = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        paragraph_sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        for sentence in paragraph_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentences.extend(split_long_sentence(sentence, target_words))
    chunks = []
    current = []
    current_words = 0
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current and current_words + sentence_words > target_words:
            chunks.append(" ".join(current).strip())
            current = []
            current_words = 0
        current.append(sentence)
        current_words += sentence_words
    if current:
        chunks.append(" ".join(current).strip())
    return chunks

def make_job_id(script, voice, speed):
    payload = f"{script}|{voice}|{speed:.4f}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

def create_job_directory(job_id):
    job_dir = get_base_work_dir() / f"job_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "chunks").mkdir(parents=True, exist_ok=True)
    return job_dir

def get_chunk_path(job_dir, index):
    return Path(job_dir) / "chunks" / f"chunk_{index:03d}.wav"

def chunk_is_complete(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        info = sf.info(str(path))
        return info.frames > 0 and info.samplerate > 0
    except Exception:
        return False

def generate_chunk(kokoro, text, voice, speed, output_path):
    samples, sample_rate = kokoro.create(text, voice=voice, speed=float(speed), lang="en-us")
    samples = np.asarray(samples, dtype=np.float32)
    sample_rate = int(sample_rate)
    sf.write(str(output_path), samples, sample_rate, subtype="PCM_16")
    del samples
    gc.collect()
    return sample_rate

def combine_wav_files(chunk_paths, output_path):
    if not chunk_paths:
        raise ValueError("No audio chunks were found.")
    first_info = sf.info(str(chunk_paths[0]))
    sample_rate = first_info.samplerate
    channels = first_info.channels
    with sf.SoundFile(str(output_path), mode="w", samplerate=sample_rate, channels=channels, subtype="PCM_16", format="WAV") as output_file:
        for path in chunk_paths:
            with sf.SoundFile(str(path), mode="r") as input_file:
                while True:
                    block = input_file.read(65536, dtype="float32")
                    if len(block) == 0:
                        break
                    output_file.write(block)
                    del block
    gc.collect()

def convert_wav_to_mp3(wav_path, mp3_path):
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("imageio-ffmpeg is not installed.") from exc
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe, "-y", "-i", str(wav_path),
        "-codec:a", "libmp3lame", "-b:a", "128k",
        "-ar", str(OUTPUT_SAMPLE_RATE), str(mp3_path),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError("FFmpeg MP3 conversion failed:\n\n" + result.stderr[-3000:])

def generate_preview_mp3(kokoro, voice, speed=1.0):
    preview_text = "Hello. This is a quick preview of this voice persona. I hope you enjoy listening."
    samples, sample_rate = kokoro.create(preview_text, voice=voice, speed=float(speed), lang="en-us")
    samples = np.asarray(samples, dtype=np.float32)
    sample_rate = int(sample_rate)
    samples_int16 = np.clip(samples, -1.0, 1.0) * 32767
    samples_int16 = samples_int16.astype(np.int16)
    raw_audio = samples_int16.tobytes()
    del samples, samples_int16
    gc.collect()
    
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("imageio-ffmpeg is not installed.") from exc
    
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe, "-y", "-f", "s16le", "-ar", str(sample_rate),
        "-ac", "1", "-i", "pipe:0", "-codec:a", "libmp3lame",
        "-b:a", "128k", "-ar", str(OUTPUT_SAMPLE_RATE), "-f", "mp3", "pipe:1",
    ]
    result = subprocess.run(command, input=raw_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError("Could not create MP3 preview.")
    return result.stdout

# ============================================================
# SESSION STATES INITIALIZATION
# ============================================================
if "image_url" not in st.session_state:
    st.session_state.image_url = None
if "current_job" not in st.session_state:
    st.session_state.current_job = None
if "audio_history" not in st.session_state:
    st.session_state.audio_history = []

# ============================================================
# HEADER NAVIGATION SECTION
# ============================================================
col_logo, col_badge = st.columns([6, 1])
with col_logo:
    st.markdown(
        "<h3 style='margin:0; font-weight:800; color:#ffffff; letter-spacing:-0.5px;'>✨ Lenchxos <span style='color:#4f46e5;'>AI Studio</span></h3>",
        unsafe_allow_html=True,
    )
with col_badge:
    st.markdown(
        "<div style='background:#1e1b4b; color:#818cf8; padding:6px 12px; border-radius:20px; font-size:12px; font-weight:600; text-align:center; border:1px solid #312e81;'>By Lencho Lemessa</div>",
        unsafe_allow_html=True,
    )

st.markdown("<hr style='margin-top:12px; margin-bottom:20px; border-color:#29292d;'>", unsafe_allow_html=True)

# ============================================================
# APP TABS SETUP
# ============================================================
tab_image, tab_audio = st.tabs(["🎨 AI Image Studio (Flux)", "🎙️ AI Audio Studio (Kokoro)"])

# ============================================================
# TAB 1: POLLINATIONS AI IMAGE STUDIO
# ============================================================
with tab_image:
    control_col, output_col = st.columns([1.1, 1.9], gap="large")

    with control_col:
        st.markdown("<h4 style='color:#f8fafc; font-weight:600; margin-bottom:6px;'>Prompt Studio</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; font-size:13px; margin-bottom:16px;'>Type your concept cleanly. The FLUX engine will synthesize your asset.</p>", unsafe_allow_html=True)

        img_prompt = st.text_area(
            "Prompt",
            value="",
            placeholder="A minimalist studio product shot of a futuristic ceramic vase, soft natural lighting...",
            label_visibility="collapsed",
            height=170,
            key="flux_prompt_input"
        )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        generate_img_btn = st.button("Generate Masterpiece", type="primary", use_container_width=True, key="flux_gen_btn")

        if generate_img_btn:
            if not img_prompt.strip():
                st.warning("Please type a prompt to initialize generation.")
            else:
                encoded_prompt = urllib.parse.quote(img_prompt)
                st.session_state["image_url"] = f"https://image.pollinations.ai/prompt/{encoded_prompt}?model=flux&width=1024&height=1024&enhance=true"

    with output_col:
        st.markdown("<h4 style='color:#f8fafc; font-weight:600; margin-bottom:6px;'>Live Canvas</h4>", unsafe_allow_html=True)

        if st.session_state.get("image_url"):
            with st.spinner("Synthesizing neural weights & rendering pixels..."):
                try:
                    img_response = requests.get(st.session_state["image_url"], timeout=15)
                    if img_response.status_code == 200 and len(img_response.content) > 1000:
                        st.markdown("<div style='background: #111114; padding: 12px; border-radius: 16px; border: 1px solid #29292d;'>", unsafe_allow_html=True)
                        st.image(img_response.content, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                        st.download_button(
                            label="📥 Download Production Asset (.jpg)",
                            data=img_response.content,
                            file_name="lenchxo_flux_masterpiece.jpg",
                            mime="image/jpeg",
                            use_container_width=True,
                        )
                    else:
                        st.error("Server took too long. Please hit generate again.")
                except Exception:
                    st.error("Connection timeout with rendering server. Please try again.")
        else:
            st.markdown(
                """
                <div style="border: 2px dashed #303036; border-radius: 16px; height: 430px; display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #111114; text-align: center; padding: 20px;">
                    <div style="font-size: 36px; margin-bottom: 12px;">🎨</div>
                    <div style="color: #f2f2f2; font-weight: 600; font-size: 15px;">Workspace Ready</div>
                    <div style="color: #94a3b8; font-size: 13px; max-width: 260px; margin-top: 4px;">Enter a description on the left panel to render your high-resolution asset.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ============================================================
# TAB 2: KOKORO TTS AUDIO STUDIO
# ============================================================
with tab_audio:
    # Sidebar control panel specifically nested or managed inside the audio session or main page
    audio_col1, audio_col2 = st.columns([1.1, 1.9], gap="large")

    with audio_col1:
        st.markdown("<h4 style='color:#f8fafc; font-weight:600; margin-bottom:6px;'>Voice Director</h4>", unsafe_allow_html=True)
        
        voice_name = st.selectbox(
            "Narrator Persona",
            list(VOICE_MAP.keys()),
            index=5,
        )
        voice_key = VOICE_MAP[voice_name]

        speed = st.slider(
            "Speech Speed",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.05,
        )

        if st.button("▶️ Preview Selected Voice", use_container_width=True):
            with st.spinner("Generating MP3 voice preview..."):
                try:
                    kokoro_eng = get_kokoro_engine()
                    preview_mp3 = generate_preview_mp3(kokoro_eng, voice_key, speed)
                    st.audio(preview_mp3, format="audio/mpeg")
                except Exception as exc:
                    st.error("Could not generate preview.")
                    st.exception(exc)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#f8fafc; font-weight:600; margin-bottom:6px;'>Narration Script</h4>", unsafe_allow_html=True)
        
        script_text = st.text_area(
            "Script Content",
            value="",
            placeholder="Type or paste your multi-paragraph article, story, or long-form script here...",
            height=230,
            label_visibility="collapsed",
            key="audio_script_input"
        )
        
        synthesize_btn = st.button("Synthesize Long-Form Audio", type="primary", use_container_width=True)

    with audio_col2:
        st.markdown("<h4 style='color:#f8fafc; font-weight:600; margin-bottom:6px;'>Studio Render Output</h4>", unsafe_allow_html=True)

        if synthesize_btn:
            if not script_text.strip():
                st.warning("Please provide a script to synthesize.")
            else:
                job_id = make_job_id(script_text, voice_key, speed)
                job_dir = create_job_directory(job_id)
                chunks = split_script_into_chunks(script_text, TARGET_WORDS_PER_CHUNK)
                
                if not chunks:
                    st.error("No valid text chunks found after normalization.")
                else:
                    st.info(f"Split script into **{len(chunks)} chunks** (~{TARGET_WORDS_PER_CHUNK} words each). Rendering audio...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        kokoro_eng = get_kokoro_engine()
                        all_chunk_paths = []
                        
                        for i, chunk in enumerate(chunks, 1):
                            chunk_path = get_chunk_path(job_dir, i)
                            status_text.text(f"Rendering chunk {i} of {len(chunks)}...")
                            
                            if not chunk_is_complete(chunk_path):
                                generate_chunk(kokoro_eng, chunk, voice_key, speed, chunk_path)
                            
                            all_chunk_paths.append(chunk_path)
                            progress_bar.progress(i / len(chunks))

                        status_text.text("Combining and mastering audio streams...")
                        final_wav = job_dir / "final_narration.wav"
                        combine_wav_files(all_chunk_paths, final_wav)

                        final_mp3 = job_dir / "final_narration.mp3"
                        convert_wav_to_mp3(final_wav, final_mp3)

                        st.session_state.current_job = {
                            "job_id": job_id,
                            "work_dir": job_dir,
                            "mp3_path": final_mp3,
                        }
                        status_text.success("Production master completed successfully!")
                        progress_bar.empty()
                    except Exception as e:
                        st.error(f"Error during audio processing: {e}")

        # Display active/rendered audio result if available
        if st.session_state.current_job and Path(st.session_state.current_job["mp3_path"]).exists():
            mp3_p = Path(st.session_state.current_job["mp3_path"])
            st.markdown(
                f"""
                <div style="background-color: #111114; padding: 20px; border-radius: 16px; border: 1px solid #29292d; margin-top: 10px;">
                    <div style="font-weight: 600; margin-bottom: 8px; color: #ffffff;">🎧 Master Audio Ready</div>
                    <div style="font-size: 13px; color: #94a3b8; margin-bottom: 14px;">File size: {mp3_p.stat().st_size / (1024*1024):.2f} MB</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            with open(mp3_p, "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                st.download_button(
                    label="📥 Download Master Narration (.mp3)",
                    data=audio_bytes,
                    file_name="lenchxo_audio_masterpiece.mp3",
                    mime="audio/mpeg",
                    use_container_width=True,
                )
        else:
            st.markdown(
                """
                <div style="border: 2px dashed #303036; border-radius: 16px; height: 380px; display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #111114; text-align: center; padding: 20px;">
                    <div style="font-size: 36px; margin-bottom: 12px;">🎙️</div>
                    <div style="color: #f2f2f2; font-weight: 600; font-size: 15px;">Audio Engine Idle</div>
                    <div style="color: #94a3b8; font-size: 13px; max-width: 280px; margin-top: 4px;">Configure your narrator settings and input text on the left to export production-grade speech.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ============================================================
# SLEEK FOOTER SECTION
# ============================================================
st.markdown(
    """
    <hr style='margin-top: 50px; margin-bottom: 20px; border-color: #29292d;'>
    <div style='display: flex; justify-content: space-between; align-items: center; color: #94a3b8; font-size: 13px; padding-bottom: 20px;'>
        <div>Designed & Developed by <span style='color: #ffffff; font-weight: 600;'>Lencho Lemessa</span></div>
        <div>AI Workflow Architect</div>
    </div>
    """,
    unsafe_allow_html=True,
)
