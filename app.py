import tempfile
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import streamlit as st

st.set_page_config(
    page_title="CogVideoX Video Generator", page_icon="🎬", layout="centered"
)

st.title("🎬 CogVideoX-2B Video Generator")
st.write(
    "A clean, native open-source video generation pipeline built with Streamlit"
    " and Diffusers."
)


# Load the model cleanly with caching so it only downloads/loads once
@st.cache_resource
def load_pipeline():
  # Using the lightweight 2B parameter open weights model
  pipe = CogVideoXPipeline.from_pretrained(
      "THUDM/CogVideoX-2b", torch_dtype=torch.float16
  )
  # Enable memory management optimizations for cloud environments
  pipe.enable_model_cpu_offload()
  pipe.vae.enable_tiling()
  return pipe


with st.spinner(
    "Loading CogVideoX-2B model into memory (this happens once)..."
):
  try:
    pipe = load_pipeline()
    model_loaded = True
  except Exception as e:
    st.error(f"Failed to load model weights: {e}")
    model_loaded = False

prompt = st.text_area(
    "Enter your video prompt:",
    placeholder=(
        "A golden retriever running through a sunflower field at sunset, slow"
        " motion..."
    ),
)

# Configuration options for clean control
col1, col2 = st.columns(2)
with col1:
  num_steps = st.slider(
      "Inference Steps", min_value=20, max_value=50, value=30, step=5
  )
with col2:
  guidance = st.slider(
      "Guidance Scale", min_value=1.0, max_value=10.0, value=6.0, step=0.5
  )

if st.button("Generate Video", type="primary"):
  if not model_loaded:
    st.error("Model is not loaded properly. Check environment dependencies.")
  elif not prompt.strip():
    st.warning("Please enter a prompt first.")
  else:
    with st.status(
        "Generating video frames (this may take a few minutes)...",
        expanded=True,
    ) as status:
      st.write("Running diffusion pipeline steps...")

      try:
        # Generate video tensor frames
        video_frames = pipe(
            prompt=prompt,
            num_frames=49,
            guidance_scale=guidance,
            num_inference_steps=num_steps,
            generator=torch.Generator("cuda").manual_seed(42),
        ).frames[0]

        st.write("Encoding frames to output file...")

        # Save to a temporary file to render cleanly in Streamlit
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ) as tmp_file:
          output_path = tmp_file.name

        export_to_video(video_frames, output_path, fps=8)

        status.update(
            label="Generation complete!", state="complete", expanded=False
        )

        st.success("Video generated successfully!")
        st.video(output_path)

      except Exception as e:
        st.error(f"Generation error: {e}")
