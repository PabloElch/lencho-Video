import os
import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(
    page_title="Wan 2.1 Custom UI", page_icon="🎬", layout="centered"
)

st.title("🎬 Wan 2.1 Custom Video Generator")
st.write("Hosted on Streamlit Cloud & powered by free open-weights APIs.")

# Grab free Hugging Face token from Streamlit secrets or user input
if "HF_TOKEN" in st.secrets:
  hf_token = st.secrets["HF_TOKEN"]
else:
  hf_token = st.text_input("Enter your free Hugging Face Token:", type="password")

prompt = st.text_area(
    "Enter your video prompt:",
    placeholder="A cinematic drone shot over a mountain...",
)

if st.button("Generate Video", type="primary"):
  if not hf_token:
    st.error(
        "Please provide a free Hugging Face token (get one at"
        " hf.co/settings/tokens)."
    )
  elif not prompt.strip():
    st.warning("Please enter a prompt first.")
  else:
    try:
      with st.status(
          "Connecting to open-source inference provider...", expanded=True
      ) as status:
        st.write("Initializing client...")

        # Initialize the Hugging Face inference client
        client = InferenceClient(token=hf_token)

        st.write("Sending generation request for Wan 2.1...")

        # Call the open-weights Wan model via HF inference routing
        # (Using the community-hosted efficient 1.3B or standard text-to-video space/endpoint)
        video_output = client.text_to_video(
            prompt=prompt, model="Wan-AI/Wan2.1-T2V-1.3B"
        )

        status.update(
            label="Generation complete!", state="complete", expanded=False
        )

      st.success("Video generated successfully!")
      st.video(video_output)

    except Exception as e:
      st.error(
          f"An error occurred. Note: Heavy models may require a free HF Pro"
          f" trial or a warm-up state if the endpoint is cold. Details: {e}"
      )
