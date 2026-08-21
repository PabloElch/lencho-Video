import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(
    page_title="Wan 2.1 Custom UI", page_icon="🎬", layout="centered"
)

st.title("🎬 Wan 2.1 Custom Video Generator")
st.write("Powered by Streamlit + Free Hugging Face SDK")

# Grab Hugging Face token from secrets or input box
if "HF_TOKEN" in st.secrets:
  hf_token = st.secrets["HF_TOKEN"]
else:
  hf_token = st.text_input("Enter your Hugging Face Token:", type="password")

prompt = st.text_input(
    "Enter your video prompt:",
    value="A red sports car driving fast down a coastal highway.",
)

if st.button("Generate Video", type="primary"):
  if not hf_token:
    st.error("Please provide your Hugging Face token.")
  elif not prompt.strip():
    st.warning("Please enter a prompt first.")
  else:
    try:
      with st.status("Generating video...", expanded=True) as status:
        st.write("Connecting through Hugging Face client...")

        # Initialize the official client securely
        client = InferenceClient(token=hf_token)

        st.write("Sending request to Wan 2.1 model...")

        # Use the official SDK text_to_video pipeline wrapper
        video_file = client.text_to_video(
            prompt=prompt, model="Wan-AI/Wan2.1-T2V-1.3B"
        )

        status.update(
            label="Generation complete!", state="complete", expanded=False
        )

      st.success("Video generated successfully!")
      st.video(video_file)

    except Exception as e:
      st.error(
          f"Generation error: {e}\n\nTip: If the model is cold-loading, wait a"
          " moment and try clicking generate again."
      )
