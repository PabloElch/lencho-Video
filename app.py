import fal_client
import streamlit as st

st.set_page_config(
    page_title="Wan 2.1 Video Generator", page_icon="🎬", layout="centered"
)

st.title("🎬 Wan 2.1 AI Video Generator")
st.write("Powered by Streamlit Cloud + Fal.ai Serverless Inference")

# Securely grab your API key from Streamlit secrets, or input it directly for testing
if "FAL_KEY" in st.secrets:
  fal_key = st.secrets["FAL_KEY"]
else:
  fal_key = st.text_input("Enter your Fal.ai API Key:", type="password")

# User inputs
prompt = st.text_area(
    "Enter your video prompt:",
    placeholder=(
        "A cinematic drone shot sweeping over a misty green valley at sunrise..."
    ),
)
resolution = st.selectbox("Select Resolution", ["480p", "720p"])

if st.button("Generate Video", type="primary"):
  if not fal_key:
    st.error("Please provide a Fal.ai API key to continue.")
  elif not prompt.strip():
    st.warning("Please enter a video prompt first.")
  else:
    # Temporarily set the key for the fal client SDK
    import os

    os.environ["FAL_KEY"] = fal_key

    try:
      with st.status(
          "Queuing job with Wan 2.1 (this takes about 1 minute)...",
          expanded=True,
      ) as status:
        st.write("Sending prompt to serverless cluster...")

        # Call the Wan 2.1 Text-to-Video endpoint on Fal.ai
        # Reference: https://fal.ai/models/fal-ai/wan-t2v
        result = fal_client.subscribe(
            "fal-ai/wan-t2v",
            arguments={"prompt": prompt, "resolution": resolution},
            on_queue_update=lambda update: st.write(f"Queue status..."),
        )

        status.update(
            label="Video generation complete!", state="complete", expanded=False
        )

      # Extract and display the video URL returned from the cloud
      if "video" in result and "url" in result["video"]:
        video_url = result["video"]["url"]
        st.success("Success!")
        st.video(video_url)
        st.markdown(f"[Download Video]({video_url})")
      else:
        st.error("Generation finished, but no video URL was returned.")

    except Exception as e:
        st.error(f"An error occurred during generation: {e}")
