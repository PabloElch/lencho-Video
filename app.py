import os
import replicate
import streamlit as st

st.set_page_config(
    page_title="Wan 2.1 Video Generator", page_icon="🎬", layout="centered"
)

st.title("🎬 Wan 2.1 Custom Video Generator")
st.write("Powered by Streamlit & Replicate API")

# Input for Replicate Token (or set via Streamlit Secrets / Environment)
if "REPLICATE_API_TOKEN" in st.secrets:
  api_token = st.secrets["REPLICATE_API_TOKEN"]
else:
  api_token = st.text_input("Enter your Replicate API Token:", type="password")

prompt = st.text_area(
    "Enter your video prompt:",
    value=(
        "A cinematic shot of a red sports car driving fast down a coastal"
        " highway."
    ),
)

if st.button("Generate Video", type="primary"):
  if not api_token:
    st.error("Please provide a Replicate API token.")
  elif not prompt.strip():
    st.warning("Please enter a prompt first.")
  else:
    # Set the token in the environment for the replicate client
    os.environ["REPLICATE_API_TOKEN"] = api_token

    try:
      with st.status(
          "Running Wan 2.1 model (this takes a moment)...", expanded=True
      ) as status:
        st.write("Sending request to Wan 2.1...")

        # Run the official Wan 2.1 model via Replicate
        output = replicate.run(
            "wavespeedai/wan-2.1-t2v-480p", input={"prompt": prompt}
        )

        status.update(
            label="Generation complete!", state="complete", expanded=False
        )

      st.success("Video generated successfully!")
      # Replicate returns a file output URL
      st.video(output)

    except Exception as e:
      st.error(f"Generation error: {e}")
