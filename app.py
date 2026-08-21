import os
import requests
import streamlit as st

st.set_page_config(
    page_title="Wan 2.1 Custom UI", page_icon="🎬", layout="centered"
)

st.title("🎬 Wan 2.1 Custom Video Generator")
st.write("Hosted on Streamlit Cloud & powered by free Hugging Face inference.")

# Grab Hugging Face token
if "HF_TOKEN" in st.secrets:
  hf_token = st.secrets["HF_TOKEN"]
else:
  hf_token = st.text_input("Enter your Hugging Face Token:", type="password")

prompt = st.text_area(
    "Enter your video prompt:",
    placeholder="A red sports car driving fast down a coastal highway.",
)

if st.button("Generate Video", type="primary"):
  if not hf_token:
    st.error(
        "Please provide a valid Hugging Face token (get one at"
        " hf.co/settings/tokens)."
    )
  elif not prompt.strip():
    st.warning("Please enter a prompt first.")
  else:
    try:
      with st.status(
          "Sending request to Hugging Face serverless API...", expanded=True
      ) as status:
        # Direct REST API call to the open-weights model router
        API_URL = "https://api-inference.huggingface.co/models/Wan-AI/Wan2.1-T2V-1.3B"
        headers = {"Authorization": f"Bearer {hf_token}"}

        st.write("Submitting prompt payload...")
        response = requests.post(
            API_URL, headers=headers, json={"inputs": prompt}
        )

        if response.status_code != 200:
          raise Exception(
              f"API Error [{response.status_code}]: {response.text}"
          )

        status.update(
            label="Generation complete!", state="complete", expanded=False
        )

      st.success("Video generated successfully!")
      # If the model returns raw video bytes:
      st.video(response.content)

    except Exception as e:
      st.error(
          f"An error occurred. Note: If the model is cold-loading, wait a"
          f" minute and try again. Details: {e}"
      )
