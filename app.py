import urllib.parse
import streamlit as st

st.set_page_config(
    page_title="Free Custom Media Generator", page_icon="🎬", layout="centered"
)

st.title("🎬 Custom AI Media Generator")
st.write(
    "A 100% free custom Streamlit UI app—bypassing all paid API limits and"
    " token errors."
)

prompt = st.text_area(
    "Enter your prompt:",
    placeholder="A cinematic drone shot over a mountain valley at sunrise...",
)

# Choose what type of media to generate natively
media_type = st.radio(
    "Select Output Format:", ["High-Quality Image (Instant)", "Visual Animation"]
)

if st.button("Generate Now", type="primary"):
  if not prompt.strip():
    st.warning("Please enter a prompt first.")
  else:
    with st.spinner("Processing generation request..."):
      encoded_prompt = urllib.parse.quote(prompt)

      if media_type == "High-Quality Image (Instant)":
        # Direct free public image endpoint (Flux/Pollinations engine)
        media_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=576&nologo=true"
        st.success("Generated successfully!")
        st.image(media_url, use_container_width=True)
      else:
        # Direct free public video/animation endpoint
        media_url = f"https://gen.pollinations.ai/video/{encoded_prompt}"
        st.success("Generated successfully!")
        st.video(media_url)

      st.markdown(f"**Direct URL:** [Open Media Link]({media_url})")
