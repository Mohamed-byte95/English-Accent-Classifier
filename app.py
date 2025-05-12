# app.py
import streamlit as st
from accent_main  import download_and_extract_audio, classify_accent, check_ffmpeg, AUDIO_FILENAME
import os

hf_token = st.secrets["huggingface"]["token"]
st.set_page_config(page_title="English Accent Classifier", layout="centered")
st.title("🎤 English Accent Classifier")
st.markdown("""
This tool analyzes a speaker's English accent from a public video URL (e.g., YouTube, Loom, direct .mp4).
It uses a deep learning model trained on multiple English accents.
""")
# Retrieve the token from secrets


url = st.text_input("Paste the video URL:", placeholder="https://...")

if st.button("Classify Accent"):
    if not url:
        st.warning("Please enter a valid video URL.")
    else:
        st.info("Checking dependencies...")
        if not check_ffmpeg():
            st.error("ffmpeg is required and was not found in PATH.")
        else:
            with st.spinner("Downloading and extracting audio..."):
                audio_file = download_and_extract_audio(url)

            if audio_file and os.path.exists(audio_file):
                with st.spinner("Classifying accent..."):
                    accent, confidence, explanation = classify_accent(audio_file)
                
                if accent:
                    st.success("Accent classification complete!")
                    st.write(f"**Predicted Accent**: {accent}")
                    st.write(f"**Confidence Score**: {confidence:.2f}%")
                    with st.expander("Explanation"):
                        st.write(explanation)
                else:
                    st.error("Could not determine the accent.")
            else:
                st.error("Failed to download or process the video. Check the URL or try another.")

        # Clean up temp file
        try:
            if os.path.exists(AUDIO_FILENAME):
                os.remove(AUDIO_FILENAME)
        except Exception as e:
            st.warning(f"Cleanup failed: {e}")
