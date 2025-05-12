import os
os.environ["PYTORCH_NO_CUSTOM_CLASS_WARNING"] = "1"
import argparse
import subprocess

import shutil # For checking ffmpeg path
import streamlit as st

# Attempt to import PyTorch and related libraries early to catch missing installations.
try:
    import torch
    from speechbrain.inference.classifiers import EncoderClassifier
    from huggingface_hub import HfFolder 
except ImportError as e:
    print(f"Error: A required library is missing: {e.name}")
    print("Please ensure you have installed all dependencies from requirements.txt:")
    print("pip install -r requirements.txt")
    exit(1)

# Global variable for the downloaded audio filename
AUDIO_FILENAME = "temp_extracted_audio.wav"

# Mapping for user-friendly accent labels
ACCENT_LABEL_MAP =  {
    "us": "American English (US)",
    "australia": "Australian English (AU)",
    "canada": "Canadian English (CA)",
    "indian": "Indian English (IN)",
    "african": "African English (AF)",
    "newzealand": "New Zealand English (NZ)",
    "ireland": "Irish English (IE)",
    "southatlandtic": "South Atlantic English (SA)",
    "SG": "Singaporean English (SG)",
    "philippines": "Philippine English (PH)",
    "england": "British English (UK)", 
    "scotland": "Scottish English (SC)",
}


def check_ffmpeg():
    """Checks if ffmpeg is installed and in PATH."""
    if shutil.which("ffmpeg") is None:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! CRITICAL: ffmpeg is NOT found or not executable.                       !!!")
        print("!!! This tool requires ffmpeg for audio extraction and processing.         !!!")
        print("!!! Please install ffmpeg and ensure it's in your system's PATH.           !!!")
        print("!!! Installation guides:                                                   !!!")
        print("!!!   - Official: https://ffmpeg.org/download.html                         !!!")
        print("!!!   - Windows: Search 'how to install ffmpeg on Windows'                 !!!")
        print("!!!   - macOS (via Homebrew): brew install ffmpeg                          !!!")
        print("!!!   - Linux (via apt): sudo apt update && sudo apt install ffmpeg        !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return False
    print("ffmpeg found.")
    return True

def download_and_extract_audio(video_url):
    print(f"\nAttempting to download and extract audio from: {video_url}...")
    # Ensure previous temp file is removed if it exists
    if os.path.exists(AUDIO_FILENAME):
        try:
            os.remove(AUDIO_FILENAME)
        except OSError as e:
            print(f"Warning: Could not remove existing temp file {AUDIO_FILENAME}: {e}")
    try:
        command = [
            "yt-dlp",
            "-x", "--audio-format", "wav",
            "--audio-quality", "0",
            "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1", 
            "-o", AUDIO_FILENAME,
            "--no-playlist",
            "--no-warnings",
            "--progress",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            video_url
        ]
        
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        
        if not os.path.exists(AUDIO_FILENAME):
            print(f"Error: yt-dlp completed but output file {AUDIO_FILENAME} was not created.")
            print(f"yt-dlp stdout: {process.stdout}")
            print(f"yt-dlp stderr: {process.stderr}")
            return None

        print(f"Audio extracted successfully to {AUDIO_FILENAME}")
        return AUDIO_FILENAME
        
    except subprocess.CalledProcessError as e:
        print(f"Error during yt-dlp execution (return code {e.returncode}):")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"stdout: {e.stdout.strip()}")
        print(f"stderr: {e.stderr.strip()}")
        if "ffmpeg" in e.stderr.lower() or "ffprobe" in e.stderr.lower():
            print("This error might be related to ffmpeg. Ensure it's correctly installed and in PATH.")
        return None
    except FileNotFoundError:
        print("Error: yt-dlp command not found. Please ensure yt-dlp is installed and in your PATH.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during download/extraction: {e}")
        return None

def classify_accent(audio_file_path):
    print(f"\nLoading accent classification model ('Jzuluaga/accent-id-commonaccent_ecapa')...")
    print("(This may take a few minutes on the first run as the model is downloaded.)")
    
    # Determine device
    device = torch.device('cpu')
    print(f"Using device: {device}")

    try:
        # Load the pre-trained accent classification model
        classifier = EncoderClassifier.from_hparams(
            source="Jzuluaga/accent-id-commonaccent_ecapa",
            savedir="pretrained_models/Jzuluaga_accent-id-commonaccent_ecapa", 
            run_opts={"device": device}
        )
        print("Accent classification model loaded successfully.")
    except Exception as e:
        print(f"Error loading the accent classification model: {e}")
        print("This could be due to network issues, an invalid model name, or compatibility problems.")
        print("Ensure you have a stable internet connection for the initial download.")
        return None, None, None

    print(f"\nClassifying accent for {audio_file_path}...")
    try:
        # Load the audio file
        signal = classifier.load_audio(audio_file_path) # Load audio explicitly for clarity
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)

        out_prob, score, index, text_lab = classifier.classify_batch(signal)
        

        predicted_label_short = text_lab[0]
        
        
        predicted_accent_friendly = ACCENT_LABEL_MAP.get(predicted_label_short, f"Unknown Accent: {predicted_label_short}")
        confidence = float(torch.max(out_prob).item() * 100)

        explanation = (
            f"The model predicted the accent as: {predicted_accent_friendly}.\n"
            f"The input audio ('{os.path.basename(audio_file_path)}') was processed, and acoustic features were extracted.\n"
            f"These features were then analyzed by a pre-trained neural network specifically designed for English accent classification.\n"
            f"The network assigned probabilities to various known accents (American, British, Australian, Canadian, Indian), "
            f"and '{predicted_accent_friendly}' received the highest probability of {confidence:.2f}%."
        )
        print(f"Accent classification complete.")
        return predicted_accent_friendly, confidence, explanation

    except FileNotFoundError:
        print(f"Error: Audio file {audio_file_path} not found for classification.")
        return None, None, None
    except RuntimeError as e:
        if "ffmpeg" in str(e).upper() or "sox" in str(e).upper(): # Check if error is related to FFMPEG/SoX
             print(f"RuntimeError during classification, possibly an audio backend issue (FFMPEG/SoX): {e}")
             print("Please ensure FFMPEG (and SoX if used by torchaudio backend) is installed and accessible.")
        else:
            print(f"RuntimeError during classification: {e}")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during accent classification: {e}")
        return None, None, None

def main():
    # Check if Hugging Face token is set
    #hf_token = st.secrets["huggingface"]["token"]
    #HfFolder.save_token(hf_token)
    HfFolder.save_token(HfFolder.get_token() or "")
    

    parser = argparse.ArgumentParser(
        description="Analyze speaker's English accent from a public video URL.",
        formatter_class=argparse.RawTextHelpFormatter 
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Public video URL (e.g., YouTube, Loom, direct MP4 link)."
    )
    args = parser.parse_args()

    print("--- Accent Analyzer Tool ---")

    if not check_ffmpeg():
        return 

    audio_file = download_and_extract_audio(args.url)

    if audio_file and os.path.exists(audio_file):
        accent, confidence, explanation = classify_accent(audio_file)
        if accent and confidence is not None:
            print("\n--- Accent Analysis Results ---")
            print(f"Accent Classification: {accent}")
            print(f"Confidence Score: {confidence:.2f}%")
            print("\n--- Explanation ---")
            print(explanation)
        else:
            print("\n--- Accent Analysis Failed ---")
            print("Could not determine the accent. Please check the logs above for errors.")
            print("Possible reasons: unsupported audio format, poor audio quality, or issues with the model.")

        # Clean up the downloaded audio file
        try:
            if os.path.exists(AUDIO_FILENAME):
                os.remove(AUDIO_FILENAME)
                print(f"\nTemporary audio file '{AUDIO_FILENAME}' has been deleted.")
        except Exception as e:
            print(f"Warning: Could not delete temporary audio file '{AUDIO_FILENAME}': {e}")
    else:
        print("\n--- Process Failed ---")
        print("Failed to download or extract audio. Cannot proceed with accent analysis.")
        if audio_file and not os.path.exists(audio_file):
            print(f"Error: Audio file {audio_file} was expected but not found after download step.")


if __name__ == "__main__":
    main()
