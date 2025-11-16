import torch
import librosa
from transformers import AutoFeatureExtractor, pipeline
from datasets import Audio

# --- CONFIGURATION ---
# Path to your finetuned model
MODEL_PATH = "./emotion_model_finetuned/final" 

# Path to an example audio file you want to test
# Let's grab a "happy" one from the Actor_01 folder
TEST_FILE = "WhatsApp Audio 2025-11-14 at 3.53.24 PM.wav"

# --- 1. Load the pipeline ---
print(f"Loading model from {MODEL_PATH}...")
device = 0 if torch.cuda.is_available() else -1 # Use 0 for GPU, -1 for CPU
pipe = pipeline(
    "audio-classification",
    model=MODEL_PATH,
    device=device
)

# --- 2. Load and preprocess the audio ---
# librosa is good for single-file loading
# We MUST resample to 16kHz for the model
print(f"Loading audio file: {TEST_FILE}")
speech_array, sampling_rate = librosa.load(TEST_FILE, sr=16000)

# --- 3. Make a prediction ---
print("Running prediction...")
result = pipe(speech_array)

print("\n--- RESULT ---")
print(result)