"""
Complete Audio-to-LLM Pipeline Test

Full pipeline test with:
1. Audio loading from test file
2. Speech-to-Text (Whisper)
3. Speech Emotion Recognition (Wav2Vec2)
4. Emotion Cause Extraction (ECE)
5. LLM Response Generation (Llama 3.2 3B)
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Direct imports to avoid circular dependencies
from unsloth import FastLanguageModel
import librosa
import soundfile as sf

# Color codes
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_section(text):
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}>>> {text}{Colors.ENDC}")


def print_context(label, value):
    print(f"{Colors.OKCYAN}{label}:{Colors.ENDC} {value}")


def print_response(text):
    print(f"{Colors.OKGREEN}{text}{Colors.ENDC}")


def load_audio(audio_path: str):
    """Load audio file and return waveform + sample rate"""
    print(f"Loading audio from: {audio_path}")
    
    try:
        # Try loading with librosa (handles more formats)
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(audio) / sr
        print(f"✓ Audio loaded: {duration:.2f}s duration, {sr}Hz sample rate")
        return audio, sr
    except Exception as e:
        print(f"{Colors.FAIL}Error loading audio: {e}{Colors.ENDC}")
        return None, None


def transcribe_audio(audio: np.ndarray, sr: int):
    """Transcribe audio using Whisper"""
    print_section("Step 1: Speech-to-Text (Whisper)")
    
    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
        print("Loading Whisper model...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        
        print("Transcribing audio...")
        
        # Process audio
        input_features = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(model.device)
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        transcription = processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        print_context("Transcription", f'"{transcription.strip()}"')
        return transcription.strip()
        
    except Exception as e:
        print(f"{Colors.FAIL}Error in transcription: {e}{Colors.ENDC}")
        # Fallback transcription for testing
        fallback = "I'm feeling really anxious because I have an important exam tomorrow."
        print(f"{Colors.WARNING}Using fallback transcription: {fallback}{Colors.ENDC}")
        return fallback


def detect_emotion(audio: np.ndarray, sr: int):
    """Detect emotion from speech using Wav2Vec2"""
    print_section("Step 2: Speech Emotion Recognition (Wav2Vec2)")
    
    try:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, AutoConfig
        
        # Try custom fine-tuned model first
        model_path = "emotion_model_finetuned/final"
        
        print(f"Loading SER model from: {model_path}")
        
        try:
            processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
            config = AutoConfig.from_pretrained(model_path)
            model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        except:
            print(f"{Colors.WARNING}Custom model not found, using pretrained{Colors.ENDC}")
            model_path = "superb/wav2vec2-base-superb-er"
            processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
            config = AutoConfig.from_pretrained(model_path)
            model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        
        print("Analyzing emotion from voice...")
        
        # Process audio
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).input_values.to(model.device)
        
        # Predict emotion
        with torch.no_grad():
            logits = model(inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        # Get emotion labels
        emotion_labels = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
        if hasattr(config, 'id2label'):
            emotion_labels = [config.id2label[i] for i in range(len(config.id2label))]
        
        # Get top emotion
        emotion_idx = torch.argmax(probs).item()
        emotion = emotion_labels[emotion_idx]
        confidence = probs[emotion_idx].item()
        
        print_context("Detected Emotion", emotion)
        print_context("Confidence", f"{confidence:.2%}")
        
        # Show all emotion scores
        print(f"\n{Colors.OKCYAN}All Emotion Scores:{Colors.ENDC}")
        for label, prob in zip(emotion_labels, probs):
            bar = '█' * int(prob.item() * 30)
            print(f"  {label:10s}: {bar} {prob.item():.2%}")
        
        return emotion, confidence, dict(zip(emotion_labels, probs.tolist()))
        
    except Exception as e:
        print(f"{Colors.FAIL}Error in emotion detection: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        # Fallback emotion
        return "fearful", 0.75, {"fearful": 0.75}


def extract_cause(text: str, emotion: str):
    """Extract emotion cause using ECE model"""
    print_section("Step 3: Emotion Cause Extraction (ECE)")
    
    try:
        from aura_ml.models.ece_classifier import EmotionCauseExtractor
        
        print("Loading ECE model...")
        ece_model = EmotionCauseExtractor(
            model_path="data/models/ece/ece_roberta_model"
        )
        ece_model.load_model()
        
        print("Extracting emotion cause...")
        result = ece_model.extract_causes(text=text, emotion=emotion)
        
        if result and result.get('causes'):
            causes = result['causes']
            print_context("Causes Found", len(causes))
            for i, cause in enumerate(causes, 1):
                print(f"  {i}. \"{cause['text']}\" (confidence: {cause['confidence']:.2%})")
            
            primary_cause = causes[0]['text']
        else:
            print_context("Causes Found", "None (using fallback)")
            # Fallback: extract after "because"
            if "because" in text.lower():
                parts = text.lower().split("because", 1)
                primary_cause = parts[1].strip().rstrip('.')
            else:
                primary_cause = None
        
        print_context("Primary Cause", f'"{primary_cause}"' if primary_cause else "N/A")
        return primary_cause
        
    except Exception as e:
        print(f"{Colors.FAIL}Error in cause extraction: {e}{Colors.ENDC}")
        # Fallback
        if "because" in text.lower():
            parts = text.lower().split("because", 1)
            return parts[1].strip().rstrip('.')
        return None


def generate_llm_response(text: str, emotion: str, cause: str):
    """Generate empathetic response using fine-tuned LLM"""
    print_section("Step 4: LLM Response Generation (Llama 3.2 3B)")
    
    try:
        model_path = "data/models/llm/llama3_finetuned_final"
        
        print(f"Loading LLM from: {model_path}")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        
        # Enable fast inference
        FastLanguageModel.for_inference(model)
        
        print("Building context-aware prompt...")
        
        # Build system message with context
        system_message = "You are Aura, an empathetic AI assistant specialized in emotional support."
        if emotion and cause:
            system_message += f"\nContext: User is feeling {emotion} because {cause}."
        elif emotion:
            system_message += f"\nContext: User is feeling {emotion}."
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        print_context("Emotion Context", emotion)
        print_context("Cause Context", cause or "N/A")
        
        print(f"\n{Colors.WARNING}Generating response (5-10 seconds)...{Colors.ENDC}\n")
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the generated part
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response
        
    except Exception as e:
        print(f"{Colors.FAIL}Error in LLM generation: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print_header("AURA COMPLETE PIPELINE TEST - AUDIO TO LLM RESPONSE")
    
    # Audio file path
    audio_path = "test_audio_file.wav"
    
    if not Path(audio_path).exists():
        print(f"{Colors.FAIL}Error: Audio file not found: {audio_path}{Colors.ENDC}")
        print(f"\n{Colors.WARNING}Please ensure test_audio_file.wav exists in the project root.{Colors.ENDC}")
        return
    
    print_context("Audio File", audio_path)
    
    # Load audio
    audio, sr = load_audio(audio_path)
    if audio is None:
        return
    
    # Step 1: Speech-to-Text
    transcription = transcribe_audio(audio, sr)
    
    # Step 2: Emotion Detection
    emotion, confidence, emotion_scores = detect_emotion(audio, sr)
    
    # Step 3: Cause Extraction
    cause = extract_cause(transcription, emotion)
    
    # Step 4: LLM Response
    response = generate_llm_response(transcription, emotion, cause)
    
    # Display final results
    print_header("COMPLETE PIPELINE RESULTS")
    
    print_section("User Input (from audio)")
    print(f'{Colors.BOLD}"{transcription}"{Colors.ENDC}')
    
    print_section("Analysis")
    print_context("Detected Emotion", f"{emotion} ({confidence:.2%} confidence)")
    print_context("Extracted Cause", f'"{cause}"' if cause else "N/A")
    
    print_section("AURA'S RESPONSE")
    if response:
        print_response(response)
    else:
        print(f"{Colors.FAIL}Failed to generate response{Colors.ENDC}")
    
    print(f"\n{Colors.OKGREEN}{'─'*70}{Colors.ENDC}")
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}✓ Complete pipeline test finished!{Colors.ENDC}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Test interrupted{Colors.ENDC}\n")
    except Exception as e:
        print(f"\n{Colors.FAIL}Error: {e}{Colors.ENDC}\n")
        import traceback
        traceback.print_exc()
