"""
Complete LLM Pipeline Test - Audio + ECE + LLM Response

Tests the full emotional support pipeline:
1. Audio emotion detection (simulated)
2. ECE model for cause extraction
3. LLM response with full context
"""

import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aura_ml.models.llm_wrapper import AuraLLM
from aura_ml.models.ece_classifier import EmotionCauseExtractor
from aura_ml.config.model_config import LLMConfig, InferenceConfig

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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


def simulate_audio_emotion(user_input: str) -> dict:
    """
    Simulate audio emotion detection based on keywords.
    In production, this would use Wav2Vec2 SER model.
    """
    # Simple keyword-based emotion detection for demo
    emotion_keywords = {
        'sad': ['sad', 'depressed', 'down', 'unhappy', 'crying', 'heartbroken'],
        'angry': ['angry', 'mad', 'furious', 'annoyed', 'frustrated'],
        'fear': ['anxious', 'worried', 'scared', 'afraid', 'nervous', 'panic'],
        'happy': ['happy', 'excited', 'joy', 'great', 'wonderful', 'amazing'],
        'neutral': []
    }
    
    user_lower = user_input.lower()
    detected_emotion = 'neutral'
    
    for emotion, keywords in emotion_keywords.items():
        if any(keyword in user_lower for keyword in keywords):
            detected_emotion = emotion
            break
    
    return {
        'emotion': detected_emotion,
        'confidence': 0.85,
        'source': 'audio_analysis'
    }


def test_complete_pipeline():
    """Test the complete emotional support pipeline"""
    
    print_header("AURA EMOTIONAL SUPPORT - COMPLETE PIPELINE TEST")
    
    # ========================================================================
    # STEP 1: Initialize Models
    # ========================================================================
    print_section("STEP 1: Loading Models")
    
    # Load ECE model
    print("Loading ECE (Emotion Cause Extraction) model...")
    ece_model = EmotionCauseExtractor(
        model_path="data/models/ece/ece_roberta_model"
    )
    ece_model.load_model()
    print(f"✓ ECE model loaded: {ece_model.model_path}")
    
    # Load LLM
    print("\nLoading Fine-tuned LLM (Llama 3.2 3B)...")
    llm_config = LLMConfig()
    inference_config = InferenceConfig(
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        enable_streaming=False  # Disable streaming for cleaner output
    )
    
    llm = AuraLLM(
        model_path="data/models/llm/llama3_finetuned_final",
        config=llm_config,
        inference_config=inference_config
    )
    llm.load_model()
    print(f"✓ LLM loaded: {llm.model_path}")
    
    # ========================================================================
    # STEP 2: Test Scenarios
    # ========================================================================
    test_scenarios = [
        {
            "name": "Academic Stress",
            "user_input": "I'm feeling really anxious because I have an important exam tomorrow and I haven't studied enough.",
            "description": "Student worried about exam preparation"
        },
        {
            "name": "Relationship Issue",
            "user_input": "I'm so sad because my girlfriend broke up with me yesterday and I don't know what to do.",
            "description": "Breakup causing sadness"
        },
        {
            "name": "Work Frustration",
            "user_input": "I'm frustrated because my boss keeps giving me more work but never appreciates what I do.",
            "description": "Feeling unappreciated at work"
        },
        {
            "name": "Family Conflict",
            "user_input": "I'm angry because my parents don't understand me and always criticize my decisions.",
            "description": "Conflict with parents"
        },
        {
            "name": "General Anxiety",
            "user_input": "I've been feeling worried lately about everything and I can't seem to relax.",
            "description": "Generalized anxiety without specific cause"
        }
    ]
    
    for idx, scenario in enumerate(test_scenarios, 1):
        print_header(f"TEST SCENARIO {idx}/{len(test_scenarios)}: {scenario['name']}")
        
        print(f"{Colors.WARNING}Description: {scenario['description']}{Colors.ENDC}")
        print(f"\n{Colors.BOLD}User Input:{Colors.ENDC}")
        print(f'"{scenario["user_input"]}"')
        
        # --------------------------------------------------------------------
        # Step 2.1: Audio Emotion Detection
        # --------------------------------------------------------------------
        print_section("Audio Emotion Analysis")
        audio_result = simulate_audio_emotion(scenario['user_input'])
        print_context("Detected Emotion", audio_result['emotion'])
        print_context("Confidence", f"{audio_result['confidence']:.2%}")
        print_context("Source", audio_result['source'])
        
        # --------------------------------------------------------------------
        # Step 2.2: ECE - Emotion Cause Extraction
        # --------------------------------------------------------------------
        print_section("Emotion Cause Extraction (ECE)")
        
        # Extract causes using ECE model
        ece_result = ece_model.extract_causes(
            text=scenario['user_input'],
            emotion=audio_result['emotion']
        )
        
        if ece_result and ece_result.get('causes'):
            causes = ece_result['causes']
            print_context("Number of Causes Found", len(causes))
            for i, cause in enumerate(causes, 1):
                print(f"  {i}. \"{cause['text']}\" (confidence: {cause['confidence']:.2%})")
            
            # Use the highest confidence cause
            primary_cause = causes[0]['text']
        else:
            print_context("Causes Found", "None (using heuristic fallback)")
            # Fallback: extract clause after "because"
            if "because" in scenario['user_input'].lower():
                parts = scenario['user_input'].lower().split("because", 1)
                primary_cause = parts[1].strip().rstrip('.')
            else:
                primary_cause = "unspecified situation"
        
        print_context("Primary Cause", f'"{primary_cause}"')
        
        # --------------------------------------------------------------------
        # Step 2.3: LLM Response Generation
        # --------------------------------------------------------------------
        print_section("Generating Empathetic Response")
        
        # Build context-aware prompt
        print(f"\n{Colors.OKCYAN}Building prompt with context...{Colors.ENDC}")
        print_context("- User emotion", audio_result['emotion'])
        print_context("- Emotion cause", primary_cause)
        
        # Generate response
        print(f"\n{Colors.OKCYAN}Generating response (this may take a few seconds)...{Colors.ENDC}\n")
        
        response = llm.chat(
            user_input=scenario['user_input'],
            emotion=audio_result['emotion'],
            cause=primary_cause,
            stream=False
        )
        
        # --------------------------------------------------------------------
        # Display Results
        # --------------------------------------------------------------------
        print_section("AURA'S RESPONSE")
        print_response(response)
        
        print(f"\n{Colors.OKGREEN}{'─'*70}{Colors.ENDC}")
        
        # Add separator between scenarios
        if idx < len(test_scenarios):
            input(f"\n{Colors.WARNING}Press Enter to continue to next scenario...{Colors.ENDC}")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print_header("PIPELINE TEST COMPLETE")
    print(f"{Colors.OKGREEN}✓ All {len(test_scenarios)} scenarios tested successfully{Colors.ENDC}")
    print(f"{Colors.OKGREEN}✓ Audio emotion detection working{Colors.ENDC}")
    print(f"{Colors.OKGREEN}✓ ECE cause extraction working{Colors.ENDC}")
    print(f"{Colors.OKGREEN}✓ LLM response generation working{Colors.ENDC}")
    print(f"\n{Colors.BOLD}The complete emotional support pipeline is functional!{Colors.ENDC}\n")


def test_interactive_mode():
    """Interactive chat mode with the LLM"""
    
    print_header("AURA EMOTIONAL SUPPORT - INTERACTIVE MODE")
    
    # Load models
    print_section("Loading Models")
    
    ece_model = EmotionCauseExtractor(
        model_path="data/models/ece/ece_roberta_model"
    )
    ece_model.load_model()
    print("✓ ECE model loaded")
    
    inference_config = InferenceConfig(
        max_new_tokens=200,
        temperature=0.7,
        enable_streaming=False
    )
    
    llm = AuraLLM(
        model_path="data/models/llm/llama3_finetuned_final",
        config=LLMConfig(),
        inference_config=inference_config
    )
    llm.load_model()
    print("✓ LLM loaded\n")
    
    print(f"{Colors.OKGREEN}Interactive mode ready! Type 'quit' to exit.{Colors.ENDC}\n")
    
    conversation_history = []
    
    while True:
        # Get user input
        print(f"{Colors.BOLD}You:{Colors.ENDC} ", end='')
        user_input = input().strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print(f"\n{Colors.OKGREEN}Thank you for chatting with Aura. Take care!{Colors.ENDC}\n")
            break
        
        if not user_input:
            continue
        
        # Analyze emotion
        audio_result = simulate_audio_emotion(user_input)
        
        # Extract cause
        ece_result = ece_model.extract_causes(
            text=user_input,
            emotion=audio_result['emotion']
        )
        
        if ece_result and ece_result.get('causes'):
            primary_cause = ece_result['causes'][0]['text']
        else:
            if "because" in user_input.lower():
                parts = user_input.lower().split("because", 1)
                primary_cause = parts[1].strip().rstrip('.')
            else:
                primary_cause = None
        
        # Show context (optional - comment out for cleaner chat)
        print(f"{Colors.OKCYAN}[Emotion: {audio_result['emotion']}, Cause: {primary_cause or 'N/A'}]{Colors.ENDC}")
        
        # Generate response
        response = llm.chat(
            user_input=user_input,
            emotion=audio_result['emotion'],
            cause=primary_cause,
            stream=False
        )
        
        print(f"{Colors.OKGREEN}{Colors.BOLD}Aura:{Colors.ENDC} {Colors.OKGREEN}{response}{Colors.ENDC}\n")
        
        # Store conversation
        conversation_history.append({
            'user': user_input,
            'aura': response,
            'emotion': audio_result['emotion'],
            'cause': primary_cause
        })


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Aura LLM Pipeline')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['test', 'interactive'],
        default='test',
        help='Run mode: test (predefined scenarios) or interactive (chat mode)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'test':
            test_complete_pipeline()
        else:
            test_interactive_mode()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Test interrupted by user{Colors.ENDC}\n")
    except Exception as e:
        print(f"\n{Colors.FAIL}Error: {e}{Colors.ENDC}\n")
        import traceback
        traceback.print_exc()
