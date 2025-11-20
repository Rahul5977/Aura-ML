#!/usr/bin/env python3
"""
CLI for Aura Chatbot
"""

import argparse
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aura_ml.models.llm_wrapper import AuraLLM
from aura_ml.inference.chatbot import AuraChatbot, interactive_loop
from aura_ml.config.model_config import LLMConfig, InferenceConfig
from aura_ml.config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Aura - Emotional Support AI Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Path to fine-tuned model (default: {settings.LLM_MODEL_PATH})"
    )
    
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate (default: 128)"
    )
    
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test and exit"
    )
    
    args = parser.parse_args()
    
    # Determine model path
    model_path = args.model if args.model else settings.LLM_MODEL_PATH
    model_path = Path(model_path)
    
    # Check if model exists
    if not model_path.exists():
        logger.error(f"❌ Model not found at {model_path}")
        logger.info("\n🔍 Looking for available models...")
        
        # Search for models
        search_paths = [
            Path("fine-tuining"),
            Path("data/models/llm"),
            Path(".")
        ]
        
        found_models = []
        for search_path in search_paths:
            if search_path.exists():
                for item in search_path.iterdir():
                    if item.is_dir() and any(x in item.name.lower() for x in ["llama", "aura", "finetuned"]):
                        found_models.append(item)
        
        if found_models:
            logger.info("\nAvailable models:")
            for model in found_models:
                logger.info(f"  • {model}")
            logger.info(f"\nUse --model <path> to specify a model")
        else:
            logger.info("No models found. Please train a model first.")
        
        sys.exit(1)
    
    # Create configurations
    llm_config = LLMConfig(
        max_seq_length=args.max_seq_length
    )
    
    inference_config = InferenceConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        enable_streaming=not args.no_stream
    )
    
    # Initialize LLM
    logger.info("🦙 Initializing Aura...")
    llm = AuraLLM(
        model_path=model_path,
        config=llm_config,
        inference_config=inference_config
    )
    
    # Load model
    llm.load_model()
    
    # Create chatbot
    chatbot = AuraChatbot(llm, inference_config)
    
    # Test mode
    if args.test:
        logger.info("\n🧪 Running quick test...\n")
        
        test_cases = [
            {
                "input": "I'm feeling overwhelmed with everything going on",
                "emotion": "stressed",
                "cause": "too many responsibilities"
            },
            {
                "input": "How can I feel better?",
                "emotion": "sad",
                "cause": "bad day at work"
            },
        ]
        
        for i, test in enumerate(test_cases, 1):
            print(f"Test {i}:")
            print(f"  Input: {test['input']}")
            print(f"  Emotion: {test['emotion']}")
            print(f"  Cause: {test['cause']}")
            print(f"\n  Aura: ", end="", flush=True)
            
            chatbot.set_emotion_context(test['emotion'], test['cause'])
            response = chatbot.chat(test['input'])
            print("\n")
        
        logger.info("✅ Test completed!")
        return
    
    # Interactive mode
    interactive_loop(chatbot)


if __name__ == "__main__":
    main()
