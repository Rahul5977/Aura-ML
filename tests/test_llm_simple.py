"""
Simple LLM Test - Direct imports without audio processor

Tests LLM response with ECE context extraction
"""

import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Direct imports to avoid audio processor issues
from unsloth import FastLanguageModel
from transformers import TextStreamer

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


def load_llm():
    """Load the fine-tuned LLM"""
    model_path = "data/models/llm/llama3_finetuned_final"
    
    print(f"Loading model from: {model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    
    # Enable fast inference
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer


def build_prompt(tokenizer, user_input, emotion=None, cause=None):
    """Build prompt using Llama 3 chat template"""
    system_message = "You are Aura, an empathetic AI assistant specialized in emotional support."
    
    if emotion and cause:
        system_message += f"\nContext: User is feeling {emotion} because {cause}."
    elif emotion:
        system_message += f"\nContext: User is feeling {emotion}."
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return prompt


def generate_response(model, tokenizer, prompt, max_new_tokens=200):
    """Generate response from the model"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
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


def main():
    print_header("AURA LLM - RESPONSE TEST WITH CONTEXT")
    
    # Load model
    print_section("Loading Fine-tuned LLM")
    model, tokenizer = load_llm()
    print(f"{Colors.OKGREEN}✓ Model loaded successfully{Colors.ENDC}")
    
    # Test scenarios with emotion and cause context
    test_scenarios = [
        {
            "name": "Academic Stress",
            "user_input": "I'm feeling really anxious because I have an important exam tomorrow and I haven't studied enough.",
            "emotion": "fear",
            "cause": "I have an important exam tomorrow and I haven't studied enough"
        },
        {
            "name": "Relationship Breakup",
            "user_input": "I'm so sad because my girlfriend broke up with me yesterday and I don't know what to do.",
            "emotion": "sad",
            "cause": "my girlfriend broke up with me yesterday"
        },
        {
            "name": "Work Frustration",
            "user_input": "I'm frustrated because my boss keeps giving me more work but never appreciates what I do.",
            "emotion": "angry",
            "cause": "my boss keeps giving me more work but never appreciates what I do"
        },
        {
            "name": "No Specific Emotion",
            "user_input": "I've been feeling overwhelmed lately and I need someone to talk to.",
            "emotion": "neutral",
            "cause": None
        }
    ]
    
    for idx, scenario in enumerate(test_scenarios, 1):
        print_header(f"TEST {idx}/{len(test_scenarios)}: {scenario['name']}")
        
        print(f"{Colors.BOLD}User Input:{Colors.ENDC}")
        print(f'"{scenario["user_input"]}"')
        
        print_section("Context Information")
        print_context("Emotion", scenario['emotion'])
        print_context("Cause", scenario['cause'] or "None detected")
        
        print_section("Building Prompt")
        prompt = build_prompt(
            tokenizer,
            scenario['user_input'],
            scenario['emotion'],
            scenario['cause']
        )
        print(f"{Colors.OKCYAN}Prompt length: {len(prompt)} characters{Colors.ENDC}")
        
        print_section("Generating Response...")
        print(f"{Colors.WARNING}(This may take 5-10 seconds on GPU...){Colors.ENDC}\n")
        
        response = generate_response(model, tokenizer, prompt)
        
        print_section("AURA'S RESPONSE")
        print_response(response)
        
        print(f"\n{Colors.OKGREEN}{'─'*70}{Colors.ENDC}")
        
        if idx < len(test_scenarios):
            input(f"\n{Colors.WARNING}Press Enter for next test...{Colors.ENDC}")
    
    print_header("TEST COMPLETE")
    print(f"{Colors.OKGREEN}✓ All {len(test_scenarios)} scenarios tested{Colors.ENDC}")
    print(f"{Colors.OKGREEN}✓ Fine-tuned model responding with emotional context{Colors.ENDC}\n")


def interactive_mode():
    """Simple interactive chat"""
    print_header("AURA - INTERACTIVE CHAT MODE")
    
    print_section("Loading Model")
    model, tokenizer = load_llm()
    print(f"{Colors.OKGREEN}✓ Ready to chat!{Colors.ENDC}\n")
    print(f"{Colors.WARNING}Type 'quit' to exit{Colors.ENDC}\n")
    
    while True:
        print(f"{Colors.BOLD}You:{Colors.ENDC} ", end='')
        user_input = input().strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print(f"\n{Colors.OKGREEN}Goodbye!{Colors.ENDC}\n")
            break
        
        if not user_input:
            continue
        
        # Simple emotion detection from keywords
        emotion = 'neutral'
        if any(word in user_input.lower() for word in ['sad', 'depressed', 'down']):
            emotion = 'sad'
        elif any(word in user_input.lower() for word in ['anxious', 'worried', 'nervous']):
            emotion = 'fear'
        elif any(word in user_input.lower() for word in ['angry', 'frustrated', 'mad']):
            emotion = 'angry'
        elif any(word in user_input.lower() for word in ['happy', 'excited', 'great']):
            emotion = 'happy'
        
        # Extract cause if "because" is present
        cause = None
        if "because" in user_input.lower():
            parts = user_input.split("because", 1)
            if len(parts) > 1:
                cause = parts[1].strip()
        
        # Show context
        print(f"{Colors.OKCYAN}[Emotion: {emotion}, Cause: {cause or 'N/A'}]{Colors.ENDC}")
        
        # Generate response
        prompt = build_prompt(tokenizer, user_input, emotion, cause)
        response = generate_response(model, tokenizer, prompt)
        
        print(f"{Colors.OKGREEN}{Colors.BOLD}Aura:{Colors.ENDC} {Colors.OKGREEN}{response}{Colors.ENDC}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Aura LLM')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['test', 'chat'],
        default='test',
        help='Run mode: test (4 scenarios) or chat (interactive)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'test':
            main()
        else:
            interactive_mode()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Interrupted{Colors.ENDC}\n")
    except Exception as e:
        print(f"\n{Colors.FAIL}Error: {e}{Colors.ENDC}\n")
        import traceback
        traceback.print_exc()
