"""
Interactive Chatbot Implementation
"""

from typing import Optional, Callable
import logging

from aura_ml.models.llm_wrapper import AuraLLM
from aura_ml.config.model_config import InferenceConfig

logger = logging.getLogger(__name__)


class AuraChatbot:
    """
    Interactive chatbot with emotion context management
    """
    
    def __init__(
        self,
        llm: AuraLLM,
        config: Optional[InferenceConfig] = None
    ):
        """
        Initialize chatbot
        
        Args:
            llm: LLM wrapper instance
            config: Inference configuration
        """
        self.llm = llm
        self.config = config or InferenceConfig()
        
        # Conversation state
        self.current_emotion: Optional[str] = None
        self.current_cause: Optional[str] = None
        self.conversation_history = []
        
    def set_emotion_context(self, emotion: str, cause: Optional[str] = None) -> None:
        """
        Set the emotional context for the conversation
        
        Args:
            emotion: Current emotion
            cause: Cause of the emotion
        """
        self.current_emotion = emotion
        self.current_cause = cause
        logger.info(f"Emotion context set: {emotion}" + (f" - {cause}" if cause else ""))
        
    def clear_emotion_context(self) -> None:
        """Clear the emotional context"""
        self.current_emotion = None
        self.current_cause = None
        logger.info("Emotion context cleared")
        
    def chat(
        self,
        user_input: str,
        stream: bool = True
    ) -> str:
        """
        Send a message and get a response
        
        Args:
            user_input: User's message
            stream: Whether to stream the response
            
        Returns:
            Bot's response
        """
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "emotion": self.current_emotion,
            "cause": self.current_cause
        })
        
        # Generate response
        response = self.llm.chat(
            user_input=user_input,
            emotion=self.current_emotion,
            cause=self.current_cause,
            stream=stream
        )
        
        # Add response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def get_conversation_history(self) -> list:
        """Get the conversation history"""
        return self.conversation_history
    
    def clear_history(self) -> None:
        """Clear the conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")


def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║      🌟 AURA - Emotional Support AI Assistant 🌟           ║
║                                                              ║
║  Powered by Llama 3.2 3B | Optimized for RTX 4050           ║
║  Fast Inference with Unsloth | Streaming Responses          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_help():
    """Print help message"""
    help_text = """
Commands:
  • Type your message to chat with Aura
  • /emotion <emotion> <cause> - Set your current emotion and cause
  • /clear - Clear the current emotion context
  • /history - Show conversation history
  • /reset - Reset conversation history
  • /help - Show this help message
  • /quit or /exit - Exit the chat

Examples:
  /emotion anxious I have an important exam tomorrow
  How can I calm down before my exam?
  
  /emotion sad my friend moved away
  I'm feeling really lonely lately

Tips:
  • Be honest about your feelings - Aura is here to help
  • You can chat without setting an emotion too!
  • Aura provides empathetic, supportive responses
"""
    print(help_text)


def interactive_loop(
    chatbot: AuraChatbot,
    on_exit: Optional[Callable] = None
) -> None:
    """
    Run the interactive chat loop
    
    Args:
        chatbot: Chatbot instance
        on_exit: Optional callback to run on exit
    """
    print_banner()
    print("Type '/help' for commands or start chatting!")
    print("─" * 64)
    print()
    
    while True:
        try:
            # Show emotion context if set
            if chatbot.current_emotion:
                context_str = f"[Context: {chatbot.current_emotion}"
                if chatbot.current_cause:
                    context_str += f" - {chatbot.current_cause}"
                context_str += "]"
                print(f"\033[90m{context_str}\033[0m")  # Gray text
            
            # Get user input
            user_input = input("\033[94m😊 You:\033[0m ").strip()  # Blue text
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                command = user_input.lower().split()[0]
                
                if command in ['/quit', '/exit']:
                    print("\n👋 Goodbye! Take care of yourself!")
                    if on_exit:
                        on_exit()
                    break
                
                elif command == '/help':
                    print_help()
                    continue
                
                elif command == '/clear':
                    chatbot.clear_emotion_context()
                    print("✅ Emotion context cleared")
                    print()
                    continue
                
                elif command == '/emotion':
                    parts = user_input.split(maxsplit=2)
                    if len(parts) >= 2:
                        emotion = parts[1]
                        cause = parts[2] if len(parts) > 2 else None
                        chatbot.set_emotion_context(emotion, cause)
                        print(f"✅ Emotion set: {emotion}")
                        if cause:
                            print(f"   Cause: {cause}")
                        print()
                    else:
                        print("❌ Usage: /emotion <emotion> <cause>")
                    continue
                
                elif command == '/history':
                    history = chatbot.get_conversation_history()
                    if history:
                        print("\n📜 Conversation History:")
                        for i, entry in enumerate(history, 1):
                            role = "You" if entry["role"] == "user" else "Aura"
                            print(f"{i}. {role}: {entry['content'][:80]}...")
                        print()
                    else:
                        print("No conversation history yet.")
                    continue
                
                elif command == '/reset':
                    chatbot.clear_history()
                    print("✅ Conversation history reset")
                    print()
                    continue
                
                else:
                    print(f"❌ Unknown command: {command}")
                    print("Type '/help' for available commands")
                    continue
            
            # Generate response with streaming
            print("\033[92m🌟 Aura:\033[0m ", end="", flush=True)  # Green text
            
            response = chatbot.chat(user_input, stream=True)
            
            print()  # Newline after streaming response
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Take care of yourself!")
            if on_exit:
                on_exit()
            break
        
        except Exception as e:
            logger.error(f"Error in chat loop: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")
            print("Please try again or type /quit to exit.\n")
