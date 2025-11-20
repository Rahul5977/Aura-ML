"""
LLM Model Wrapper for Aura
"""

import torch
from typing import Optional, List, Dict
from pathlib import Path
import logging

from unsloth import FastLanguageModel
from transformers import TextStreamer

from aura_ml.config.model_config import LLMConfig, InferenceConfig

logger = logging.getLogger(__name__)


class AuraLLM:
    """
    Wrapper for Llama 3.2 3B fine-tuned model with Unsloth optimizations
    """
    
    def __init__(
        self,
        model_path: str | Path,
        config: Optional[LLMConfig] = None,
        inference_config: Optional[InferenceConfig] = None
    ):
        """
        Initialize the LLM wrapper
        
        Args:
            model_path: Path to the fine-tuned model
            config: LLM configuration
            inference_config: Inference configuration
        """
        self.model_path = Path(model_path)
        self.config = config or LLMConfig()
        self.inference_config = inference_config or InferenceConfig()
        
        self.model = None
        self.tokenizer = None
        self.streamer = None
        
    def load_model(self) -> None:
        """Load the model and tokenizer"""
        logger.info(f"Loading model from {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        # Load model with Unsloth
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(self.model_path),
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=self.config.load_in_4bit,
            trust_remote_code=True,
        )
        
        # Enable fast inference
        if self.inference_config.use_fast_inference:
            logger.info("Enabling fast inference mode")
            FastLanguageModel.for_inference(self.model)
        
        # Create streamer if enabled
        if self.inference_config.enable_streaming:
            self.streamer = TextStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
        
        logger.info("Model loaded successfully")
        
    def build_prompt(
        self,
        user_input: str,
        emotion: Optional[str] = None,
        cause: Optional[str] = None,
        system_message: Optional[str] = None
    ) -> str:
        """
        Build prompt using Llama 3 chat template
        
        Args:
            user_input: User's message
            emotion: Current emotion
            cause: Cause of the emotion
            system_message: Custom system message
            
        Returns:
            Formatted prompt
        """
        if system_message is None:
            system_message = "You are Aura, an empathetic AI assistant specialized in emotional support."
            
            if emotion and cause:
                system_message += f"\nContext: User is feeling {emotion} because {cause}."
            elif emotion:
                system_message += f"\nContext: User is feeling {emotion}."
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: bool = True
    ) -> str:
        """
        Generate response from prompt
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stream: Whether to stream output
            
        Returns:
            Generated text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        max_new_tokens = max_new_tokens or self.inference_config.max_new_tokens
        temperature = temperature or self.inference_config.temperature
        top_p = top_p or self.inference_config.top_p
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=self.inference_config.do_sample,
                streamer=self.streamer if stream else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response
    
    def chat(
        self,
        user_input: str,
        emotion: Optional[str] = None,
        cause: Optional[str] = None,
        **generation_kwargs
    ) -> str:
        """
        High-level chat interface
        
        Args:
            user_input: User's message
            emotion: Current emotion
            cause: Cause of the emotion
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Model's response
        """
        prompt = self.build_prompt(user_input, emotion, cause)
        response = self.generate(prompt, **generation_kwargs)
        return response
