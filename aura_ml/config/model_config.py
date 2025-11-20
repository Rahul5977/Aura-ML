"""
Model configuration for training and inference
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ECEModelConfig:
    """Configuration for ECE (Emotion Cause Extraction) Model"""
    
    # Model architecture
    base_model: str = "roberta-base"
    num_labels: int = 2
    dropout: float = 0.1
    
    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Data
    max_length: int = 128
    train_test_split: float = 0.2
    
    # Output
    model_name: str = "ece_roberta_model"


@dataclass
class LLMConfig:
    """Configuration for LLM Fine-tuning"""
    
    # Model settings
    base_model: str = "unsloth/Llama-3.2-3B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = None
    
    # Training settings (for 6GB VRAM)
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10
    
    # Training duration
    num_train_epochs: Optional[int] = 3
    max_steps: Optional[int] = None
    
    # Optimizer
    optim: str = "adamw_8bit"
    
    # Logging and checkpointing
    logging_steps: int = 5
    save_steps: int = 100
    save_total_limit: int = 3
    
    # Output
    output_dir: str = "llama3_finetuned"
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


@dataclass
class InferenceConfig:
    """Configuration for inference"""
    
    # Generation parameters
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # Streaming
    enable_streaming: bool = True
    
    # Performance
    use_fast_inference: bool = True
