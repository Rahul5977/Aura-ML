"""
Global configuration settings for Aura ML
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application settings"""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = DATA_DIR / "models"
    OUTPUTS_DIR: Path = DATA_DIR / "outputs"
    
    # Model paths
    ECE_MODEL_PATH: Path = MODELS_DIR / "ece" / "ece_roberta_model"
    LLM_MODEL_PATH: Path = MODELS_DIR / "llm" / "llama3_finetuned_final"
    
    # Training settings
    RANDOM_SEED: int = 42
    USE_GPU: bool = True
    MAX_GPU_MEMORY: Optional[str] = "6GB"
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    API_RELOAD: bool = False
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[Path] = OUTPUTS_DIR / "aura_ml.log"
    
    # Environment
    ENV: str = "development"  # development, staging, production
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Create necessary directories
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
