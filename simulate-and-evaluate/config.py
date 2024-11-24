import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class EnvironmentConfig:
    # Model configurations
    MODEL_TYPE: str = os.getenv("MODEL_TYPE", "llama")  # "llama" or "gpt4"
    LLAMA_MODEL_PATH: str = os.getenv("LLAMA_MODEL_PATH", "meta-llama/Llama-2-7b")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", None)
    
    # Training configurations
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "1e-5"))
    MAX_STEPS: int = int(os.getenv("MAX_STEPS", "10000"))
    
    # Hardware configurations
    DEVICE: str = "cuda" if os.getenv("USE_CPU", "0") == "0" else "cpu"
    NUM_GPUS: int = int(os.getenv("NUM_GPUS", "1"))
    
    # Logging configurations
    WANDB_PROJECT: str = os.getenv("WANDB_PROJECT", "llm-behavior-comparison")
    WANDB_ENTITY: str = os.getenv("WANDB_ENTITY", None)
    
    # Data configurations
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "outputs")
    
    def validate(self):
        """Validate the configuration"""
        if self.MODEL_TYPE not in ["llama", "gpt4"]:
            raise ValueError(f"Invalid model type: {self.MODEL_TYPE}")
            
        if self.MODEL_TYPE == "gpt4" and not self.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is required for GPT-4")
            
        # Create necessary directories
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)