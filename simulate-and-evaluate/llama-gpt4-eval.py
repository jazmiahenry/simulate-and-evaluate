import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer
)
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from dataclasses import dataclass
import logging
import wandb
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Enhanced configuration for training parameters"""
    model_type: str = "llama"  # "llama" or "gpt4"
    model_path: str = "meta-llama/Llama-2-7b"  # Path for LLaMA model
    batch_size: int = 32
    learning_rate: float = 1e-5
    max_steps: int = 10000
    beta: float = 0.1
    eval_interval: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    openai_api_key: Optional[str] = None
    max_length: int = 512
    early_stopping_threshold: float = 0.01
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01

class ModelWrapper:
    """Wrapper class to provide a uniform interface for different models"""
    def __init__(self, config: TrainingConfig):
        self.config = config
        if config.model_type == "llama":
            self.tokenizer = LlamaTokenizer.from_pretrained(config.model_path)
            self.model = LlamaForCausalLM.from_pretrained(
                config.model_path,
                torch_dtype=torch.float16 if config.device == "cuda" else torch.float32
            ).to(config.device)
        elif config.model_type == "gpt4":
            assert config.openai_api_key is not None, "OpenAI API key required for GPT-4"
            openai.api_key = config.openai_api_key
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")  # For tokenization
            self.model = None  # GPT-4 accessed via API
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def generate(self, prompt: Union[str, List[str]], **kwargs) -> List[str]:
        """Generate completions using either LLaMA or GPT-4"""
        if self.config.model_type == "llama":
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.config.max_length
            ).to(self.config.device)
            
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
            
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
        else:  # GPT-4
            if isinstance(prompt, str):
                prompt = [prompt]
                
            responses = []
            for p in prompt:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": p}],
                    max_tokens=kwargs.get("max_tokens", 100),
                    temperature=kwargs.get("temperature", 0.7)
                )
                responses.append(response.choices[0].message.content)
            
            return responses

class PreferenceDataset(Dataset):
    """Dataset for preference pairs with support for both LLaMA and GPT-4"""
    def __init__(self, prompts: List[str], chosen: List[str], rejected: List[str], 
                 model_wrapper: ModelWrapper):
        self.model_wrapper = model_wrapper
        self.prompts = prompts
        self.chosen = chosen
        self.rejected = rejected
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        # For LLaMA, tokenize normally
        if self.model_wrapper.config.model_type == "llama":
            return {
                "prompt": self.model_wrapper.tokenizer(
                    self.prompts[idx],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.model_wrapper.config.max_length
                ),
                "chosen": self.model_wrapper.tokenizer(
                    self.chosen[idx],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.model_wrapper.config.max_length
                ),
                "rejected": self.model_wrapper.tokenizer(
                    self.rejected[idx],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.model_wrapper.config.max_length
                )
            }
        # For GPT-4, store raw text
        else:
            return {
                "prompt": self.prompts[idx],
                "chosen": self.chosen[idx],
                "rejected": self.rejected[idx]
            }

class GPT4RewardModel:
    """Reward model using GPT-4 for evaluation"""
    def __init__(self, api_key: str):
        openai.api_key = api_key
        
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def __call__(self, prompt: str, completion: str) -> float:
        """Compute reward using GPT-4's assessment"""
        evaluation_prompt = f"""
        Rate the following completion for the given prompt on a scale of 0 to 1,
        where 1 is perfect and 0 is completely inappropriate.
        
        Prompt: {prompt}
        Completion: {completion}
        
        Output only the numerical score between 0 and 1.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": evaluation_prompt}],
            max_tokens=10,
            temperature=0
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1
        except ValueError:
            logger.warning("Failed to parse GPT-4 reward score")
            return 0.0

class LLaMADPOTrainer(DPOTrainer):
    """DPO Trainer specifically optimized for LLaMA"""
    def __init__(self, model_wrapper: ModelWrapper, ref_model_wrapper: ModelWrapper, 
                 config: TrainingConfig):
        super().__init__(model_wrapper.model, ref_model_wrapper.model, config)
        self.tokenizer = model_wrapper.tokenizer
        
    def compute_dpo_loss(self, batch, weights: Optional[torch.Tensor] = None):
        """Compute DPO loss with LLaMA-specific optimizations"""
        # Implementation similar to base DPO but with LLaMA-specific handling
        # [Previous DPO loss computation with LLaMA optimizations]
        pass

class GPT4DPOTrainer:
    """DPO Trainer for GPT-4 using API calls"""
    def __init__(self, model_wrapper: ModelWrapper, config: TrainingConfig):
        self.model_wrapper = model_wrapper
        self.config = config
        
    def train_step(self, batch):
        """Simulate training step for GPT-4"""
        # Since we can't actually train GPT-4, we'll log performance metrics
        logger.info("Note: GPT-4 cannot be trained, only evaluated")
        return None

def main():
    # Initialize config with model-specific settings
    config = TrainingConfig(
        model_type="llama",  # or "gpt4"
        model_path="meta-llama/Llama-2-7b",
        openai_api_key="your-api-key"  # if using GPT-4
    )
    
    # Initialize model wrappers
    model_wrapper = ModelWrapper(config)
    ref_model_wrapper = ModelWrapper(config)  # For LLaMA reference model
    
    # Initialize datasets
    train_dataset = PreferenceDataset([], [], [], model_wrapper)
    eval_dataset = PreferenceDataset([], [], [], model_wrapper)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size)
    
    # Initialize appropriate trainer based on model type
    if config.model_type == "llama":
        trainer = LLaMADPOTrainer(model_wrapper, ref_model_wrapper, config)
    else:
        trainer = GPT4DPOTrainer(model_wrapper, config)
    
    # Train and evaluate
    wandb.init(project="llm-behavior-comparison")
    train_and_evaluate(trainer, train_dataloader, eval_dataloader, config)
    wandb.finish()

if __name__ == "__main__":
    main()