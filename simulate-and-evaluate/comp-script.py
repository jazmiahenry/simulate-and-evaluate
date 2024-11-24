import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass
import logging
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    batch_size: int = 32
    learning_rate: float = 1e-5
    max_steps: int = 10000
    beta: float = 0.1  # KL penalty coefficient
    eval_interval: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class PreferenceDataset(Dataset):
    """Dataset for preference pairs (x, y_w, y_l)"""
    def __init__(self, prompts: List[str], chosen: List[str], rejected: List[str], tokenizer):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.chosen = chosen
        self.rejected = rejected
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt_tokens = self.tokenizer(self.prompts[idx], return_tensors="pt", padding=True, truncation=True)
        chosen_tokens = self.tokenizer(self.chosen[idx], return_tensors="pt", padding=True, truncation=True)
        rejected_tokens = self.tokenizer(self.rejected[idx], return_tensors="pt", padding=True, truncation=True)
        
        return {
            "prompt": prompt_tokens,
            "chosen": chosen_tokens,
            "rejected": rejected_tokens
        }

class EMMIA:
    """Implementation of EM-MIA membership inference"""
    def __init__(self, model):
        self.model = model
        self.membership_scores = {}
        
    def compute_membership_score(self, x: str, y: str) -> float:
        """Compute membership score for input-output pair"""
        # Simplified implementation - in practice, this would use more sophisticated
        # techniques to determine membership likelihood
        with torch.no_grad():
            inputs = self.model.tokenizer(x, return_tensors="pt")
            outputs = self.model(inputs.input_ids)
            logits = outputs.logits
            # Calculate probability of generating y given x
            score = F.softmax(logits, dim=-1).mean().item()
        return score
    
    def update_membership_scores(self, dataset: PreferenceDataset):
        """Update membership scores for entire dataset"""
        for i in range(len(dataset)):
            x = dataset.prompts[i]
            y_w = dataset.chosen[i]
            self.membership_scores[(x, y_w)] = self.compute_membership_score(x, y_w)

class DPOTrainer:
    """Direct Preference Optimization Trainer"""
    def __init__(self, model, ref_model, config: TrainingConfig):
        self.model = model
        self.ref_model = ref_model
        self.config = config
        
    def compute_dpo_loss(self, batch, weights: Optional[torch.Tensor] = None):
        """Compute DPO loss for a batch"""
        chosen_logps = self.model(**batch["chosen"]).logits
        rejected_logps = self.model(**batch["rejected"]).logits
        
        with torch.no_grad():
            ref_chosen_logps = self.ref_model(**batch["chosen"]).logits
            ref_rejected_logps = self.ref_model(**batch["rejected"]).logits
        
        advantage = self.config.beta * (
            (chosen_logps - ref_chosen_logps) - 
            (rejected_logps - ref_rejected_logps)
        )
        
        loss = -F.logsigmoid(advantage)
        if weights is not None:
            loss = loss * weights
            
        return loss.mean()

class EMDPOTrainer(DPOTrainer):
    """EM-MIA enhanced DPO Trainer"""
    def __init__(self, model, ref_model, config: TrainingConfig):
        super().__init__(model, ref_model, config)
        self.emmia = EMMIA(model)
        
    def train_step(self, batch):
        """Training step with EM-MIA weighted loss"""
        # Get membership scores for batch
        weights = torch.tensor([
            self.emmia.membership_scores.get((x, y_w), 1.0)
            for x, y_w in zip(batch["prompt"], batch["chosen"])
        ])
        
        return self.compute_dpo_loss(batch, weights)

class RLHFTrainer:
    """RLHF Trainer using PPO"""
    def __init__(self, model, reward_model, config: TrainingConfig):
        self.model = model
        self.reward_model = reward_model
        self.config = config
        
    def compute_ppo_loss(self, batch):
        """Compute PPO loss"""
        # Generate responses
        with torch.no_grad():
            old_outputs = self.model(**batch["prompt"])
            old_logprobs = F.log_softmax(old_outputs.logits, dim=-1)
        
        # Get current policy outputs
        new_outputs = self.model(**batch["prompt"])
        new_logprobs = F.log_softmax(new_outputs.logits, dim=-1)
        
        # Compute rewards
        rewards = self.reward_model(batch["prompt"], new_outputs.logits)
        
        # Calculate PPO loss
        ratio = torch.exp(new_logprobs - old_logprobs)
        surr1 = ratio * rewards
        surr2 = torch.clamp(ratio, 1 - self.config.beta, 1 + self.config.beta) * rewards
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        return policy_loss

def train_and_evaluate(trainer, train_dataloader, eval_dataloader, config: TrainingConfig):
    """Training and evaluation loop"""
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=config.learning_rate)
    
    for step in range(config.max_steps):
        trainer.model.train()
        batch = next(iter(train_dataloader))
        
        # Move batch to device
        batch = {k: v.to(config.device) for k, v in batch.items()}
        
        # Compute loss
        if isinstance(trainer, EMDPOTrainer):
            loss = trainer.train_step(batch)
        elif isinstance(trainer, DPOTrainer):
            loss = trainer.compute_dpo_loss(batch)
        else:  # RLHFTrainer
            loss = trainer.compute_ppo_loss(batch)
            
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log metrics
        if step % config.eval_interval == 0:
            eval_metrics = evaluate(trainer.model, eval_dataloader)
            wandb.log({
                "step": step,
                "train_loss": loss.item(),
                **eval_metrics
            })
            
def compute_kl_divergence(model_logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between model and reference model distributions
    """
    model_probs = F.softmax(model_logits, dim=-1)
    ref_probs = F.softmax(ref_logits, dim=-1)
    return F.kl_div(
        model_probs.log(),
        ref_probs,
        reduction='batchmean',
        log_target=False
    )

def compute_reward(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute simple reward based on token overlap with target
    For real applications, this should be replaced with a trained reward model
    """
    pred_tokens = torch.argmax(outputs, dim=-1)
    matches = (pred_tokens == targets).float()
    return matches.mean(dim=-1)

def evaluate(model, ref_model, eval_dataloader, reward_model=None):
    """
    Evaluation function with multiple metrics:
    - KL divergence from reference model
    - Reward achieved
    - Token accuracy
    - Loss on preferred vs non-preferred completions
    """
    model.eval()
    ref_model.eval()
    
    metrics = {
        'kl_divergence': 0.0,
        'reward': 0.0,
        'preference_accuracy': 0.0,
        'token_accuracy': 0.0,
        'eval_loss': 0.0
    }
    
    n_batches = len(eval_dataloader)
    
    with torch.no_grad():
        for batch in eval_dataloader:
            # 1. Basic model outputs
            chosen_outputs = model(**batch["chosen"])
            rejected_outputs = model(**batch["rejected"])
            
            # Reference model outputs
            ref_chosen_outputs = ref_model(**batch["chosen"])
            
            # 2. KL Divergence
            kl_div = compute_kl_divergence(
                chosen_outputs.logits,
                ref_chosen_outputs.logits
            )
            metrics['kl_divergence'] += kl_div.item()
            
            # 3. Reward
            if reward_model is not None:
                # Use actual reward model if available
                reward = reward_model(batch["prompt"], chosen_outputs.logits)
            else:
                # Fallback to simple token overlap reward
                reward = compute_reward(
                    chosen_outputs.logits,
                    batch["chosen"]["input_ids"]
                )
            metrics['reward'] += reward.mean().item()
            
            # 4. Preference Accuracy
            # Check if model assigns higher likelihood to chosen vs rejected completions
            chosen_likelihood = chosen_outputs.logits.mean(dim=-1)
            rejected_likelihood = rejected_outputs.logits.mean(dim=-1)
            preference_correct = (chosen_likelihood > rejected_likelihood).float()
            metrics['preference_accuracy'] += preference_correct.mean().item()
            
            # 5. Token Accuracy
            pred_tokens = torch.argmax(chosen_outputs.logits, dim=-1)
            token_matches = (pred_tokens == batch["chosen"]["input_ids"]).float()
            metrics['token_accuracy'] += token_matches.mean().item()
            
            # 6. Loss
            loss = compute_dpo_loss(batch)  # Using the same loss as training
            metrics['eval_loss'] += loss.item()
    
    # Average metrics over batches
    for key in metrics:
        metrics[key] /= n_batches
        
    # Add some derived metrics
    metrics['reward_per_kl'] = (
        metrics['reward'] / (metrics['kl_divergence'] + 1e-8)
    )  # Efficiency metric
    
    # Log detailed metrics
    logger.info("Evaluation Results:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.4f}")
    
    return metrics

def compute_dpo_loss(batch):
    """Simplified DPO loss computation for evaluation"""
    chosen_logits = batch["chosen_outputs"].logits
    rejected_logits = batch["rejected_outputs"].logits
    
    # Compute log probabilities
    chosen_logprobs = F.log_softmax(chosen_logits, dim=-1)
    rejected_logprobs = F.log_softmax(rejected_logits, dim=-1)
    
    # Basic preference loss
    loss = -F.logsigmoid(chosen_logprobs.mean() - rejected_logprobs.mean())
    return loss

class TrainingLoop:
    """Wrapper class for training with evaluation"""
    def __init__(self, model, ref_model, config: TrainingConfig, reward_model=None):
        self.model = model
        self.ref_model = ref_model
        self.config = config
        self.reward_model = reward_model
        
    def train_epoch(self, train_dataloader, eval_dataloader):
        for step, batch in enumerate(train_dataloader):
            # Training step [previous implementation]
            
            # Evaluation
            if step % self.config.eval_interval == 0:
                eval_metrics = evaluate(
                    self.model,
                    self.ref_model,
                    eval_dataloader,
                    self.reward_model
                )
                
                # Log metrics to wandb
                wandb.log({
                    "step": step,
                    **eval_metrics
                })
                
                # Early stopping check could be added here
                if eval_metrics['reward_per_kl'] < self.config.early_stopping_threshold:
                    logger.info("Early stopping triggered")
                    break

def main():
    # Initialize config
    config = TrainingConfig()
    
    # Initialize models
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(config.device)
    ref_model = GPT2LMHeadModel.from_pretrained("gpt2").to(config.device)
    
    # Initialize datasets
    # Note: You'll need to implement data loading logic
    train_dataset = PreferenceDataset([], [], [], tokenizer)
    eval_dataset = PreferenceDataset([], [], [], tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size)
    
    # Initialize trainers
    dpo_trainer = DPOTrainer(model, ref_model, config)
    emdpo_trainer = EMDPOTrainer(model, ref_model, config)
    rlhf_trainer = RLHFTrainer(model, None, config)  # Need to implement reward model
    
    # Train and evaluate each method
    for trainer in [dpo_trainer, emdpo_trainer, rlhf_trainer]:
        wandb.init(project="llm-behavior-comparison")
        train_and_evaluate(trainer, train_dataloader, eval_dataloader, config)
        wandb.finish()

if __name__ == "__main__":
    main()