"""
Training configuration dataclass for Sinhala-CharBERT.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Hyperparameter configuration for training Sinhala-CharBERT."""
    output_dir: str = "checkpoints/sinhala_charbert"
    resume_from_checkpoint: Optional[str] = None
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 10000
    max_steps: int = 320000
    batch_size: int = 32
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    mlm_probability: float = 0.10
    nlm_probability: float = 0.15
    mlm_loss_weight: float = 1.0
    nlm_loss_weight: float = 1.0
    save_steps: int = 10000
    eval_steps: int = 5000
    logging_steps: int = 500
    seed: int = 42
    dataloader_num_workers: int = 2
    ddp_find_unused_parameters: bool = True
