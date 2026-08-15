"""
Model configuration dataclass for Sinhala-CharBERT architecture.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SinhalaCharBERTConfig:
    """Hyperparameter configuration for Sinhala-CharBERT model."""
    vocab_size: int = 32000
    char_vocab_size: int = 1500
    nlm_vocab_size: int = 32000
    hidden_size: int = 786
    char_embedding_dim: int = 128
    char_gru_hidden_size: int = 393  # 393 per direction -> 786 bidirectional
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    intermediate_size: int = 1024
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 256
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    hi_kernel_sizes: List[int] = field(default_factory=lambda: [1, 3, 5])
    pad_token_id: int = 0
    char_pad_token_id: int = 0
    backbone_model_name_or_path: Optional[str] = "Ransaka/sinhala-bert-medium-v2"
