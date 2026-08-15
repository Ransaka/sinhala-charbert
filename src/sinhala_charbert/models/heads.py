"""
Prediction and Pre-Training Output Heads for Sinhala-CharBERT.
Implements Token MLM Head (Subword Masked Language Modeling) and Char NLM Head (Noisy Language Modeling Denoising).
"""

import torch
import torch.nn as nn
from transformers.activations import get_activation

from sinhala_charbert.config.model_config import SinhalaCharBERTConfig


class SinhalaTokenMLMHead(nn.Module):
    """
    Subword Token Channel Masked Language Modeling (MLM) Prediction Head.
    Computes vocabulary logits over the subword tokenizer vocabulary.
    """

    def __init__(self, config: SinhalaCharBERTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = get_activation(config.hidden_act)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : torch.Tensor
            Token channel hidden states of shape (batch_size, seq_len, hidden_size).

        Returns
        -------
        torch.Tensor
            Logits over subword vocabulary of shape (batch_size, seq_len, vocab_size).
        """
        x = self.dense(hidden_states)
        x = self.act(x)
        x = self.layer_norm(x)
        logits = self.decoder(x)
        return logits


class SinhalaCharNLMHead(nn.Module):
    """
    Character Channel Noisy Language Modeling (NLM) Prediction Head.
    Predicts correct canonical words from corrupted token-aligned character representations.
    """

    def __init__(self, config: SinhalaCharBERTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = get_activation(config.hidden_act)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.nlm_vocab_size, bias=True)

    def forward(self, char_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        char_hidden_states : torch.Tensor
            Character channel hidden states of shape (batch_size, seq_len, hidden_size).

        Returns
        -------
        torch.Tensor
            Logits over NLM candidate word vocabulary of shape (batch_size, seq_len, nlm_vocab_size).
        """
        x = self.dense(char_hidden_states)
        x = self.act(x)
        x = self.layer_norm(x)
        logits = self.decoder(x)
        return logits
