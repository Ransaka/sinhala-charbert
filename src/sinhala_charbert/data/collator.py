"""
Dynamic batch collator for dual-channel (Subword + Phonological Akshara) tensors with MLM masking.
"""

import random
from typing import Any, Dict, List, Optional
import torch

from sinhala_charbert.data.alignment import AlignedSequence


class DualChannelDataCollator:
    """
    Collates aligned sequences into padded PyTorch batch tensors for dual-channel training.
    Applies Masked Language Modeling (MLM) masking on the token channel.
    """

    def __init__(
        self,
        subword_pad_token_id: int = 0,
        subword_mask_token_id: int = 103,
        subword_vocab_size: int = 30000,
        char_pad_token_id: int = 0,
        mlm_probability: float = 0.15,
    ):
        self.subword_pad_token_id = subword_pad_token_id
        self.subword_mask_token_id = subword_mask_token_id
        self.subword_vocab_size = subword_vocab_size
        self.char_pad_token_id = char_pad_token_id
        self.mlm_probability = mlm_probability

    def __call__(self, batch: List[AlignedSequence]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        if batch_size == 0:
            raise ValueError("Empty batch provided to DualChannelDataCollator.")

        # Determine max lengths in current batch
        max_subword_len = max(len(seq.input_ids) for seq in batch)
        max_char_len = max(len(seq.char_input_ids) for seq in batch)

        # Allocate padded tensors
        input_ids = torch.full((batch_size, max_subword_len), self.subword_pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_subword_len), dtype=torch.long)
        
        char_input_ids = torch.full((batch_size, max_char_len), self.char_pad_token_id, dtype=torch.long)
        char_attention_mask = torch.zeros((batch_size, max_char_len), dtype=torch.long)
        
        start_char_idx = torch.zeros((batch_size, max_subword_len), dtype=torch.long)
        end_char_idx = torch.zeros((batch_size, max_subword_len), dtype=torch.long)

        mlm_labels = torch.full((batch_size, max_subword_len), -100, dtype=torch.long)

        for b_idx, seq in enumerate(batch):
            m_len = len(seq.input_ids)
            n_len = len(seq.char_input_ids)

            # Copy token channel values
            input_ids[b_idx, :m_len] = torch.tensor(seq.input_ids, dtype=torch.long)
            attention_mask[b_idx, :m_len] = torch.tensor(seq.attention_mask, dtype=torch.long)

            # Copy character channel values
            char_input_ids[b_idx, :n_len] = torch.tensor(seq.char_input_ids, dtype=torch.long)
            char_attention_mask[b_idx, :n_len] = torch.tensor(seq.char_attention_mask, dtype=torch.long)

            # Copy alignment indices
            start_char_idx[b_idx, :m_len] = torch.tensor(seq.start_char_idx, dtype=torch.long)
            end_char_idx[b_idx, :m_len] = torch.tensor(seq.end_char_idx, dtype=torch.long)

            # Apply MLM Masking to token channel (ignoring special tokens at 0 and m_len-1)
            for t_idx in range(1, m_len - 1):
                if random.random() < self.mlm_probability:
                    mlm_labels[b_idx, t_idx] = seq.input_ids[t_idx]
                    rand_action = random.random()
                    if rand_action < 0.80:
                        input_ids[b_idx, t_idx] = self.subword_mask_token_id
                    elif rand_action < 0.90:
                        input_ids[b_idx, t_idx] = random.randint(104, min(self.subword_vocab_size - 1, 29999))
                    # Remaining 10% retains original token

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "char_input_ids": char_input_ids,
            "char_attention_mask": char_attention_mask,
            "start_char_idx": start_char_idx,
            "end_char_idx": end_char_idx,
            "mlm_labels": mlm_labels,
        }
