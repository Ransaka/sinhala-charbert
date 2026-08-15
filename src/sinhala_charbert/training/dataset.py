"""
Dynamic Dual-Channel Pre-training Dataset and Batch Collator with Noise Curriculum integration.
"""

import random
from typing import Any, Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer

from sinhala_charbert.data.alignment import SequenceAlignmentEngine, AlignedSequence
from sinhala_charbert.data.char_tokenizer import SinhalaCharTokenizer
from sinhala_charbert.data.syntypo import SinhalaTypoSynthesizer
from sinhala_charbert.data.collator import DualChannelDataCollator
from sinhala_charbert.training.curriculum import NoiseCurriculumScheduler
from sinhala_charbert.training.dictionary import SinhalaNLMDictionary


class SinhalaCharBERTPretrainDataset(Dataset):
    """
    Dynamic Pre-Training Dataset for Sinhala-CharBERT.
    Injects noise on-the-fly according to the active training curriculum step,
    aligns dual-channel sequences, and constructs MLM and NLM supervision targets.
    """

    def __init__(
        self,
        texts: List[str],
        subword_tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer, Any],
        char_tokenizer: SinhalaCharTokenizer,
        alignment_engine: SequenceAlignmentEngine,
        nlm_dictionary: SinhalaNLMDictionary,
        curriculum_scheduler: Optional[NoiseCurriculumScheduler] = None,
        nlm_probability: float = 0.15,
    ):
        self.texts = texts
        self.subword_tokenizer = subword_tokenizer
        self.char_tokenizer = char_tokenizer
        self.alignment_engine = alignment_engine
        self.nlm_dictionary = nlm_dictionary
        self.curriculum = curriculum_scheduler or NoiseCurriculumScheduler()
        self.nlm_probability = nlm_probability
        self.current_step = 0

    def set_step(self, step: int) -> None:
        """Updates the current pre-training step to shift the noise curriculum."""
        self.current_step = step

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        clean_text = self.texts[idx]
        if not clean_text or not isinstance(clean_text, str):
            clean_text = "සිංහල"

        # 1. Sample noise profile from active curriculum stage
        noise_profile = self.curriculum.get_profile(self.current_step)
        synthesizer = SinhalaTypoSynthesizer(profile=noise_profile)

        # 2. Synthesize noisy text
        noise_result = synthesizer.generate_pair(clean_text)
        noisy_text = noise_result["source_noisy"]
        clean_text = noise_result["target_clean"]

        # Fallback if noisy string is empty
        if not noisy_text.strip():
            noisy_text = clean_text

        # 3. Align sequence across subword and character channels
        aligned: AlignedSequence = self.alignment_engine.align(noisy_text)
        m_len = len(aligned.input_ids)

        # 4. Generate NLM (Noisy Language Modeling) character-channel targets
        # NLM targets predict clean words for corrupted or sampled token spans
        nlm_labels = [-100] * m_len
        clean_words = clean_text.split()

        for t_idx in range(1, m_len - 1):  # Skip [CLS] and [SEP]
            if random.random() < self.nlm_probability and clean_words:
                target_word = random.choice(clean_words).strip(".,;:!?\"'()[]{}«»-")
                if target_word and target_word in self.nlm_dictionary.vocab_map:
                    nlm_labels[t_idx] = self.nlm_dictionary.word_to_id(target_word)

        return {
            "aligned_seq": aligned,
            "nlm_labels": nlm_labels,
        }


class PretrainDualChannelCollator:
    """
    Collation wrapper extending DualChannelDataCollator with NLM target tensor batching.
    """

    def __init__(
        self,
        subword_pad_token_id: int = 0,
        subword_mask_token_id: int = 103,
        subword_vocab_size: int = 32000,
        char_pad_token_id: int = 0,
        mlm_probability: float = 0.15,
    ):
        self.base_collator = DualChannelDataCollator(
            subword_pad_token_id=subword_pad_token_id,
            subword_mask_token_id=subword_mask_token_id,
            subword_vocab_size=subword_vocab_size,
            char_pad_token_id=char_pad_token_id,
            mlm_probability=mlm_probability,
        )

    def __call__(self, batch_items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        aligned_seqs = [item["aligned_seq"] for item in batch_items]
        batch_tensors = self.base_collator(aligned_seqs)

        batch_size = len(batch_items)
        max_m_len = batch_tensors["input_ids"].shape[1]

        nlm_labels_tensor = torch.full((batch_size, max_m_len), -100, dtype=torch.long)
        for b_idx, item in enumerate(batch_items):
            labels = item.get("nlm_labels", [])
            length = min(len(labels), max_m_len)
            if length > 0:
                nlm_labels_tensor[b_idx, :length] = torch.tensor(labels[:length], dtype=torch.long)

        batch_tensors["nlm_labels"] = nlm_labels_tensor
        return batch_tensors
