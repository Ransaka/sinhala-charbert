"""
Standard Comparison Baselines for Sinhala Typo Correction Benchmarking.
Includes:
1. Identity Baseline (No-op)
2. Rule-Based Heuristic & Dictionary Corrector
3. Standard BERT Masked LM Corrector
"""

import re
from typing import Any, Dict, List, Optional, Set, Union
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from sinlib.utils.preprocessing import normalize_sinhala
from sinhala_charbert.training.dictionary import SinhalaNLMDictionary


class IdentityBaseline:
    """
    No-op baseline that outputs the source text unmodified.
    Establishes the upper-bound error rate of the uncorrected dataset.
    """

    def correct(self, text: str) -> str:
        return normalize_sinhala(text)

    def correct_batch(self, texts: List[str]) -> List[str]:
        return [self.correct(t) for t in texts]


class RuleBasedSinhalaCorrector:
    """
    Classical rule-based and lexicon-assisted Sinhala spellchecker.
    Applies heuristic confusion rules (Murdhaja/Dantaja, ZWJ repair, dialectal shifts)
    and verifies candidate words against the Sinhala word dictionary.
    """

    # Classical confusion pairs
    ORTHOGRAPHIC_PAIRS = [
        ("න", "ණ"), ("ණ", "න"),
        ("ල", "ළ"), ("ළ", "ල"),
        ("ස", "ශ"), ("ශ", "ස"),
        ("ස", "ෂ"), ("ෂ", "ස"),
        ("ක", "ඛ"), ("ඛ", "ක"),
        ("ග", "ඝ"), ("ඝ", "ග"),
        ("ච", "ඡ"), ("ඡ", "ච"),
        ("ට", "ඨ"), ("ඨ", "ට"),
        ("ඩ", "ඪ"), ("ඪ", "ඩ"),
        ("ත", "ථ"), ("ථ", "ත"),
        ("ද", "ධ"), ("ධ", "ද"),
        ("ප", "ඵ"), ("ඵ", "ප"),
        ("බ", "භ"), ("භ", "බ"),
    ]

    # Dialectal & Morphology rules
    DIALECT_RULES = [
        (r"(\w+)න්ඩ\b", r"\1න්න"),
        (r"(\w+)න්ට\b", r"\1න්නට"),
        (r"(\w+)තමා\b", r"\1තමයි"),
        (r"(\w+)තමෙයි\b", r"\1තමයි"),
    ]

    # ZWJ Ligature Repairs
    LIGATURE_REPAIRS = [
        ("ක්ර", "ක්‍ර"),
        ("ග්ර", "ග්‍ර"),
        ("ත්ර", "ත්‍ර"),
        ("ද්ර", "ද්‍ර"),
        ("ප්ර", "ප්‍ර"),
        ("බ්ර", "බ්‍ර"),
        ("ව්ර", "ව්‍ර"),
        ("ශ්ර", "ශ්‍ර"),
        ("ක්ය", "ක්‍ය"),
        ("ත්‍ය", "ත්‍ය"),
        ("ද්‍ය", "ද්‍ය"),
    ]

    def __init__(self, dictionary: Optional[SinhalaNLMDictionary] = None):
        self.dictionary = dictionary or SinhalaNLMDictionary()

    def correct_word(self, word: str) -> str:
        """Corrects a single word using rules and dictionary lookup."""
        cleaned = word.strip(".,;:!?\"'()[]{}«»-")
        if not cleaned:
            return word

        # 1. If already in dictionary, keep it
        if cleaned in self.dictionary.vocab_map:
            return word

        # 2. Try dialectal shifts
        for pat, repl in self.DIALECT_RULES:
            cand = re.sub(pat, repl, cleaned)
            if cand != cleaned and cand in self.dictionary.vocab_map:
                return word.replace(cleaned, cand)

        # 3. Try Murdhaja / Dantaja and Sibilant substitutions (one occurrence at a time)
        for orig_c, swap_c in self.ORTHOGRAPHIC_PAIRS:
            if orig_c in cleaned:
                idx = 0
                while True:
                    found = cleaned.find(orig_c, idx)
                    if found == -1:
                        break
                    cand = cleaned[:found] + swap_c + cleaned[found + len(orig_c):]
                    if cand in self.dictionary.vocab_map:
                        return word.replace(cleaned, cand)
                    idx = found + 1

        return word

    def correct(self, text: str) -> str:
        norm_text = normalize_sinhala(text)

        # 1. Apply global ligature repairs
        for broken, fixed in self.LIGATURE_REPAIRS:
            norm_text = norm_text.replace(broken, fixed)

        # 2. Word-by-word dictionary check
        words = norm_text.split()
        corrected_words = [self.correct_word(w) for w in words]
        return " ".join(corrected_words)

    def correct_batch(self, texts: List[str]) -> List[str]:
        return [self.correct(t) for t in texts]


class StandardBERTMLMCorrector:
    """
    Standard BERT Masked Language Modeling (MLM) typo corrector.
    Uses pre-trained BERT embeddings and encoder without character channels.
    Masks subword tokens and predicts replacements via MLM cross-entropy logits.
    """

    def __init__(
        self,
        model_name_or_path: str = "Ransaka/sinhala-bert-medium-v2",
        device: Optional[torch.device] = None,
        confidence_threshold: float = 0.50,
    ):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        self.confidence_threshold = confidence_threshold

    def correct(self, text: str) -> str:
        norm_text = normalize_sinhala(text)
        encoding = self.tokenizer(norm_text, return_tensors="pt", truncation=True, max_length=256)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        seq_len = input_ids.size(1)
        if seq_len <= 2:
            return norm_text

        orig_ids = input_ids.clone()
        modified_ids = input_ids.clone()

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (1, seq_len, vocab_size)
            probs = torch.softmax(logits, dim=-1)

        for t_idx in range(1, seq_len - 1):
            curr_id = orig_ids[0, t_idx].item()
            curr_prob = probs[0, t_idx, curr_id].item()

            # If current token has low probability, check top candidate
            if curr_prob < self.confidence_threshold:
                top_prob, top_id = torch.max(probs[0, t_idx], dim=-1)
                if top_prob.item() > self.confidence_threshold and top_id.item() != curr_id:
                    modified_ids[0, t_idx] = top_id

        return self.tokenizer.decode(modified_ids[0], skip_special_tokens=True)

    def correct_batch(self, texts: List[str]) -> List[str]:
        return [self.correct(t) for t in texts]
