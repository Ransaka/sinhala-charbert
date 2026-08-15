"""
Mode A: Bounded Word-Level Denoising Corrector using CharBERT character representations and NLM dictionary.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from sinhala_charbert.data.alignment import SequenceAlignmentEngine, AlignedSequence
from sinhala_charbert.data.char_tokenizer import SinhalaCharTokenizer
from sinhala_charbert.models.modeling_charbert import SinhalaCharBERTForPreTraining, SinhalaCharBERTModel
from sinhala_charbert.training.dictionary import SinhalaNLMDictionary


@dataclass
class WordCorrectionCandidate:
    """Represents a predicted replacement candidate for a token span."""
    token_index: int
    original_token: str
    predicted_word: str
    confidence: float
    is_modified: bool


class BoundedWordCorrector:
    """
    Mode A Typo Corrector: Bounded Word-Level Denoising.
    Extracts character channel representations from Sinhala-CharBERT,
    projects them through the Noisy Language Modeling (NLM) head,
    and maps candidate distributions back into valid Sinhala words from the frequent lexicon.
    """

    def __init__(
        self,
        model: Union[SinhalaCharBERTForPreTraining, SinhalaCharBERTModel],
        subword_tokenizer: Any,
        char_tokenizer: SinhalaCharTokenizer,
        alignment_engine: SequenceAlignmentEngine,
        nlm_dictionary: SinhalaNLMDictionary,
        device: Optional[torch.device] = None,
        confidence_threshold: float = 0.40,
    ):
        self.model = model
        self.subword_tokenizer = subword_tokenizer
        self.char_tokenizer = char_tokenizer
        self.alignment_engine = alignment_engine
        self.nlm_dictionary = nlm_dictionary
        self.confidence_threshold = confidence_threshold

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

    def correct_sequence(
        self,
        text: str,
        confidence_threshold: Optional[float] = None,
    ) -> Tuple[str, List[WordCorrectionCandidate]]:
        """
        Corrects words in a single text sequence using bounded NLM dictionary predictions.
        """
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        aligned: AlignedSequence = self.alignment_engine.align(text)

        m_len = len(aligned.input_ids)
        if m_len <= 2:  # Only [CLS] and [SEP]
            return text, []

        input_ids = torch.tensor([aligned.input_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([aligned.attention_mask], dtype=torch.long, device=self.device)
        char_input_ids = torch.tensor([aligned.char_input_ids], dtype=torch.long, device=self.device)
        char_attention_mask = torch.tensor([aligned.char_attention_mask], dtype=torch.long, device=self.device)
        start_char_idx = torch.tensor([aligned.start_char_idx], dtype=torch.long, device=self.device)
        end_char_idx = torch.tensor([aligned.end_char_idx], dtype=torch.long, device=self.device)

        with torch.no_grad():
            if isinstance(self.model, SinhalaCharBERTForPreTraining):
                outputs = self.model(
                    input_ids=input_ids,
                    char_input_ids=char_input_ids,
                    start_char_idx=start_char_idx,
                    end_char_idx=end_char_idx,
                    attention_mask=attention_mask,
                    char_attention_mask=char_attention_mask,
                )
                char_logits = outputs.char_logits  # (1, seq_len, nlm_vocab_size)
            else:
                backbone_out = self.model(
                    input_ids=input_ids,
                    char_input_ids=char_input_ids,
                    start_char_idx=start_char_idx,
                    end_char_idx=end_char_idx,
                    attention_mask=attention_mask,
                    char_attention_mask=char_attention_mask,
                )
                # If using backbone directly without pre-training wrapper
                return text, []

            probs = F.softmax(char_logits[0], dim=-1)  # (seq_len, nlm_vocab_size)
            top_probs, top_ids = torch.max(probs, dim=-1)

        candidates: List[WordCorrectionCandidate] = []
        tokens = list(aligned.tokens)
        modified_tokens = list(tokens)

        for t_idx in range(1, m_len - 1):
            tok_str = tokens[t_idx]
            clean_tok_str = tok_str.replace("##", "")
            
            prob = float(top_probs[t_idx].item())
            pred_id = int(top_ids[t_idx].item())
            predicted_word = self.nlm_dictionary.id_to_word(pred_id)

            # Skip special tokens or empty predictions
            if not predicted_word or predicted_word.startswith("[") or predicted_word.startswith("<"):
                candidates.append(
                    WordCorrectionCandidate(
                        token_index=t_idx,
                        original_token=tok_str,
                        predicted_word=clean_tok_str,
                        confidence=prob,
                        is_modified=False,
                    )
                )
                continue

            # Check if prediction exceeds confidence threshold and improves token
            is_mod = (
                prob >= threshold
                and predicted_word != clean_tok_str
                and len(clean_tok_str) > 1
                and not clean_tok_str.isnumeric()
            )

            if is_mod:
                modified_tokens[t_idx] = predicted_word if not tok_str.startswith("##") else f"##{predicted_word}"

            candidates.append(
                WordCorrectionCandidate(
                    token_index=t_idx,
                    original_token=tok_str,
                    predicted_word=predicted_word,
                    confidence=prob,
                    is_modified=is_mod,
                )
            )

        # Reconstruct sentence from subwords
        corrected_text = self.subword_tokenizer.convert_tokens_to_string(modified_tokens[1:-1])
        return corrected_text, candidates
