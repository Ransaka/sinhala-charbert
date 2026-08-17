"""
Mode A: Bounded Word-Level Denoising Corrector using CharBERT character representations and NLM dictionary.

Reconstruction Strategy:
    Subword tokens are grouped by their originating word using `subword_offsets` from the alignment engine.
    The NLM prediction is taken from the **first subword token** of each word group (which captures the
    word-start character boundary representation). The full word is then replaced atomically, avoiding
    the duplication bug where per-subword replacement repeats the full predicted word for each piece.

    An edit distance ratio guard prevents semantic substitutions (e.g., replacing a valid word with a
    contextually plausible but character-dissimilar word). Only corrections with edit_distance / max_len
    below `max_edit_ratio` are applied.
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
    """Represents a predicted replacement candidate for a word span."""
    word_index: int
    original_word: str
    predicted_word: str
    confidence: float
    is_modified: bool
    token_indices: List[int]


class BoundedWordCorrector:
    """
    Mode A Typo Corrector: Bounded Word-Level Denoising.
    Extracts character channel representations from Sinhala-CharBERT,
    projects them through the Noisy Language Modeling (NLM) head,
    and maps candidate distributions back into valid Sinhala words from the frequent lexicon.

    Crucially, subword tokens are grouped by word and the NLM prediction is aggregated
    at the word level to produce a single atomic replacement per input word.
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
        self.max_edit_ratio = 0.5

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

    @staticmethod
    def _char_edit_distance(s1: str, s2: str) -> int:
        """Computes Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return BoundedWordCorrector._char_edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                cost = 0 if c1 == c2 else 1
                curr_row.append(min(
                    curr_row[j] + 1,
                    prev_row[j + 1] + 1,
                    prev_row[j] + cost,
                ))
            prev_row = curr_row
        return prev_row[-1]

    def _is_plausible_correction(self, original: str, predicted: str) -> bool:
        """Returns True if the predicted word is character-similar enough to be a typo correction.

        Blocks semantic substitutions where the edit distance ratio exceeds max_edit_ratio.
        Examples:
            'gurvarayaa' -> 'guruvarayaa'  (ratio ~0.1)  -> PASS (minor insertion)
            'havasata'   -> 'dahaval'      (ratio ~0.6)  -> BLOCK (semantic rewrite)
        """
        dist = self._char_edit_distance(original, predicted)
        max_len = max(len(original), len(predicted), 1)
        ratio = dist / max_len
        return ratio <= self.max_edit_ratio

    def _group_tokens_by_word(
        self,
        tokens: List[str],
        offsets: List[Tuple[int, int]],
        text: str,
    ) -> List[Dict]:
        """Groups subword tokens into word-level spans using offset continuity.

        Returns a list of dicts, each with:
            - 'token_indices': indices into the tokens list
            - 'original_word': the original text substring for this word
            - 'char_start': character start in original text
            - 'char_end': character end in original text
        """
        word_groups: List[Dict] = []
        current_group: Optional[Dict] = None

        for t_idx in range(len(tokens)):
            tok_start, tok_end = offsets[t_idx] if t_idx < len(offsets) else (0, 0)

            # Skip special tokens ([CLS], [SEP], [PAD])
            if tok_start == 0 and tok_end == 0:
                continue

            token_str = tokens[t_idx]
            is_continuation = token_str.startswith("##")

            if is_continuation and current_group is not None:
                # Extend the current word group
                current_group["token_indices"].append(t_idx)
                current_group["char_end"] = tok_end
            else:
                # Start a new word group
                if current_group is not None:
                    word_groups.append(current_group)
                current_group = {
                    "token_indices": [t_idx],
                    "char_start": tok_start,
                    "char_end": tok_end,
                }

        if current_group is not None:
            word_groups.append(current_group)

        # Extract original word text from the raw string
        for group in word_groups:
            group["original_word"] = text[group["char_start"]:group["char_end"]]

        return word_groups

    def correct_sequence(
        self,
        text: str,
        confidence_threshold: Optional[float] = None,
    ) -> Tuple[str, List[WordCorrectionCandidate]]:
        """
        Corrects words in a single text sequence using bounded NLM dictionary predictions.

        Word-level reconstruction:
            1. Tokenize input into subwords, get NLM logits from the character channel.
            2. Group subwords by word using offset continuity.
            3. For each word group, take the NLM prediction from the first token (word-start boundary).
            4. If the prediction exceeds the confidence threshold and differs from the original word,
               replace the entire word atomically.
            5. Reconstruct the output by substituting corrected words into the original text.
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
                return text, []

            probs = F.softmax(char_logits[0], dim=-1)  # (seq_len, nlm_vocab_size)
            top_probs, top_ids = torch.max(probs, dim=-1)

        # Group subword tokens by word using offset-based continuity
        word_groups = self._group_tokens_by_word(
            tokens=list(aligned.tokens),
            offsets=aligned.subword_offsets,
            text=aligned.text,
        )

        candidates: List[WordCorrectionCandidate] = []
        # Build replacement list: (char_start, char_end, replacement_word)
        replacements: List[Tuple[int, int, str]] = []

        for w_idx, group in enumerate(word_groups):
            original_word = group["original_word"]
            first_token_idx = group["token_indices"][0]

            # Use the NLM prediction from the first subword token of this word
            prob = float(top_probs[first_token_idx].item())
            pred_id = int(top_ids[first_token_idx].item())
            predicted_word = self.nlm_dictionary.id_to_word(pred_id)

            # Skip special or empty predictions
            if not predicted_word or predicted_word.startswith("[") or predicted_word.startswith("<"):
                candidates.append(WordCorrectionCandidate(
                    word_index=w_idx,
                    original_word=original_word,
                    predicted_word=original_word,
                    confidence=prob,
                    is_modified=False,
                    token_indices=group["token_indices"],
                ))
                continue

            # Decide whether to apply the correction.
            # The edit distance guard prevents semantic substitutions where the model
            # predicts a contextually plausible but character-dissimilar word.
            is_mod = (
                prob >= threshold
                and predicted_word != original_word
                and len(original_word) > 1
                and not original_word.isnumeric()
                and self._is_plausible_correction(original_word, predicted_word)
            )

            if is_mod:
                replacements.append((group["char_start"], group["char_end"], predicted_word))

            candidates.append(WordCorrectionCandidate(
                word_index=w_idx,
                original_word=original_word,
                predicted_word=predicted_word,
                confidence=prob,
                is_modified=is_mod,
                token_indices=group["token_indices"],
            ))

        # Reconstruct the corrected text by applying replacements right-to-left
        # (to preserve character offsets)
        corrected_text = aligned.text
        for char_start, char_end, replacement in reversed(replacements):
            corrected_text = corrected_text[:char_start] + replacement + corrected_text[char_end:]

        return corrected_text, candidates
