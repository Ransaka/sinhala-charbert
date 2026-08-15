"""
Unified High-Level Inference Pipeline and Typo Corrector API for Sinhala-CharBERT.
"""

from dataclasses import dataclass, field
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from sinlib.utils.preprocessing import normalize_sinhala
from sinhala_charbert.config.model_config import SinhalaCharBERTConfig
from sinhala_charbert.data.alignment import SequenceAlignmentEngine, AlignedSequence
from sinhala_charbert.data.char_tokenizer import SinhalaCharTokenizer
from sinhala_charbert.models.modeling_charbert import SinhalaCharBERTForPreTraining, SinhalaCharBERTModel
from sinhala_charbert.models.word_corrector import BoundedWordCorrector
from sinhala_charbert.models.seq2seq_decoder import SinhalaCharBERTSeq2SeqModel
from sinhala_charbert.training.dictionary import SinhalaNLMDictionary


@dataclass
class EditOp:
    """Represents a discrete correction edit operation."""
    op_type: str  # 'replace', 'insert', 'delete', 'equal'
    original: str
    corrected: str
    start_pos: int
    end_pos: int
    category: str = "general"


@dataclass
class CorrectionResult:
    """Structured output from SinhalaCharBERTCorrector."""
    text: str
    original_text: str
    edits: List[EditOp] = field(default_factory=list)
    confidence: float = 1.0
    mode: str = "seq2seq"

    def summary(self) -> str:
        """Returns a formatted human-readable summary of applied edits."""
        if not self.edits or all(e.op_type == "equal" for e in self.edits):
            return "No typos detected. Text is clean."

        lines = [f"Original : {self.original_text}", f"Corrected: {self.text}", "Edits:"]
        for e in self.edits:
            if e.op_type != "equal":
                lines.append(f"  - [{e.op_type.upper()}] '{e.original}' -> '{e.corrected}' ({e.category})")
        return "\n".join(lines)


class SinhalaCharBERTCorrector:
    """
    High-level end-to-end Typo Correction Pipeline for Sinhala.
    Supports Mode A (Bounded Word-Level Denoising) and Mode B (Open-Vocabulary Seq2Seq Decoder).
    """

    def __init__(
        self,
        model: Union[SinhalaCharBERTForPreTraining, SinhalaCharBERTSeq2SeqModel],
        subword_tokenizer: Any,
        char_tokenizer: SinhalaCharTokenizer,
        alignment_engine: SequenceAlignmentEngine,
        nlm_dictionary: Optional[SinhalaNLMDictionary] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.subword_tokenizer = subword_tokenizer
        self.char_tokenizer = char_tokenizer
        self.alignment_engine = alignment_engine
        self.nlm_dictionary = nlm_dictionary or SinhalaNLMDictionary()

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

        # Initialize Mode A word corrector helper
        self.word_corrector = BoundedWordCorrector(
            model=self.model,
            subword_tokenizer=self.subword_tokenizer,
            char_tokenizer=self.char_tokenizer,
            alignment_engine=self.alignment_engine,
            nlm_dictionary=self.nlm_dictionary,
            device=self.device,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path_or_name: Union[str, Path],
        subword_tokenizer_name: str = "Ransaka/sinhala-bert-medium-v2",
        device: Optional[torch.device] = None,
        is_seq2seq: bool = False,
    ) -> "SinhalaCharBERTCorrector":
        """
        Loads pre-trained or fine-tuned Sinhala-CharBERT corrector from checkpoint.
        """
        subword_tok = AutoTokenizer.from_pretrained(subword_tokenizer_name)
        char_tok = SinhalaCharTokenizer()
        nlm_dict = SinhalaNLMDictionary()

        path = Path(model_path_or_name)
        if (path / "nlm_dict.json").exists():
            nlm_dict.load(path / "nlm_dict.json")
        if (path / "char_vocab.json").exists():
            char_tok.load(path / "char_vocab.json")

        align_engine = SequenceAlignmentEngine(
            subword_tokenizer=subword_tok,
            char_tokenizer=char_tok,
        )

        config = SinhalaCharBERTConfig(
            vocab_size=subword_tok.vocab_size,
            char_vocab_size=char_tok.vocab_size or 1500,
            nlm_vocab_size=nlm_dict.vocab_size or 32000,
        )

        if is_seq2seq:
            model = SinhalaCharBERTSeq2SeqModel(config)
        else:
            model = SinhalaCharBERTForPreTraining(config)

        weight_file = path / "pytorch_model.bin"
        if weight_file.exists():
            state_dict = torch.load(weight_file, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)

        return cls(
            model=model,
            subword_tokenizer=subword_tok,
            char_tokenizer=char_tok,
            alignment_engine=align_engine,
            nlm_dictionary=nlm_dict,
            device=device,
        )

    def _categorize_edit(self, orig: str, corr: str) -> str:
        """Classifies edit operation into linguistic / orthographic / keystroke categories."""
        if "\u200d" in corr and "\u200d" not in orig:
            return "ligature_repair"
        if ("ණ" in corr and "න" in orig) or ("න" in corr and "ණ" in orig):
            return "murdhaja_dantaja"
        if ("ළ" in corr and "ල" in orig) or ("ල" in corr and "ළ" in orig):
            return "retroflex_dental"
        if ("ශ" in corr or "ෂ" in corr) and "ස" in orig:
            return "sibilant_confusion"
        if ("න්ඩ" in orig and "න්න" in corr) or ("න්ට" in orig and "න්නට" in corr) or (orig == "ඩ" and corr == "න"):
            return "dialectal_morphology"
        return "orthographic"

    def _compute_edits(self, original_text: str, corrected_text: str) -> List[EditOp]:
        """Calculates exact edit spans between original and corrected strings using SequenceMatcher."""
        matcher = difflib.SequenceMatcher(None, original_text, corrected_text)
        edits: List[EditOp] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            orig_slice = original_text[i1:i2]
            corr_slice = corrected_text[j1:j2]
            cat = self._categorize_edit(orig_slice, corr_slice) if tag != "equal" else "none"
            edits.append(
                EditOp(
                    op_type=tag,
                    original=orig_slice,
                    corrected=corr_slice,
                    start_pos=i1,
                    end_pos=i2,
                    category=cat,
                )
            )
        return edits

    def correct(
        self,
        text: str,
        mode: str = "seq2seq",
        confidence_threshold: float = 0.40,
        max_length: int = 128,
    ) -> CorrectionResult:
        """
        Corrects typos in input Sinhala text.
        Args:
            text: Input Sinhala sentence (with potential typos).
            mode: 'seq2seq' (Mode B: sound-by-sound open-vocab) or 'word_denoise' (Mode A: fast dictionary).
            confidence_threshold: Minimum probability required to apply a correction in Mode A.
            max_length: Maximum target generation length for Mode B.
        """
        norm_text = normalize_sinhala(text.strip())
        if not norm_text:
            return CorrectionResult(text=text, original_text=text, edits=[], confidence=1.0, mode=mode)

        if mode == "word_denoise" or isinstance(self.model, SinhalaCharBERTForPreTraining):
            corrected_text, candidates = self.word_corrector.correct_sequence(
                norm_text, confidence_threshold=confidence_threshold
            )
            # Estimate mean confidence over modified tokens
            mod_confs = [c.confidence for c in candidates if c.is_modified]
            overall_conf = float(sum(mod_confs) / len(mod_confs)) if mod_confs else 1.0

        elif mode == "seq2seq" and isinstance(self.model, SinhalaCharBERTSeq2SeqModel):
            aligned = self.alignment_engine.align(norm_text)
            input_ids = torch.tensor([aligned.input_ids], dtype=torch.long, device=self.device)
            attention_mask = torch.tensor([aligned.attention_mask], dtype=torch.long, device=self.device)
            char_input_ids = torch.tensor([aligned.char_input_ids], dtype=torch.long, device=self.device)
            char_attention_mask = torch.tensor([aligned.char_attention_mask], dtype=torch.long, device=self.device)
            start_char_idx = torch.tensor([aligned.start_char_idx], dtype=torch.long, device=self.device)
            end_char_idx = torch.tensor([aligned.end_char_idx], dtype=torch.long, device=self.device)

            generated_ids = self.model.generate(
                input_ids=input_ids,
                char_input_ids=char_input_ids,
                start_char_idx=start_char_idx,
                end_char_idx=end_char_idx,
                attention_mask=attention_mask,
                char_attention_mask=char_attention_mask,
                max_length=max_length,
                bos_token_id=self.char_tokenizer.bos_token_id,
                eos_token_id=self.char_tokenizer.eos_token_id,
            )

            gen_list = generated_ids[0].tolist()
            corrected_text = self.char_tokenizer.decode(gen_list, skip_special_tokens=True).strip()
            overall_conf = 0.95

        else:
            # Fallback
            corrected_text = norm_text
            overall_conf = 1.0

        edits = self._compute_edits(text, corrected_text)
        return CorrectionResult(
            text=corrected_text,
            original_text=text,
            edits=edits,
            confidence=overall_conf,
            mode=mode,
        )

    def correct_batch(
        self,
        texts: List[str],
        mode: str = "seq2seq",
        confidence_threshold: float = 0.40,
    ) -> List[CorrectionResult]:
        """Processes a batch of sentences sequentially."""
        return [
            self.correct(t, mode=mode, confidence_threshold=confidence_threshold)
            for t in texts
        ]
