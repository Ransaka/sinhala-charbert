"""
Interactive CLI for Sinhala-CharBERT Typo Detection & Correction.
Supports Mode A (Word-Level Denoising) and Mode B (Open-Vocabulary Seq2Seq).
"""

import argparse
from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer

from sinhala_charbert.config.model_config import SinhalaCharBERTConfig
from sinhala_charbert.data.alignment import SequenceAlignmentEngine
from sinhala_charbert.data.char_tokenizer import SinhalaCharTokenizer
from sinhala_charbert.models.modeling_charbert import SinhalaCharBERTForPreTraining
from sinhala_charbert.models.seq2seq_decoder import SinhalaCharBERTSeq2SeqModel
from sinhala_charbert.models.pipeline import SinhalaCharBERTCorrector
from sinhala_charbert.training.dictionary import SinhalaNLMDictionary


def main():
    parser = argparse.ArgumentParser(description="Sinhala-CharBERT Typo Correction CLI")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/sinhala_charbert/final_model",
        help="Path to pre-trained/fine-tuned model checkpoint directory.",
    )
    parser.add_argument(
        "--subword_tokenizer",
        type=str,
        default="Ransaka/sinhala-bert-medium-v2",
        help="Subword tokenizer name or local path.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single text string to correct. If not provided, enters interactive mode.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["word_denoise", "seq2seq"],
        default="word_denoise",
        help="Correction mode: 'word_denoise' (Mode A) or 'seq2seq' (Mode B).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.40,
        help="Confidence threshold for applying word replacements in Mode A.",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("Sinhala-CharBERT Downstream Typo Correction Pipeline")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"Loading resources on device: {device}...")

    # Load Tokenizers and Dict
    subword_tokenizer = AutoTokenizer.from_pretrained(args.subword_tokenizer)
    char_tokenizer = SinhalaCharTokenizer()
    nlm_dict = SinhalaNLMDictionary()

    ckpt_path = Path(args.checkpoint_path)
    if (ckpt_path / "nlm_dict.json").exists():
        nlm_dict = SinhalaNLMDictionary.load(ckpt_path / "nlm_dict.json")
    if (ckpt_path / "char_vocab.json").exists():
        char_tokenizer = SinhalaCharTokenizer.load(ckpt_path / "char_vocab.json")

    char_vocab_size = char_tokenizer.vocab_size
    nlm_vocab_size = nlm_dict.vocab_size

    weight_file = ckpt_path / "pytorch_model.bin"
    state_dict = None
    if weight_file.exists():
        print(f"Loading weights from '{weight_file}'...")
        state_dict = torch.load(weight_file, map_location="cpu")
        if "charbert.char_embeddings.char_embeddings.weight" in state_dict:
            char_vocab_size = state_dict["charbert.char_embeddings.char_embeddings.weight"].shape[0]
        if "nlm_head.decoder.weight" in state_dict:
            nlm_vocab_size = state_dict["nlm_head.decoder.weight"].shape[0]
        elif "target_char_embeddings.weight" in state_dict:
            char_vocab_size = state_dict["target_char_embeddings.weight"].shape[0]
    else:
        print(f"Warning: Checkpoint '{weight_file}' not found. Using initialized model.")

    align_engine = SequenceAlignmentEngine(
        subword_tokenizer=subword_tokenizer,
        char_tokenizer=char_tokenizer,
    )

    is_seq2seq = (args.mode == "seq2seq")
    config = SinhalaCharBERTConfig(
        vocab_size=subword_tokenizer.vocab_size,
        char_vocab_size=char_vocab_size,
        nlm_vocab_size=nlm_vocab_size,
        max_position_embeddings=256,
    )

    if is_seq2seq:
        model = SinhalaCharBERTSeq2SeqModel(config)
    else:
        model = SinhalaCharBERTForPreTraining(config)

    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)

    corrector = SinhalaCharBERTCorrector(
        model=model,
        subword_tokenizer=subword_tokenizer,
        char_tokenizer=char_tokenizer,
        alignment_engine=align_engine,
        nlm_dictionary=nlm_dict,
        device=device,
    )

    print("Model successfully loaded!\n")

    def run_correction(input_str: str):
        res = corrector.correct(input_str, mode=args.mode, confidence_threshold=args.confidence)
        print("-" * 65)
        print(f"Input Text  : {res.original_text}")
        print(f"Clean Output: {res.text}")
        print(f"Mode        : {res.mode} | Confidence: {res.confidence:.2%}")
        if res.edits and any(e.op_type != "equal" for e in res.edits):
            print("Detected Edits:")
            for edit in res.edits:
                if edit.op_type != "equal":
                    print(f"  * [{edit.op_type.upper()}] '{edit.original}' -> '{edit.corrected}' ({edit.category})")
        else:
            print("No typos detected.")
        print("-" * 65)

    if args.text:
        run_correction(args.text)
    else:
        print("Entering interactive mode. Type your Sinhala text and press Enter (or 'q' to quit):")
        while True:
            try:
                user_input = input("\nSinhala input > ").strip()
                if user_input.lower() in ("q", "quit", "exit"):
                    print("Exiting.")
                    break
                if not user_input:
                    continue
                run_correction(user_input)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break


if __name__ == "__main__":
    main()
