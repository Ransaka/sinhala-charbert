"""
Automated Benchmarking and Evaluation CLI for Sinhala-CharBERT and Baselines.
"""

import argparse
import json
from pathlib import Path
import time
from datasets import load_dataset
import torch
from transformers import AutoTokenizer

from sinhala_charbert.config.model_config import SinhalaCharBERTConfig
from sinhala_charbert.data.alignment import SequenceAlignmentEngine
from sinhala_charbert.data.char_tokenizer import SinhalaCharTokenizer
from sinhala_charbert.data.syntypo import SinhalaTypoSynthesizer
from sinhala_charbert.evaluation.baselines import (
    IdentityBaseline,
    RuleBasedSinhalaCorrector,
    StandardBERTMLMCorrector,
)
from sinhala_charbert.evaluation.evaluator import TypoCorrectionEvaluator
from sinhala_charbert.models.modeling_charbert import SinhalaCharBERTForPreTraining
from sinhala_charbert.models.pipeline import SinhalaCharBERTCorrector
from sinhala_charbert.training.dictionary import SinhalaNLMDictionary


def generate_benchmark_testset(texts: list, num_samples: int = 100) -> list:
    """Generates a diverse synthetic evaluation set with category tags."""
    synthesizer = SinhalaTypoSynthesizer()
    samples = []

    for text in texts[:num_samples]:
        pair = synthesizer.generate_pair(text)
        # Determine primary error category from metadata if available
        edits = pair.get("metadata", {}).get("edits", [])
        cat = edits[0].get("stage", "general") if edits else "clean"

        samples.append({
            "source_noisy": pair["source_noisy"],
            "target_clean": pair["target_clean"],
            "error_category": cat,
        })
    return samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sinhala-CharBERT and Baselines.")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/sinhala_charbert/final_model")
    parser.add_argument("--dataset_name", type=str, default="Ransaka/sinhala-450M-sample")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_file", type=str, default="benchmark_results.json")
    parser.add_argument("--subword_tokenizer", type=str, default="Ransaka/sinhala-bert-medium-v2")
    args = parser.parse_args()

    print("=" * 70)
    print("Sinhala-CharBERT Comprehensive Evaluation & Benchmarking Suite")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Prepare Test Data
    print(f"Loading {args.num_samples} evaluation samples from '{args.dataset_name}'...")
    ds = load_dataset(args.dataset_name, split="train")
    test_texts = ds.select(range(min(args.num_samples, len(ds))))["text"]
    test_samples = generate_benchmark_testset(test_texts, num_samples=args.num_samples)
    print(f"Generated {len(test_samples)} evaluation test pairs.")

    # 2. Setup Resources & Dictionaries
    subword_tokenizer = AutoTokenizer.from_pretrained(args.subword_tokenizer)
    char_tokenizer = SinhalaCharTokenizer()
    char_tokenizer.train_on_corpus(test_texts[:200])

    nlm_dict = SinhalaNLMDictionary()
    nlm_dict.build_from_corpus(test_texts[:500], max_words=5000)

    align_engine = SequenceAlignmentEngine(
        subword_tokenizer=subword_tokenizer,
        char_tokenizer=char_tokenizer,
    )

    # 3. Initialize Baseline Models
    identity_model = IdentityBaseline()
    rule_model = RuleBasedSinhalaCorrector(dictionary=nlm_dict)
    bert_mlm_model = StandardBERTMLMCorrector(
        model_name_or_path=args.subword_tokenizer, device=device
    )

    # 4. Initialize Sinhala-CharBERT
    ckpt_path = Path(args.checkpoint_path)
    char_vocab_size = char_tokenizer.vocab_size
    nlm_vocab_size = nlm_dict.vocab_size

    weight_file = ckpt_path / "pytorch_model.bin"
    state_dict = None
    if weight_file.exists():
        state_dict = torch.load(weight_file, map_location="cpu")
        if "charbert.char_embeddings.char_embeddings.weight" in state_dict:
            char_vocab_size = state_dict["charbert.char_embeddings.char_embeddings.weight"].shape[0]
        if "nlm_head.decoder.weight" in state_dict:
            nlm_vocab_size = state_dict["nlm_head.decoder.weight"].shape[0]

    charbert_config = SinhalaCharBERTConfig(
        vocab_size=subword_tokenizer.vocab_size,
        char_vocab_size=char_vocab_size,
        nlm_vocab_size=nlm_vocab_size,
        max_position_embeddings=256,
    )
    charbert_model = SinhalaCharBERTForPreTraining(charbert_config)
    if state_dict is not None:
        charbert_model.load_state_dict(state_dict, strict=False)

    charbert_corrector = SinhalaCharBERTCorrector(
        model=charbert_model,
        subword_tokenizer=subword_tokenizer,
        char_tokenizer=char_tokenizer,
        alignment_engine=align_engine,
        nlm_dictionary=nlm_dict,
        device=device,
    )

    # 5. Run Comparative Benchmark
    models_to_evaluate = {
        "Identity (No-op Baseline)": identity_model.correct_batch,
        "Rule-Based Spellchecker": rule_model.correct_batch,
        "Standard BERT (MLM Head)": bert_mlm_model.correct_batch,
        "Sinhala-CharBERT (Mode A: Word Denoise)": lambda texts: [
            charbert_corrector.correct(t, mode="word_denoise").text for t in texts
        ],
    }

    evaluator = TypoCorrectionEvaluator(test_samples)
    benchmark_results = evaluator.run_benchmark(models_to_evaluate)

    # 6. Format and Display Results
    markdown_table = evaluator.format_markdown_table(benchmark_results)
    print("\n" + "=" * 70)
    print("BENCHMARK EVALUATION RESULTS")
    print("=" * 70)
    print(markdown_table)
    print("=" * 70)

    # Save to file
    out_file = Path(args.output_file)
    with open(out_file, "w", encoding="utf-8") as f:
        # Save serializable metrics
        serializable_res = {
            k: {
                "overall": v["overall"],
                "categories": v["categories"],
                "latency_ms_per_sample": v["latency_ms_per_sample"],
            }
            for k, v in benchmark_results.items()
        }
        json.dump(serializable_res, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed JSON results successfully saved to '{out_file}'.")


if __name__ == "__main__":
    main()
