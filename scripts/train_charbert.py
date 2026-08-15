"""
Pre-training CLI script for Sinhala-CharBERT with Joint MLM + NLM Dual Loss and Noise Curriculum.
"""

import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

from sinhala_charbert.config.training_config import TrainingConfig
from sinhala_charbert.data.char_tokenizer import SinhalaCharTokenizer
from sinhala_charbert.data.alignment import SequenceAlignmentEngine
from sinhala_charbert.models.modeling_charbert import SinhalaCharBERTForPreTraining
from sinhala_charbert.training.curriculum import NoiseCurriculumScheduler
from sinhala_charbert.training.dictionary import SinhalaNLMDictionary
from sinhala_charbert.training.dataset import SinhalaCharBERTPretrainDataset
from sinhala_charbert.training.trainer import SinhalaCharBERTTrainer


def main():
    parser = argparse.ArgumentParser(description="Pre-train Sinhala-CharBERT with Joint MLM + NLM Loss.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Ransaka/sinhala-450M-sample",
        help="HuggingFace dataset repository or local text dataset path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split name.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of training samples to load (default: all).",
    )
    parser.add_argument(
        "--backbone_path",
        type=str,
        default="Ransaka/sinhala-bert-medium-v2",
        help="Path or HuggingFace identifier for pre-trained BERT backbone weights.",
    )
    parser.add_argument(
        "--subword_tokenizer",
        type=str,
        default="Ransaka/sinhala-bert-medium-v2",
        help="Subword tokenizer identifier or local directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/sinhala_charbert",
        help="Directory to save pre-trained model checkpoints.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Peak learning rate with cosine decay schedule.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100000,
        help="Maximum pre-training optimization steps.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5000,
        help="Number of linear learning rate warmup steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
        help="Checkpoint saving frequency in steps.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Logging interval in training steps.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Sinhala-CharBERT Pre-Training Pipeline")
    print("=" * 60)

    # 1. Load Dataset
    print(f"Loading corpus from '{args.dataset_name}'...")
    ds = load_dataset(args.dataset_name, split=args.split)
    if args.num_samples is not None:
        ds = ds.select(range(min(args.num_samples, len(ds))))
    raw_texts = ds["text"]
    print(f"Total training texts: {len(raw_texts):,}")

    # 2. Tokenizers and Alignment Engine Setup
    print("Initializing Subword and Character Tokenizers...")
    subword_tokenizer = AutoTokenizer.from_pretrained(args.subword_tokenizer)
    char_tokenizer = SinhalaCharTokenizer()
    char_tokenizer.train_on_corpus(raw_texts[:5000])

    alignment_engine = SequenceAlignmentEngine(
        subword_tokenizer=subword_tokenizer,
        char_tokenizer=char_tokenizer,
    )

    # 3. Build NLM Candidate Dictionary
    print("Building NLM Word Dictionary...")
    nlm_dict = SinhalaNLMDictionary()
    nlm_dict.build_from_corpus(raw_texts[:10000], max_words=32000)
    print(f"NLM Dictionary vocabulary size: {nlm_dict.vocab_size:,}")

    # 4. Curriculum and Dataset Setup
    curriculum = NoiseCurriculumScheduler(total_steps=args.max_steps)
    train_dataset = SinhalaCharBERTPretrainDataset(
        texts=raw_texts,
        subword_tokenizer=subword_tokenizer,
        char_tokenizer=char_tokenizer,
        alignment_engine=alignment_engine,
        nlm_dictionary=nlm_dict,
        curriculum_scheduler=curriculum,
    )

    # 5. Model Initialization
    if args.backbone_path:
        print(f"Initializing Sinhala-CharBERT from backbone '{args.backbone_path}'...")
        model = SinhalaCharBERTForPreTraining.from_pretrained_backbone(args.backbone_path)
    else:
        print("Initializing Sinhala-CharBERT from scratch...")
        from sinhala_charbert.config.model_config import SinhalaCharBERTConfig
        model_config = SinhalaCharBERTConfig(
            vocab_size=subword_tokenizer.vocab_size,
            char_vocab_size=char_tokenizer.vocab_size,
            nlm_vocab_size=nlm_dict.vocab_size,
        )
        model = SinhalaCharBERTForPreTraining(model_config)

    # 6. Training Configuration and Execution
    train_config = TrainingConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )

    trainer = SinhalaCharBERTTrainer(
        model=model,
        train_dataset=train_dataset,
        config=train_config,
    )

    trainer.train()


if __name__ == "__main__":
    main()
