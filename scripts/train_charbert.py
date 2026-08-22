"""
Pre-training CLI script for Sinhala-CharBERT with Joint MLM + NLM Dual Loss and Noise Curriculum.
Supports single-GPU and multi-GPU DDP training via ``torchrun --nproc_per_node=N``.
"""

import argparse
import os
from pathlib import Path
import torch
import torch.distributed as dist
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


def _setup_distributed() -> bool:
    """Initialize distributed process group if launched via torchrun.

    Returns True when DDP is active, False for single-device training.
    """
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) <= 1:
        return False

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True


def _teardown_distributed(is_distributed: bool) -> None:
    """Cleanly shuts down the distributed process group."""
    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _is_main_process() -> bool:
    """Returns True on rank 0 or when running single-device."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


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
        default=8,
        help="Per-device training micro-batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of update steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--max_subword_length",
        type=int,
        default=256,
        help="Maximum subword token sequence length.",
    )
    parser.add_argument(
        "--max_char_length",
        type=int,
        default=512,
        help="Maximum phonological character sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Peak learning rate for backbone (warm-started) parameters.",
    )
    parser.add_argument(
        "--char_channel_lr",
        type=float,
        default=None,
        help="Peak learning rate for character channel (randomly-initialized) parameters. "
             "If not set, uses the same value as --learning_rate.",
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of DataLoader worker processes per GPU.",
    )
    parser.add_argument(
        "--nlm_max_words",
        type=int,
        default=64000,
        help="Maximum vocabulary size for the NLM word dictionary.",
    )
    parser.add_argument(
        "--nlm_min_freq",
        type=int,
        default=2,
        help="Minimum word frequency to include in the NLM dictionary.",
    )
    parser.add_argument(
        "--mlm_loss_weight",
        type=float,
        default=1.0,
        help="Scalar weight for the MLM loss term in the combined objective.",
    )
    parser.add_argument(
        "--nlm_loss_weight",
        type=float,
        default=1.0,
        help="Scalar weight for the NLM loss term in the combined objective.",
    )
    parser.add_argument(
        "--find_unused_parameters",
        action="store_true",
        default=True,
        help="Enable unused parameter detection in DDP for auxiliary/downstream heads.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from (e.g. 'checkpoints/sinhala_charbert/checkpoint-5000').",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------
    # Distributed Setup
    # -------------------------------------------------------------------
    is_distributed = _setup_distributed()

    if _is_main_process():
        print("=" * 60)
        print("Sinhala-CharBERT Pre-Training Pipeline")
        if is_distributed:
            print(f"  Mode: Distributed Data Parallel ({dist.get_world_size()} GPUs)")
        else:
            print("  Mode: Single Device")
        print("=" * 60)

    # 1. Load Dataset
    if _is_main_process():
        print(f"Loading corpus from '{args.dataset_name}'...")
    ds = load_dataset(args.dataset_name, split=args.split)
    if args.num_samples is not None:
        ds = ds.select(range(min(args.num_samples, len(ds))))
    raw_texts = ds["text"]
    if _is_main_process():
        print(f"Total training texts: {len(raw_texts):,}")

    # 2. Tokenizers and Alignment Engine Setup
    if _is_main_process():
        print("Initializing Subword and Character Tokenizers...")
    subword_tokenizer = AutoTokenizer.from_pretrained(args.subword_tokenizer)

    # Check if vocabulary files exist in resume checkpoint directory
    resume_ckpt_dir = Path(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    char_vocab_path = (resume_ckpt_dir / "char_vocab.json") if resume_ckpt_dir else None
    nlm_dict_path = (resume_ckpt_dir / "nlm_dict.json") if resume_ckpt_dir else None

    if char_vocab_path and char_vocab_path.exists():
        if _is_main_process():
            print(f"Loading Character Tokenizer from checkpoint: '{char_vocab_path}'")
        char_tokenizer = SinhalaCharTokenizer.load(char_vocab_path)
    else:
        char_tokenizer = SinhalaCharTokenizer()
        char_tokenizer.train_on_corpus(raw_texts[:5000])

    alignment_engine = SequenceAlignmentEngine(
        subword_tokenizer=subword_tokenizer,
        char_tokenizer=char_tokenizer,
        max_subword_length=args.max_subword_length,
        max_char_length=args.max_char_length,
    )

    # 3. NLM Candidate Dictionary Setup (load from checkpoint or build from corpus)
    if nlm_dict_path and nlm_dict_path.exists():
        if _is_main_process():
            print(f"Loading NLM Word Dictionary from checkpoint: '{nlm_dict_path}'")
        nlm_dict = SinhalaNLMDictionary.load(nlm_dict_path)
    else:
        if _is_main_process():
            print("Building NLM Word Dictionary from corpus...")
        nlm_dict = SinhalaNLMDictionary()
        nlm_dict.build_from_corpus(raw_texts, max_words=args.nlm_max_words, min_freq=args.nlm_min_freq)

    if _is_main_process():
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
    if _is_main_process():
        print(f"Total training documents: {len(train_dataset):,} (lazy on-the-fly sentence chunking enabled)")

    # 5. Model Initialization
    if args.backbone_path:
        if _is_main_process():
            print(f"Initializing Sinhala-CharBERT from backbone '{args.backbone_path}'...")
        model = SinhalaCharBERTForPreTraining.from_pretrained_backbone(
            args.backbone_path,
            char_vocab_size=char_tokenizer.vocab_size,
            nlm_vocab_size=nlm_dict.vocab_size,
        )
    else:
        if _is_main_process():
            print("Initializing Sinhala-CharBERT from scratch...")
        from sinhala_charbert.config.model_config import SinhalaCharBERTConfig
        model_config = SinhalaCharBERTConfig(
            vocab_size=subword_tokenizer.vocab_size,
            char_vocab_size=char_tokenizer.vocab_size,
            nlm_vocab_size=nlm_dict.vocab_size,
            max_position_embeddings=args.max_subword_length,
        )
        model = SinhalaCharBERTForPreTraining(model_config)

    # 6. Training Configuration and Execution
    train_config = TrainingConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        char_channel_lr=args.char_channel_lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.num_workers,
        mlm_loss_weight=args.mlm_loss_weight,
        nlm_loss_weight=args.nlm_loss_weight,
        ddp_find_unused_parameters=args.find_unused_parameters,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    trainer = SinhalaCharBERTTrainer(
        model=model,
        train_dataset=train_dataset,
        config=train_config,
    )

    trainer.train()

    # -------------------------------------------------------------------
    # Distributed Teardown
    # -------------------------------------------------------------------
    _teardown_distributed(is_distributed)


if __name__ == "__main__":
    main()
