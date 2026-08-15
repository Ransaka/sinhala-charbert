"""
Fine-tuning CLI for Mode B: Sinhala-CharBERT Open-Vocabulary Seq2Seq Decoder.
Trains the autoregressive Transformer Decoder to reconstruct clean Akshara sequences from noisy inputs.
"""

import argparse
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from sinhala_charbert.config.model_config import SinhalaCharBERTConfig
from sinhala_charbert.data.alignment import SequenceAlignmentEngine
from sinhala_charbert.data.char_tokenizer import SinhalaCharTokenizer
from sinhala_charbert.data.syntypo import SinhalaTypoSynthesizer
from sinhala_charbert.models.seq2seq_decoder import SinhalaCharBERTSeq2SeqModel
from sinhala_charbert.training.curriculum import NoiseCurriculumScheduler


class Seq2SeqCorrectionDataset(Dataset):
    """Generates on-the-fly noisy/clean paired sequences for Seq2Seq fine-tuning."""

    def __init__(
        self,
        texts: List[str],
        alignment_engine: SequenceAlignmentEngine,
        char_tokenizer: SinhalaCharTokenizer,
        curriculum: Optional[NoiseCurriculumScheduler] = None,
        max_target_len: int = 128,
    ):
        self.texts = texts
        self.alignment_engine = alignment_engine
        self.char_tokenizer = char_tokenizer
        self.curriculum = curriculum or NoiseCurriculumScheduler()
        self.max_target_len = max_target_len
        self.current_step = 0

    def set_step(self, step: int) -> None:
        self.current_step = step

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        clean_text = self.texts[idx]
        if not clean_text or not isinstance(clean_text, str):
            clean_text = "සිංහල"

        # 1. Synthesize typo
        profile = self.curriculum.get_profile(self.current_step)
        synth = SinhalaTypoSynthesizer(profile=profile)
        pair = synth.generate_pair(clean_text)
        noisy_text = pair["source_noisy"]
        target_clean = pair["target_clean"]

        # 2. Source alignment
        aligned = self.alignment_engine.align(noisy_text)

        # 3. Target character encoding
        target_ids = self.char_tokenizer.encode(
            target_clean, add_special_tokens=True, max_length=self.max_target_len
        )

        return {
            "aligned": aligned,
            "target_ids": target_ids,
        }


def seq2seq_collate_fn(batch_items: List[Dict[str, Any]], pad_char_id: int = 0) -> Dict[str, torch.Tensor]:
    batch_size = len(batch_items)
    aligned_list = [b["aligned"] for b in batch_items]
    target_ids_list = [b["target_ids"] for b in batch_items]

    max_subword_len = max(len(a.input_ids) for a in aligned_list)
    max_char_len = max(len(a.char_input_ids) for a in aligned_list)
    max_target_len = max(len(t) for t in target_ids_list)

    input_ids = torch.zeros((batch_size, max_subword_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_subword_len), dtype=torch.long)
    char_input_ids = torch.full((batch_size, max_char_len), pad_char_id, dtype=torch.long)
    char_attention_mask = torch.zeros((batch_size, max_char_len), dtype=torch.long)
    start_char_idx = torch.zeros((batch_size, max_subword_len), dtype=torch.long)
    end_char_idx = torch.zeros((batch_size, max_subword_len), dtype=torch.long)

    decoder_input_ids = torch.full((batch_size, max_target_len - 1), pad_char_id, dtype=torch.long)
    labels = torch.full((batch_size, max_target_len - 1), -100, dtype=torch.long)

    for i, (aligned, target_ids) in enumerate(zip(aligned_list, target_ids_list)):
        m_len = len(aligned.input_ids)
        n_len = len(aligned.char_input_ids)

        input_ids[i, :m_len] = torch.tensor(aligned.input_ids, dtype=torch.long)
        attention_mask[i, :m_len] = torch.tensor(aligned.attention_mask, dtype=torch.long)
        char_input_ids[i, :n_len] = torch.tensor(aligned.char_input_ids, dtype=torch.long)
        char_attention_mask[i, :n_len] = torch.tensor(aligned.char_attention_mask, dtype=torch.long)
        start_char_idx[i, :m_len] = torch.tensor(aligned.start_char_idx, dtype=torch.long)
        end_char_idx[i, :m_len] = torch.tensor(aligned.end_char_idx, dtype=torch.long)

        # Decoder inputs: target[0:-1], Labels: target[1:]
        t_len = len(target_ids)
        if t_len > 1:
            dec_in = target_ids[:-1]
            lab = target_ids[1:]
            decoder_input_ids[i, : len(dec_in)] = torch.tensor(dec_in, dtype=torch.long)
            labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "char_input_ids": char_input_ids,
        "char_attention_mask": char_attention_mask,
        "start_char_idx": start_char_idx,
        "end_char_idx": end_char_idx,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Mode B Seq2Seq Typo Corrector")
    parser.add_argument("--dataset_name", type=str, default="Ransaka/sinhala-450M-sample")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--subword_tokenizer", type=str, default="Ransaka/sinhala-bert-medium-v2")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sinhala_charbert_seq2seq")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--logging_steps", type=int, default=50)
    args = parser.parse_args()

    print("=" * 65)
    print("Sinhala-CharBERT Mode B Seq2Seq Fine-Tuning Pipeline")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load corpus
    ds = load_dataset(args.dataset_name, split="train")
    if args.num_samples is not None:
        ds = ds.select(range(min(args.num_samples, len(ds))))
    raw_texts = ds["text"]

    # Setup tokenizers
    subword_tokenizer = AutoTokenizer.from_pretrained(args.subword_tokenizer)
    char_tokenizer = SinhalaCharTokenizer()
    char_tokenizer.train_on_corpus(raw_texts[:5000])

    align_engine = SequenceAlignmentEngine(
        subword_tokenizer=subword_tokenizer,
        char_tokenizer=char_tokenizer,
        max_subword_length=256,
        max_char_length=512,
    )

    curriculum = NoiseCurriculumScheduler(total_steps=args.max_steps)
    train_ds = Seq2SeqCorrectionDataset(
        texts=raw_texts,
        alignment_engine=align_engine,
        char_tokenizer=char_tokenizer,
        curriculum=curriculum,
    )

    dataloader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: seq2seq_collate_fn(b, pad_char_id=char_tokenizer.pad_token_id),
    )

    config = SinhalaCharBERTConfig(
        vocab_size=subword_tokenizer.vocab_size,
        char_vocab_size=char_tokenizer.vocab_size,
        max_position_embeddings=256,
    )
    model = SinhalaCharBERTSeq2SeqModel(config=config, num_decoder_layers=4)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)

    global_step = 0
    total_loss = 0.0
    start_time = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting Seq2Seq training: {args.max_steps} steps | Batch Size: {args.batch_size} (Eff: {args.batch_size * args.gradient_accumulation_steps})")

    model.train()
    while global_step < args.max_steps:
        for batch in dataloader:
            if global_step >= args.max_steps:
                break

            train_ds.set_step(global_step)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            char_input_ids = batch["char_input_ids"].to(device)
            char_attention_mask = batch["char_attention_mask"].to(device)
            start_char_idx = batch["start_char_idx"].to(device)
            end_char_idx = batch["end_char_idx"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                char_input_ids=char_input_ids,
                start_char_idx=start_char_idx,
                end_char_idx=end_char_idx,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                char_attention_mask=char_attention_mask,
                labels=labels,
                label_smoothing=0.05,
            )

            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            total_loss += outputs.loss.item()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if device.type == "mps" and (global_step + 1) % 10 == 0:
                    torch.mps.empty_cache()

            global_step += 1

            if global_step % args.logging_steps == 0:
                avg_loss = total_loss / args.logging_steps
                elapsed = time.time() - start_time
                lr = scheduler.get_last_lr()[0]
                print(f"Step [{global_step:5d}/{args.max_steps}] | Seq2Seq Loss: {avg_loss:.4f} | LR: {lr:.2e} | {elapsed:.1f}s")
                total_loss = 0.0

            if global_step % args.save_steps == 0:
                save_path = out_dir / f"checkpoint-{global_step}"
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path / "pytorch_model.bin")
                print(f"Saved checkpoint to '{save_path}'")

    torch.save(model.state_dict(), out_dir / "pytorch_model.bin")
    print(f"Seq2Seq training complete! Saved model to '{out_dir}'")


if __name__ == "__main__":
    main()
