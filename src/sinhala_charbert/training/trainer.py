"""
Sinhala-CharBERT Pre-Training Engine with Joint Dual-Objective Optimization and Noise Curriculum.
"""

import math
import os
from pathlib import Path
import time
from typing import Dict, Optional, Union
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from sinhala_charbert.config.training_config import TrainingConfig
from sinhala_charbert.models.modeling_charbert import SinhalaCharBERTForPreTraining
from sinhala_charbert.training.dataset import SinhalaCharBERTPretrainDataset, PretrainDualChannelCollator


class SinhalaCharBERTTrainer:
    """
    Pre-trainer for Sinhala-CharBERT.
    Jointly optimizes Masked Language Modeling (MLM) and Noisy Language Modeling (NLM)
    under an adaptive 3-phase SynTypo-SI noise curriculum.
    """

    def __init__(
        self,
        model: SinhalaCharBERTForPreTraining,
        train_dataset: SinhalaCharBERTPretrainDataset,
        config: Optional[TrainingConfig] = None,
        eval_dataset: Optional[SinhalaCharBERTPretrainDataset] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.cfg = config or TrainingConfig()

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

        # Setup Optimizer (AdamW) with weight decay exclusions for biases and LayerNorm
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.cfg.learning_rate,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
            eps=self.cfg.adam_epsilon,
        )

        # Setup Learning Rate Scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=self.cfg.max_steps,
        )

        # Mixed precision scaler (for CUDA devices)
        self.scaler = torch.amp.GradScaler("cuda") if (self.cfg.fp16 and self.device.type == "cuda") else None

        # Data Collator
        self.collator = PretrainDualChannelCollator(
            subword_pad_token_id=self.model.config.pad_token_id,
            subword_vocab_size=self.model.config.vocab_size,
            char_pad_token_id=self.model.config.char_pad_token_id,
            mlm_probability=self.cfg.mlm_probability,
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Executes a single optimization forward and backward step."""
        self.model.train()

        # Move tensors to active device
        input_ids = batch["input_ids"].to(self.device)
        char_input_ids = batch["char_input_ids"].to(self.device)
        start_char_idx = batch["start_char_idx"].to(self.device)
        end_char_idx = batch["end_char_idx"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        char_attention_mask = batch["char_attention_mask"].to(self.device)
        mlm_labels = batch["mlm_labels"].to(self.device)
        nlm_labels = batch["nlm_labels"].to(self.device)

        # Mixed precision execution
        if self.scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = self.model(
                    input_ids=input_ids,
                    char_input_ids=char_input_ids,
                    start_char_idx=start_char_idx,
                    end_char_idx=end_char_idx,
                    attention_mask=attention_mask,
                    char_attention_mask=char_attention_mask,
                    mlm_labels=mlm_labels,
                    nlm_labels=nlm_labels,
                    mlm_loss_weight=self.cfg.mlm_loss_weight,
                    nlm_loss_weight=self.cfg.nlm_loss_weight,
                )
                loss = outputs.loss / self.cfg.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
        else:
            outputs = self.model(
                input_ids=input_ids,
                char_input_ids=char_input_ids,
                start_char_idx=start_char_idx,
                end_char_idx=end_char_idx,
                attention_mask=attention_mask,
                char_attention_mask=char_attention_mask,
                mlm_labels=mlm_labels,
                nlm_labels=nlm_labels,
                mlm_loss_weight=self.cfg.mlm_loss_weight,
                nlm_loss_weight=self.cfg.nlm_loss_weight,
            )
            loss = outputs.loss / self.cfg.gradient_accumulation_steps
            loss.backward()

        return {
            "loss": outputs.loss.item() if outputs.loss is not None else 0.0,
            "mlm_loss": outputs.mlm_loss.item() if outputs.mlm_loss is not None else 0.0,
            "nlm_loss": outputs.nlm_loss.item() if outputs.nlm_loss is not None else 0.0,
        }

    def train(self) -> None:
        """Executes the full pre-training loop over the specified maximum steps."""
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=0,
        )

        global_step = 0
        total_loss = 0.0
        start_time = time.time()

        print(f"Starting Sinhala-CharBERT Pre-Training on device '{self.device}'...")
        print(f"Total Steps: {self.cfg.max_steps} | Batch Size: {self.cfg.batch_size} | LR: {self.cfg.learning_rate}")

        while global_step < self.cfg.max_steps:
            for batch in dataloader:
                if global_step >= self.cfg.max_steps:
                    break

                # Advance noise curriculum step
                self.train_dataset.set_step(global_step)

                step_metrics = self.train_step(batch)
                total_loss += step_metrics["loss"]

                if (global_step + 1) % self.cfg.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if self.device.type == "mps" and (global_step + 1) % 10 == 0:
                        torch.mps.empty_cache()

                global_step += 1

                # Logging
                if global_step % self.cfg.logging_steps == 0:
                    avg_loss = total_loss / self.cfg.logging_steps
                    elapsed = time.time() - start_time
                    lr = self.scheduler.get_last_lr()[0]
                    curriculum_phase = self.train_dataset.curriculum.get_phase_name(global_step)
                    print(
                        f"Step [{global_step:6d}/{self.cfg.max_steps}] | "
                        f"Loss: {avg_loss:.4f} (MLM: {step_metrics['mlm_loss']:.4f}, NLM: {step_metrics['nlm_loss']:.4f}) | "
                        f"LR: {lr:.2e} | {curriculum_phase} | {elapsed:.1f}s"
                    )
                    total_loss = 0.0

                # Periodic Evaluation
                if self.eval_dataset is not None and global_step % self.cfg.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    print(f"\n--- Validation [Step {global_step}] ---")
                    print(f"Val Loss: {eval_metrics['val_loss']:.4f} (MLM: {eval_metrics['val_mlm_loss']:.4f}, NLM: {eval_metrics['val_nlm_loss']:.4f})\n")

                # Periodic Checkpoint Saving
                if global_step % self.cfg.save_steps == 0:
                    ckpt_path = Path(self.cfg.output_dir) / f"checkpoint-{global_step}"
                    self.save_checkpoint(ckpt_path)

        print(f"Pre-training complete! Saving final checkpoint to '{self.cfg.output_dir}'...")
        self.save_checkpoint(Path(self.cfg.output_dir) / "final_model")

    def evaluate(self) -> Dict[str, float]:
        """Evaluates model on the validation dataset."""
        if self.eval_dataset is None:
            return {}

        self.model.eval()
        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )

        total_loss = 0.0
        total_mlm = 0.0
        total_nlm = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                char_input_ids = batch["char_input_ids"].to(self.device)
                start_char_idx = batch["start_char_idx"].to(self.device)
                end_char_idx = batch["end_char_idx"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                char_attention_mask = batch["char_attention_mask"].to(self.device)
                mlm_labels = batch["mlm_labels"].to(self.device)
                nlm_labels = batch["nlm_labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    char_input_ids=char_input_ids,
                    start_char_idx=start_char_idx,
                    end_char_idx=end_char_idx,
                    attention_mask=attention_mask,
                    char_attention_mask=char_attention_mask,
                    mlm_labels=mlm_labels,
                    nlm_labels=nlm_labels,
                )

                if outputs.loss is not None:
                    total_loss += outputs.loss.item()
                    total_mlm += outputs.mlm_loss.item() if outputs.mlm_loss is not None else 0.0
                    total_nlm += outputs.nlm_loss.item() if outputs.nlm_loss is not None else 0.0
                    num_batches += 1

        self.model.train()
        return {
            "val_loss": total_loss / max(num_batches, 1),
            "val_mlm_loss": total_mlm / max(num_batches, 1),
            "val_nlm_loss": total_nlm / max(num_batches, 1),
        }

    def save_checkpoint(self, output_dir: Union[str, Path]) -> None:
        """Saves model weights, optimizer, vocabularies, and training configuration."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), out_path / "pytorch_model.bin")
        torch.save(self.optimizer.state_dict(), out_path / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), out_path / "scheduler.pt")

        # Save vocabularies and dictionaries for standalone downstream inference
        if hasattr(self.train_dataset, "char_tokenizer") and self.train_dataset.char_tokenizer is not None:
            self.train_dataset.char_tokenizer.save(out_path / "char_vocab.json")
        if hasattr(self.train_dataset, "nlm_dictionary") and self.train_dataset.nlm_dictionary is not None:
            self.train_dataset.nlm_dictionary.save(out_path / "nlm_dict.json")

        print(f"Checkpoint successfully saved to '{out_path}'")
