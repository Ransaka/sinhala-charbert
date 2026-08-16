"""
Sinhala-CharBERT Pre-Training Engine with Joint Dual-Objective Optimization and Noise Curriculum.
Supports single-device (CPU/CUDA/MPS) and multi-GPU Distributed Data Parallel (DDP) training.
"""

import math
import os
from pathlib import Path
import time
from typing import Dict, Optional, Union
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from sinhala_charbert.config.training_config import TrainingConfig
from sinhala_charbert.models.modeling_charbert import SinhalaCharBERTForPreTraining
from sinhala_charbert.training.dataset import SinhalaCharBERTPretrainDataset, PretrainDualChannelCollator


class SinhalaCharBERTTrainer:
    """
    Pre-trainer for Sinhala-CharBERT.
    Jointly optimizes Masked Language Modeling (MLM) and Noisy Language Modeling (NLM)
    under an adaptive 3-phase SynTypo-SI noise curriculum.

    Automatically detects and activates PyTorch DistributedDataParallel (DDP) when
    launched via ``torchrun --nproc_per_node=N``. Falls back to single-device training
    otherwise.
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

        # Distributed training state
        self._setup_distributed(device)

        self.model.to(self.device)

        # Wrap model with DDP when running distributed
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.cfg.ddp_find_unused_parameters,
            )

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
        model_config = self._unwrapped_model.config
        self.collator = PretrainDualChannelCollator(
            subword_pad_token_id=model_config.pad_token_id,
            subword_vocab_size=model_config.vocab_size,
            char_pad_token_id=model_config.char_pad_token_id,
            mlm_probability=self.cfg.mlm_probability,
        )

    # ------------------------------------------------------------------
    # Distributed Utilities
    # ------------------------------------------------------------------

    def _setup_distributed(self, device: Optional[torch.device]) -> None:
        """Detects and configures distributed training environment."""
        self.is_distributed = dist.is_available() and dist.is_initialized()

        if self.is_distributed:
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = dist.get_world_size()
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            if device is not None:
                self.device = device
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

    @property
    def is_main_process(self) -> bool:
        """Returns True on rank 0 (the primary process for logging and checkpointing)."""
        return self.rank == 0

    @property
    def _unwrapped_model(self) -> SinhalaCharBERTForPreTraining:
        """Returns the underlying model without the DDP wrapper."""
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _log(self, msg: str) -> None:
        """Prints a message only on the main process."""
        if self.is_main_process:
            print(msg)

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------

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
        # Setup sampler: DistributedSampler for multi-GPU, None for single device
        sampler = None
        shuffle = True
        if self.is_distributed:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            shuffle = False  # Sampler handles shuffling

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=self.collator,
            num_workers=self.cfg.dataloader_num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        global_step = 0
        total_loss = 0.0
        epoch = 0
        start_time = time.time()

        self._log(
            f"Starting Sinhala-CharBERT Pre-Training on device '{self.device}'"
            + (f" (DDP: {self.world_size} GPUs)" if self.is_distributed else "") + "..."
        )
        self._log(f"Total Steps: {self.cfg.max_steps} | Batch Size: {self.cfg.batch_size} | LR: {self.cfg.learning_rate}")

        while global_step < self.cfg.max_steps:
            # Update sampler epoch for proper shuffling across DDP ranks
            if sampler is not None:
                sampler.set_epoch(epoch)

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

                # Logging (rank 0 only)
                if global_step % self.cfg.logging_steps == 0 and self.is_main_process:
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

                # Periodic Evaluation (rank 0 only)
                if self.eval_dataset is not None and global_step % self.cfg.eval_steps == 0 and self.is_main_process:
                    eval_metrics = self.evaluate()
                    print(f"\n--- Validation [Step {global_step}] ---")
                    print(f"Val Loss: {eval_metrics['val_loss']:.4f} (MLM: {eval_metrics['val_mlm_loss']:.4f}, NLM: {eval_metrics['val_nlm_loss']:.4f})\n")

                # Periodic Checkpoint Saving (rank 0 only)
                if global_step % self.cfg.save_steps == 0 and self.is_main_process:
                    ckpt_path = Path(self.cfg.output_dir) / f"checkpoint-{global_step}"
                    self.save_checkpoint(ckpt_path)

            epoch += 1

        self._log(f"Pre-training complete! Saving final checkpoint to '{self.cfg.output_dir}'...")
        if self.is_main_process:
            self.save_checkpoint(Path(self.cfg.output_dir) / "final_model")

        # Synchronize all processes before exit
        if self.is_distributed:
            dist.barrier()

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

        # Unwrap DDP model to save clean state_dict loadable on any device
        model_to_save = self._unwrapped_model
        torch.save(model_to_save.state_dict(), out_path / "pytorch_model.bin")
        torch.save(self.optimizer.state_dict(), out_path / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), out_path / "scheduler.pt")

        # Save vocabularies and dictionaries for standalone downstream inference
        if hasattr(self.train_dataset, "char_tokenizer") and self.train_dataset.char_tokenizer is not None:
            self.train_dataset.char_tokenizer.save(out_path / "char_vocab.json")
        if hasattr(self.train_dataset, "nlm_dictionary") and self.train_dataset.nlm_dictionary is not None:
            self.train_dataset.nlm_dictionary.save(out_path / "nlm_dict.json")

        self._log(f"Checkpoint successfully saved to '{out_path}'")
