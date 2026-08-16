"""
Sinhala-CharBERT Pre-Training Engine with Joint Dual-Objective Optimization and Noise Curriculum.
Supports single-device (CPU/CUDA/MPS) and multi-GPU Distributed Data Parallel (DDP) training.
"""

import json
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

        # Setup Optimizer (AdamW) with differential learning rates
        # Backbone (warm-started): token_embeddings, encoder.layers.*.transformer_layer, mlm_head
        # Char channel (random-init): char_embeddings, char_encoder, encoder.layers.*.hi_module,
        #                             fused_proj, layer_norm, nlm_head
        char_channel_prefixes = (
            "char_embeddings", "char_encoder", "hi_module",
            "fused_proj", "layer_norm", "nlm_head",
        )
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

        char_lr = self.cfg.char_channel_lr or self.cfg.learning_rate

        def _is_char_channel(name: str) -> bool:
            return any(prefix in name for prefix in char_channel_prefixes)

        backbone_decay = []
        backbone_no_decay = []
        char_decay = []
        char_no_decay = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_no_decay = any(nd in name for nd in no_decay)
            if _is_char_channel(name):
                (char_no_decay if is_no_decay else char_decay).append(param)
            else:
                (backbone_no_decay if is_no_decay else backbone_decay).append(param)

        optimizer_grouped_parameters = [
            {"params": backbone_decay, "weight_decay": self.cfg.weight_decay, "lr": self.cfg.learning_rate},
            {"params": backbone_no_decay, "weight_decay": 0.0, "lr": self.cfg.learning_rate},
            {"params": char_decay, "weight_decay": self.cfg.weight_decay, "lr": char_lr},
            {"params": char_no_decay, "weight_decay": 0.0, "lr": char_lr},
        ]

        # Filter out empty groups
        optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g["params"]) > 0]

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.cfg.learning_rate,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
            eps=self.cfg.adam_epsilon,
        )

        if self.is_main_process and self.cfg.char_channel_lr:
            backbone_count = len(backbone_decay) + len(backbone_no_decay)
            char_count = len(char_decay) + len(char_no_decay)
            self._log(
                f"Differential LR: backbone ({backbone_count} params) @ {self.cfg.learning_rate:.1e}, "
                f"char channel ({char_count} params) @ {char_lr:.1e}"
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
    # Checkpoint Resumption
    # ------------------------------------------------------------------

    def resume_from_checkpoint(self, checkpoint_dir: Union[str, Path]) -> dict:
        """Loads model, optimizer, scheduler, and scaler state from a checkpoint directory.

        Returns a dict with 'global_step' and 'epoch' for the training loop to resume from.
        """
        ckpt_path = Path(checkpoint_dir)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

        # 1. Load model weights
        model_state_path = ckpt_path / "pytorch_model.bin"
        if model_state_path.exists():
            state_dict = torch.load(model_state_path, map_location=self.device, weights_only=True)
            self._unwrapped_model.load_state_dict(state_dict)
            self._log(f"Resumed model weights from '{model_state_path}'")
        else:
            raise FileNotFoundError(f"Model weights not found at '{model_state_path}'")

        # 2. Load optimizer state
        opt_path = ckpt_path / "optimizer.pt"
        if opt_path.exists():
            opt_state = torch.load(opt_path, map_location=self.device, weights_only=True)
            self.optimizer.load_state_dict(opt_state)
            self._log(f"Resumed optimizer state from '{opt_path}'")

        # 3. Load LR scheduler state
        sched_path = ckpt_path / "scheduler.pt"
        if sched_path.exists():
            sched_state = torch.load(sched_path, map_location=self.device, weights_only=True)
            self.scheduler.load_state_dict(sched_state)
            self._log(f"Resumed scheduler state from '{sched_path}'")

        # 4. Load GradScaler state
        scaler_path = ckpt_path / "scaler.pt"
        if self.scaler is not None and scaler_path.exists():
            scaler_state = torch.load(scaler_path, map_location=self.device, weights_only=True)
            self.scaler.load_state_dict(scaler_state)
            self._log(f"Resumed GradScaler state from '{scaler_path}'")

        # 5. Load training state metadata (global_step, epoch)
        training_state = {"global_step": 0, "epoch": 0}
        state_path = ckpt_path / "training_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                training_state = json.load(f)
            self._log(
                f"Resuming from global_step={training_state['global_step']}, "
                f"epoch={training_state['epoch']}"
            )

        return training_state

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
        """Executes the full pre-training loop over the specified maximum steps.

        When ``self.cfg.resume_from_checkpoint`` is set, loads all training state
        from the checkpoint directory and fast-forwards the dataloader past
        already-completed steps before resuming gradient updates.
        """
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

        # Resume from checkpoint if specified
        if self.cfg.resume_from_checkpoint:
            resumed_state = self.resume_from_checkpoint(self.cfg.resume_from_checkpoint)
            global_step = resumed_state.get("global_step", 0)
            epoch = resumed_state.get("epoch", 0)
            self._log(f"Resuming training from step {global_step}, epoch {epoch}")

        start_time = time.time()

        self._log(
            f"Starting Sinhala-CharBERT Pre-Training on device '{self.device}'"
            + (f" (DDP: {self.world_size} GPUs)" if self.is_distributed else "") + "..."
        )
        self._log(f"Total Steps: {self.cfg.max_steps} | Batch Size: {self.cfg.batch_size} | LR: {self.cfg.learning_rate}")

        # Calculate number of batches per epoch for fast-forward during resume
        steps_per_epoch = len(dataloader)
        skip_batches = 0
        if global_step > 0 and steps_per_epoch > 0:
            skip_batches = global_step % steps_per_epoch
            self._log(f"Fast-forwarding past {skip_batches} batches in epoch {epoch}")

        while global_step < self.cfg.max_steps:
            # Update sampler epoch for proper shuffling across DDP ranks
            if sampler is not None:
                sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(dataloader):
                # Fast-forward past already-completed batches on resume
                if skip_batches > 0:
                    skip_batches -= 1
                    continue

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
                    self.save_checkpoint(ckpt_path, global_step=global_step, epoch=epoch)

            epoch += 1

        self._log(f"Pre-training complete! Saving final checkpoint to '{self.cfg.output_dir}'...")
        if self.is_main_process:
            self.save_checkpoint(
                Path(self.cfg.output_dir) / "final_model",
                global_step=global_step,
                epoch=epoch,
            )

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

    def save_checkpoint(
        self,
        output_dir: Union[str, Path],
        global_step: int = 0,
        epoch: int = 0,
    ) -> None:
        """Saves model weights, optimizer, scheduler, scaler, and training state."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Unwrap DDP model to save clean state_dict loadable on any device
        model_to_save = self._unwrapped_model
        torch.save(model_to_save.state_dict(), out_path / "pytorch_model.bin")
        torch.save(self.optimizer.state_dict(), out_path / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), out_path / "scheduler.pt")

        # Save GradScaler state for mixed precision resumption
        if self.scaler is not None:
            torch.save(self.scaler.state_dict(), out_path / "scaler.pt")

        # Save training loop state for exact resumption
        training_state = {
            "global_step": global_step,
            "epoch": epoch,
            "max_steps": self.cfg.max_steps,
            "learning_rate": self.cfg.learning_rate,
        }
        with open(out_path / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)

        # Save vocabularies and dictionaries for standalone downstream inference
        if hasattr(self.train_dataset, "char_tokenizer") and self.train_dataset.char_tokenizer is not None:
            self.train_dataset.char_tokenizer.save(out_path / "char_vocab.json")
        if hasattr(self.train_dataset, "nlm_dictionary") and self.train_dataset.nlm_dictionary is not None:
            self.train_dataset.nlm_dictionary.save(out_path / "nlm_dict.json")

        self._log(f"Checkpoint successfully saved to '{out_path}'")
