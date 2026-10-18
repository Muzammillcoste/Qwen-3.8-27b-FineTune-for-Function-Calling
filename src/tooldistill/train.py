"""QLoRA adapter training (Phase 3A pilot scope). One adapter per run from the frozen base.

GPU path: 4-bit NF4 + double quant, FP16 compute (T4 has no bf16), LoRA on all linear layers
except the LM head, gradient checkpointing (non-reentrant), paged 8-bit AdamW, completion-only
loss (labels prepared by ``format_qwen.build_training_example``).

CPU path (tests only): same code on a tiny randomly-initialised model, no quantization.
Never silently changes model, length or batch settings: problems raise or are reported.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

from .format_qwen import IGNORE_INDEX

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainConfig:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 1e-4
    epochs: float = 1.0
    per_device_batch: int = 1
    grad_accum: int = 8
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    max_seq_len: int = 1536
    seed: int = 42
    save_steps: int = 5
    logging_steps: int = 1
    optim_gpu: str = "paged_adamw_8bit"
    optim_cpu: str = "adamw_torch"

    @classmethod
    def from_yaml(cfg_cls, y: dict[str, Any]) -> TrainConfig:
        lo, tr = y["lora"], y["training"]
        return cfg_cls(lora_r=lo["r"], lora_alpha=lo["alpha"], lora_dropout=lo["dropout"],
                       learning_rate=tr["learning_rate"], epochs=tr["num_train_epochs"],
                       per_device_batch=tr["per_device_train_batch_size"],
                       grad_accum=tr["gradient_accumulation_steps"],
                       warmup_ratio=tr["warmup_steps"], max_grad_norm=tr["max_grad_norm"],
                       max_seq_len=tr["max_seq_length"], seed=y["seed"],
                       save_steps=tr["save_steps"], logging_steps=tr["logging_steps"],
                       optim_gpu=tr["optim"])


# ----------------------------------------------------------------------------- loading


def load_quantized_model(model_id: str, revision: str, compute_dtype=torch.float16):
    """4-bit NF4 base model on ONE GPU (device_map pinned to cuda:0)."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype)
    return AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, quantization_config=bnb, dtype=compute_dtype,
        device_map={"": 0}, attn_implementation="sdpa")


def add_lora(model, cfg: TrainConfig, quantized: bool):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if quantized:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    # GradScaler (fp16) requires fp32 trainable params
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    return model


def trainable_summary(model) -> dict[str, Any]:
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    dtypes = sorted({str(p.dtype) for p in model.parameters() if p.requires_grad})
    lm_head_trainable = any(p.requires_grad for n, p in model.named_parameters() if "lm_head" in n)
    return {"trainable": tr, "total": tot, "trainable_dtypes": dtypes,
            "lm_head_trainable": lm_head_trainable}


# ----------------------------------------------------------------------------- data


class ListDataset(Dataset):
    def __init__(self, examples: list[dict[str, list[int]]]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        e = self.examples[i]
        return {"input_ids": e["input_ids"], "labels": e["labels"],
                "attention_mask": e["attention_mask"]}


@dataclass
class PadCollator:
    """Right-pads for training. Padding positions: attention 0, label -100 (never supervised)."""

    pad_token_id: int

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        n = max(len(b["input_ids"]) for b in batch)
        ids, lab, att = [], [], []
        for b in batch:
            pad = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_token_id] * pad)
            lab.append(b["labels"] + [IGNORE_INDEX] * pad)
            att.append(b["attention_mask"] + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}


# ----------------------------------------------------------------------------- monitoring


class NonFiniteLossError(RuntimeError):
    pass


@dataclass
class StepLog:
    rows: list[dict[str, Any]] = field(default_factory=list)
    nonfinite: list[int] = field(default_factory=list)


class MonitorCallback(TrainerCallback):
    """Per-optimizer-step: loss, grad norm, lr, step time, tokens/s, peak VRAM. Stops on NaN/Inf."""

    def __init__(self, log: StepLog):
        self.log = log
        self._t0 = None
        self._tok0 = 0

    def on_step_begin(self, args, state, control, **kw):
        self._t0 = time.perf_counter()
        self._tok0 = int(state.num_input_tokens_seen or 0)

    def on_step_end(self, args, state, control, **kw):
        dt = time.perf_counter() - (self._t0 or time.perf_counter())
        toks = int(state.num_input_tokens_seen or 0) - self._tok0
        row = {"step": state.global_step, "step_seconds": round(dt, 3), "tokens": toks,
               "tokens_per_s": round(toks / dt, 1) if dt > 0 else None}
        if torch.cuda.is_available():
            row["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
            row["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
        self.log.rows.append(row)

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs or not self.log.rows:
            return
        row = self.log.rows[-1]
        for k in ("loss", "grad_norm", "learning_rate"):
            if k in logs:
                row[k] = logs[k]
        for k in ("loss", "grad_norm"):
            v = logs.get(k)
            if v is not None and not math.isfinite(float(v)):
                self.log.nonfinite.append(state.global_step)
                control.should_training_stop = True


class InterruptAtStep(TrainerCallback):
    """Simulates a session ending: checkpoint at ``step`` then stop. Used to test true resume."""

    def __init__(self, step: int):
        self.step = step

    def on_step_end(self, args, state, control, **kw):
        if state.global_step == self.step:
            control.should_save = True
            control.should_training_stop = True


# ----------------------------------------------------------------------------- training


def training_args(out_dir: str, cfg: TrainConfig, use_gpu: bool) -> TrainingArguments:
    return TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=cfg.warmup_ratio,          # float in [0,1) = ratio of total steps (v5 API)
        max_grad_norm=cfg.max_grad_norm,
        optim=cfg.optim_gpu if use_gpu else cfg.optim_cpu,
        fp16=use_gpu,
        bf16=False,
        gradient_checkpointing=False,           # enabled on the model in add_lora
        logging_steps=cfg.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=3,
        seed=cfg.seed,
        data_seed=cfg.seed,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        include_num_input_tokens_seen="non_padding",
        use_cpu=not use_gpu,
    )


def run_training(model, tok, examples, out_dir: str, cfg: TrainConfig, use_gpu: bool,
                 interrupt_at: int | None = None, resume: bool | str = False) -> dict[str, Any]:
    log = StepLog()
    callbacks: list[TrainerCallback] = [MonitorCallback(log)]
    if interrupt_at:
        callbacks.append(InterruptAtStep(interrupt_at))
    trainer = Trainer(model=model, args=training_args(out_dir, cfg, use_gpu),
                      train_dataset=ListDataset(examples),
                      data_collator=PadCollator(tok.pad_token_id), callbacks=callbacks)
    if use_gpu:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = trainer.train(resume_from_checkpoint=resume or None)
    wall = time.perf_counter() - t0
    if log.nonfinite:
        raise NonFiniteLossError(f"non-finite loss/grad at steps {log.nonfinite}")
    st = trainer.state
    res = {"global_step": st.global_step, "max_steps": st.max_steps,
           "tokens_seen_non_padding": int(st.num_input_tokens_seen or 0),
           "wall_seconds": round(wall, 2), "train_loss": out.training_loss,
           "steps": log.rows, "log_history": st.log_history}
    if use_gpu:
        res["peak_vram_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        res["peak_vram_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
    return res, trainer
