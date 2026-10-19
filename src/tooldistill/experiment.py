"""Phase 3B/4 runner on Kaggle (one T4). One job per session, chosen by MODE / RUN_ID.

MODE:
  preflight      GPU/env + data hashes + template + mask checks for the arm (no model download)
  eval_base_val  untuned base on the fixed val_eval_400 set
  train_eval     train ONE arm (RUN_ID) from the ORIGINAL base with a FRESH LoRA (same seeded
                 init for every arm; never warm-started from pilot_100 or another arm), save the
                 adapter, then evaluate it on val_eval_400
  eval_test      refused while configs/eval.yaml test.sealed is true (the default)
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

from .io import sha256_file
from .phase3a import _dump, _load, data_and_mask_checks, evaluate_model

MODES = ("preflight", "eval_base_val", "train_eval", "eval_test")
ARMS = ("random_1k", "diversity_1k")


class SealedTestError(RuntimeError):
    pass


def check_request(mode: str, run_id: str | None, eval_cfg: dict[str, Any]) -> None:
    if mode not in MODES:
        raise ValueError(f"MODE must be one of {MODES}")
    if mode == "eval_test" and eval_cfg.get("test", {}).get("sealed", True):
        raise SealedTestError("test_eval_1000 is SEALED until the evaluation protocol and "
                              "training configs are frozen and the researcher unseals it.")
    if mode in ("train_eval", "preflight") and run_id not in ARMS:
        raise ValueError(f"RUN_ID must be one of {ARMS} for MODE={mode}")
    if mode == "eval_base_val" and run_id is not None:
        raise ValueError("eval_base_val evaluates the untuned base; set RUN_ID = None")


def _verify(dd: Path, manifest: dict[str, Any], run_id: str | None) -> None:
    p3b = manifest["phase3b"]
    if sha256_file(dd / p3b["val_eval_400"]["file"]) != p3b["val_eval_400"]["sha256"]:
        raise RuntimeError("val_eval_400.jsonl sha256 mismatch")
    if run_id and sha256_file(dd / p3b["arms"][run_id]["file"]) != p3b["arms"][run_id]["sha256"]:
        raise RuntimeError(f"train_{run_id}.jsonl sha256 mismatch")


def run(mode: str, run_id: str | None, data_dir: str, out_dir: str, train_cfg: dict[str, Any],
        eval_cfg: dict[str, Any]) -> dict[str, Any]:
    check_request(mode, run_id, eval_cfg)
    import torch
    import transformers
    from transformers import AutoTokenizer

    from .hardware import preflight
    from .train import TrainConfig, add_lora, load_quantized_model, run_training, trainable_summary

    t_start = time.perf_counter()
    dd, out = Path(data_dir), Path(out_dir) / (run_id or "base")
    out.mkdir(parents=True, exist_ok=True)
    rep: dict[str, Any] = {"mode": mode, "run_id": run_id, "phase": "3B/4",
                           "preflight": preflight(str(out), require_gpu=True)}
    manifest = json.loads((dd / "manifest.json").read_text("utf-8"))
    _verify(dd, manifest, run_id)
    val = _load(dd / "val_eval_400.jsonl")
    train_recs = _load(dd / f"train_{run_id}.jsonl") if run_id else []
    cfg = TrainConfig.from_yaml(train_cfg)
    cfg.save_steps = int(train_cfg["checkpointing"]["arm_save_steps"])
    model_id, rev = train_cfg["model_name"], train_cfg["model_revision"]
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
    examples, rep["data_checks"] = data_and_mask_checks(
        tok, train_recs or val[:1], val, manifest, cfg.max_seq_len)
    if not run_id:
        examples = []
    rep["train_config"] = cfg.__dict__
    _dump(out / "report.json", rep)
    if mode == "preflight":
        rep["status"] = "preflight_ok"
        _dump(out / "report.json", rep)
        return rep

    base = load_quantized_model(model_id, rev)
    if mode == "eval_base_val":
        rep["val_eval_400"], _ = evaluate_model(base, tok, val, out, "base_val400")
        rep["status"] = "eval_base_val_done"
    else:  # train_eval
        transformers.set_seed(cfg.seed)              # identical LoRA init for every arm
        model = add_lora(base, cfg, quantized=True)
        rep["lora"] = trainable_summary(model)
        ckpt = out / "checkpoints"
        resume = any(ckpt.glob("checkpoint-*"))      # same-session restart -> true resume
        try:
            tr, trainer = run_training(model, tok, examples, str(ckpt), cfg, use_gpu=True,
                                       resume=resume)
        except torch.cuda.OutOfMemoryError as e:
            rep.update(status="cuda_oom", oom_detail=str(e)[:500],
                       oom_peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 3))
            _dump(out / "report.json", rep)
            raise
        steps = tr["steps"]
        sec = sum(s["step_seconds"] for s in steps)
        rep["training"] = {
            "resumed": resume, "global_step": tr["global_step"], "max_steps": tr["max_steps"],
            "non_padding_tokens": tr["tokens_seen_non_padding"],
            "supervised_tokens": rep["data_checks"]["pilot_supervised_tokens"],
            "train_seconds": round(sec, 1), "wall_seconds": tr["wall_seconds"],
            "tokens_per_s": round(tr["tokens_seen_non_padding"] / sec, 1) if sec else None,
            "peak_vram_allocated_gb": tr.get("peak_vram_allocated_gb"),
            "peak_vram_reserved_gb": tr.get("peak_vram_reserved_gb"),
            "train_loss": tr["train_loss"],
            "losses": [s.get("loss") for s in steps],
            "grad_norms": [s.get("grad_norm") for s in steps],
        }
        adapter = out / "adapter"
        trainer.model.save_pretrained(str(adapter))
        _dump(out / "report.json", rep)
        # Identical inference path for every adapter (same as Phase 3A): fresh base + adapter.
        del trainer, model, base
        gc.collect()
        torch.cuda.empty_cache()
        from peft import PeftModel

        base = PeftModel.from_pretrained(load_quantized_model(model_id, rev), str(adapter))
        rep["val_eval_400"], _ = evaluate_model(base, tok, val, out, f"{run_id}_val400")
        rep["status"] = "train_eval_done"
    del base
    gc.collect()
    torch.cuda.empty_cache()
    rep["total_wall_minutes"] = round((time.perf_counter() - t_start) / 60, 1)
    _dump(out / "report.json", rep)
    return rep
