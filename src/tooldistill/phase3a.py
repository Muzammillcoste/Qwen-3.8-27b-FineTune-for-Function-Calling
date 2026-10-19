"""Phase 3A: Kaggle technical feasibility (one T4). Run from the Kaggle notebook.

MODE:
  preflight     environment + data + tokenizer/mask checks only (no model download)
  format_check  + load 4-bit base, generate on 20 val examples, parse
  baseline      + base model on the fixed 100-example val sample
  phase3a       everything: format check, baseline, 100-example QLoRA pilot with a simulated
                interruption + true resume, adapter reload, pilot eval on the SAME val sample,
                measured throughput -> GPU-time projections for the remaining runs

The 100-example pilot is a feasibility check. It is NOT an experimental result.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .evaluate import aggregate, paired_comparison, score_example, score_to_dict
from .format_qwen import build_training_example, prompt_ids
from .io import iter_jsonl, sha256_file, write_jsonl
from .schemas import NormalizedRecord

MODES = ("preflight", "format_check", "baseline", "phase3a")


def _load(path: Path) -> list[NormalizedRecord]:
    return [NormalizedRecord.from_dict(d) for d in iter_jsonl(path)]


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n", "utf-8")
    if path.name == "report.json":
        (path.parent / "report.md").write_text(render_report_md(obj), "utf-8")


def render_report_md(rep: dict[str, Any]) -> str:
    """Human-readable summary. Every number here is MEASURED in this run unless labelled."""
    L = [f"# Phase 3A report — status `{rep.get('status', 'in_progress')}`\n",
         "Feasibility only; pilot numbers are NOT experimental findings.\n"]
    pf = rep.get("preflight", {})
    L.append(f"- GPU: {pf.get('gpu_name')} (cc {pf.get('compute_capability')}), total "
             f"{pf.get('gpu_total_gb')} GB, free {pf.get('gpu_free_gb')} GB, bf16 "
             f"{pf.get('bf16_supported')}; torch {pf.get('packages', {}).get('torch')}, "
             f"transformers {pf.get('packages', {}).get('transformers')}, bitsandbytes "
             f"{pf.get('packages', {}).get('bitsandbytes')}")
    dc = rep.get("data_checks", {})
    if dc:
        L.append(f"- data/mask checks: template sha `{dc['chat_template_sha256'][:12]}`, "
                 f"{dc['pilot_examples']} pilot examples, supervised tokens "
                 f"{dc['pilot_supervised_tokens']} ({100 * dc['supervised_fraction']:.1f}% of "
                 f"tokens), eval prompt == train prefix: {dc['eval_prompt_equals_train_prefix']}")
    if "model_load" in rep:
        L.append(f"- model load: {rep['model_load']['seconds']} s, VRAM after load "
                 f"{rep['model_load']['vram_after_load_gb']} GB")
    rows = [("format_check (20 val, base)", "format_check"),
            ("baseline (100 val, base)", "baseline_val100"),
            ("pilot adapter (100 val)", "pilot_val100"),
            (f"{rep.get('run_id') or 'base'} on val_eval_400", "val_eval_400")]
    tr = rep.get("training")
    if tr:
        L.append(f"- training ({rep.get('run_id')}): steps {tr['global_step']}/{tr['max_steps']} "
                 f"(resumed: {tr['resumed']}), {tr['non_padding_tokens']} tokens "
                 f"(supervised {tr['supervised_tokens']}) in {tr['train_seconds']} s = "
                 f"**{tr['tokens_per_s']} tok/s**, **peak VRAM {tr['peak_vram_allocated_gb']} GB** "
                 f"(reserved {tr['peak_vram_reserved_gb']}), train loss {tr['train_loss']:.4f}")
        L.append(f"- losses: {tr['losses']}\n- grad norms: {tr['grad_norms']}")
    if any(k in rep for _, k in rows):
        L.append("\n| eval | exact acc | parse | tool name | schema-valid pred | hallucinated "
                 "| s/example | new tok/s |\n|---|---|---|---|---|---|---|---|")
        for label, k in rows:
            if k in rep:
                a = rep[k]
                g = a["generation"]
                L.append(f"| {label} | {a['exact_structured_call_accuracy']['k']}/"
                         f"{a['exact_structured_call_accuracy']['n']} | "
                         f"{a['parse_rate']['rate']} | {a['tool_name_accuracy']['rate']} | "
                         f"{a['schema_valid_prediction_rate']['rate']} | "
                         f"{a['hallucinated_tool_rate']['rate']} | {g['seconds_per_example']} | "
                         f"{g['new_tokens_per_s']} |")
    pt = rep.get("pilot_training", {})
    if pt:
        m = pt["measured"]
        L.append(f"\n- pilot training: steps {m['optimizer_steps']} (interrupted at "
                 f"{pt['interrupted_at_step']}, resumed to {pt['resumed_to_step']}/"
                 f"{pt['max_steps']}; optimizer/scheduler/RNG restored: "
                 f"{pt['resume_restored_optimizer_scheduler_rng']}), {m['non_padding_tokens']} "
                 f"tokens in {m['train_seconds_excl_eval']} s = **{m['tokens_per_s']} tok/s**, "
                 f"mean step {m['mean_step_seconds']} s, **peak VRAM {m['peak_vram_allocated_gb']} "
                 f"GB** (reserved {m['peak_vram_reserved_gb']} GB)")
        L.append(f"- losses: {m['losses']}\n- grad norms: {m['grad_norms']}")
    if "paired_base_vs_pilot" in rep:
        L.append(f"- paired exact-match (base vs pilot, same 100): {rep['paired_base_vs_pilot']}")
    if "projections" in rep:
        p = rep["projections"]
        L.append(f"\n**Projections (ESTIMATES from measured pilot throughput; {p['basis']})**")
        L.append(f"- training hours: {p['training_hours']}")
        L.append(f"- eval hours per model by #examples: {p['eval_hours_per_model']}")
    if rep.get("status") == "cuda_oom":
        L.append(f"\n**CUDA OOM** — {rep.get('oom_detail')} (peak reserved "
                 f"{rep.get('oom_peak_reserved_gb')} GB). Not auto-mitigated.")
    if "total_wall_minutes" in rep:
        L.append(f"\n- total wall time: {rep['total_wall_minutes']} min")
    return "\n".join(L) + "\n"


def data_and_mask_checks(tok, pilot, val, manifest, max_len) -> tuple[list[dict], dict[str, Any]]:
    """Template hash, eval-prompt == train-prefix, label masking. Fails loudly."""
    sha = hashlib.sha256((tok.chat_template or "").encode()).hexdigest()
    if sha != manifest["chat_template_sha256"]:
        raise RuntimeError(f"chat template sha {sha[:12]} != manifest "
                           f"{manifest['chat_template_sha256'][:12]}: prompt inconsistency")
    examples, n_target, n_prompt = [], 0, 0
    for r in pilot:
        e = build_training_example(r, tok, max_len)
        assert prompt_ids(r, tok) == e["input_ids"][: e["n_prompt"]]
        assert all(x == -100 for x in e["labels"][: e["n_prompt"]])
        assert all(x != -100 for x in e["labels"][e["n_prompt"]: e["n_prompt"] + e["n_target"]])
        examples.append(e)
        n_target += e["n_target"]
        n_prompt += e["n_prompt"]
    target_text = tok.decode([x for x in examples[0]["labels"] if x != -100])
    for r in val:  # eval prompts must equal the training-format prefix
        e = build_training_example(r, tok, max_len)
        assert prompt_ids(r, tok) == e["input_ids"][: e["n_prompt"]]
    return examples, {
        "chat_template_sha256": sha, "pilot_examples": len(examples),
        "pilot_prompt_tokens": n_prompt, "pilot_supervised_tokens": n_target,
        "supervised_fraction": round(n_target / (n_target + n_prompt), 4),
        "example_supervised_text": target_text,
        "eval_prompt_equals_train_prefix": True,
    }


def evaluate_model(model, tok, recs, out: Path, tag: str, batch_size: int = 1):
    from .infer import generate

    items = [(r.sample_id, prompt_ids(r, tok)) for r in recs]
    gens = generate(model, tok, items, max_new_tokens=256, batch_size=batch_size)
    by = {r.sample_id: r for r in recs}
    scores = [score_example(by[g.sample_id], g.text) for g in gens]
    write_jsonl(out / f"predictions_{tag}.jsonl",
                ({**score_to_dict(s), "raw_output": g.text, "new_tokens": g.new_tokens,
                  "seconds": g.seconds, "hit_max_new_tokens": g.hit_max_new_tokens,
                  "run_id": tag} for s, g in zip(scores, gens, strict=True)))
    new_tok = sum(g.new_tokens for g in gens)
    secs = sum(g.seconds for g in gens)
    agg = aggregate(scores)
    agg["generation"] = {"batch_size": batch_size, "new_tokens": new_tok,
                         "seconds": round(secs, 2),
                         "new_tokens_per_s": round(new_tok / secs, 2) if secs else None,
                         "seconds_per_example": round(secs / len(gens), 3),
                         "hit_max_new_tokens": sum(g.hit_max_new_tokens for g in gens)}
    return agg, scores


def projections(train_tok_s: float, gen_s_per_example: float, arms: dict[str, int],
                eval_sizes=(100, 1000, 2813)) -> dict[str, Any]:
    """MEASURED-throughput projections (still estimates: assume pilot speed holds)."""
    runs = {
        "random_1k (1 ep)": arms["random_1k"], "diversity_1k (1 ep)": arms["diversity_1k"],
        "random_3k (1 ep)": arms["random_3k"], "random_1k x3 ep (compute-matched)": 3 * arms["random_1k"],
    }
    return {
        "basis": f"pilot train {train_tok_s:.1f} non-pad tokens/s; "
                 f"{gen_s_per_example:.2f} s/example generation (batch 1)",
        "training_hours": {k: round(v / train_tok_s / 3600, 2) for k, v in runs.items()},
        "eval_hours_per_model": {str(n): round(n * gen_s_per_example / 3600, 2)
                                 for n in eval_sizes},
    }


def run(mode: str, data_dir: str, out_dir: str, train_cfg: dict[str, Any],
        arm_tokens: dict[str, int] | None = None, interrupt_at: int = 6) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"MODE must be one of {MODES}")
    import torch
    from transformers import AutoTokenizer

    from .hardware import preflight
    from .train import (
        TrainConfig,
        add_lora,
        load_quantized_model,
        run_training,
        trainable_summary,
    )

    t_start = time.perf_counter()
    dd, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep: dict[str, Any] = {"mode": mode, "phase": "3A", "note":
                           "Feasibility only. Pilot metrics are NOT experimental findings."}
    rep["preflight"] = preflight(str(out), require_gpu=True)
    _dump(out / "report.json", rep)

    manifest = json.loads((dd / "manifest.json").read_text("utf-8"))
    for key in ("pilot_train", "val_sample"):
        f = dd / manifest[key]["file"]
        if sha256_file(f) != manifest[key]["sha256"]:
            raise RuntimeError(f"{f.name}: sha256 mismatch vs manifest")
    pilot, val = _load(dd / "pilot_train_100.jsonl"), _load(dd / "val_sample_100.jsonl")
    fmt = [r for r in val if r.sample_id in set(manifest["val_sample"]["format_check_ids"])]
    model_id, rev = train_cfg["model_name"], train_cfg["model_revision"]
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
    cfg = TrainConfig.from_yaml(train_cfg)
    if cfg.max_seq_len != int(manifest["max_seq_len"]):
        raise RuntimeError(f"max_seq_len {cfg.max_seq_len} != data manifest "
                           f"{manifest['max_seq_len']}: eligibility/training mismatch")
    rep["train_config"] = cfg.__dict__
    examples, rep["data_checks"] = data_and_mask_checks(tok, pilot, val, manifest, cfg.max_seq_len)
    _dump(out / "report.json", rep)
    if mode == "preflight":
        rep["status"] = "preflight_ok"
        _dump(out / "report.json", rep)
        return rep

    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    base = load_quantized_model(model_id, rev)
    rep["model_load"] = {"seconds": round(time.perf_counter() - t0, 1),
                         "vram_after_load_gb": round(torch.cuda.memory_allocated() / 1e9, 3)}
    rep["format_check"], _ = evaluate_model(base, tok, fmt, out, "base_format20")
    _dump(out / "report.json", rep)
    if mode == "format_check":
        rep["status"] = "format_check_done"
        _dump(out / "report.json", rep)
        return rep

    rep["baseline_val100"], base_scores = evaluate_model(base, tok, val, out, "base_val100")
    _dump(out / "report.json", rep)
    if mode == "baseline":
        rep["status"] = "baseline_done"
        _dump(out / "report.json", rep)
        return rep

    # ---------------- pilot: interrupted at `interrupt_at`, then resumed from the checkpoint
    model = add_lora(base, cfg, quantized=True)
    rep["lora"] = trainable_summary(model)
    ckpt_dir = str(out / "pilot_checkpoints")
    try:
        part1, trainer1 = run_training(model, tok, examples, ckpt_dir, cfg, use_gpu=True,
                                       interrupt_at=interrupt_at)
        # Simulated session end: discard EVERYTHING in memory, then resume from disk into a
        # freshly loaded base + freshly initialised LoRA (weights must come from the checkpoint).
        del trainer1, model, base
        gc.collect()
        torch.cuda.empty_cache()
        base = load_quantized_model(model_id, rev)
        model = add_lora(base, cfg, quantized=True)
        part2, trainer = run_training(model, tok, examples, ckpt_dir, cfg, use_gpu=True,
                                      resume=True)
    except torch.cuda.OutOfMemoryError as e:
        rep["status"] = "cuda_oom"
        rep["oom_detail"] = str(e)[:500]
        rep["oom_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
        _dump(out / "report.json", rep)
        raise
    ckpt_files = sorted(p.name for p in Path(ckpt_dir).glob(f"checkpoint-{interrupt_at}/*"))
    rep["pilot_training"] = {
        "interrupted_at_step": part1["global_step"], "resumed_to_step": part2["global_step"],
        "max_steps": part2["max_steps"],
        "checkpoint_files_at_interrupt": ckpt_files,
        "resume_restored_optimizer_scheduler_rng": all(
            any(f.startswith(x) for f in ckpt_files) for x in ("optimizer", "scheduler", "rng")),
        "part1": {k: v for k, v in part1.items() if k != "log_history"},
        "part2": {k: v for k, v in part2.items() if k != "log_history"},
    }
    steps = part1["steps"] + part2["steps"]
    tok_total = sum(s["tokens"] for s in steps)
    sec_total = sum(s["step_seconds"] for s in steps)
    rep["pilot_training"]["measured"] = {
        "optimizer_steps": len(steps), "non_padding_tokens": tok_total,
        "train_seconds_excl_eval": round(sec_total, 1),
        "tokens_per_s": round(tok_total / sec_total, 1) if sec_total else None,
        "mean_step_seconds": round(sec_total / len(steps), 2) if steps else None,
        "peak_vram_allocated_gb": max(part1.get("peak_vram_allocated_gb", 0),
                                      part2.get("peak_vram_allocated_gb", 0)),
        "peak_vram_reserved_gb": max(part1.get("peak_vram_reserved_gb", 0),
                                     part2.get("peak_vram_reserved_gb", 0)),
        "losses": [s.get("loss") for s in steps], "grad_norms": [s.get("grad_norm") for s in steps],
    }
    adapter_dir = out / "pilot_adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    _dump(out / "report.json", rep)

    # ---------------- adapter reload into a FRESH base (tests save/load path)
    del trainer, model, base
    gc.collect()
    torch.cuda.empty_cache()
    from peft import PeftModel

    base2 = load_quantized_model(model_id, rev)
    tuned = PeftModel.from_pretrained(base2, str(adapter_dir))
    rep["adapter_reload"] = {"ok": True, "adapter_files": sorted(p.name for p in adapter_dir.iterdir())}
    rep["pilot_val100"], pilot_scores = evaluate_model(tuned, tok, val, out, "pilot100_val100")
    rep["paired_base_vs_pilot"] = paired_comparison(base_scores, pilot_scores)

    m = rep["pilot_training"]["measured"]
    if arm_tokens and m["tokens_per_s"]:
        rep["projections"] = projections(m["tokens_per_s"],
                                         rep["baseline_val100"]["generation"]["seconds_per_example"],
                                         arm_tokens)
    rep["total_wall_minutes"] = round((time.perf_counter() - t_start) / 60, 1)
    rep["status"] = "phase3a_done"
    _dump(out / "report.json", rep)
    return rep
