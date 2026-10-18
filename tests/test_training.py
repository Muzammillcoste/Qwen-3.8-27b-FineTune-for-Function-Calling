"""Phase 3A mechanics on CPU with a TINY RANDOMLY-INITIALISED Qwen3 model (no weights downloaded).

Uses the real Qwen tokenizer from the local HF cache (skipped if unavailable). Verifies: loss
mask, padding mask, finite training, checkpoint + true resume into a fresh model, NaN stop,
adapter save/load, generation plumbing, and safe preflight failure (spec L #15, #17).
Speeds measured here are meaningless; only mechanics are tested.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from tooldistill.format_qwen import (  # noqa: E402
    IGNORE_INDEX,
    OverLengthError,
    build_training_example,
    prompt_ids,
)
from tooldistill.io import load_yaml  # noqa: E402
from tooldistill.train import (  # noqa: E402
    MonitorCallback,
    PadCollator,
    StepLog,
    TrainConfig,
    add_lora,
    run_training,
)


@pytest.fixture(scope="module")
def tok():
    t = load_yaml("configs/data.yaml")["tokenizer"]
    try:
        return transformers.AutoTokenizer.from_pretrained(t["repo_id"], revision=t["revision"],
                                                          local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not in local HF cache")


def tiny_model(seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=8, tie_word_embeddings=True, max_position_embeddings=4096)
    return Qwen3ForCausalLM(cfg)


@pytest.fixture(scope="module")
def examples(tok, request):
    by_id = request.getfixturevalue("by_id_module")
    ids = ["fixture:0", "fixture:33659@row3", "fixture:33700", "fixture:40003"]
    return [build_training_example(by_id[i], tok, 1536) for i in ids]


@pytest.fixture(scope="module")
def by_id_module():
    from tooldistill.io import read_records
    from tooldistill.normalize_xlam import normalize_rows

    from .conftest import FIXTURE, FIXTURE_OPTS

    recs, _ = normalize_rows(read_records(FIXTURE), FIXTURE_OPTS)
    return {r.sample_id: r for r in recs}


def test_loss_mask_prompt_masked_target_active(tok, by_id_module):
    rec = by_id_module["fixture:40003"]
    e = build_training_example(rec, tok, 1536)
    n = e["n_prompt"]
    assert all(x == IGNORE_INDEX for x in e["labels"][:n])
    target = [x for x in e["labels"] if x != IGNORE_INDEX]
    assert len(target) == e["n_target"]
    text = tok.decode(target)
    assert text.startswith("<tool_call>") and text.endswith("<|im_end|>")
    assert "weather.forecast.v2.get" in text and rec.query not in text
    assert e["labels"][-1] == IGNORE_INDEX  # newline after EOS is not supervised
    assert prompt_ids(rec, tok) == e["input_ids"][:n]  # eval prompt == train prefix


def test_over_length_is_refused_not_truncated(tok, by_id_module):
    with pytest.raises(OverLengthError):
        build_training_example(by_id_module["fixture:40016"], tok, 50)


def test_collator_masks_padding(examples, tok):
    b = PadCollator(tok.pad_token_id)([examples[0], examples[2]])
    short = 0 if len(examples[0]["input_ids"]) < len(examples[2]["input_ids"]) else 1
    pad_pos = b["attention_mask"][short] == 0
    assert pad_pos.any()
    assert (b["labels"][short][pad_pos] == IGNORE_INDEX).all()


def _lora_state(model):
    return {k: v.detach().clone() for k, v in model.named_parameters() if "lora_" in k}


def test_train_checkpoint_and_true_resume_match_uninterrupted(tok, examples, tmp_path):
    cfg = TrainConfig(grad_accum=2, save_steps=1, lora_dropout=0.0, epochs=1)
    transformers.set_seed(42)
    full = add_lora(tiny_model(), cfg, quantized=False)
    res_full, _ = run_training(full, tok, examples, str(tmp_path / "full"), cfg, use_gpu=False)
    assert res_full["global_step"] == 2
    assert all(math.isfinite(s["loss"]) for s in res_full["steps"] if "loss" in s)
    assert res_full["tokens_seen_non_padding"] == sum(len(e["input_ids"]) for e in examples)
    assert all(s["tokens"] > 0 and s["step_seconds"] > 0 for s in res_full["steps"])

    transformers.set_seed(42)
    m1 = add_lora(tiny_model(), cfg, quantized=False)
    part1, _ = run_training(m1, tok, examples, str(tmp_path / "res"), cfg, use_gpu=False,
                            interrupt_at=1)
    assert part1["global_step"] == 1
    ck = tmp_path / "res" / "checkpoint-1"
    files = {p.name for p in ck.iterdir()}
    assert {"optimizer.pt", "scheduler.pt"} <= files and any(f.startswith("rng") for f in files)
    # fresh model with DIFFERENT random LoRA init: weights must come from the checkpoint
    transformers.set_seed(123)
    m2 = add_lora(tiny_model(), cfg, quantized=False)
    part2, _ = run_training(m2, tok, examples, str(tmp_path / "res"), cfg, use_gpu=False,
                            resume=True)
    assert part2["global_step"] == 2
    a, b = _lora_state(full), _lora_state(m2)
    assert a.keys() == b.keys()
    assert all(torch.allclose(a[k], b[k], atol=1e-5) for k in a)


def test_adapter_save_and_reload_reproduces_outputs(tok, examples, tmp_path):
    from peft import PeftModel

    cfg = TrainConfig(grad_accum=2, save_steps=10, lora_dropout=0.0)
    transformers.set_seed(42)
    m = add_lora(tiny_model(), cfg, quantized=False)
    run_training(m, tok, examples, str(tmp_path / "t"), cfg, use_gpu=False)
    m.save_pretrained(str(tmp_path / "adapter"))
    reloaded = PeftModel.from_pretrained(tiny_model(), str(tmp_path / "adapter"))
    x = torch.tensor([examples[0]["input_ids"][:40]])
    m.eval()
    reloaded.eval()
    with torch.no_grad():
        assert torch.allclose(m(input_ids=x).logits, reloaded(input_ids=x).logits, atol=1e-5)


def test_lora_targets_exclude_lm_head():
    cfg = TrainConfig()
    m = add_lora(tiny_model(), cfg, quantized=False)
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)
    assert not any("lm_head" in n or "embed_tokens" in n for n in names)
    assert {str(p.dtype) for p in m.parameters() if p.requires_grad} == {"torch.float32"}


def test_phase3a_data_checks_mask_and_prompt_consistency(tok, by_id_module):
    import hashlib

    from tooldistill.phase3a import data_and_mask_checks

    pilot = [by_id_module[i] for i in ("fixture:0", "fixture:33700")]
    val = [by_id_module[i] for i in ("fixture:40003", "fixture:40002")]
    manifest = {"chat_template_sha256": hashlib.sha256(tok.chat_template.encode()).hexdigest()}
    ex, rep = data_and_mask_checks(tok, pilot, val, manifest, 1536)
    assert len(ex) == 2 and rep["eval_prompt_equals_train_prefix"]
    assert rep["example_supervised_text"].startswith("<tool_call>")
    assert 0 < rep["supervised_fraction"] < 0.5
    with pytest.raises(RuntimeError, match="prompt inconsistency"):
        data_and_mask_checks(tok, pilot, val, {"chat_template_sha256": "0" * 64}, 1536)


def test_report_markdown_renders_partial_and_full_reports():
    from tooldistill.phase3a import render_report_md

    assert "status `preflight_ok`" in render_report_md({"status": "preflight_ok",
                                                        "preflight": {"gpu_name": "Tesla T4"}})
    agg = {"exact_structured_call_accuracy": {"k": 5, "n": 10, "rate": 0.5},
           "parse_rate": {"rate": 1.0}, "tool_name_accuracy": {"rate": 0.9},
           "schema_valid_prediction_rate": {"rate": 0.9}, "hallucinated_tool_rate": {"rate": 0.0},
           "generation": {"seconds_per_example": 2.0, "new_tokens_per_s": 20.0}}
    md = render_report_md({"status": "x", "baseline_val100": agg, "pilot_val100": agg})
    assert "| baseline (100 val, base) | 5/10 |" in md


def test_projection_math():
    from tooldistill.phase3a import projections

    p = projections(500.0, 2.0, {"random_1k": 554_547, "diversity_1k": 555_718,
                                 "random_3k": 1_671_748})
    assert p["training_hours"]["random_1k (1 ep)"] == round(554_547 / 500 / 3600, 2)
    assert p["eval_hours_per_model"]["1000"] == round(1000 * 2.0 / 3600, 2)


def test_yaml_config_maps_to_train_config():
    cfg = TrainConfig.from_yaml(load_yaml("configs/train_qlora.yaml"))
    assert (cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout) == (8, 16, 0.05)
    assert (cfg.per_device_batch, cfg.grad_accum, cfg.max_seq_len) == (1, 8, 1536)
    assert cfg.optim_gpu == "paged_adamw_8bit" and cfg.warmup_ratio == 0.03 and cfg.seed == 42
    from tooldistill.train import training_args

    a = training_args("unused_out", cfg, use_gpu=False)
    assert a.lr_scheduler_type == "cosine" and a.gradient_accumulation_steps == 8


def test_nonfinite_loss_stops_training():
    log = StepLog()
    cb = MonitorCallback(log)
    log.rows.append({"step": 3})
    control = SimpleNamespace(should_training_stop=False)
    cb.on_log(None, SimpleNamespace(global_step=3), control, logs={"loss": float("nan")})
    assert control.should_training_stop and log.nonfinite == [3]


def test_generation_plumbing(tok, by_id_module):
    from tooldistill.infer import generate

    rec = by_id_module["fixture:0"]
    gens = generate(tiny_model(), tok, [(rec.sample_id, prompt_ids(rec, tok))] * 2,
                    max_new_tokens=5, batch_size=2)
    assert len(gens) == 2 and all(0 < g.new_tokens <= 5 for g in gens)


class _FakeCuda:
    def __init__(self, avail, free_gb=15.0):
        self.avail, self.free = avail, free_gb

    def is_available(self):
        return self.avail

    def device_count(self):
        return 1

    def get_device_name(self, i):
        return "Tesla T4"

    def get_device_capability(self, i):
        return (7, 5)

    def mem_get_info(self, i):
        return int(self.free * 1e9), int(15.6e9)

    def is_bf16_supported(self):
        return False


def test_preflight_fails_safely_without_gpu_or_memory(tmp_path):
    from tooldistill.hardware import InsufficientGPUMemoryError, NoGPUError, preflight

    with pytest.raises(NoGPUError):
        preflight(str(tmp_path), cuda=_FakeCuda(False))
    with pytest.raises(InsufficientGPUMemoryError):
        preflight(str(tmp_path), cuda=_FakeCuda(True, free_gb=4.0))
    ok = preflight(str(tmp_path), cuda=_FakeCuda(True))
    assert ok["status"] == "ok" and ok["bf16_supported"] is False
