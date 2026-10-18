"""Deterministic generation (do_sample=False) with fixed settings for base and adapters (spec I1).

Prompts come from ``format_qwen.prompt_ids`` -- the exact prefix used in training.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass
class Generation:
    sample_id: str
    text: str
    new_tokens: int
    seconds: float
    hit_max_new_tokens: bool


@torch.inference_mode()
def generate(model, tok, items: list[tuple[str, list[int]]], max_new_tokens: int = 256,
             batch_size: int = 1) -> list[Generation]:
    """items: (sample_id, prompt_ids). Left-padded batches; greedy; stops at <|im_end|>."""
    model.eval()
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    pad = tok.pad_token_id
    device = next(model.parameters()).device
    out: list[Generation] = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        n = max(len(p) for _, p in chunk)
        ids = torch.tensor([[pad] * (n - len(p)) + p for _, p in chunk], device=device)
        att = torch.tensor([[0] * (n - len(p)) + [1] * len(p) for _, p in chunk], device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen = model.generate(input_ids=ids, attention_mask=att, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             max_new_tokens=max_new_tokens, eos_token_id=eos_ids,
                             pad_token_id=pad)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        for j, (sid, _) in enumerate(chunk):
            new = gen[j, n:].tolist()
            # drop padding emitted after EOS for finished sequences in a batch
            n_new = next((k + 1 for k, t in enumerate(new) if t in eos_ids), len(new))
            text = tok.decode(new[:n_new], skip_special_tokens=False)
            out.append(Generation(sid, text, n_new, round(dt / len(chunk), 4),
                                  n_new >= max_new_tokens and new[n_new - 1] not in eos_ids))
    return out
