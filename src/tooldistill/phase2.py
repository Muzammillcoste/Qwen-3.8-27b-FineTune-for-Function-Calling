"""Phase 2 orchestration: single-call pool -> render -> features -> arms -> analysis -> audit sample.

No training, no model weights (tokenizer files only), no API calls.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .analysis import balance_vs, describe, overlap, wilson
from .format_qwen import RENDER_VERSION, load_chat_tokenizer, render, template_sha256, truncation_stats
from .io import iter_jsonl, resolve, write_jsonl
from .manifests import LICENSE_ACKNOWLEDGEMENT, code_commit
from .manual_audit import build_audit_sample, write_audit_files
from .schemas import NormalizedRecord
from .select import (
    ARMS,
    SELECTION_VERSION,
    build_pool,
    check_teacher_balance,
    record_features,
    select_all,
)
from .split import SPLITS


def load_split(norm_dir: Path, name: str) -> list[NormalizedRecord]:
    return [NormalizedRecord.from_dict(d) for d in iter_jsonl(norm_dir / "splits" / f"{name}.jsonl")]


def _single(records: list[NormalizedRecord]) -> list[NormalizedRecord]:
    return [r for r in records if r.call_count == 1 and r.duplicate_of is None]


def _status_table(recs: list[NormalizedRecord]) -> dict[str, Any]:
    n = len(recs)
    st = Counter(r.metadata.get("label_status") for r in recs)
    return {"n": n, "label_status": dict(sorted(st.items())),
            "strict_valid_pct": round(100 * sum(bool(r.metadata.get("strict_schema_valid"))
                                                for r in recs) / n, 2),
            "teacher": dict(sorted(Counter(r.teacher_source for r in recs).items()))}


def run_phase2(data_cfg: dict[str, Any], research_cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    nd = resolve(data_cfg["paths"]["normalized_dir"])
    splits = {s: load_split(nd, s) for s in SPLITS}
    pool_recs = build_pool(splits["train"])
    held = {s: _single(splits[s]) for s in ("val", "test")}
    multi_preserved = {s: sum(r.call_count > 1 for r in splits[s]) for s in SPLITS}

    t = data_cfg["tokenizer"]
    tok = load_chat_tokenizer(t["repo_id"], t["revision"])
    rendered = {}
    for r in pool_recs + held["val"] + held["test"]:
        rendered[r.sample_id] = render(r, tok)
    by_sid = {r.sample_id: r for r in pool_recs + held["val"] + held["test"]}

    def rows(recs):
        return [rendered[r.sample_id] for r in recs]

    render_checks = {
        "n": len(rendered),
        "prefix_ok": sum(x.prefix_ok for x in rendered.values()),
        "calls_rendered": sum(x.calls_rendered for x in rendered.values()),
    }
    trunc = {name: {L: truncation_stats(rows(recs), L) for L in (1024, 1536, 2048)}
             for name, recs in (("train_pool", pool_recs), ("val", held["val"]),
                                ("test", held["test"]))}

    # Length eligibility (decided on TRAIN lengths only): a record whose full render exceeds the
    # proposed training max length would have its assistant target truncated -> never eligible.
    max_tok = research_cfg["primary_pool"].get("max_rendered_tokens")
    over_length = [r.sample_id for r in pool_recs
                   if max_tok and rendered[r.sample_id].total_tokens > max_tok]
    pool_recs = [r for r in pool_recs if r.sample_id not in set(over_length)]

    feats = {r.sample_id: record_features(r, rendered[r.sample_id].total_tokens) for r in pool_recs}
    subsets, sel_params = select_all(feats, seed)
    max_dev = float(research_cfg["source_matching"]["max_teacher_ratio_deviation"])
    teacher_dev = check_teacher_balance(subsets, feats, max_dev)

    # never select held-out records / groups
    held_groups = {r.source_group_id for s in ("val", "test") for r in splits[s]}
    for s in subsets.values():
        assert not set(s.sample_ids) & {r.sample_id for r in held["val"] + held["test"]}
        assert not {feats[i]["group"] for i in s.sample_ids} & held_groups

    cuts = sel_params["length_cuts"]
    completion = {sid: x.completion_tokens for sid, x in rendered.items()}
    held_fn = Counter(r.expected_calls[0].name for r in held["test"])
    pool_ids = sorted(feats)
    tables = {"pool": describe(pool_ids, feats, cuts, held_fn, completion)}
    for name in ARMS:
        tables[name] = describe(subsets[name].sample_ids, feats, cuts, held_fn, completion)
    balance = {name: {"vs_pool": balance_vs(subsets[name].sample_ids, pool_ids, feats),
                      "vs_random_1k": balance_vs(subsets[name].sample_ids,
                                                 subsets["random_1k"].sample_ids, feats)}
               for name in ARMS}
    overlaps = {f"{a}~{b}": overlap(subsets[a].sample_ids, subsets[b].sample_ids)
                for i, a in enumerate(ARMS) for b in ARMS[i + 1:]}

    # schema-consistent arm: does it differ meaningfully from random_1k?
    r1, sc = set(subsets["random_1k"].sample_ids), set(subsets["schema_consistent_1k"].sample_ids)
    n_incons_pool = sum(not f["schema_consistent"] for f in feats.values())
    gate = research_cfg["schema_arm_gate"]
    differing = len(r1 - sc)
    schema_gate = {
        "pool_inconsistent": n_incons_pool,
        "pool_inconsistent_pct": round(100 * n_incons_pool / len(feats), 2),
        "pool_inconsistent_pct_95ci": wilson(n_incons_pool, len(feats)),
        "pool_inconsistent_by_status": dict(Counter(
            f["label_status"] for f in feats.values() if not f["schema_consistent"])),
        "random_1k_inconsistent": sum(not feats[i]["schema_consistent"] for i in r1),
        "schema_consistent_1k_inconsistent": sum(not feats[i]["schema_consistent"] for i in sc),
        "records_differing": differing,
        "records_differing_pct": round(100 * differing / len(r1), 2),
        "min_difference_pct": gate["min_difference_pct"],
        "meaningful": 100 * differing / len(r1) >= gate["min_difference_pct"],
    }

    # pool-level recalculation (single-call subset)
    group_sizes = Counter(f["group"] for f in feats.values())
    pool_stats = {
        "train_single_call": _status_table(pool_recs),
        "val_single_call": _status_table(held["val"]),
        "test_single_call": _status_table(held["test"]),
        "multi_call_preserved_not_used": multi_preserved,
        "max_rendered_tokens": max_tok,
        "excluded_over_length": len(over_length),
        "excluded_over_length_ids": over_length,
        "pool_groups": len(group_sizes),
        "pool_records_in_multi_member_groups": sum(c for c in group_sizes.values() if c > 1),
        "pool_largest_group": max(group_sizes.values()),
        "pool_distinct_functions": len({f["function"] for f in feats.values()}),
        "test_functions_unseen_in_pool_pct": round(100 * sum(
            c for f_, c in held_fn.items() if f_ not in {f["function"] for f in feats.values()})
            / sum(held_fn.values()), 2),
        "length_cuts_rendered_tokens": cuts,
    }

    # ---------------------------------------------------------------- outputs
    sel_dir = resolve(data_cfg["paths"]["selected_dir"])
    man_dir = resolve("reports/phase2_manifests")
    man_dir.mkdir(parents=True, exist_ok=True)
    pool_by_id = {r.sample_id: r for r in pool_recs}
    tpl_sha = template_sha256(tok)
    manifests = {}
    for name, s in subsets.items():
        write_jsonl(sel_dir / f"train_{name}.jsonl",
                    (pool_by_id[i].to_dict() for i in sorted(s.sample_ids)))
        manifests[name] = {
            "arm": name, "n": len(s.sample_ids), "algorithm": s.algorithm,
            "selection_version": SELECTION_VERSION, "seed": seed,
            "sample_ids_sha256": s.ids_sha256, "sample_ids": sorted(s.sample_ids),
            "source_dataset": data_cfg["source"]["repo_id"],
            "source_revision": data_cfg["source"]["revision"],
            "render_version": RENDER_VERSION, "chat_template_sha256": tpl_sha,
            "tokenizer": f"{t['repo_id']}@{t['revision']}",
            "tokens_total": tables[name]["tokens_sum_total"],
            "tokens_completion": tables[name]["tokens_sum_completion"],
            "teacher_deepseek_pct": tables[name]["teacher_deepseek_pct"],
            "code_commit": code_commit(), "license_acknowledgement": LICENSE_ACKNOWLEDGEMENT,
        }
        (man_dir / f"{name}.json").write_text(json.dumps(manifests[name], indent=1) + "\n",
                                              "utf-8")
    write_jsonl(resolve(data_cfg["paths"]["audits_dir"]) / "rendered_lengths.jsonl",
                ({"sample_id": x.sample_id, "prompt": x.prompt_tokens,
                  "completion": x.completion_tokens, "total": x.total_tokens,
                  "split": by_sid[x.sample_id].split} for x in rendered.values()))

    # blinded manual audit sample
    one_k = {a: subsets[a].sample_ids for a in ("random_1k", "schema_consistent_1k",
                                                "diversity_1k")}
    in_any = set().union(*map(set, one_k.values()))
    diag_pool = {
        "schema_conflict": [i for i, f in feats.items()
                            if f["label_status"] == "schema_conflict" and i not in in_any],
        "ambiguous": [i for i, f in feats.items()
                      if f["label_status"] == "ambiguous" and i not in in_any],
    }
    aud = research_cfg["semantic_audit"]
    items = build_audit_sample(one_k, pool_by_id, diag_pool, seed, aud["per_arm"],
                               aud["diagnostic"])
    audit_files = write_audit_files(items, pool_by_id,
                                    Path(data_cfg["paths"]["audits_dir"]) / "semantic_audit")

    # compute-aware cost table (tokens are measured; throughput values are ASSUMPTIONS)
    tr = research_cfg["compute"]
    eff_batch = tr["per_device_batch"] * tr["grad_accum"]
    costs = {}
    for name in ARMS:
        n = len(subsets[name].sample_ids)
        tok_epoch = tables[name]["tokens_sum_total"]
        costs[name] = {"n": n, "steps_per_epoch": math.ceil(n / eff_batch),
                       "tokens_per_epoch": tok_epoch,
                       "hours_per_epoch_at_assumed_tok_s": {
                           str(v): round(tok_epoch / v / 3600, 2)
                           for v in tr["assumed_train_tokens_per_s"]}}
    n_test = len(held["test"])
    mean_prompt = sum(rendered[r.sample_id].prompt_tokens for r in held["test"]) / n_test
    mean_compl = sum(rendered[r.sample_id].completion_tokens for r in held["test"]) / n_test
    eval_cost = {"test_single_call_n": n_test, "mean_prompt_tokens": round(mean_prompt, 1),
                 "mean_reference_completion_tokens": round(mean_compl, 1),
                 "hours_per_model_full_test_at_assumed_gen_tok_s": {
                     str(v): round(n_test * mean_compl / v / 3600, 2)
                     for v in tr["assumed_generate_tokens_per_s_batch1"]}}

    return {
        "seed": seed, "pool_stats": pool_stats, "render_checks": render_checks,
        "chat_template_sha256": tpl_sha, "truncation": trunc, "selection_params": sel_params,
        "teacher_deviation_vs_pool": teacher_dev, "tables": tables, "balance": balance,
        "overlaps": overlaps, "schema_gate": schema_gate, "manifests": {
            k: {kk: vv for kk, vv in v.items() if kk != "sample_ids"} for k, v in manifests.items()},
        "audit": {"n_items": len(items), "files": {k: str(v) for k, v in audit_files.items()},
                  "strata": dict(Counter(s for e in items for s in e["strata"]))},
        "costs": costs, "eval_cost": eval_cost,
    }


# ------------------------------------------------------------------------------ report


def _row(cells: list[Any]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def render_phase2_md(res: dict[str, Any]) -> str:
    T, P = res["tables"], res["pool_stats"]
    cols = ["pool", *ARMS]
    L = ["# Phase 2 — Candidate Subsets and Selection Audit\n",
         "Descriptive analysis only. No training, no model weights (tokenizer files only), no "
         "quality score, no semantic-correctness claim. Framing: **Option A — data selection, "
         "diversity, coverage and quantity.** Option B (label noise) deferred.\n"]

    L.append("## 1. Primary pool (TRAIN ∩ single-call ∩ non-duplicate)\n")
    L.append(_row(["set", "n", "teacher", "label_status", "strict valid %"]))
    L.append(_row(["---"] * 5))
    for k in ("train_single_call", "val_single_call", "test_single_call"):
        v = P[k]
        L.append(_row([k, v["n"], v["teacher"], v["label_status"], v["strict_valid_pct"]]))
    L.append(f"\n- train row = eligible pool AFTER excluding {P['excluded_over_length']} records "
             f"whose full chat-template render exceeds {P['max_rendered_tokens']} tokens "
             "(their assistant target would be truncated); val/test rows are unfiltered.")
    L.append("- multi-call records preserved but unused (per split): "
             f"{P['multi_call_preserved_not_used']}. Multi-call = several calls in ONE "
             "assistant response; every xLAM record is a single user turn (no multi-turn data).")
    L.append(f"- pool groups: {P['pool_groups']}; records in multi-member (near-)duplicate groups: "
             f"{P['pool_records_in_multi_member_groups']}; largest group {P['pool_largest_group']}")
    L.append(f"- distinct called functions in pool: {P['pool_distinct_functions']}; "
             f"single-call TEST records whose function never appears in the pool: "
             f"{P['test_functions_unseen_in_pool_pct']}%")
    L.append(f"- rendered-length tertile cuts (train pool): {P['length_cuts_rendered_tokens']}\n")

    rc = res["render_checks"]
    L.append("## 2. Final chat-template token lengths (real Qwen template)\n")
    L.append(f"- chat template sha256 `{res['chat_template_sha256'][:16]}…`; "
             f"prompt is exact token prefix of full render: {rc['prefix_ok']}/{rc['n']}; "
             f"every expected call rendered verbatim: {rc['calls_rendered']}/{rc['n']}\n")
    L.append(_row(["set", "max_len", "truncated", "truncated %", "prompt alone >= max_len"]))
    L.append(_row(["---"] * 5))
    for name, by_len in res["truncation"].items():
        for lmax, s in by_len.items():
            L.append(_row([name, lmax, s["truncated"], s["truncated_pct"],
                           s["prompt_alone_exceeds"]]))
    L.append("\nRight-truncation always cuts the assistant target (it is last). Records over the "
             "limit must be excluded from training or the limit raised; never train on cut targets.\n")

    def table(title: str, keys: list[tuple[str, str]]):
        L.append(f"### {title}\n")
        L.append(_row(["metric", *cols]))
        L.append(_row(["---"] * (len(cols) + 1)))
        for key, label in keys:
            L.append(_row([label, *[T[c][key] for c in cols]]))
        L.append("")

    L.append("## 3. Comparison tables\n")
    table("Sources and tokens", [("n", "records"), ("teacher_deepseek_pct", "DeepSeek %"),
                                 ("tokens_sum_total", "rendered tokens (sum)"),
                                 ("tokens_sum_completion", "target tokens (sum)"),
                                 ("tokens_mean", "tokens mean"), ("tokens_p50", "tokens p50"),
                                 ("tokens_p90", "tokens p90"), ("tokens_max", "tokens max")])
    table("Difficulty proxies", [("tools_bucket", "available tools"), ("args_bucket", "call args"),
                                 ("length_bin", "length tertile"), ("mean_n_tools", "mean tools"),
                                 ("mean_n_args", "mean args"),
                                 ("structured_arg_pct", "list/dict arg %")])
    table("Quality warnings (descriptive; not semantic correctness)",
          [("label_status", "label_status"), ("strict_schema_valid_pct", "strict valid %"),
           ("schema_consistent_pct", "schema-consistent %"),
           ("type_conflict_records", "type-conflict records"),
           ("warning_records", "records with warnings"),
           ("default_mismatch_records_pct", "default-mismatch records %"),
           ("unknown_param_records_pct", "UNKNOWN-param records %")])
    table("Coverage and diversity",
          [("distinct_functions", "distinct functions"),
           ("distinct_arg_signatures", "distinct arg signatures"),
           ("distinct_groups", "distinct dedup groups"),
           ("function_entropy_bits", "function entropy (bits)"),
           ("max_records_one_function", "max records / function"),
           ("heldout_records_function_seen_pct", "test records w/ function seen %"),
           ("query_word_types", "query word types"),
           ("query_type_token_ratio", "query type/token"),
           ("mean_pairwise_query_jaccard", "mean pairwise query Jaccard (lower = more diverse)")])

    L.append("### Balance (standardized mean difference; |SMD| < 0.1 = balanced)\n")
    feats = list(next(iter(res["balance"].values()))["vs_pool"])
    L.append(_row(["arm", "reference", *feats]))
    L.append(_row(["---"] * (len(feats) + 2)))
    for arm, b in res["balance"].items():
        for ref in ("vs_pool", "vs_random_1k"):
            L.append(_row([arm, ref, *[b[ref][f] for f in feats]]))
    L.append(f"\n- teacher share deviation vs pool: {res['teacher_deviation_vs_pool']}")
    L.append("\n### Overlap between arms\n")
    L.append(_row(["pair", "intersection", "Jaccard"]))
    L.append(_row(["---"] * 3))
    for k, v in res["overlaps"].items():
        L.append(_row([k, v["intersection"], v["jaccard"]]))

    g = res["schema_gate"]
    L.append("\n## 4. Schema-consistent arm: does it differ meaningfully from random_1k?\n")
    L.append(f"- pool schema-inconsistent: {g['pool_inconsistent']} "
             f"({g['pool_inconsistent_pct']}%, 95% CI {g['pool_inconsistent_pct_95ci']}) by "
             f"status {g['pool_inconsistent_by_status']}")
    L.append(f"- inconsistent records in random_1k: {g['random_1k_inconsistent']}; in "
             f"schema_consistent_1k: {g['schema_consistent_1k_inconsistent']}")
    L.append(f"- records differing between the two arms: **{g['records_differing']} "
             f"({g['records_differing_pct']}%)**; gate threshold {g['min_difference_pct']}% -> "
             f"meaningful: **{g['meaningful']}**")
    L.append("- Schema consistency is NOT evidence of higher semantic correctness; only the "
             "blinded human audit can test that.\n")

    L.append("## 5. Manifests\n")
    L.append(_row(["arm", "n", "algorithm", "sample_ids_sha256", "rendered tokens"]))
    L.append(_row(["---"] * 5))
    for k, m in res["manifests"].items():
        L.append(_row([k, m["n"], m["algorithm"], f"`{m['sample_ids_sha256'][:16]}…`",
                       m["tokens_total"]]))
    L.append(f"\nFull ID lists: `reports/phase2_manifests/<arm>.json`. Selection params: "
             f"{res['selection_params']['n_strata']} strata, seed {res['seed']}.\n")

    a = res["audit"]
    L.append("## 6. Blinded semantic audit sample\n")
    L.append(f"- {a['n_items']} unique items; strata (an item can serve several arms): "
             f"{a['strata']}")
    L.append("- annotator sheet and hidden key in `data/audits/semantic_audit/` (gitignored; "
             "contains xLAM text). Protocol: `docs/SEMANTIC_AUDIT_PROTOCOL.md`.\n")

    L.append("## 7. Training-cost estimates (tokens measured; throughput ASSUMED)\n")
    L.append(_row(["arm", "n", "steps/epoch (eff. batch 8)", "tokens/epoch",
                   "hours/epoch @ assumed tok/s"]))
    L.append(_row(["---"] * 5))
    for k, c in res["costs"].items():
        L.append(_row([k, c["n"], c["steps_per_epoch"], c["tokens_per_epoch"],
                       c["hours_per_epoch_at_assumed_tok_s"]]))
    e = res["eval_cost"]
    L.append(f"\n- evaluation: {e['test_single_call_n']} single-call test records, mean prompt "
             f"{e['mean_prompt_tokens']} tokens, mean reference completion "
             f"{e['mean_reference_completion_tokens']} tokens; hours per model (batch 1) at assumed "
             f"generation speed: {e['hours_per_model_full_test_at_assumed_gen_tok_s']}")
    return "\n".join(L) + "\n"
