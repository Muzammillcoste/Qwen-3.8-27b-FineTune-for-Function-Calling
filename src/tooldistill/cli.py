"""Command-line entry points. Scripts in ``scripts/`` are thin wrappers around these."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .io import load_yaml, read_records, resolve

DEFAULT_DATA_CONFIG = "configs/data.yaml"


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


# ------------------------------------------------------------------ inspect


def cmd_inspect(argv: list[str] | None = None) -> int:
    from .hf_download import check_access, stream_sample
    from .inspection import build_inspection, render_markdown
    from .pipeline import options_from_config

    ap = argparse.ArgumentParser(
        prog="inspect_xlam",
        description="Inspect xLAM schema/IDs/fields. --source is an HF repo id OR a local "
                    "JSON/JSONL file (e.g. tests/fixtures/miniature_xlam.jsonl).",
    )
    ap.add_argument("--source", required=True)
    ap.add_argument("--sample", type=int, default=100, help="rows to inspect (HF: streamed)")
    ap.add_argument("--config", default=DEFAULT_DATA_CONFIG)
    ap.add_argument("--out", default=None, help="markdown report path")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    local = resolve(args.source)
    if local.exists():
        rows = read_records(local)[: args.sample]
        is_fixture = "fixtures" in local.parts
        revision = "local-file"
        label = local.name
        out = args.out or ("reports/fixture_inspection.md" if is_fixture
                           else cfg["paths"]["inspection_report"])
    else:
        revision = cfg["source"]["revision"]
        status = check_access(args.source, revision)
        if not status.ok:
            print(f"[inspect] access status: {status.status} (token present: "
                  f"{status.token_present})\n{status.message}", file=sys.stderr)
            return 3  # nonfatal, documented exit code for "access not granted"
        rows = stream_sample(args.source, revision, args.sample, cfg["source"]["split"])
        is_fixture, label = False, args.source
        out = args.out or cfg["paths"]["inspection_report"]

    opts = options_from_config(cfg, fixture=is_fixture)
    ins = build_inspection(rows, opts)
    md = render_markdown(ins, label, revision, is_fixture)
    p = resolve(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md, "utf-8")
    summary = {k: ins[k] for k in ("n_source_rows", "n_normalized", "n_rejected",
                                   "answers_count_distribution", "rejections_by_reason",
                                   "teacher_source_counts")}
    summary["report"] = str(Path(out).as_posix())
    summary["is_fixture"] = is_fixture
    _print(summary)
    return 0


# ------------------------------------------------------------------ access / download


def cmd_check_access(argv: list[str] | None = None) -> int:
    from .hf_download import check_access

    ap = argparse.ArgumentParser(prog="check_access",
                                 description="Check gated HF dataset access (never prints tokens)")
    ap.add_argument("--config", default=DEFAULT_DATA_CONFIG)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    st = check_access(cfg["source"]["repo_id"], cfg["source"]["revision"])
    _print({k: v for k, v in st.to_dict().items() if k != "message"})
    print(st.message)
    return 0 if st.ok else 3


def cmd_download(argv: list[str] | None = None) -> int:
    from .hf_download import download_raw

    ap = argparse.ArgumentParser(
        prog="download_xlam",
        description="PHASE 1: download the raw xLAM file at the pinned revision (requires accepted "
                    "access + HF login). Raw file is stored read-only under data/raw/<revision>/.",
    )
    ap.add_argument("--config", default=DEFAULT_DATA_CONFIG)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    s = cfg["source"]
    try:
        meta = download_raw(s["repo_id"], s["revision"], s["data_file"],
                            cfg["paths"]["raw_dir"], s["license"])
    except PermissionError as e:
        print(str(e), file=sys.stderr)
        return 3
    _print(meta)
    return 0


# ------------------------------------------------------------------ prepare


def cmd_prepare(argv: list[str] | None = None) -> int:
    from .pipeline import prepare

    ap = argparse.ArgumentParser(
        prog="prepare_dataset",
        description="normalize -> validate -> dedup/group -> split -> leakage audit. "
                    "Selection is NOT performed here (Phase 2).",
    )
    ap.add_argument("--config", default=DEFAULT_DATA_CONFIG)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--input", default=None,
                    help="raw JSON/JSONL; default data/raw/<revision>/<data_file>")
    ap.add_argument("--out", default=None, help="default: data/normalized")
    ap.add_argument("--fixture", action="store_true", help="mark outputs as fixture-derived")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    seed = args.seed if args.seed is not None else int(cfg["split"]["seed"])
    inp = args.input or (Path(cfg["paths"]["raw_dir"]) / cfg["source"]["revision"]
                         / cfg["source"]["data_file"])
    if not resolve(inp).exists():
        print(f"input not found: {inp}. Run scripts/download_xlam.py first (Phase 1).",
              file=sys.stderr)
        return 2
    out = args.out or cfg["paths"]["normalized_dir"]
    res = prepare(cfg, inp, out, seed, fixture=args.fixture)
    _print({"counts": res["manifest"]["counts"],
            "split_counts": {k: v["n_records"] for k, v in res["split_summary"]["splits"].items()},
            "dedup": {k: res["dedup_report"][k] for k in ("n_exact_duplicates", "n_groups",
                                                         "largest_group", "edges")},
            "leakage_ok": res["leakage_audit"]["ok"],
            "teacher_card_check": {k: res["provenance_check"].get(k)
                                   for k in ("computed", "ok", "message")},
            "out": str(out)})
    return 0


# ------------------------------------------------------------------ phase 1 audit


def cmd_audit_data(argv: list[str] | None = None) -> int:
    from .audit import build_audit, load_records, write_audit
    from .io import write_jsonl
    from .token_lengths import TOKEN_METHOD, char_lengths, count_tokens, load_qwen_tokenizer

    ap = argparse.ArgumentParser(
        prog="audit_dataset",
        description="Phase 1 factual data audit from prepare_dataset outputs (no selection). "
                    "Token lengths use ONLY Qwen tokenizer.json (no model weights).",
    )
    ap.add_argument("--config", default=DEFAULT_DATA_CONFIG)
    ap.add_argument("--normalized", default=None, help="default: paths.normalized_dir")
    ap.add_argument("--no-tokenizer", action="store_true", help="use character counts instead")
    ap.add_argument("--out-md", default="reports/phase1_audit.md")
    ap.add_argument("--out-json", default="data/audits/phase1_audit.json")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    nd = args.normalized or cfg["paths"]["normalized_dir"]
    meta_path = resolve(cfg["paths"]["raw_dir"]) / cfg["source"]["revision"] / "raw_metadata.json"
    raw_meta = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else None

    records = load_records(resolve(nd) / "normalized.jsonl")
    if args.no_tokenizer:
        rows, method = char_lengths(records), "characters (tokenizer disabled)"
    else:
        t = cfg["tokenizer"]
        tok = load_qwen_tokenizer(t["repo_id"], t["revision"], t["filename"])
        rows = count_tokens(records, tok)
        method = f"{TOKEN_METHOD} ({t['repo_id']}@{t['revision'][:12]}, {t['filename']})"
    write_jsonl(resolve(cfg["paths"]["audits_dir"]) / "token_lengths.jsonl", rows)
    audit = build_audit(nd, raw_meta, rows, method)
    write_audit(audit, args.out_md, args.out_json)
    _print({"report": args.out_md, "records": audit["n_records"],
            "label_status": audit["label_status"], "teacher": audit["teacher"],
            "card_match": audit["provenance"]["ok"], "leakage_ok": audit["leakage"]["ok"]})
    return 0


# ------------------------------------------------------------------ phase 2


def cmd_audit_selection(argv: list[str] | None = None) -> int:
    from .phase2 import render_phase2_md, run_phase2

    ap = argparse.ArgumentParser(
        prog="audit_selection",
        description="PHASE 2: single-call pool, chat-template lengths, candidate subsets, "
                    "comparison tables, blinded audit sample. No training; tokenizer files only.",
    )
    ap.add_argument("--config", default="configs/research.yaml")
    ap.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-md", default="reports/phase2_selection.md")
    ap.add_argument("--out-json", default="data/audits/phase2_analysis.json")
    args = ap.parse_args(argv)
    research = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)
    seed = args.seed if args.seed is not None else int(research["study"]["primary_seed"])
    res = run_phase2(data_cfg, research, seed)
    for path, text in ((args.out_md, render_phase2_md(res)),
                       (args.out_json, json.dumps(res, indent=1, default=str) + "\n")):
        p = resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, "utf-8")
    _print({"report": args.out_md, "schema_gate": res["schema_gate"],
            "render_checks": res["render_checks"],
            "arms": {k: v["n"] for k, v in res["manifests"].items()}})
    return 0


# ------------------------------------------------------------------ phase 3A


def cmd_export_kaggle(argv: list[str] | None = None) -> int:
    from .kaggle_export import export

    ap = argparse.ArgumentParser(
        prog="export_kaggle",
        description="PHASE 3A: build the PRIVATE Kaggle data bundle (100 pilot TRAIN + 100 fixed "
                    "VALIDATION examples) and the project source zip under data/kaggle_upload/.",
    )
    ap.add_argument("--config", default=DEFAULT_DATA_CONFIG)
    ap.add_argument("--research-config", default="configs/research.yaml")
    ap.add_argument("--train-config", default="configs/train_qlora.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/kaggle_upload")
    args = ap.parse_args(argv)
    res = export(load_yaml(args.config), load_yaml(args.research_config),
                 load_yaml(args.train_config), args.seed, args.out)
    _print(res)
    return 0


# ------------------------------------------------------------------ later phases


def not_yet(prog: str, phase: int, description: str):
    def _cmd(argv: list[str] | None = None) -> int:
        ap = argparse.ArgumentParser(prog=prog, description=f"[Phase {phase}] {description}")
        ap.add_argument("--config", default=None)
        ap.parse_known_args(argv)
        print(f"{prog}: scheduled for Phase {phase}; not implemented. Requires explicit human "
              f"approval of Phase {phase} (master spec section K).", file=sys.stderr)
        return 2
    return _cmd


COMMANDS = {
    "inspect": cmd_inspect,
    "check-access": cmd_check_access,
    "download": cmd_download,
    "prepare": cmd_prepare,
    "audit-data": cmd_audit_data,
    "audit-selection": cmd_audit_selection,
    "export-kaggle": cmd_export_kaggle,
    "kaggle-preflight": not_yet("kaggle_preflight", 3,
                                "use notebooks/kaggle_train_eval.ipynb with MODE='preflight'"),
    "train-adapter": not_yet("train_adapter", 4, "1k/3k arm training (needs approval)"),
    "eval-adapter": not_yet("eval_adapter", 4, "full held-out evaluation (needs approval)"),
    "summarize-results": not_yet("summarize_results", 5, "results summary"),
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print("usage: tooldistill {" + ",".join(COMMANDS) + "} [args]", file=sys.stderr)
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
