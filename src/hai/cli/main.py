from __future__ import annotations

import argparse
import json
from pathlib import Path

from hai.data.pipeline import (
    DataGovernanceError,
    fetch_dataset,
    inspect_config,
    prepare_dataset,
    verify_dataset,
)
from hai.evaluation.benchmark import (
    BenchmarkError,
    evaluate_benchmark,
    generate_benchmark,
    verify_benchmark,
)
from hai.models.gru import (
    GRUGovernanceError,
    GRULanguageModel,
    compare_errors,
    evaluate_gru,
    train_gru,
)
from hai.models.transformer_smoke import (
    TinyCausalDecoder,
    TransformerGovernanceError,
    evaluate_model,
    generate_text,
    load_model_config,
    train_model,
)
from hai.symbolic.reasoning import (
    SymbolicError,
    exact_arithmetic,
    solve_linear_equation_text,
)
from hai.tokenization.bpe import (
    TokenizerGovernanceError,
    inspect_tokenizer,
    train_tokenizer,
    verify_tokenizer,
)


def env_check() -> int:
    try:
        import torch
    except ImportError as exc:
        print(json.dumps({"ok": False, "error": f"torch import failed: {exc}"}, indent=2))
        return 2

    result = {
        "ok": bool(torch.cuda.is_available()),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            result["devices"].append({"index": idx, "name": torch.cuda.get_device_name(idx)})
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="hai")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("env-check", help="show PyTorch/ROCm GPU visibility")
    data = sub.add_parser("data", help="inspect and verify governed dataset inputs")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    inspect = data_sub.add_parser("inspect-config", help="validate a dataset config")
    inspect.add_argument("--config", type=Path, required=True)
    for command, help_text in (
        ("fetch", "acquire the allowlisted dataset into immutable raw JSONL"),
        ("prepare", "create deterministic processed splits"),
        ("verify", "verify manifest, checksums, and split disjointness"),
    ):
        action = data_sub.add_parser(command, help=help_text)
        action.add_argument("--config", type=Path, required=True)
    tokenizer = sub.add_parser("tokenizer", help="train and verify project-owned tokenizers")
    tokenizer_sub = tokenizer.add_subparsers(dest="tokenizer_command", required=True)
    train = tokenizer_sub.add_parser("train", help="train a BPE tokenizer from TRAIN only")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--dataset-manifest", type=Path, required=True)
    inspect_tokenizer_parser = tokenizer_sub.add_parser("inspect", help="inspect encoded text")
    inspect_tokenizer_parser.add_argument("--tokenizer", type=Path, required=True)
    inspect_tokenizer_parser.add_argument("--text", required=True)
    verify_tokenizer_parser = tokenizer_sub.add_parser(
        "verify", help="verify tokenizer provenance and reload"
    )
    verify_tokenizer_parser.add_argument("--tokenizer", type=Path, required=True)
    model = sub.add_parser("model", help="create and inspect small research models")
    model_sub = model.add_subparsers(dest="model_command", required=True)
    create = model_sub.add_parser("create", help="report a randomly initialized model")
    create.add_argument("--config", type=Path, required=True)
    create.add_argument("--tokenizer", type=Path, required=True)
    train_model_parser = sub.add_parser("train", help="train the tiny causal decoder")
    train_model_parser.add_argument("--model", type=Path, required=True)
    train_model_parser.add_argument("--tokenizer", type=Path, required=True)
    train_model_parser.add_argument("--dataset-manifest", type=Path, required=True)
    train_model_parser.add_argument("--max-steps", type=int, required=True)
    train_model_parser.add_argument("--experiment", required=True)
    train_model_parser.add_argument("--resume", type=Path)
    evaluate_parser = sub.add_parser("evaluate", help="evaluate a checkpoint on validation")
    evaluate_parser.add_argument("--model", type=Path, required=True)
    evaluate_parser.add_argument("--tokenizer", type=Path, required=True)
    evaluate_parser.add_argument("--dataset-manifest", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    generate_parser = sub.add_parser("generate", help="generate fixed greedy text")
    generate_parser.add_argument("--model", type=Path, required=True)
    generate_parser.add_argument("--tokenizer", type=Path, required=True)
    generate_parser.add_argument("--checkpoint", type=Path, required=True)
    generate_parser.add_argument("--prompt", required=True)
    compare_parser = sub.add_parser(
        "compare-errors", help="compare paired Transformer/GRU token errors"
    )
    compare_parser.add_argument("--transformer-model", type=Path, required=True)
    compare_parser.add_argument("--gru-model", type=Path, required=True)
    compare_parser.add_argument("--tokenizer", type=Path, required=True)
    compare_parser.add_argument("--dataset-manifest", type=Path, required=True)
    compare_parser.add_argument("--transformer-checkpoint", type=Path, required=True)
    compare_parser.add_argument("--gru-checkpoint", type=Path, required=True)
    compare_parser.add_argument("--max-blocks", type=int, default=128)
    symbolic = sub.add_parser("symbolic", help="run deterministic symbolic reasoning")
    symbolic_sub = symbolic.add_subparsers(dest="symbolic_command", required=True)
    solve = symbolic_sub.add_parser("solve", help="solve a safe arithmetic or linear equation")
    solve.add_argument("expression")
    symbolic_sub.add_parser("self-test", help="run deterministic symbolic self-tests")
    benchmark = sub.add_parser("benchmark", help="generate and score objective benchmarks")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)
    for command, help_text in (
        ("generate", "generate held-out benchmark partitions"),
        ("verify", "verify benchmark hashes and split isolation"),
    ):
        action = benchmark_sub.add_parser(command, help=help_text)
        action.add_argument("--config", type=Path, required=True)
    benchmark_evaluate = benchmark_sub.add_parser(
        "evaluate", help="evaluate one expert through the common harness"
    )
    benchmark_evaluate.add_argument("--config", type=Path, required=True)
    benchmark_evaluate.add_argument("--expert", required=True)
    benchmark_evaluate.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    args = parser.parse_args()
    if args.command == "env-check":
        return env_check()
    if args.command == "data":
        try:
            if args.data_command == "inspect-config":
                result = inspect_config(args.config)
            elif args.data_command == "fetch":
                result = fetch_dataset(args.config, Path.cwd())
            elif args.data_command == "prepare":
                result = prepare_dataset(args.config, Path.cwd())
            elif args.data_command == "verify":
                result = verify_dataset(args.config, Path.cwd())
            else:
                return 2
            print(json.dumps(result, indent=2, sort_keys=True))
        except (DataGovernanceError, OSError) as exc:
            parser.error(str(exc))
        return 0
    if args.command == "tokenizer":
        try:
            if args.tokenizer_command == "train":
                result = train_tokenizer(args.config, args.dataset_manifest, Path.cwd())
            elif args.tokenizer_command == "inspect":
                result = inspect_tokenizer(args.tokenizer, args.text)
            elif args.tokenizer_command == "verify":
                result = verify_tokenizer(args.tokenizer, Path.cwd())
            else:
                return 2
            print(json.dumps(result, indent=2, sort_keys=True))
        except (TokenizerGovernanceError, OSError) as exc:
            parser.error(str(exc))
        return 0
    if args.command == "symbolic":
        try:
            if args.symbolic_command == "solve":
                result = (
                    solve_linear_equation_text(args.expression)
                    if "=" in args.expression
                    else exact_arithmetic(args.expression)
                )
                print(json.dumps({"value": result.value, "trace": result.trace}, indent=2))
            elif args.symbolic_command == "self-test":
                cases = [
                    exact_arithmetic("1 + 2 * 3").value == "7",
                    solve_linear_equation_text("3*x + 4 = 19").value == "5",
                ]
                print(
                    json.dumps(
                        {"ok": all(cases), "cases_passed": sum(cases), "cases_total": len(cases)}
                    )
                )
            else:
                return 2
        except (SymbolicError, OSError) as exc:
            parser.error(str(exc))
        return 0
    if args.command == "benchmark":
        try:
            if args.benchmark_command == "generate":
                result = generate_benchmark(args.config, Path.cwd())
            elif args.benchmark_command == "verify":
                result = verify_benchmark(args.config, Path.cwd())
            elif args.benchmark_command == "evaluate":
                result = evaluate_benchmark(args.config, Path.cwd(), args.expert, args.split)
            else:
                return 2
            print(json.dumps(result, indent=2, sort_keys=True))
        except (BenchmarkError, OSError, KeyError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        return 0
    if args.command == "model":
        try:
            config = load_model_config(args.config)
            tokenizer_config = json.loads(args.tokenizer.read_text(encoding="utf-8"))
            config["vocab_size"] = len(tokenizer_config["model"]["vocab"])
            model_class = (
                GRULanguageModel
                if config.get("architecture") == "gru_language_model"
                else TinyCausalDecoder
            )
            result = {
                "initialization": config["initialization"],
                "parameter_count": model_class(config).parameter_count(),
            }
            print(json.dumps(result, indent=2, sort_keys=True))
        except (TransformerGovernanceError, OSError, KeyError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        return 0
    try:
        if args.command == "train":
            if load_model_config(args.model).get("architecture") == "gru_language_model":
                result = train_gru(
                    args.model,
                    args.tokenizer,
                    args.dataset_manifest,
                    Path("artifacts/checkpoints") / args.experiment,
                    args.max_steps,
                    args.resume,
                )
            else:
                result = train_model(
                    args.model,
                    args.tokenizer,
                    args.dataset_manifest,
                    Path("artifacts/checkpoints") / args.experiment,
                    args.max_steps,
                    args.resume,
                )
        elif args.command == "evaluate":
            if load_model_config(args.model).get("architecture") == "gru_language_model":
                result = evaluate_gru(
                    args.model, args.tokenizer, args.dataset_manifest, args.checkpoint
                )
            else:
                result = evaluate_model(
                    args.model, args.tokenizer, args.dataset_manifest, args.checkpoint
                )
        elif args.command == "generate":
            result = generate_text(args.model, args.tokenizer, args.checkpoint, args.prompt)
        elif args.command == "compare-errors":
            result = compare_errors(
                args.transformer_model,
                args.gru_model,
                args.tokenizer,
                args.dataset_manifest,
                args.transformer_checkpoint,
                args.gru_checkpoint,
                args.max_blocks,
            )
        else:
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
    except (
        TransformerGovernanceError,
        GRUGovernanceError,
        OSError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        parser.error(str(exc))
    return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
