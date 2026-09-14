from __future__ import annotations

import argparse
import json
from pathlib import Path

from hai.calibration.engine import (
    CalibrationError,
    calibrate,
    evaluate_consensus,
    write_report,
)
from hai.common.tracking import log_cli_run
from hai.data.pipeline import (
    DataGovernanceError,
    fetch_dataset,
    inspect_config,
    prepare_dataset,
    verify_dataset,
)
from hai.evaluation.ablation import AblationError, milestone_report, run_ablation
from hai.evaluation.benchmark import (
    BenchmarkError,
    evaluate_benchmark,
    evaluate_system,
    generate_benchmark,
    verify_benchmark,
)
from hai.experts.protocol import core_experts
from hai.graph.store import GraphError, GraphStore
from hai.graph.store import self_test as graph_self_test
from hai.memory.store import MemoryError, MemoryStore
from hai.memory.store import self_test as memory_self_test
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
from hai.retrieval.engine import (
    RetrievalError,
    build_index,
    evaluate_retrieval,
    train_encoder,
)
from hai.routing.router import (
    RouterError,
    build_router_dataset,
    evaluate_router,
    route_prompt,
    train_router,
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
from hai.verification.layer import self_test as verification_self_test


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


def _benchmark_config_path(value: str) -> Path:
    direct = Path(value)
    if direct.is_file():
        return direct
    aliases = {"core-objective-v1": Path("configs/benchmarks/core.yaml")}
    try:
        return aliases[value]
    except KeyError as exc:
        raise OSError(f"unknown benchmark config: {value}") from exc


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
    benchmark_evaluate_group = benchmark_evaluate.add_mutually_exclusive_group(required=True)
    benchmark_evaluate_group.add_argument("--expert")
    benchmark_evaluate_group.add_argument("--system", choices=("router-plus-verifier",))
    benchmark_evaluate.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    expert = sub.add_parser("expert", help="run unified expert protocol checks")
    expert_sub = expert.add_subparsers(dest="expert_command", required=True)
    expert_self_test = expert_sub.add_parser("self-test", help="verify all core expert adapters")
    expert_self_test.add_argument("--all", action="store_true")
    verify = sub.add_parser("verify", help="run independent answer verification")
    verify_sub = verify.add_subparsers(dest="verify_command", required=True)
    verify_sub.add_parser("self-test", help="exercise verification and rejection paths")
    router = sub.add_parser("router", help="train and evaluate the task router")
    router_sub = router.add_subparsers(dest="router_command", required=True)
    router_dataset = router_sub.add_parser("build-dataset", help="build TRAIN-only router metadata")
    router_dataset.add_argument("--benchmark", required=True)
    router_train = router_sub.add_parser("train", help="train the interpretable router")
    router_train.add_argument("--config", type=Path, required=True)
    router_train.add_argument("--benchmark", required=True)
    router_train.add_argument("--split", choices=("train",), default="train")
    router_evaluate = router_sub.add_parser(
        "evaluate", help="evaluate router utility and expert cost"
    )
    router_evaluate.add_argument("--config", type=Path, required=True)
    router_evaluate.add_argument("--benchmark", required=True)
    router_evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    router_route = router_sub.add_parser("route", help="route one task with a decision trace")
    router_route.add_argument("--config", type=Path, required=True)
    router_route.add_argument("--category", required=True)
    router_route.add_argument("--prompt", required=True)
    calibrate_parser = sub.add_parser(
        "calibrate", help="fit validation-only confidence calibration"
    )
    calibrate_parser.add_argument(
        "--config", type=Path, default=Path("configs/benchmarks/core.yaml")
    )
    calibrate_parser.add_argument("--split", choices=("validation",), default="validation")
    calibrate_parser.add_argument(
        "--method", choices=("all", "temperature", "logistic", "isotonic"), default="all"
    )
    consensus = sub.add_parser("consensus", help="evaluate evidence-aware consensus")
    consensus_sub = consensus.add_subparsers(dest="consensus_command", required=True)
    consensus_evaluate = consensus_sub.add_parser(
        "evaluate", help="measure coverage and abstention"
    )
    consensus_evaluate.add_argument(
        "--config", type=Path, default=Path("configs/benchmarks/core.yaml")
    )
    consensus_evaluate.add_argument("--split", choices=("validation",), default="validation")
    evaluation = sub.add_parser("evaluation", help="produce calibration and reliability reports")
    evaluation_sub = evaluation.add_subparsers(dest="evaluation_command", required=True)
    reliability = evaluation_sub.add_parser(
        "reliability-report", help="write validation reliability metrics"
    )
    reliability.add_argument("--config", type=Path, default=Path("configs/benchmarks/core.yaml"))
    reliability.add_argument("--split", choices=("validation",), default="validation")
    ablation = sub.add_parser("ablation", help="run frozen milestone ablation suites")
    ablation_sub = ablation.add_subparsers(dest="ablation_command", required=True)
    ablation_run = ablation_sub.add_parser("run", help="evaluate a frozen suite on TEST")
    ablation_run.add_argument("--suite", choices=("core-m2",), required=True)
    ablation_run.add_argument("--split", choices=("test",), default="test")
    report = sub.add_parser("report", help="read milestone reports")
    report_sub = report.add_subparsers(dest="report_command", required=True)
    milestone = report_sub.add_parser("milestone", help="show an ablation milestone report")
    milestone.add_argument("name", choices=("M2",))
    retrieval = sub.add_parser("retrieval", help="train and evaluate semantic retrieval")
    retrieval_sub = retrieval.add_subparsers(dest="retrieval_command", required=True)
    retrieval_train = retrieval_sub.add_parser(
        "train-encoder", help="train a random-initialized dual encoder on TRAIN"
    )
    retrieval_train.add_argument(
        "--config", type=Path, default=Path("configs/models/embedding.yaml")
    )
    retrieval_index = retrieval_sub.add_parser("build-index", help="build the HNSW index")
    retrieval_index.add_argument(
        "--config", type=Path, default=Path("configs/models/embedding.yaml")
    )
    retrieval_index.add_argument("--split", choices=("validation",), default=None)
    retrieval_evaluate = retrieval_sub.add_parser(
        "evaluate", help="compare random, lexical, dense, and hybrid retrieval"
    )
    retrieval_evaluate.add_argument(
        "--config", type=Path, default=Path("configs/models/embedding.yaml")
    )
    retrieval_evaluate.add_argument("--split", choices=("validation",), default="validation")
    memory = sub.add_parser("memory", help="manage explicit memory stores")
    memory_sub = memory.add_subparsers(dest="memory_command", required=True)
    memory_init = memory_sub.add_parser("init", help="initialize the memory database")
    memory_init.add_argument("--db", type=Path, default=Path("artifacts/memory/memory.sqlite"))
    memory_sub.add_parser("self-test", help="exercise memory trust and retention controls")
    memory_inspect = memory_sub.add_parser("inspect", help="inspect non-deleted memory items")
    memory_inspect.add_argument(
        "--type", choices=("working", "episodic", "semantic", "procedural")
    )
    memory_inspect.add_argument("--db", type=Path, default=Path("artifacts/memory/memory.sqlite"))
    graph = sub.add_parser("graph", help="manage provenance-aware knowledge graphs")
    graph_sub = graph.add_subparsers(dest="graph_command", required=True)
    graph_init = graph_sub.add_parser("init", help="initialize the graph database")
    graph_init.add_argument("--db", type=Path, default=Path("artifacts/graphs/knowledge.sqlite"))
    graph_fixture = graph_sub.add_parser(
        "import-fixture", help="import deterministic JSONL graph records"
    )
    graph_fixture.add_argument("path", type=Path)
    graph_fixture.add_argument("--db", type=Path, default=Path("artifacts/graphs/knowledge.sqlite"))
    graph_query = graph_sub.add_parser("query", help="list trusted outgoing relations")
    graph_query.add_argument("--subject")
    graph_query.add_argument("--entity", dest="subject")
    graph_query.add_argument("--predicate")
    graph_query.add_argument("--relation", dest="predicate")
    graph_query.add_argument(
        "--min-trust", choices=("verified", "candidate", "untrusted"), default="verified"
    )
    graph_query.add_argument("--db", type=Path, default=Path("artifacts/graphs/knowledge.sqlite"))
    graph_path = graph_sub.add_parser("path", help="find a deterministic trusted path")
    graph_path.add_argument("--start", required=True)
    graph_path.add_argument("--goal", required=True)
    graph_path.add_argument("--max-hops", type=int, default=4)
    graph_path.add_argument(
        "--min-trust", choices=("verified", "candidate", "untrusted"), default="verified"
    )
    graph_path.add_argument("--db", type=Path, default=Path("artifacts/graphs/knowledge.sqlite"))
    graph_sub.add_parser("self-test", help="exercise graph provenance and trust controls")
    args = parser.parse_args()
    if args.command == "env-check":
        return env_check()
    if args.command == "memory":
        try:
            if args.memory_command == "init":
                store = MemoryStore(args.db)
                store.close()
                result = {"ok": True, "path": str(args.db)}
            elif args.memory_command == "self-test":
                result = memory_self_test()
            elif args.memory_command == "inspect":
                store = MemoryStore(args.db)
                try:
                    items = store.inspect(args.type)
                    result = {"items": items, "count": len(items)}
                finally:
                    store.close()
            else:
                return 2
            result["tracking"] = log_cli_run(
                f"memory-{args.memory_command}",
                result,
                tags={"phase": "P16", "command": f"memory-{args.memory_command}"},
            )
            print(json.dumps(result, indent=2, sort_keys=True))
        except (MemoryError, OSError, KeyError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        return 0
    if args.command == "graph":
        try:
            if args.graph_command == "init":
                store = GraphStore(args.db)
                store.close()
                result = {"ok": True, "path": str(args.db)}
            elif args.graph_command == "import-fixture":
                store = GraphStore(args.db)
                try:
                    result = store.import_fixture(args.path)
                finally:
                    store.close()
            elif args.graph_command == "query":
                if not args.subject:
                    parser.error("graph query requires --subject or --entity")
                store = GraphStore(args.db)
                try:
                    matches = store.find_entities(args.subject)
                    if not matches:
                        raise GraphError(f"entity label not found: {args.subject}")
                    relations = []
                    for entity in matches:
                        relations.extend(
                            store.neighbors(
                                entity["id"],
                                relation_type=args.predicate,
                                min_trust=args.min_trust,
                            )
                        )
                    result = {"entities": matches, "relations": relations, "count": len(relations)}
                finally:
                    store.close()
            elif args.graph_command == "path":
                store = GraphStore(args.db)
                try:
                    result = store.path(
                        args.start, args.goal, max_hops=args.max_hops, min_trust=args.min_trust
                    )
                finally:
                    store.close()
            elif args.graph_command == "self-test":
                result = graph_self_test()
            else:
                return 2
            result["tracking"] = log_cli_run(
                f"graph-{args.graph_command}",
                result,
                tags={"phase": "P17", "command": f"graph-{args.graph_command}"},
                params={"min_trust": getattr(args, "min_trust", "none")},
            )
            print(json.dumps(result, indent=2, sort_keys=True))
        except (GraphError, OSError, KeyError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        return 0
    if args.command == "retrieval":
        try:
            if args.retrieval_command == "train-encoder":
                result = train_encoder(args.config, Path.cwd())
            elif args.retrieval_command == "build-index":
                result = build_index(args.config, Path.cwd(), args.split)
            elif args.retrieval_command == "evaluate":
                result = evaluate_retrieval(args.config, Path.cwd(), args.split)
            else:
                return 2
            result["tracking"] = log_cli_run(
                f"retrieval-{args.retrieval_command}",
                result,
                tags={"phase": "P15", "command": f"retrieval-{args.retrieval_command}"},
                params={"split": getattr(args, "split", "none")},
            )
            print(json.dumps(result, indent=2, sort_keys=True))
        except (RetrievalError, OSError, KeyError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        return 0
    if args.command == "ablation":
        if args.ablation_command != "run":
            return 2
        try:
            result = run_ablation(
                Path("configs/ablation") / f"{args.suite}.yaml", Path.cwd(), args.split
            )
            print(
                json.dumps(
                    {
                        **write_report(result, Path("artifacts/ablations") / f"{args.suite}.json"),
                        "tracking": log_cli_run(
                            f"ablation-{args.suite}-{args.split}",
                            result,
                            tags={"phase": "P14", "command": "ablation", "split": args.split},
                            params={"suite": args.suite, "split": args.split},
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        except (AblationError, OSError, KeyError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        return 0
    if args.command == "report":
        if args.report_command != "milestone":
            return 2
        try:
            result = milestone_report(Path("artifacts/ablations/core-m2.json"))
            print(json.dumps(result, indent=2, sort_keys=True))
        except (AblationError, OSError, KeyError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        return 0
    if args.command == "calibrate":
        try:
            result = calibrate(args.config, Path.cwd(), args.split, args.method)
            report_result = write_report(result, Path("artifacts/calibration/calibration-v1.json"))
            report_result["tracking"] = log_cli_run(
                f"calibrate-{args.split}",
                result,
                tags={"phase": "P13", "command": "calibrate", "split": args.split},
                params={"method": args.method, "split": args.split},
            )
            print(
                json.dumps(report_result, indent=2, sort_keys=True)
            )
        except (CalibrationError, OSError, KeyError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        return 0
    if args.command == "consensus":
        if args.consensus_command != "evaluate":
            return 2
        try:
            result = evaluate_consensus(args.config, Path.cwd(), args.split)
            report_result = write_report(result, Path("artifacts/calibration/consensus-v1.json"))
            report_result["tracking"] = log_cli_run(
                f"consensus-{args.split}",
                result,
                tags={"phase": "P13", "command": "consensus", "split": args.split},
                params={"split": args.split},
            )
            print(
                json.dumps(report_result, indent=2, sort_keys=True)
            )
        except (CalibrationError, OSError, KeyError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        return 0
    if args.command == "evaluation":
        if args.evaluation_command != "reliability-report":
            return 2
        try:
            result = calibrate(args.config, Path.cwd(), args.split, "all")
            report_result = write_report(result, Path("artifacts/calibration/reliability-v1.json"))
            report_result["tracking"] = log_cli_run(
                f"reliability-{args.split}",
                result,
                tags={"phase": "P13", "command": "reliability-report", "split": args.split},
                params={"split": args.split},
            )
            print(
                json.dumps(report_result, indent=2, sort_keys=True)
            )
        except (CalibrationError, OSError, KeyError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        return 0
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
                result = (
                    evaluate_system(args.config, Path.cwd(), args.system, args.split)
                    if args.system
                    else evaluate_benchmark(args.config, Path.cwd(), args.expert, args.split)
                )
            else:
                return 2
            result["tracking"] = log_cli_run(
                f"benchmark-{args.benchmark_command}",
                result,
                tags={"phase": "evaluation", "command": f"benchmark-{args.benchmark_command}"},
                params={"split": getattr(args, "split", "none")},
            )
            print(json.dumps(result, indent=2, sort_keys=True))
        except (BenchmarkError, OSError, KeyError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        return 0
    if args.command == "verify":
        if args.verify_command != "self-test":
            return 2
        result = verification_self_test()
        result["tracking"] = log_cli_run(
            "verification-self-test",
            result,
            tags={"phase": "P12", "command": "verify-self-test"},
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "expert":
        if args.expert_command != "self-test":
            return 2
        results = []
        for expert_id, adapter in core_experts().items():
            result = adapter.answer(f"self-test:{expert_id}", "3*x + 4 = 19")
            results.append(
                {
                    "expert": expert_id,
                    "status": result.status,
                    "serialized": bool(result.to_dict()),
                    "model_version": result.model_version,
                    "has_evidence": bool(result.evidence),
                }
            )
        print(
            json.dumps(
                {
                    "ok": all(item["serialized"] and item["has_evidence"] for item in results),
                    "experts": results,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "router":
        try:
            benchmark_config = (
                _benchmark_config_path(args.benchmark)
                if args.router_command != "route"
                else None
            )
            if args.router_command == "build-dataset":
                result = build_router_dataset(benchmark_config, Path.cwd())
            elif args.router_command == "train":
                result = train_router(args.config, benchmark_config, Path.cwd(), args.split)
            elif args.router_command == "evaluate":
                result = evaluate_router(args.config, benchmark_config, Path.cwd(), args.split)
            elif args.router_command == "route":
                result = route_prompt(args.config, args.prompt, args.category, Path.cwd())
            else:
                return 2
            result["tracking"] = log_cli_run(
                f"router-{args.router_command}",
                result,
                tags={"phase": "P11", "command": f"router-{args.router_command}"},
                params={"split": getattr(args, "split", "none")},
            )
            print(json.dumps(result, indent=2, sort_keys=True))
        except (RouterError, OSError, KeyError, json.JSONDecodeError) as exc:
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
