from __future__ import annotations

import argparse
import json
from pathlib import Path

from structmem.analysis import analyze
from structmem.config import load_config
from structmem.llm import LLMClient
from structmem.preflight import preflight
from structmem.prepare import (
    automatic_manifests_from_memories,
    failure_audit_from_episodes,
    matrix_from_memories,
    memory_from_episodes,
    queries_from_episodes,
)
from structmem.runner import ExperimentRunner
from structmem.report import aggregate_four_models
from structmem.probe import probe_models
from structmem.storage import JSONLStore, write_json


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="structmem")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name == "preflight":
            command.add_argument("--output", type=Path)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--episodes", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--failure-audit", type=Path)
    memory_parser = commands.add_parser("build-memory")
    memory_parser.add_argument("--episodes", type=Path, required=True)
    memory_parser.add_argument("--output", type=Path, required=True)
    memory_parser.add_argument("--include-failures", action="store_true")
    query_parser = commands.add_parser("build-queries")
    query_parser.add_argument("--episodes", type=Path, required=True)
    query_parser.add_argument("--output", type=Path, required=True)
    matrix_parser = commands.add_parser("build-matrix")
    matrix_parser.add_argument("--queries", type=Path, required=True)
    matrix_parser.add_argument("--sources", type=Path, required=True)
    matrix_parser.add_argument("--output", type=Path, required=True)
    matrix_parser.add_argument("--semantic-threshold", type=float, default=0.25)
    matrix_parser.add_argument("--structural-threshold", type=float, default=0.65)
    matrix_parser.add_argument("--per-cell", type=int, default=300)
    matrix_parser.add_argument("--seed", type=int, default=42)
    auto_parser = commands.add_parser("build-auto-manifests")
    auto_parser.add_argument("--queries", type=Path, required=True)
    auto_parser.add_argument("--sources", type=Path, required=True)
    auto_parser.add_argument("--output-dir", type=Path, required=True)
    auto_parser.add_argument("--semantic-threshold", type=float, default=0.25)
    auto_parser.add_argument("--structural-threshold", type=float, default=0.65)
    auto_parser.add_argument("--per-cell", type=int, default=300)
    auto_parser.add_argument("--seed", type=int, default=42)
    audit_parser = commands.add_parser("build-failure-audit")
    audit_parser.add_argument("--episodes", type=Path, action="append", required=True)
    audit_parser.add_argument("--config", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    audit_parser.add_argument("--limit", type=int, default=50)
    audit_parser.add_argument("--observation-chars", type=int, default=700)
    aggregate_parser = commands.add_parser("aggregate-four-models")
    aggregate_parser.add_argument("--root", type=Path, required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)
    probe_parser = commands.add_parser("probe-models")
    probe_parser.add_argument("--api", type=Path, required=True)
    probe_parser.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "preflight":
        result = preflight(load_config(args.config))
        if args.output:
            write_json(args.output, result)
    elif args.command == "run":
        config = load_config(args.config)
        readiness = preflight(config)
        write_json(config.output_dir / "preflight.json", readiness)
        if not readiness["ready"]:
            raise SystemExit("Preflight failed; see preflight.json. No benchmark scores were generated.")
        ExperimentRunner(config, args.config.resolve()).run()
        result = {"output_dir": str(config.output_dir), "status": "finished"}
    elif args.command == "analyze":
        failure_audit = (
            json.loads(args.failure_audit.read_text(encoding="utf-8"))
            if args.failure_audit
            else None
        )
        result = analyze(JSONLStore(args.episodes).read(), failure_audit=failure_audit)
        write_json(args.output, result)
    elif args.command == "build-memory":
        result = memory_from_episodes(args.episodes, args.output, include_failures=args.include_failures)
    elif args.command == "build-queries":
        result = queries_from_episodes(args.episodes, args.output)
    elif args.command == "build-matrix":
        result = matrix_from_memories(
            args.queries,
            args.sources,
            args.output,
            semantic_threshold=args.semantic_threshold,
            structural_threshold=args.structural_threshold,
            per_cell=args.per_cell,
            seed=args.seed,
        )
    elif args.command == "build-auto-manifests":
        result = automatic_manifests_from_memories(
            args.queries,
            args.sources,
            args.output_dir,
            semantic_threshold=args.semantic_threshold,
            structural_threshold=args.structural_threshold,
            per_cell=args.per_cell,
            seed=args.seed,
        )
    elif args.command == "build-failure-audit":
        config = load_config(args.config)
        result = failure_audit_from_episodes(
            args.episodes,
            args.output,
            llm=LLMClient(config.api),
            limit=args.limit,
            observation_chars=args.observation_chars,
        )
    elif args.command == "aggregate-four-models":
        result = aggregate_four_models(args.root, args.output)
    elif args.command == "probe-models":
        result = probe_models(args.api, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["ready"]:
            raise SystemExit("One or more configured models failed the API probe; full execution was not started.")
        return
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "preflight" and not result["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
