from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_dataset_config
from .data.manifest import read_manifest, summarize_manifest, write_manifest
from .data.sample import sample_by_stage
from .data.split import assign_subject_splits
from .evaluation.report import evaluate_prediction_file
from .provenance import write_json


def _targets(value: str) -> dict[str, int]:
    result = {}
    try:
        for item in value.split(","):
            label, count = item.split("=", maxsplit=1)
            result[label.strip().upper()] = int(count)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "targets must look like W=100,N1=100,N2=100,N3=100,REM=100"
        ) from error
    return result


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sleepvlm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("--manifest", required=True)

    prepare = subparsers.add_parser("prepare-dataset")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--failures")
    prepare.add_argument("--limit", type=int)

    split = subparsers.add_parser("split-manifest")
    split.add_argument("--manifest", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=2024)

    sample = subparsers.add_parser("sample-manifest")
    sample.add_argument("--manifest", required=True)
    sample.add_argument("--output", required=True)
    sample.add_argument("--targets", required=True, type=_targets)
    sample.add_argument("--split")
    sample.add_argument("--cohort")
    sample.add_argument("--seed", type=int, default=2024)

    render = subparsers.add_parser("render-manifest")
    render.add_argument("--manifest", required=True)
    render.add_argument("--output-root", required=True)
    render.add_argument("--output-manifest", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--parsed-output")
    evaluate.add_argument("--parser", choices=("first", "last", "majority"), default="first")

    aggregate = subparsers.add_parser("aggregate-reports")
    aggregate.add_argument("--reports", required=True, nargs="+")
    aggregate.add_argument("--output", required=True)

    run_api = subparsers.add_parser("run-openai-compatible")
    run_api.add_argument("--manifest", required=True)
    run_api.add_argument("--prompt", required=True)
    run_api.add_argument("--output-root", default="outputs")
    run_api.add_argument("--run-id", required=True)
    run_api.add_argument("--model-id", required=True)
    run_api.add_argument("--model-revision", default="")
    run_api.add_argument("--api-key-env", required=True)
    run_api.add_argument("--base-url")
    run_api.add_argument("--split", default="test")
    run_api.add_argument("--image-size", type=int, default=336)
    run_api.add_argument("--max-new-tokens", type=int, default=2048)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-manifest":
        _print_json(summarize_manifest(read_manifest(args.manifest)))
        return 0

    if args.command == "prepare-dataset":
        from .data.cohorts import prepare_dataset

        config = load_dataset_config(args.config)
        records, failures = prepare_dataset(config, args.output_root, limit=args.limit)
        if not records:
            if args.failures:
                write_json(args.failures, failures)
            raise RuntimeError("dataset preparation produced no valid records")
        write_manifest(records, args.manifest)
        failures_path = args.failures or str(Path(args.manifest).with_suffix(".failures.json"))
        write_json(failures_path, failures)
        _print_json({"manifest": summarize_manifest(records), "failures": len(failures)})
        return 0

    if args.command == "split-manifest":
        records = assign_subject_splits(read_manifest(args.manifest), seed=args.seed)
        write_manifest(records, args.output)
        _print_json(summarize_manifest(records))
        return 0

    if args.command == "sample-manifest":
        records = sample_by_stage(
            read_manifest(args.manifest),
            args.targets,
            split=args.split,
            cohort=args.cohort,
            seed=args.seed,
        )
        write_manifest(records, args.output)
        _print_json(summarize_manifest(records))
        return 0

    if args.command == "render-manifest":
        from .data.render import render_manifest

        records = render_manifest(read_manifest(args.manifest), args.output_root)
        write_manifest(records, args.output_manifest)
        _print_json(summarize_manifest(records))
        return 0

    if args.command == "evaluate":
        report = evaluate_prediction_file(
            args.predictions,
            args.output,
            parser_mode=args.parser,
            parsed_output_path=args.parsed_output,
        )
        _print_json(report["metrics"])
        return 0

    if args.command == "aggregate-reports":
        from .evaluation.table import reports_to_csv

        reports_to_csv(args.reports, args.output)
        print(args.output)
        return 0

    if args.command == "run-openai-compatible":
        from .inference.openai_compatible import OpenAICompatibleBackend
        from .inference.runner import run_inference

        backend = OpenAICompatibleBackend(
            model_id=args.model_id,
            model_revision=args.model_revision,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            image_size=args.image_size,
            max_new_tokens=args.max_new_tokens,
        )
        path = run_inference(
            backend=backend,
            manifest_path=args.manifest,
            prompt_path=args.prompt,
            output_dir=args.output_root,
            run_id=args.run_id,
            split=args.split,
        )
        print(path)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
