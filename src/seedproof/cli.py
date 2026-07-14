"""Command line interface for seedproof.

Subcommands:

- ``seedproof ingest`` — convert a captured stream into a canonical record.
- ``seedproof show``   — pretty-print one record.
- ``seedproof ls``     — list records under a directory.
- ``seedproof diff``   — first-divergence forensics between two records.
- ``seedproof matrix`` — group N records into output classes, analyze axes.
- ``seedproof check``  — CI gate: exit 1 unless all records are identical.

Exit codes follow ``diff(1)`` conventions: 0 = success / no divergence,
1 = divergence found, 2 = usage or file errors. Errors go to stderr as one
readable line, never as a raw traceback.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from .align import DEFAULT_TIE_EPSILON
from .errors import SeedproofError
from .forensics import diagnose
from .ingest import FORMATS, ingest
from .matrix import build_matrix
from .record import RunConfig, RunRecord, dumps, load, save
from .report import (
    format_table,
    matrix_to_dict,
    pluralize,
    render_diff,
    render_matrix,
    render_show,
)

EXIT_OK = 0
EXIT_DIVERGENT = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the ``seedproof`` argument parser (exposed for testing)."""
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="seedproof",
        description=(
            "Token-stream divergence forensics: compare recorded generations "
            "across seeds, backends, and quants."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"seedproof {__version__}"
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    p_ingest = sub.add_parser(
        "ingest", help="convert a captured token stream into a run record"
    )
    p_ingest.add_argument("input", help="capture file, or '-' for stdin")
    p_ingest.add_argument(
        "--format", required=True, choices=FORMATS, help="capture format"
    )
    p_ingest.add_argument("-o", "--output", help="record file to write (default: stdout)")
    p_ingest.add_argument("--name", default="", help="record label")
    p_ingest.add_argument("--prompt", default="", help="the prompt the run answered")
    p_ingest.add_argument("--note", default="", help="free-form note stored in the record")
    p_ingest.add_argument("--model", default="", help="model identifier")
    p_ingest.add_argument("--backend", default="", help="runtime backend (cpu, cuda, metal, ...)")
    p_ingest.add_argument("--device", default="", help="device detail (gpu0, ...)")
    p_ingest.add_argument("--quant", default="", help="quantization (fp16, q4_k_m, ...)")
    p_ingest.add_argument("--seed", type=int, default=None, help="sampler seed")
    p_ingest.add_argument("--sampler", default="greedy", help="sampler name (default: greedy)")
    p_ingest.add_argument(
        "--temperature", type=float, default=0.0,
        help="sampling temperature (default: 0.0)",
    )
    p_ingest.add_argument(
        "--top-k", type=int, default=0, dest="top_k",
        help="top-k sampling cutoff (default: 0 = disabled)",
    )
    p_ingest.add_argument(
        "--top-p", type=float, default=1.0, dest="top_p",
        help="nucleus sampling cutoff (default: 1.0 = disabled)",
    )

    p_show = sub.add_parser("show", help="pretty-print one run record")
    p_show.add_argument("record", help="path to a record JSON file")
    p_show.add_argument("--limit", type=int, default=8, help="tokens to print (default: 8)")

    p_ls = sub.add_parser("ls", help="list run records under a directory")
    p_ls.add_argument("directory", help="directory to scan recursively for records")

    p_diff = sub.add_parser(
        "diff", help="pinpoint and explain the first divergent token between two runs"
    )
    p_diff.add_argument("a", help="first record")
    p_diff.add_argument("b", help="second record")
    p_diff.add_argument(
        "--basis", default="auto", choices=("auto", "id", "text"),
        help="compare token ids or texts (default: auto — ids when both have them)",
    )
    p_diff.add_argument(
        "--context", type=int, default=3,
        help="tokens of context around the divergence (default: 3)",
    )
    p_diff.add_argument(
        "--tie-epsilon", type=float, default=DEFAULT_TIE_EPSILON, dest="tie_epsilon",
        help=f"logprob gap (nats) treated as a tie (default: {DEFAULT_TIE_EPSILON})",
    )
    p_diff.add_argument("--json", action="store_true", help="machine-readable output")

    p_matrix = sub.add_parser(
        "matrix", help="group runs into output classes and analyze config axes"
    )
    p_matrix.add_argument("paths", nargs="+", help="record files and/or directories")
    p_matrix.add_argument(
        "--basis", default="auto", choices=("auto", "id", "text"),
        help="compare token ids or texts (default: auto — ids when all runs have them)",
    )
    p_matrix.add_argument("--json", action="store_true", help="machine-readable output")

    p_check = sub.add_parser(
        "check", help="CI gate: exit 1 unless every run produced the same stream"
    )
    p_check.add_argument("paths", nargs="+", help="record files and/or directories")
    p_check.add_argument(
        "--basis", default="auto", choices=("auto", "id", "text"),
        help="compare token ids or texts (default: auto — ids when all runs have them)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_ERROR
    handlers = {
        "ingest": _cmd_ingest,
        "show": _cmd_show,
        "ls": _cmd_ls,
        "diff": _cmd_diff,
        "matrix": _cmd_matrix,
        "check": _cmd_check,
    }
    try:
        return handlers[args.command](args)
    except SeedproofError as exc:
        print(f"seedproof: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except OSError as exc:
        print(f"seedproof: error: {exc}", file=sys.stderr)
        return EXIT_ERROR


# -- helpers ------------------------------------------------------------------


def _collect_records(paths: List[str]) -> List[RunRecord]:
    """Load records from a mix of files and directories (scanned recursively)."""
    records: List[RunRecord] = []
    for path in paths:
        if os.path.isdir(path):
            found = []
            for root, _dirs, files in sorted(os.walk(path)):
                for filename in sorted(files):
                    if filename.endswith(".json"):
                        found.append(os.path.join(root, filename))
            if not found:
                raise SeedproofError(f"no .json records found under {path}")
            for file_path in found:
                records.append(load(file_path))
        else:
            records.append(load(path))
    return records


# -- ingest -------------------------------------------------------------------


def _cmd_ingest(args: argparse.Namespace) -> int:
    if args.input == "-":
        text = sys.stdin.read()
    else:
        with open(args.input, "r", encoding="utf-8") as handle:
            text = handle.read()
    config = RunConfig(
        model=args.model,
        backend=args.backend,
        device=args.device,
        quant=args.quant,
        seed=args.seed,
        sampler=args.sampler,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    name = args.name
    if not name and args.output:
        name = os.path.splitext(os.path.basename(args.output))[0]
    record = ingest(
        text,
        args.format,
        config=config,
        prompt=args.prompt,
        name=name,
        notes=args.note,
    )
    if args.output:
        save(record, args.output)
        print(
            f"wrote {args.output} ({pluralize(len(record.tokens), 'token')}, "
            f"stream {record.stream_fingerprint('text')})"
        )
    else:
        sys.stdout.write(dumps(record))
    return EXIT_OK


# -- show / ls ----------------------------------------------------------------


def _cmd_show(args: argparse.Namespace) -> int:
    record = load(args.record)
    for line in render_show(record, limit=args.limit):
        print(line)
    return EXIT_OK


def _cmd_ls(args: argparse.Namespace) -> int:
    if not os.path.isdir(args.directory):
        raise SeedproofError(f"not a directory: {args.directory}")
    rows = []
    for root, _dirs, files in sorted(os.walk(args.directory)):
        for filename in sorted(files):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(root, filename)
            rel = os.path.relpath(path, args.directory)
            try:
                record = load(path)
            except SeedproofError as exc:
                rows.append((rel, "-", "-", "-", "-", f"skipped: {exc}"))
                continue
            config = record.config
            rows.append(
                (
                    rel,
                    config.backend or "-",
                    config.quant or "-",
                    "-" if config.seed is None else str(config.seed),
                    str(len(record.tokens)),
                    record.stream_fingerprint("text"),
                )
            )
    if not rows:
        print(f"no records found under {args.directory}")
        return EXIT_OK
    header = ("RECORD", "BACKEND", "QUANT", "SEED", "TOKENS", "STREAM")
    for line in format_table(header, rows):
        print(line)
    return EXIT_OK


# -- diff ---------------------------------------------------------------------


def _cmd_diff(args: argparse.Namespace) -> int:
    record_a, record_b = load(args.a), load(args.b)
    diagnosis = diagnose(
        record_a, record_b, basis=args.basis, tie_epsilon=args.tie_epsilon
    )
    if args.json:
        payload = diagnosis.to_dict()
        payload["a"], payload["b"] = record_a.name, record_b.name
        print(json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2))
    else:
        for line in render_diff(diagnosis, record_a, record_b, context=args.context):
            print(line)
    return EXIT_DIVERGENT if diagnosis.divergence.diverged else EXIT_OK


# -- matrix / check -----------------------------------------------------------


def _cmd_matrix(args: argparse.Namespace) -> int:
    records = _collect_records(args.paths)
    report = build_matrix(records, basis=args.basis)
    if args.json:
        print(json.dumps(matrix_to_dict(report), sort_keys=True, ensure_ascii=False, indent=2))
    else:
        for line in render_matrix(report):
            print(line)
    return EXIT_OK


def _cmd_check(args: argparse.Namespace) -> int:
    records = _collect_records(args.paths)
    report = build_matrix(records, basis=args.basis)
    if report.consistent:
        print(
            f"OK: {len(records)} runs, 1 stream "
            f"({report.classes[0].fingerprint}, basis {report.basis})"
        )
        return EXIT_OK
    print(
        f"FAIL: {len(records)} runs split into {len(report.classes)} distinct streams"
    )
    for line in render_matrix(report):
        print(line)
    return EXIT_DIVERGENT


if __name__ == "__main__":  # pragma: no cover - exercised via console script
    sys.exit(main())
