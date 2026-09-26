"""Create-only clustering CLI over semantic-profile artifacts.

Two subcommands:

  profiles  Expand record rows ({id, state}) into frozen battery probe rows
            suitable for scoring with ``semif-score --mode shared``.
  cluster   Cluster a scored-probe artifact into a versioned cluster catalog.

Model scoring itself stays in ``semif-score``; this CLI never loads a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import digest
from . import clustering, profiles


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as destination:
        destination.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def _cmd_profiles(args: argparse.Namespace) -> None:
    records = _read_jsonl(args.input)
    if not records:
        raise SystemExit("Input is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        for record in records:
            for row in profiles.probe_rows(record):
                destination.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _cmd_cluster(args: argparse.Namespace) -> None:
    scorings = _read_jsonl(args.input)
    if not scorings:
        raise SystemExit("Input is empty")
    artifact = profiles.aggregate(scorings)
    params: dict = {}
    if args.method == "kmeans":
        if args.k is None:
            raise SystemExit("--k is required for --method kmeans")
        params["k"] = args.k
    else:
        params["min_cluster_size"] = args.min_cluster_size
        if args.min_samples is not None:
            params["min_samples"] = args.min_samples
    params["seed"] = args.seed
    result = clustering.assign(artifact, method=args.method, **params)
    if "catalog_version" not in result:
        provenance = [result.get("battery_sha256"), result.get("source_sha256"),
                      result.get("method"), result.get("params")]
        result["catalog_version"] = args.catalog_version or digest(
            json.dumps(provenance, ensure_ascii=False))[:16]
    _write_json(args.output, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    profile_parser = subcommands.add_parser(
        "profiles", help="Expand {id, state} records into frozen probe rows for semif-score")
    profile_parser.add_argument("--input", type=Path, required=True,
                                help="JSONL of records with 'id' and 'state' fields")
    profile_parser.add_argument("--output", type=Path, required=True,
                                help="New JSONL of probe rows to score with semif-score --mode shared")
    profile_parser.set_defaults(handler=_cmd_profiles)

    cluster_parser = subcommands.add_parser(
        "cluster", help="Cluster scored probe rows into a versioned catalog")
    cluster_parser.add_argument("--input", type=Path, required=True,
                                help="JSONL scorings produced by running semif-score over probe rows")
    cluster_parser.add_argument("--output", type=Path, required=True,
                                help="New JSON cluster artifact (create-only)")
    cluster_parser.add_argument("--method", choices=("kmeans", "hdbscan"), default="kmeans")
    cluster_parser.add_argument("--k", type=int, help="Cluster count for --method kmeans")
    cluster_parser.add_argument("--min-cluster-size", type=int, default=5,
                                help="Minimum cluster size for --method hdbscan (default: 5)")
    cluster_parser.add_argument("--min-samples", type=int,
                                help="HDBSCAN min_samples (default: min_cluster_size)")
    cluster_parser.add_argument("--seed", type=int, default=217)
    cluster_parser.add_argument("--catalog-version",
                                help="Stable catalog version string (default: derived from content)")
    cluster_parser.set_defaults(handler=_cmd_cluster)

    args = parser.parse_args()
    for path in (getattr(args, "output", None),):
        if path is not None and path.exists():
            parser.error(f"Output must be new: {path}")
    try:
        args.handler(args)
    except (ValueError, KeyError) as error:
        print(f"semif-cluster: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
