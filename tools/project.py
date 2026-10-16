#!/usr/bin/env python3
"""Audit/export source or verify/download weights. No implicit Git writes."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.maintenance.publishing import audit, compare_to_remote, export
from tools.maintenance.weights import ensure_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("audit", help="Preview the exact source publication list")
    export_parser = commands.add_parser("export", help="Create a new release candidate")
    export_parser.add_argument("--destination", type=Path, required=True)
    weights_parser = commands.add_parser("weights", help="Verify existing checkpoints")
    weights_parser.add_argument("--download", action="store_true")
    weights_parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args()
    if args.command in (None, "audit"):
        result = audit(args.root)
    elif args.command == "export":
        result = export(args.root, args.destination)
        result["comparison"] = compare_to_remote(args.root, args.destination)
    else:
        result = ensure_weights(args.root, args.download, args.include_optional)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
