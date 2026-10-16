"""Plan, apply, or reverse a local-only migration; planning is the default.

Run from the project root: python -m tools.maintenance.cli --help
"""

import argparse
import json
from pathlib import Path

from .inventory import build_plan
from .migration import apply_plan, rollback_plan, save_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser(
        "plan", help="Read-only inventory; write local review manifest"
    )
    plan.add_argument("--root", type=Path, required=True)
    plan.add_argument(
        "--code-root", type=Path, help="Final source tree used for dependency closure"
    )
    plan.add_argument("--manifest", type=Path, required=True)
    for name in ("apply", "rollback"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "plan":
        outputs = [
            arguments.manifest,
            arguments.manifest.with_suffix(".members.json"),
            arguments.manifest.with_suffix(".csv"),
            arguments.manifest.with_suffix(".compact.csv"),
            arguments.manifest.with_suffix(".md"),
        ]
        if any(output.exists() for output in outputs):
            parser.error("Inventory output already exists; choose a new review file")
        result = build_plan(arguments.root, arguments.code_root)
        save_plan(result, arguments.manifest, tables=True)
    elif arguments.command == "apply":
        result = apply_plan(arguments.manifest)
    else:
        result = rollback_plan(arguments.manifest)
    print(
        json.dumps(
            {
                "manifest": str(arguments.manifest),
                "entries": len(result["entries"]),
                "bytes": sum(entry["size"] for entry in result["entries"]),
                "protected_candidates": len(result["blocked"]),
                "notes": result["notes"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
