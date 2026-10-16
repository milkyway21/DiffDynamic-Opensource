"""Unified command dispatcher; historical command arguments pass through."""

import importlib
import sys

COMMANDS = {
    "sample": "diffdynamic.sampling.application",
    "batch": "diffdynamic.batch.application",
    "evaluate": "diffdynamic.evaluation.molecule.application",
    "pocket": "diffdynamic.analysis.pocket.application",
    "extract": "diffdynamic.cli.extract",
    "cascade": "diffdynamic.sampling.cascade",
    "plot": "diffdynamic.analysis.plots.plot_generic_results",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: diffdynamic {" + ",".join(COMMANDS) + "} [arguments]")
        return 0
    command = sys.argv[1]
    if command not in COMMANDS:
        print("Unknown command: " + command, file=sys.stderr)
        return 2
    original = sys.argv[:]
    try:
        sys.argv = [original[0] + " " + command] + original[2:]
        return importlib.import_module(COMMANDS[command]).main()
    finally:
        sys.argv = original
