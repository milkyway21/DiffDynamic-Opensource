"""Auditable rule-based molecular processing independent of the web server."""

RULE_VERSION = "medchem-2"


def ensure_repair_package():
    import sys
    from diffdynamic.io.paths import project_root

    path = project_root() / "rdkit-structure-repair" / "src"
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
