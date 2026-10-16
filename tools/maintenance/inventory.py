"""Read-only discovery of production references and archive candidates.

Only paths and reference locations are recorded; database field contents and
process arguments are never exported. Symbolic links are not traversed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS = {
    "__init__.py",
    "sample_diffusion.py",
    "sample_for_pocket.py",
    "evaluate_diffusion.py",
    "check_model_type.py",
    "setup_pt_loader.sh",
}
ROOT_PYTHON = {
    "batch_sampleandeval_parallel.py",
    "evaluate_pt_with_correct_reconstruct.py",
    "evaluate_pocket_quality.py",
    "extract_pt_to_sdf_excel.py",
    "scaffold_cascade_pipeline.py",
    "run_prudent_generations.py",
    "refine_saved_pt_baseline_then_eval.py",
    "split_and_eval_parallel.py",
    "show_sampling_steps.py",
    "_diffdynamic_bootstrap.py",
}
TESTS = {
    "auth_prefix",
    "diffdynamic_security",
    "generation_visualization",
    "go_live_security",
    "molecule_ingest_dirs",
    "pt_access_control",
    "quota_daily",
    "reconstruct_scaffold_bonds",
    "scaffold_dual_masks",
    "scaffold_placement_web",
    "scaffold_sites",
    "smoke_api",
    "targetdiff_baseline_lock",
    "visual_sim",
    "web_limits",
    "gspt1_scaffold_prior",
    "prudent_terminal_rollout",
    "prudent_targetdiff_reconstruction",
}
RESEARCH_HELPERS = {
    "gspt1_class_setup.py",
    "gspt1_interface_setup.py",
    "ikzf2_scaffold_setup.py",
}
RESEARCH_ROOTS = {
    "ablationstudy",
    "analysis",
    "batchsummary",
    "diagnostics",
    "eval",
    "experiments",
    "fig7",
    "gspt1_handoff",
    "hsvpol",
    "jsdpt3010",
    "jsdpt3010_pilot_aggressive",
    "jsdpt3010_pilot_control",
    "jsdpt3010_pilot_medchem",
    "jsdpt3010_rdkit_repair_chem",
    "molecules_score_65-70",
    "molecules_score_70-80",
    "molecules_score_80plus",
    "pics",
    "pocket_quality_vis",
    "rdkit-structure-repair",
    "runs",
}
RESEARCH_COLLECTIONS = {
    "ablationstudy",
    "analysis",
    "diagnostics",
    "eval",
    "experiments",
    "pocket_quality_vis",
    "runs",
}
LOCAL_ROOTS = {"local", ".git", "__pycache__", ".cache", "tmp", "tmp_bench", "docktmp"}
PATH_PATTERN = re.compile(r"(?<![\w])(?:/[^\s\"\'<>]+|(?:[.\w-]+/)+[^\s\"\'<>]+)")


def absolute(path):
    """Normalize a path lexically, preserving symlink identity."""
    return Path(os.path.abspath(os.fspath(path)))


def under(path, parent):
    try:
        absolute(path).relative_to(absolute(parent))
        return True
    except ValueError:
        return False


def paths_in(value, root, base=None):
    """Extract path-like strings, including nested JSON, without retaining text."""
    if isinstance(value, dict):
        for child in value.values():
            yield from paths_in(child, root, base)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from paths_in(child, root, base)
    elif isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            decoded = None
        if isinstance(decoded, (dict, list)):
            yield from paths_in(decoded, root, base)
            return
        candidates = [value] + PATH_PATTERN.findall(value)
        for token in candidates:
            token = token.strip().rstrip(",;:)]}").lstrip('([{"').rstrip('"')
            if not token or "\n" in token or "\x00" in token or "://" in token:
                continue
            if token.startswith("/") or "/" in token or os.path.lexists(root / token):
                candidate = absolute(
                    token if token.startswith("/") else (base or root) / token
                )
                if under(candidate, root) and candidate != root:
                    yield candidate
                    resolved = candidate.resolve(strict=False)
                    if resolved != candidate and under(resolved, root):
                        yield resolved


def walk(path):
    """Yield a tree in stable order without following links or embedded .git."""
    path = absolute(path)
    yield path
    if path.is_dir() and not path.is_symlink():
        for child in sorted(path.iterdir(), key=lambda item: item.name):
            if child.name == ".git":
                continue
            yield from walk(child)


def tree_record(path):
    path = absolute(path)
    members = []
    for item in walk(path):
        stat = item.lstat()
        kind = (
            "symlink" if item.is_symlink() else "directory" if item.is_dir() else "file"
        )
        members.append(
            {
                "relative": "." if item == path else str(item.relative_to(path)),
                "kind": kind,
                "size": stat.st_size if kind != "directory" else 0,
                "mtime_ns": stat.st_mtime_ns,
                "target": os.readlink(item) if kind == "symlink" else None,
            }
        )
    fingerprint = hashlib.sha256(
        json.dumps(members, sort_keys=True).encode()
    ).hexdigest()
    return {
        "members": members,
        "size": sum(item["size"] for item in members),
        "fingerprint": fingerprint,
    }


def production_files(code_root):
    """Initial release sources; expand to imported Python and literal CLI paths."""
    code_root = absolute(code_root)
    selected = set()
    # Publication metadata is maintained source, never a historical result.
    # Share the explicit release allowlist instead of guessing by extension.
    if (code_root / "publish-manifest.json").is_file():
        from .publishing import source_files

        selected.update(source_files(code_root).values())
    for directory in (
        "src",
        "server",
        "models",
        "datasets",
        "utils",
        "paper_plot_scripts",
        "ui",
        "tools",
    ):
        base = code_root / directory
        if not base.exists():
            continue
        for item in walk(base):
            if (
                item.is_file()
                and "__pycache__" not in item.parts
                and item.suffix != ".pyc"
            ):
                if directory == "utils" and item.name in RESEARCH_HELPERS:
                    continue
                selected.add(item)
    for name in ROOT_PYTHON | {"start_server.sh"}:
        if (code_root / name).is_file():
            selected.add(code_root / name)
    for directory, names in (
        ("scripts", SCRIPTS),
        ("tests", {"conftest.py"} | {"test_" + name + ".py" for name in TESTS}),
    ):
        for name in names:
            if (code_root / directory / name).is_file():
                selected.add(code_root / directory / name)
    tests = code_root / "tests"
    if tests.exists():
        selected.update(tests.glob("test_refactor_*.py"))
    for name in (
        "sampling.yml",
        "benchmark_reference.yml",
        "sampling_strategies",
        "ablation/prudent_terminal_rollout",
        "ablation/prudent_rounds",
    ):
        target = code_root / "configs" / name
        if target.exists():
            selected.update(item for item in walk(target) if item.is_file())
    pending = list(selected)
    while pending:
        item = pending.pop()
        if item.suffix != ".py" or item.is_symlink():
            continue
        try:
            tree = ast.parse(item.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeError, OSError):
            continue
        candidates = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                candidates.update(alias.name.replace(".", "/") for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = (node.module or "").replace(".", "/")
                if node.level:
                    parent = item.parent
                    for _ in range(node.level - 1):
                        parent = parent.parent
                    base = str(parent.relative_to(code_root) / base)
                candidates.add(base)
                candidates.update(
                    base + "/" + alias.name for alias in node.names if alias.name != "*"
                )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                token = node.value.strip()
                if (
                    len(token) < 512
                    and "\n" not in token
                    and token.endswith((".py", ".sh", ".yml", ".yaml"))
                ):
                    if token.startswith(str(code_root)):
                        candidates.add(str(Path(token).relative_to(code_root)))
                    elif not token.startswith("/"):
                        candidates.add(token)
                        candidates.add(str(item.parent.relative_to(code_root) / token))
        for token in candidates:
            for suffix in ("", ".py", "/__init__.py"):
                candidate = absolute(code_root / (token + suffix))
                if (
                    under(candidate, code_root)
                    and candidate.is_file()
                    and candidate not in selected
                ):
                    selected.add(candidate)
                    pending.append(candidate)
    return selected


def collect_references(root, code_root=None):
    root = absolute(root)
    references = {}
    notes = []

    def protect(candidate, reason):
        candidate = absolute(candidate)
        if under(candidate, root) and candidate != root:
            references.setdefault(str(candidate), set()).add(reason)

    for name in (
        "diffdynamic.db",
        "diffdynamic.db-wal",
        "diffdynamic.db-shm",
        "data",
        "uploads",
        "$upload",
        "outputs",
        "pretrained_models",
        "evalresults record.xlsx",
        "evalresults record.xlsx.lock",
        "model_comparison_data.xlsx",
        "batchsummary",
        "molecules_score_65-70",
        "molecules_score_70-80",
        "molecules_score_80plus",
        "local",
        "server/server_config.json",
        "server/.internal_api_key",
    ):
        protect(root / name, "runtime-resource")
    config = root / "server/server_config.json"
    if config.exists():
        try:
            value = json.loads(config.read_text())
            for candidate in paths_in(value, root):
                protect(candidate, "runtime-config")
        except (OSError, ValueError) as error:
            notes.append("runtime-config-read-error: " + type(error).__name__)
    database = root / "diffdynamic.db"
    if database.exists():
        try:
            connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
            try:
                connection.execute("PRAGMA query_only=ON")
                for table, wanted in {
                    "runs": {
                        "output_path",
                        "parameters",
                        "config_snapshot",
                        "progress_detail",
                    },
                    "molecules": {"sdf_path"},
                    "evaluations": {"details"},
                    "history": {"details"},
                }.items():
                    columns = {
                        row[1]
                        for row in connection.execute(
                            'PRAGMA table_info("' + table + '")'
                        )
                    }
                    for column in sorted(columns & wanted):
                        for (value,) in connection.execute(
                            'SELECT "'
                            + column
                            + '" FROM "'
                            + table
                            + '" WHERE "'
                            + column
                            + '" IS NOT NULL'
                        ):
                            for candidate in paths_in(value, root):
                                protect(candidate, "database:" + table + "." + column)
            finally:
                connection.close()
        except sqlite3.Error as error:
            notes.append("database-read-error: " + type(error).__name__)
    source = absolute(code_root or root)
    for item in production_files(source):
        relative = item.relative_to(source)
        protect(root / relative, "production-dependency")
        if item.is_symlink():
            for candidate in paths_in(str(item.resolve(strict=False)), root):
                protect(candidate, "production-symlink")
    proc = Path("/proc")
    if proc.exists():
        for process in proc.iterdir():
            if not process.name.isdigit() or int(process.name) == os.getpid():
                continue
            try:
                cwd = absolute(os.readlink(process / "cwd"))
                if under(cwd, root) and cwd != root:
                    protect(cwd, "process-cwd:" + process.name)
                environment = (process / "environ").read_bytes().split(b"\x00")
                for entry in environment:
                    key, _, value = entry.partition(b"=")
                    if key in {
                        b"DD_DB_PATH",
                        b"DD_DATA",
                        b"DD_OUTPUT",
                        b"DD_MODELS",
                        b"DD_UPLOAD",
                        b"DD_SAMPLING_YML",
                    }:
                        for candidate in paths_in(
                            value.decode(errors="replace"), root, base=cwd
                        ):
                            protect(candidate, "process-environment:" + process.name)
                arguments = (
                    (process / "cmdline")
                    .read_bytes()
                    .decode(errors="replace")
                    .split("\x00")
                )
                for argument in arguments:
                    for candidate in paths_in(argument, root, base=cwd):
                        if under(candidate, root):
                            protect(candidate, "process-argument:" + process.name)
                for descriptor in (process / "fd").iterdir():
                    try:
                        target = os.readlink(descriptor)
                        if target.startswith("/"):
                            protect(
                                target.removesuffix(" (deleted)")
                                if hasattr(target, "removesuffix")
                                else target.replace(" (deleted)", ""),
                                "process-fd:" + process.name,
                            )
                    except OSError:
                        pass
            except (OSError, PermissionError):
                continue
    for variable in (
        "DD_DB_PATH",
        "DD_DATA",
        "DD_OUTPUT",
        "DD_MODELS",
        "DD_UPLOAD",
        "DD_SAMPLING_YML",
    ):
        for candidate in paths_in(os.environ.get(variable, ""), root):
            protect(candidate, "environment:" + variable)
    visited = set()
    while True:
        configs = [
            Path(value)
            for value in references
            if value not in visited and Path(value).suffix in {".json", ".yml", ".yaml"}
        ]
        if not configs:
            break
        for config in configs:
            visited.add(str(config))
            # A migration journal contains old/new paths for the whole project.
            # Reading it as a job config would protect every migration source.
            if under(config, root / "local" / "inventory") or config.name in {
                "publish-manifest.json",
                "weights-manifest.json",
            }:
                continue
            if not config.is_file() or config.stat().st_size > 2 * 1024 * 1024:
                continue
            try:
                for candidate in paths_in(config.read_text(encoding="utf-8"), root):
                    protect(
                        candidate, "referenced-config:" + str(config.relative_to(root))
                    )
            except (OSError, UnicodeError) as error:
                notes.append(
                    "referenced-config-read-error:"
                    + str(config.relative_to(root))
                    + ":"
                    + type(error).__name__
                )
    return {
        path: sorted(reasons) for path, reasons in sorted(references.items())
    }, notes


def archive_candidates(root, code_root=None):
    root = absolute(root)
    production = {
        item.relative_to(absolute(code_root or root))
        for item in production_files(code_root or root)
    }
    for directory in ("scripts", "tests", "configs"):
        base = root / directory
        if not base.exists():
            continue
        for item in walk(base):
            if item.is_dir() and not item.is_symlink():
                continue
            if "__pycache__" in item.parts or item.suffix == ".pyc":
                continue
            if item.relative_to(root) not in production:
                yield item, "archive", "historical-task-source"
    for name in RESEARCH_HELPERS:
        if (root / "utils" / name).exists() and Path("utils", name) not in production:
            yield root / "utils" / name, "archive", "historical-research-helper"
    for item in sorted(root.iterdir()):
        if item.name in RESEARCH_ROOTS:
            if (
                item.name in RESEARCH_COLLECTIONS
                and item.is_dir()
                and not item.is_symlink()
            ):
                for topic in sorted(item.iterdir()):
                    if topic.name in {"__pycache__", ".cache"}:
                        continue
                    if (
                        item.name == "runs"
                        and topic.name == "diffdynamic_cross_backbone_100p"
                        and topic.is_dir()
                        and not topic.is_symlink()
                    ):
                        for run in sorted(topic.iterdir()):
                            yield run, "research", "historical-research-tree"
                    else:
                        yield topic, "research", "historical-research-tree"
            else:
                yield item, "research", "historical-research-tree"
        elif item.name == "obsolete":
            yield item, "archive", "previous-archive"
        elif item.is_file() and item.relative_to(root) not in production:
            if item.suffix in {".py", ".sh"} or item.name in {
                "SEEDFORGE.md",
                "PRUDENT_VS_NORMAL_REPORT.md",
            }:
                yield item, "archive", "historical-task-source"
            elif item.suffix in {".xlsx", ".csv", ".json"} or item.name in {
                "pt_result.txt",
                "pt_inspect_result.txt",
            }:
                yield item, "research", "historical-summary"


def build_plan(root, code_root=None):
    root = absolute(root)
    references, notes = collect_references(root, code_root)
    protected = [Path(value) for value in references]
    entries, blocked = [], []

    def choose(item, category, purpose):
        direct = [value for value in protected if under(item, value)]
        descendants = [value for value in protected if under(value, item)]
        if direct:
            blocked.append(
                {
                    "path": str(item),
                    "reason": "protected-reference",
                    "references": [str(value) for value in direct],
                }
            )
            return
        if descendants:
            blocked.append(
                {
                    "path": str(item),
                    "reason": "protected-topic-tree",
                    "references": [str(value) for value in descendants],
                }
            )
            return
        record = tree_record(item)
        destination = root / "local" / category / item.relative_to(root)
        entries.append(
            {
                "old_path": str(item),
                "new_path": str(destination),
                "category": category,
                "purpose": purpose,
                "github": "local-only",
                "historical_direct_run": "not-maintained"
                if category == "archive"
                else "data-only",
                "status": "planned",
                **record,
            }
        )

    for item, category, purpose in archive_candidates(root, code_root):
        choose(item, category, purpose)
    symlinks = []
    for child in sorted(root.iterdir()):
        if child.name in LOCAL_ROOTS or child.name.startswith(".venv"):
            continue
        for item in walk(child):
            if item.is_symlink():
                symlinks.append(
                    {
                        "path": str(item),
                        "target": os.readlink(item),
                        "existed": item.exists(),
                    }
                )
    classifications = []
    moved_paths = {entry["old_path"] for entry in entries}
    public_source_paths = {
        str(root / item.relative_to(absolute(code_root or root)))
        for item in production_files(code_root or root)
    }
    source_dirs = {
        "src",
        "server",
        "models",
        "datasets",
        "utils",
        "ui",
        "paper_plot_scripts",
        "tools",
        "docs",
        "third_party",
    }
    for item in sorted(root.iterdir()):
        if str(item) in moved_paths:
            continue
        reasons = references.get(str(item), [])
        runtime = (
            "runtime-resource" in reasons
            or item.name in LOCAL_ROOTS
            or item.name.startswith(".venv")
        )
        if runtime:
            category, github, purpose = (
                "runtime",
                "local-only",
                "retained-runtime-resource",
            )
        elif item.name in {"scripts", "tests", "configs"}:
            category, github, purpose = (
                "mixed-source",
                "selected",
                "retained-production-and-protected-children",
            )
        elif (
            str(item) in public_source_paths
            or item.name in source_dirs
            or item.name in ROOT_PYTHON
            or item.name
            in {
                "start_server.sh",
                "README.md",
                "README.zh-CN.md",
                "requirements.txt",
                "requirements-web.txt",
                "pyproject.toml",
                "LICENSE",
                ".gitignore",
            }
        ):
            category, github, purpose = (
                "production-source",
                "required",
                "retained-production-source",
            )
        else:
            category, github, purpose = (
                "local",
                "local-only",
                "retained-local-or-research-content",
            )
        classifications.append(
            {
                "old_path": str(item),
                "new_path": str(item),
                "category": category,
                "github": github,
                "purpose": purpose,
                "status": "retained",
                "size": ""
                if item.is_dir() and not item.is_symlink()
                else item.lstat().st_size,
                "kind": "symlink"
                if item.is_symlink()
                else "directory"
                if item.is_dir()
                else "file",
                "historical_direct_run": "unchanged",
            }
        )
    classified = {entry["old_path"] for entry in classifications}
    for entry in blocked:
        if entry["path"] in classified:
            continue
        item = Path(entry["path"])
        classifications.append(
            {
                "old_path": str(item),
                "new_path": str(item),
                "category": "protected",
                "github": "local-only",
                "purpose": entry["reason"],
                "status": "retained",
                "size": ""
                if item.is_dir() and not item.is_symlink()
                else item.lstat().st_size,
                "kind": "symlink"
                if item.is_symlink()
                else "directory"
                if item.is_dir()
                else "file",
                "historical_direct_run": "unchanged",
            }
        )
    return {
        "classifications": classifications,
        "version": 1,
        "root": str(root),
        "code_root": str(absolute(code_root or root)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
        "blocked": blocked,
        "references": references,
        "symlinks": symlinks,
        "symlink_edits": [],
        "notes": notes,
        "events": [],
    }
