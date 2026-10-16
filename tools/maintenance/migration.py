"""Journaled, same-filesystem migration without overwriting user files."""

from __future__ import annotations

import csv
import ctypes
import errno
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .inventory import absolute, collect_references, tree_record, under


def save_plan(plan, filename, tables=False):
    """Atomically persist state and refresh both human-readable location maps."""
    filename = absolute(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    initial = "members_file" not in plan
    members_name = plan.setdefault(
        "members_file", filename.with_suffix(".members.json").name
    )
    members_path = filename.parent / members_name
    if initial and members_path.exists():
        raise FileExistsError("Member inventory already exists: " + str(members_path))
    if not members_path.exists():
        with members_path.open("x", encoding="utf-8") as stream:
            json.dump(
                {entry["old_path"]: entry["members"] for entry in plan["entries"]},
                stream,
                ensure_ascii=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
    serialized = {
        **plan,
        "entries": [
            {key: value for key, value in entry.items() if key != "members"}
            for entry in plan["entries"]
        ],
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=filename.parent,
        prefix=filename.name + ".",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(serialized, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if initial:
            rename_without_replace(temporary, filename)
        else:
            os.replace(temporary, filename)
    finally:
        if temporary.exists():
            temporary.unlink()
    if not tables:
        return
    table = filename.with_suffix(".csv")
    fields = [
        "old_path",
        "new_path",
        "category",
        "purpose",
        "size",
        "status",
        "github",
        "historical_direct_run",
        "kind",
    ]
    with table.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for entry in plan["entries"]:
            for member in entry["members"]:
                row = {field: entry.get(field, "") for field in fields}
                for field in ("old_path", "new_path"):
                    row[field] = str(Path(entry[field]) / member["relative"])
                row.update(size=member["size"], kind=member["kind"])
                writer.writerow(row)
        for entry in plan.get("classifications", []):
            writer.writerow({field: entry.get(field, "") for field in fields})
    compact = filename.with_suffix(".compact.csv")
    with compact.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for entry in plan["entries"]:
            row = {field: entry.get(field, "") for field in fields}
            row["kind"] = entry["members"][0]["kind"]
            writer.writerow(row)
        for entry in plan.get("classifications", []):
            writer.writerow({field: entry.get(field, "") for field in fields})
    markdown = filename.with_suffix(".md")
    with markdown.open("w", encoding="utf-8") as stream:
        stream.write("# Local file migration inventory\n\n")
        stream.write(
            "Full file and symlink positions are in the companion CSV. Absolute paths are local-only.\n\n"
        )
        for entry in plan["entries"]:
            stream.write(
                "- **{}** · {} · {} bytes · {} objects\n  - `{}` → `{}`\n".format(
                    entry["status"],
                    entry["category"],
                    entry["size"],
                    len(entry["members"]),
                    entry["old_path"],
                    entry["new_path"],
                )
            )
        stream.write("\n## Retained resources and project classification\n\n")
        for entry in plan.get("classifications", []):
            stream.write(
                "- `{}` → same location · {} · {} · {}\n".format(
                    entry["old_path"],
                    entry["category"],
                    entry["github"],
                    entry["purpose"],
                )
            )
        stream.write("\n## Protected candidates\n\n")
        for entry in plan["blocked"]:
            stream.write("- `{}`: {}\n".format(entry["path"], entry["reason"]))
        if plan["notes"]:
            stream.write("\n## Discovery limitations\n\n")
            stream.writelines("- " + note + "\n" for note in plan["notes"])


def load_plan(filename):
    with open(filename, encoding="utf-8") as stream:
        plan = json.load(stream)
    if plan.get("version") != 1:
        raise ValueError("Unsupported migration manifest version")
    members_name = plan.get("members_file")
    if not members_name or Path(members_name).name != members_name:
        raise ValueError("Invalid member inventory reference")
    with (absolute(filename).parent / members_name).open(encoding="utf-8") as stream:
        members = json.load(stream)
    for entry in plan["entries"]:
        entry["members"] = members[entry["old_path"]]
    root = absolute(plan["root"])
    for entry in plan["entries"]:
        if entry["category"] not in {"archive", "research"}:
            raise ValueError("Unknown migration category")
        old, new = absolute(entry["old_path"]), absolute(entry["new_path"])
        expected = root / "local" / entry["category"] / old.relative_to(root)
        retained = (
            entry["status"] == "retained"
            and new == old
            and absolute(entry.get("planned_new_path", old)) == expected
        )
        if (
            not under(old, root)
            or under(old, root / "local")
            or (new != expected and not retained)
        ):
            raise ValueError("Manifest path leaves the migration boundary")
    return plan


def event(plan, operation, path, status, detail=""):
    plan["events"].append(
        {
            "time": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "path": str(path),
            "status": status,
            "detail": detail,
        }
    )


def retain_referenced_entry(plan, entry):
    """Keep a newly active tree in place and record its actual location."""
    reason = "Source gained a production, database, or process reference"
    entry.update(
        planned_new_path=entry["new_path"],
        new_path=entry["old_path"],
        status="retained",
        retained_reason=reason,
        purpose="retained-new-production-reference",
        historical_direct_run="unchanged",
    )
    if not any(item["path"] == entry["old_path"] for item in plan["blocked"]):
        plan["blocked"].append({"path": entry["old_path"], "reason": reason})
    event(plan, "retain", entry["old_path"], "retained", reason)


def rename_without_replace(source, destination):
    """Use Linux renameat2(RENAME_NOREPLACE); never degrade to unsafe rename."""
    source, destination = absolute(source), absolute(destination)
    if os.path.lexists(destination):
        raise FileExistsError(str(destination))
    ancestor = destination.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if source.parent.resolve() != source.parent:
        raise ValueError("Source parents must not be symlinks")
    if source.lstat().st_dev != ancestor.stat().st_dev:
        raise OSError(errno.EXDEV, "Migration must remain on one filesystem")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.resolve() != destination.parent:
        raise ValueError("Destination parents must not be symlinks")
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise RuntimeError("Atomic no-overwrite renameat2 is unavailable")
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def mapped_path(path, entries, reverse=False):
    path = absolute(path)
    for entry in sorted(
        entries, key=lambda value: len(value["old_path"]), reverse=True
    ):
        source, destination = (
            (entry["new_path"], entry["old_path"])
            if reverse
            else (entry["old_path"], entry["new_path"])
        )
        if under(path, source):
            return absolute(Path(destination) / path.relative_to(source))
    return path


def replace_link(path, expected, target):
    """Replace only an unchanged existing symlink, using an exclusive temp link."""
    if not path.is_symlink() or os.readlink(path) != expected:
        raise ValueError("Symbolic link changed after inventory: " + str(path))
    temporary = path.with_name(path.name + ".dd-migration-link")
    os.symlink(target, temporary)
    try:
        if not path.is_symlink() or os.readlink(path) != expected:
            raise ValueError("Symbolic link changed during migration: " + str(path))
        os.replace(temporary, path)
    finally:
        if temporary.is_symlink():
            temporary.unlink()


def update_symlinks(plan, filename):
    moved = [entry for entry in plan["entries"] if entry["status"] == "moved"]
    existing_edits = {entry["old_path"]: entry for entry in plan["symlink_edits"]}
    for link in plan["symlinks"]:
        original = Path(link["path"])
        current = mapped_path(original, moved)
        if not current.is_symlink():
            if original in [Path(entry["old_path"]) for entry in moved]:
                raise ValueError("Moved symlink is missing: " + str(current))
            continue
        target = absolute(
            link["target"]
            if os.path.isabs(link["target"])
            else original.parent / link["target"]
        )
        new_target_path = mapped_path(target, moved)
        new_target = (
            str(new_target_path)
            if os.path.isabs(link["target"])
            else os.path.relpath(new_target_path, current.parent)
        )
        if new_target == link["target"]:
            continue
        prior = existing_edits.get(str(original))
        if prior and os.readlink(current) == new_target:
            prior["status"] = "updated"
            continue
        edit = prior or {
            "old_path": str(original),
            "new_path": str(current),
            "old_target": link["target"],
            "new_target": new_target,
            "status": "updating",
        }
        expected = os.readlink(current)
        allowed = {link["target"], new_target}
        if prior:
            allowed.update(
                {prior["new_target"], prior.get("previous_target", prior["old_target"])}
            )
        if expected not in allowed:
            raise ValueError("Symbolic link changed after inventory: " + str(current))
        edit.update(
            new_path=str(current),
            new_target=new_target,
            previous_target=expected,
            status="updating",
        )
        if not prior:
            plan["symlink_edits"].append(edit)
        save_plan(plan, filename)
        replace_link(current, expected, new_target)
        edit["status"] = "updated"
        event(plan, "symlink", current, "updated")
        save_plan(plan, filename)
    broken = []
    for link in plan["symlinks"]:
        current = mapped_path(link["path"], moved)
        if link["existed"] and current.is_symlink() and not current.exists():
            broken.append(str(current))
    if broken:
        raise ValueError("Migration introduced broken links: " + ", ".join(broken))


def apply_plan(filename):
    plan = load_plan(filename)
    if any(entry["status"] == "rolled-back" for entry in plan["entries"]):
        raise ValueError("Create a new plan after rollback")
    for entry in plan["entries"]:
        if entry["status"] == "retained" and not os.path.lexists(entry["old_path"]):
            raise ValueError("Retained source is missing: " + entry["old_path"])
    references, notes = collect_references(plan["root"], plan["code_root"])
    if notes or plan["notes"]:
        raise ValueError(
            "Reference discovery failed; inspect manifest notes before applying"
        )
    failures = []
    for entry in plan["entries"]:
        if entry["status"] == "retained":
            continue
        source, destination = Path(entry["old_path"]), Path(entry["new_path"])
        try:
            if entry["status"] == "moved":
                if os.path.lexists(source) or not os.path.lexists(destination):
                    raise ValueError("Previously moved paths no longer match journal")
                continue
            if (
                entry["status"] == "moving"
                and not os.path.lexists(source)
                and os.path.lexists(destination)
            ):
                if tree_record(destination)["fingerprint"] != entry["fingerprint"]:
                    raise ValueError(
                        "Cannot recover an interrupted move: destination changed"
                    )
                entry["status"] = "moved"
                save_plan(plan, filename)
                continue
            if any(
                under(source, value) or under(value, source) for value in references
            ):
                retain_referenced_entry(plan, entry)
                save_plan(plan, filename)
                continue
            if tree_record(source)["fingerprint"] != entry["fingerprint"]:
                raise ValueError("Source changed after planning")
            entry["status"] = "moving"
            save_plan(plan, filename)
            rename_without_replace(source, destination)
            entry["status"] = "moved"
            event(plan, "move", source, "moved")
        except (OSError, ValueError, RuntimeError) as error:
            entry["status"] = "failed"
            failures.append(str(source))
            event(plan, "move", source, "failed", str(error))
        save_plan(plan, filename)
    try:
        update_symlinks(plan, filename)
    except (OSError, ValueError) as error:
        event(plan, "symlink", plan["root"], "failed", str(error))
        failures.append("symlink-update")
    for entry in plan["entries"]:
        if entry["status"] == "moved":
            entry["after_fingerprint"] = tree_record(entry["new_path"])["fingerprint"]
    save_plan(plan, filename, tables=True)
    if failures:
        raise RuntimeError(
            "Migration partially applied; inspect journal and rollback or retry: "
            + ", ".join(failures)
        )
    return plan


def rollback_plan(filename):
    plan = load_plan(filename)
    failures = []
    for entry in reversed(plan["entries"]):
        if entry["status"] not in {"moved", "rolling-back"}:
            continue
        source, destination = Path(entry["new_path"]), Path(entry["old_path"])
        try:
            if (
                entry["status"] == "rolling-back"
                and not os.path.lexists(source)
                and os.path.lexists(destination)
            ):
                entry["status"] = "rolled-back"
                save_plan(plan, filename)
                continue
            current = tree_record(source)["fingerprint"]
            if current != entry.get("after_fingerprint", entry["fingerprint"]):
                entry["rollback_blocked_by_change"] = True
                raise ValueError("Moved files changed; refusing to overwrite user work")
            entry["status"] = "rolling-back"
            save_plan(plan, filename)
            rename_without_replace(source, destination)
            entry["status"] = "rolled-back"
            event(plan, "rollback", destination, "rolled-back")
        except (OSError, ValueError, RuntimeError) as error:
            entry["status"] = "moved"
            failures.append(str(source))
            event(plan, "rollback", destination, "failed", str(error))
        save_plan(plan, filename)
    remaining = [entry for entry in plan["entries"] if entry["status"] == "moved"]
    if failures:
        try:
            update_symlinks(plan, filename)
        except (OSError, ValueError) as error:
            event(plan, "rollback-link", plan["root"], "failed", str(error))
        for entry in remaining:
            if not entry.get("rollback_blocked_by_change"):
                entry["after_fingerprint"] = tree_record(entry["new_path"])[
                    "fingerprint"
                ]
        save_plan(plan, filename, tables=True)
        raise RuntimeError(
            "Rollback needs attention; inspect journal: " + ", ".join(failures)
        )
    for edit in reversed(plan["symlink_edits"]):
        if edit["status"] == "rolled-back":
            continue
        current = mapped_path(edit["old_path"], remaining)
        try:
            if current.is_symlink() and os.readlink(current) == edit["old_target"]:
                edit["status"] = "rolled-back"
                continue
            replace_link(current, edit["new_target"], edit["old_target"])
            edit["status"] = "rolled-back"
            event(plan, "rollback-link", current, "rolled-back")
        except (OSError, ValueError) as error:
            failures.append(str(current))
            event(plan, "rollback-link", current, "failed", str(error))
        save_plan(plan, filename)
    save_plan(plan, filename, tables=True)
    if failures:
        raise RuntimeError(
            "Rollback needs attention; inspect journal: " + ", ".join(failures)
        )
    return plan
