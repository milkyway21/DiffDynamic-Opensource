"""Loading responsibilities extracted from scaffold_prior.py."""

from __future__ import annotations

from diffdynamic.chemistry.profiles.state import Any

from diffdynamic.chemistry.profiles.state import (
    Path,
    SUPPORTED_GENERATION_ELEMENTS,
    SUPPORTED_PROFILE_VERSIONS,
    json,
)


def load_scaffold_profile(path: str | Path) -> dict[str, Any]:
    """Load and validate only the public coarse-prior fields."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    profile_version = int(payload.get("profile_version", -1))
    if profile_version not in SUPPORTED_PROFILE_VERSIONS:
        raise ValueError(f"unsupported scaffold profile version: {path}")
    n_scaffold = int(payload.get("n_scaffold", 0))
    if n_scaffold <= 0:
        raise ValueError(f"invalid scaffold size in profile: {path}")
    preserved_amine_path_length = int(payload.get("preserved_amine_path_length", 0))
    if preserved_amine_path_length < 0 or preserved_amine_path_length > n_scaffold:
        raise ValueError(f"invalid preserved amine prefix length in profile: {path}")
    weights = payload.get("exit_site_weights") or {}
    clean_weights = {
        str(int(slot)): max(float(weight), 0.0)
        for slot, weight in weights.items()
        if 0 <= int(slot) < n_scaffold
    }
    values = [int(v) for v in payload.get("n_extra_values", [])]
    size_weights = [float(v) for v in payload.get("n_extra_weights", [])]
    if len(values) != len(size_weights) or not values:
        values, size_weights = [], []
    element_counts = {
        str(symbol): max(float(count), 0.0)
        for symbol, count in (
            payload.get("reference_extra_element_counts") or {}
        ).items()
    }
    aromatic_counts = {
        str(key): max(float(count), 0.0)
        for key, count in (
            payload.get("reference_extra_aromatic_element_counts") or {}
        ).items()
    }
    allocation_patterns = []
    for raw_pattern in payload.get("allocation_patterns") or []:
        site_counts = {
            str(int(slot)): int(count)
            for slot, count in (raw_pattern.get("site_counts") or {}).items()
            if 0 <= int(slot) < n_scaffold and int(count) > 0
        }
        n_extra = int(raw_pattern.get("n_extra", sum(site_counts.values())))
        weight = max(float(raw_pattern.get("weight", 1.0)), 0.0)
        if site_counts and sum(site_counts.values()) == n_extra and weight > 0:
            allocation_patterns.append(
                {
                    "n_extra": n_extra,
                    "site_counts": site_counts,
                    "weight": weight,
                }
            )
    extra_type_patterns = []
    for raw_pattern in payload.get("extra_type_patterns") or []:
        class_counts = {}
        for raw_key, raw_count in (raw_pattern.get("class_counts") or {}).items():
            try:
                symbol, aromatic_text = str(raw_key).rsplit("|", 1)
                aromatic = int(aromatic_text)
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            if (
                symbol not in SUPPORTED_GENERATION_ELEMENTS
                or aromatic not in (0, 1)
                or count <= 0
            ):
                continue
            class_counts[f"{symbol}|{aromatic}"] = count
        n_extra = int(raw_pattern.get("n_extra", sum(class_counts.values())))
        weight = max(float(raw_pattern.get("weight", 1.0)), 0.0)
        if class_counts and sum(class_counts.values()) == n_extra and weight > 0:
            extra_type_patterns.append(
                {
                    "n_extra": n_extra,
                    "class_counts": class_counts,
                    "weight": weight,
                }
            )
    fragment_patterns = []
    for raw_pattern in payload.get("fragment_patterns") or []:
        try:
            n_extra = int(raw_pattern.get("n_extra", 0))
            weight = max(float(raw_pattern.get("weight", 0.0)), 0.0)
        except (TypeError, ValueError):
            continue
        if n_extra <= 0 or weight <= 0.0:
            continue
        site_fragments = {}
        valid = True
        for raw_slot, raw_fragments in (
            raw_pattern.get("site_fragments") or {}
        ).items():
            try:
                slot = str(int(raw_slot))
            except (TypeError, ValueError):
                valid = False
                break
            if not (0 <= int(slot) < n_scaffold):
                valid = False
                break
            clean_fragments = []
            for raw_fragment in raw_fragments or []:
                kind = str(raw_fragment.get("kind", "point"))
                ring_sizes = []
                for raw_size in raw_fragment.get("ring_sizes") or []:
                    try:
                        ring_size = int(raw_size)
                    except (TypeError, ValueError):
                        valid = False
                        break
                    if ring_size < 3 or ring_size > 8:
                        valid = False
                        break
                    ring_sizes.append(ring_size)
                if not valid:
                    break
                atom_counts = {}
                for raw_key, raw_count in (
                    raw_fragment.get("atom_counts") or {}
                ).items():
                    try:
                        symbol, aromatic_text = str(raw_key).rsplit("|", 1)
                        aromatic = int(aromatic_text)
                        count = int(raw_count)
                    except (TypeError, ValueError):
                        valid = False
                        break
                    if (
                        symbol not in SUPPORTED_GENERATION_ELEMENTS
                        or aromatic not in (0, 1)
                        or count <= 0
                    ):
                        valid = False
                        break
                    atom_counts[f"{symbol}|{aromatic}"] = count
                if not valid or not atom_counts:
                    valid = False
                    break
                clean_fragments.append(
                    {
                        "kind": kind,
                        "ring_sizes": ring_sizes,
                        "atom_counts": atom_counts,
                    }
                )
            if not valid or not clean_fragments:
                valid = False
                break
            site_fragments[slot] = clean_fragments
        if not valid or not site_fragments:
            continue
        if (
            sum(
                sum(int(count) for count in fragment["atom_counts"].values())
                for fragments in site_fragments.values()
                for fragment in fragments
            )
            != n_extra
        ):
            continue
        fragment_patterns.append(
            {
                "n_extra": n_extra,
                "site_fragments": site_fragments,
                "weight": weight,
            }
        )
    return {
        "profile_version": profile_version,
        "profile_kind": str(payload.get("profile_kind", "coarse")),
        "class_name": str(payload.get("class_name", "")),
        "n_scaffold": n_scaffold,
        "preserve_to_amine": bool(payload.get("preserve_to_amine", False)),
        "preserved_amine_path_length": preserved_amine_path_length,
        "exit_site_weights": clean_weights,
        "n_extra_values": values,
        "n_extra_weights": size_weights,
        "reference_extra_element_counts": element_counts,
        "reference_extra_aromatic_element_counts": aromatic_counts,
        "unsupported_extra_element_counts": {
            str(symbol): max(float(count), 0.0)
            for symbol, count in (
                payload.get("unsupported_extra_element_counts") or {}
            ).items()
        },
        "allocation_patterns": allocation_patterns,
        "extra_type_patterns": extra_type_patterns,
        "fragment_patterns": fragment_patterns,
        "included_reference_indices": [
            int(index) for index in payload.get("included_reference_indices", [])
        ],
        "excluded_reference_indices": [
            int(index) for index in payload.get("excluded_reference_indices", [])
        ],
        "matched_reference_records": int(payload.get("matched_reference_records", 0)),
        "unmatched_reference_indices": list(
            payload.get("unmatched_reference_indices", [])
        ),
    }


def write_scaffold_profile(profile: dict[str, Any], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(profile, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
