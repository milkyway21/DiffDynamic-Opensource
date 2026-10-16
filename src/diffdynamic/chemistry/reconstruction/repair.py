"""Repair responsibilities extracted from reconstruct.py."""


def _maybe_apply_rdkit_structure_repair(rd_mol, repair_cfg):
    """Optional deterministic topology repair (rdkit-structure-repair package).

    Runs after OB→RDKit postprocess. Default off. Config shape::
        {enable: bool, config: path, on_reject: keep_original|raise}
    """
    from diffdynamic.chemistry.reconstruction.atoms import MolReconsError

    if not repair_cfg:
        return rd_mol
    # Support OmegaConf / dict / plain object
    try:
        enable = bool(repair_cfg.get("enable", False))
    except Exception:
        enable = bool(getattr(repair_cfg, "enable", False))
    if not enable:
        return rd_mol

    import os
    import sys

    try:
        cfg_path = repair_cfg.get("config", None)
        on_reject = repair_cfg.get("on_reject", "keep_original")
    except Exception:
        cfg_path = getattr(repair_cfg, "config", None)
        on_reject = getattr(repair_cfg, "on_reject", "keep_original")

    # Ensure package importable when running from DiffDynamic root
    from diffdynamic.io.paths import project_root

    dd_root = str(project_root())
    pkg_src = os.path.join(dd_root, "rdkit-structure-repair", "src")
    if os.path.isdir(pkg_src) and pkg_src not in sys.path:
        sys.path.insert(0, pkg_src)

    if cfg_path and not os.path.isabs(str(cfg_path)):
        cfg_path = os.path.join(dd_root, str(cfg_path))

    try:
        from structure_repair import repair_molecule  # type: ignore
        from structure_repair.config import load_config  # type: ignore

        config = load_config(cfg_path) if cfg_path else load_config()
        result = repair_molecule(rd_mol, config=config, molecule_id="reconstruct")
        if result.status == "REPAIRED" and result.repaired_mol is not None:
            return result.repaired_mol
        if result.status == "UNCHANGED" and result.repaired_mol is not None:
            # May still have cleared maps / standardize applied
            return result.repaired_mol
        if result.status in ("AMBIGUOUS", "REJECTED"):
            if on_reject == "raise":
                raise MolReconsError(
                    f"rdkit_structure_repair {result.status}: {result.reject_reason}"
                )
            return rd_mol
        return result.repaired_mol if result.repaired_mol is not None else rd_mol
    except MolReconsError:
        raise
    except Exception:
        # Never break reconstruction if the optional repair package fails
        return rd_mol
