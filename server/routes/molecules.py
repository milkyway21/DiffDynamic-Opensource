"""Molecules HTTP routes; business behavior is compatibility-preserving."""

from typing import Optional

from fastapi import APIRouter, Depends, Request

from server.dependencies import dependencies as deps

router = APIRouter()

__all__ = ["browse_molecules", "dashboard_stats"]


@router.get("/api/molecules")
def browse_molecules(
    request: Request,
    run_id: Optional[int] = None,
    smiles: Optional[str] = None,
    protein: Optional[str] = None,
    min_vina: Optional[float] = None,
    max_vina: Optional[float] = None,
    lipinski: Optional[bool] = None,
    limit: int = 200,
    offset: int = 0,
    _auth: None = Depends(deps.verify_internal_key),
):
    limit = deps._clamp(limit, 1, 500, 200)
    offset = max(0, offset)
    min_lip = 4 if lipinski else None
    owner = deps._owner_filter(request)
    if run_id is not None:
        deps._require_run(run_id, request)
    mols = deps.query_molecules(
        run_id=run_id,
        smiles_like=smiles,
        min_vina=min_vina,
        max_vina=max_vina,
        min_lipinski=min_lip,
        pocket_id=protein,
        limit=limit,
        offset=offset,
        owner=owner,
    )
    total = deps.count_molecules(
        run_id=run_id,
        smiles_like=smiles,
        min_vina=min_vina,
        max_vina=max_vina,
        min_lipinski=min_lip,
        pocket_id=protein,
        owner=owner,
    )
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [
            {
                "id": m.id,
                "run_id": m.run_id,
                "pocket_id": m.pocket_id,
                "smiles": m.smiles,
                "vina_score": m.vina_score,
                "qed": m.qed,
                "sa": m.sa,
                "logp": m.logp,
                "tpsa": m.tpsa,
                "comprehensive_score": m.comprehensive_score,
                "lipinski_pass": m.lipinski_pass,
                "pains_pass": m.pains_pass,
                "lilly_passed": m.lilly_passed,
                "lilly_demerit": m.lilly_demerit,
                "lilly_description": m.lilly_description,
                "conformer_energy": m.conformer_energy,
                "rdkit_valid": m.rdkit_valid,
                "molecule_stable": m.molecule_stable,
                "n_heavy_atoms": m.n_heavy_atoms,
                "tanimoto": m.tanimoto,
                "sdf_path": m.sdf_path,
            }
            for m in mols
        ],
    }


@router.get("/api/stats")
def dashboard_stats(request: Request, _auth: None = Depends(deps.verify_internal_key)):
    scheduler = deps.get_scheduler()
    owner = deps._owner_filter(request)
    stats = deps.get_dashboard_stats(owner=owner)
    if deps._request_is_admin(request):
        stats["gpus"] = deps.get_gpu_info()
    jobs = scheduler.list_jobs(owner=owner)
    stats["active_jobs"] = sum(1 for j in jobs if j["status"] == "running")
    stats["pending_jobs"] = sum(1 for j in jobs if j["status"] == "pending")
    stats["queue_length"] = (
        scheduler._queue_position()
        if deps._request_is_admin(request)
        else stats["pending_jobs"]
    )
    return stats
