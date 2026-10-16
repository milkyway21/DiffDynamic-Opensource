"""CPU task submission for independent and generated medchem processing."""

from server.medchem_schemas import MedchemOptions
from server.submissions.dependencies import dependencies as deps


class MedchemJobs:
    def submit_medchem(self, input_path=None, smiles=None, source_run_id=None,
                       protein_path=None, reference_ligand_path=None,
                       input_aligned=False, options=None, triggered_by="web_ui",
                       skip_daily_quota=False, **internal):
        options = MedchemOptions(**(options or {})).dict()
        dock = options["evaluation"]["mode"] == "vina_dock" or internal.get("auto_evaluate", False)
        run_type = "medchem_dock" if dock else "medchem"
        params = dict(input_path=input_path, smiles=smiles,
                      source_run_id=source_run_id, protein_path=protein_path,
                      reference_ligand_path=reference_ligand_path,
                      input_aligned=input_aligned, options=options)
        # Internal values are supplied only by generated-task orchestration.
        for key in ("source_pt", "source_pts", "source_label", "auto_evaluate", "auto_extract",
                    "max_records", "protein_root", "remove_fragments"):
            if key in internal:
                params[key] = internal[key]
        owner = self._attach_owner_params(
            params, triggered_by, daily_bucket="eval" if dock else None,
            skip_daily_quota=skip_daily_quota,
        )
        run_id = deps.create_run(run_type=run_type, parameters=params,
                                 triggered_by=triggered_by, owner=owner)
        job_id = self._next_job_id(run_type)
        rec = deps.JobRecord(job_id, run_id, run_type, -1, owner=owner)
        from server.runners.medchem import run_medchem

        return self._submit_cpu(rec, run_medchem, self, rec, params)
