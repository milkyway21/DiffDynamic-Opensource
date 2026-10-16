"""Scaffold job submissions, independent of queue/resource implementation."""

from typing import Optional

from server.submissions.dependencies import dependencies as deps


class ScaffoldJobs:
    """Prepare and dispatch scaffold jobs through JobScheduler hooks."""

    def submit_optimization(
        self,
        data_id: Optional[int] = None,
        protein_path: Optional[str] = None,
        ligand_path: Optional[str] = None,
        molecule_path: Optional[str] = None,
        use_test_set: bool = True,
        gpu_id: Optional[int] = None,
        config_path: Optional[str] = None,
        auto_evaluate: bool = False,
        auto_extract: bool = False,
        remove_fragments: bool = True,
        num_samples: int = 2,
        start_t: int = 16,
        stride: int = 2,
        step_size: float = 0.2,
        cycles: int = 5,
        schedule: str = "linear",
        after_dynamic: bool = False,
        min_qed: float = 0.3,
        min_sa: float = 0.3,
        max_survivors_per_cycle: int = 10,
        max_survivors_total: int = 300,
        triggered_by: str = "web_ui",
        skip_daily_quota: bool = False,
    ) -> dict:
        cfg = deps.get_config()
        parameters = {
            "data_id": data_id,
            "protein_path": protein_path,
            "ligand_path": ligand_path,
            "molecule_path": molecule_path,
            "use_test_set": use_test_set,
            "auto_evaluate": auto_evaluate,
            "auto_extract": auto_extract,
            "remove_fragments": remove_fragments,
            "max_samples": 5,
            "vina_timeout": 20,
            "num_samples": num_samples,
            "start_t": start_t,
            "stride": stride,
            "step_size": step_size,
            "cycles": cycles,
            "schedule": schedule,
            "after_dynamic": after_dynamic,
            "min_qed": min_qed,
            "min_sa": min_sa,
            "max_survivors_per_cycle": max_survivors_per_cycle,
            "max_survivors_total": max_survivors_total,
            "preferred_gpu": gpu_id,
        }
        opt_patches = deps.build_optimization_patches(parameters)
        base_cfg = self._validate_config_path(config_path, cfg)
        job_id = self._next_job_id("opt")
        patched_config = self._write_patched_config(base_cfg, opt_patches, job_id)
        parameters["patched_config"] = patched_config

        owner = self._attach_owner_params(
            parameters,
            triggered_by,
            daily_bucket="generate",
            skip_daily_quota=skip_daily_quota,
        )
        run_id = deps.create_run(
            run_type="optimization",
            parameters=parameters,
            config_snapshot=self._read_config_snapshot(base_cfg),
            triggered_by=triggered_by,
            owner=owner,
        )
        deps.log_history(
            "start_optimization", {"run_id": run_id, "data_id": data_id}, user=owner
        )
        rec = deps.JobRecord(job_id, run_id, "optimization", -1, owner=owner)

        def starter(rec, gpu):
            parameters["gpu_id"] = gpu
            deps.run_optimization(self, rec, parameters)

        return self._acquire_or_queue(gpu_id, rec, starter)

    def submit_scaffold(
        self,
        workflow: str = "legacy",
        seed: int = 42,
        placement: Optional[dict] = None,
        data_id: Optional[int] = None,
        protein_path: Optional[str] = None,
        ligand_path: Optional[str] = None,
        molecule_path: Optional[str] = None,
        use_test_set: bool = True,
        gpu_id: Optional[int] = None,
        config_path: Optional[str] = None,
        auto_evaluate: bool = False,
        auto_extract: bool = False,
        remove_fragments: bool = True,
        scaffold_mode: str = "grow",
        scaffold_source: str = "auto_murcko",
        scaffold_skip_refine: bool = True,
        scaffold_enable_refine: Optional[bool] = None,
        scaffold_atom_indices: Optional[str] = None,
        scaffold_smarts: Optional[str] = None,
        fix_scaffold_pos: bool = True,
        fix_scaffold_type: bool = True,
        after_dynamic: bool = False,
        qed_weight: float = 1.0,
        sa_weight: float = 1.0,
        diversity_weight: float = 0.5,
        rmsd_penalty_weight: float = 0.0,
        min_qed: float = 0.15,
        min_sa: float = 0.15,
        grow_num_samples: int = 200,
        grow_start_t: int = 357,
        grow_stride: int = 15,
        grow_step_size: float = 0.3262,
        grow_lambda_a: int = 47,
        grow_lambda_b: int = 11,
        grow_n_extra_mode: str = "prior_minus_scaffold",
        grow_n_extra_fixed: int = 8,
        grow_n_extra_min: int = 3,
        grow_n_extra_max: int = 20,
        grow_n_extra_min_clamp: int = 6,
        grow_n_extra_max_clamp: int = 20,
        grow_filter_incomplete: bool = False,
        grow_min_qed: float = 0.2,
        grow_min_sa: float = 0.2,
        evolve_population_size: int = 50,
        evolve_n_generations: int = 5,
        evolve_children_per_parent: int = 10,
        evolve_start_t_high: int = 200,
        evolve_start_t_low: int = 30,
        evolve_stride: int = 2,
        evolve_step_size: float = 0.2,
        evolve_lambda_a: int = 10,
        evolve_lambda_b: int = 1,
        diversity_filter_enable: bool = True,
        max_tanimoto: float = 0.9,
        # Scaffold-Prudent
        prudent_n_generations: int = 5,
        prudent_n_chains_per_seed: int = 4,
        prudent_advance_top_k: int = 2,
        prudent_renoise_t: int = 600,
        prudent_qed_weight: float = 0.2,
        prudent_sa_weight: float = 0.2,
        prudent_vina_weight: float = 0.6,
        prudent_lipinski_weight: float = 0.0,
        prudent_lilly_weight: float = 0.0,
        prudent_vina_exhaustiveness: int = 8,
        prudent_min_qed_for_docking: float = 0.15,
        prudent_min_sa_for_docking: float = 0.15,
        # 门控：adaptive（默认）或 manual（人工数值）
        adaptive_gates_enable: bool = True,
        adaptive_gates_actives: Optional[str] = None,
        gate_mode: str = "adaptive",
        manual_max_logp: Optional[float] = 5.0,
        manual_max_molwt: Optional[float] = 850.0,
        manual_max_heavy_atoms: Optional[int] = 60,
        manual_max_rings: Optional[int] = 7,
        triggered_by: str = "web_ui",
        skip_daily_quota: bool = False,
    ) -> dict:
        placement = placement or {"mode": "auto", "regions": []}
        if scaffold_enable_refine is None:
            scaffold_enable_refine = workflow != "repaint"
        scaffold_request = dict(locals())
        for key in ("self", "triggered_by", "skip_daily_quota"):
            scaffold_request.pop(key, None)
        # Only persist fields accepted by the public schema for later replay.
        from server.schemas import ScaffoldRequest

        scaffold_options = dict(scaffold_request)
        public_fields = ScaffoldRequest.__fields__
        scaffold_request = {
            k: v for k, v in scaffold_request.items() if k in public_fields
        }
        if scaffold_mode not in deps.VALID_SCAFFOLD_MODES:
            return {
                "error": f"Invalid scaffold_mode={scaffold_mode!r}; "
                f"expected one of {sorted(deps.VALID_SCAFFOLD_MODES)}",
            }

        cfg = deps.get_config()
        parameters = {
            "data_id": data_id,
            "protein_path": protein_path,
            "ligand_path": ligand_path,
            "molecule_path": molecule_path,
            "use_test_set": use_test_set,
            "auto_evaluate": auto_evaluate,
            "auto_extract": auto_extract,
            "remove_fragments": remove_fragments,
            "max_samples": 5,
            "vina_timeout": 20,
            "scaffold_mode": scaffold_mode,
            "scaffold_source": scaffold_source,
            "preferred_gpu": gpu_id,
            "prudent_n_generations": prudent_n_generations,
            "prudent_n_chains_per_seed": prudent_n_chains_per_seed,
            "prudent_advance_top_k": prudent_advance_top_k,
            "prudent_renoise_t": prudent_renoise_t,
            "adaptive_gates_enable": adaptive_gates_enable,
            "adaptive_gates_actives": adaptive_gates_actives,
            "gate_mode": gate_mode,
            "manual_max_logp": manual_max_logp,
            "manual_max_molwt": manual_max_molwt,
            "manual_max_heavy_atoms": manual_max_heavy_atoms,
            "manual_max_rings": manual_max_rings,
            "scaffold_request": scaffold_request,
            "workflow": workflow,
            "seed": seed,
            "placement": placement,
        }

        sc_patches, metadata = deps.build_scaffold_patches(scaffold_options)
        parameters.update(metadata)

        base_cfg = self._validate_config_path(config_path, cfg)
        job_id = self._next_job_id(f"sc_{scaffold_mode}")
        patched_config = self._write_patched_config(base_cfg, sc_patches, job_id)
        parameters["patched_config"] = patched_config

        owner = self._attach_owner_params(
            parameters,
            triggered_by,
            daily_bucket="generate",
            skip_daily_quota=skip_daily_quota,
        )
        run_id = deps.create_run(
            run_type=f"scaffold_{scaffold_mode}",
            parameters=parameters,
            config_snapshot=self._read_config_snapshot(patched_config),
            triggered_by=triggered_by,
            owner=owner,
        )
        deps.log_history(
            "start_scaffold",
            {"run_id": run_id, "scaffold_mode": scaffold_mode},
            user=owner,
        )
        rec = deps.JobRecord(
            job_id, run_id, f"scaffold_{scaffold_mode}", -1, owner=owner
        )

        def starter(rec, gpu):
            parameters["gpu_id"] = gpu
            deps.run_scaffold(self, rec, parameters)

        return self._acquire_or_queue(gpu_id, rec, starter)

    def submit_scaffold_cascade(
        self,
        data_id: Optional[int] = None,
        protein_path: Optional[str] = None,
        ligand_path: Optional[str] = None,
        gpu_id: Optional[int] = None,
        config_path: Optional[str] = None,
        samples_per_round: int = 5,
        rounds: int = 2,
        auto_evaluate: bool = False,
        auto_extract: bool = False,
        remove_fragments: bool = True,
        triggered_by: str = "web_ui",
        skip_daily_quota: bool = False,
    ) -> dict:
        parameters = {
            "data_id": data_id,
            "protein_path": protein_path,
            "ligand_path": ligand_path,
            "samples_per_round": samples_per_round,
            "rounds": rounds,
            "config_path": config_path,
            "auto_evaluate": auto_evaluate,
            "auto_extract": auto_extract,
            "remove_fragments": remove_fragments,
            "max_samples": 5,
            "vina_timeout": 20,
            "preferred_gpu": gpu_id,
        }
        owner = self._attach_owner_params(
            parameters,
            triggered_by,
            daily_bucket="generate",
            skip_daily_quota=skip_daily_quota,
        )
        run_id = deps.create_run(
            run_type="scaffold_cascade",
            parameters=parameters,
            triggered_by=triggered_by,
            owner=owner,
        )
        deps.log_history(
            "start_scaffold_cascade", {"run_id": run_id, "rounds": rounds}, user=owner
        )
        job_id = self._next_job_id("cascade")
        rec = deps.JobRecord(job_id, run_id, "scaffold_cascade", -1, owner=owner)

        def starter(rec, gpu):
            parameters["gpu_id"] = gpu
            deps.run_scaffold_cascade(self, rec, parameters)

        return self._acquire_or_queue(gpu_id, rec, starter)
