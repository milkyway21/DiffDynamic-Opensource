// ── Scaffold Tab ────────────────────────────────────────────────────────

let scJobId = null;
let scPollTimer = null;

// Merged scaffold-optimization tab: show the form panel matching the
// selected function (SDEdit optimization / scaffold / cascade).
function selectOptFunc(name) {
  ['optimization', 'scaffold', 'cascade'].forEach(func => {
    const wrap = document.getElementById('optfunc-' + func);
    if (wrap) wrap.style.display = func === name ? '' : 'none';
    const btn = document.getElementById('optfunc-btn-' + func);
    if (btn) btn.classList.toggle('active', func === name);
  });
}

function toggleScaffoldMode() {
  const mode = $('#sc-scaffold-mode').value;
  const isGrow = mode === 'grow';
  const isDynamic = mode === 'dynamic_locked';
  const isPrudent = mode === 'prudent';
  const isGrowLike = isGrow || isDynamic || isPrudent;

  $('#sc-grow-section').style.display = isGrowLike ? '' : 'none';
  $('#sc-grow-diffusion').style.display = isGrow ? '' : 'none';
  $('#sc-evolve-section').style.display = mode === 'evolve' ? '' : 'none';
  $('#sc-prudent-section').style.display = isPrudent ? '' : 'none';

  const summary = $('#sc-grow-summary');
  if (summary) {
    summary.textContent = isGrow ? t('sc.grow_title') : t('sc.sample_title');
  }

  const note = $('#sc-mode-note');
  if (note) {
    if (isDynamic) {
      note.textContent = t('sc.note_dynamic_locked');
      note.style.display = '';
    } else if (isPrudent) {
      note.textContent = t('sc.note_prudent');
      note.style.display = '';
    } else {
      note.style.display = 'none';
    }
  }

  const fixPos = $('#sc-fix-pos');
  const fixType = $('#sc-fix-type');
  const lockPosType = isDynamic || isPrudent;
  if (fixPos) {
    fixPos.checked = true;
    fixPos.disabled = lockPosType;
  }
  if (fixType) {
    fixType.checked = !lockPosType;
    fixType.disabled = lockPosType;
  }
  if (window.ScaffoldPlacement) ScaffoldPlacement.updateControls();
}

function toggleScaffoldSource() {
  const src = $('#sc-source').value;
  $('#sc-atom-indices-group').style.display = src === 'atom_indices' ? '' : 'none';
  $('#sc-smarts-group').style.display = src === 'smarts' ? '' : 'none';
}

function toggleGateMode() {
  const mode = ($('#sc-gate-mode') || {}).value || 'adaptive';
  const adaptivePanel = $('#sc-adaptive-panel');
  const manualPanel = $('#sc-manual-gates-panel');
  if (adaptivePanel) adaptivePanel.style.display = mode === 'adaptive' ? '' : 'none';
  if (manualPanel) manualPanel.style.display = mode === 'manual' ? '' : 'none';
}

async function previewAdaptiveGates() {
  const lig = ($('#sc-molecule').value || '').trim();
  const pre = $('#sc-adaptive-preview');
  if (!lig) {
    setLocalizedContent('sc-adaptive-preview', () => escHtml(t('sc.need_ligand')));
    return;
  }
  pre.textContent = 'loading...';
  try {
    const q = new URLSearchParams({ ligand_path: lig });
    const data = await api('/api/adaptive_gates/preview?' + q.toString());
    pre.textContent = JSON.stringify({
      ligand: data.ligand,
      gates: data.gates,
      rationale: data.rationale,
    }, null, 2);
  } catch (err) {
    pre.textContent = String(err);
  }
}

function buildScaffoldRequest(contextOnly = false) {
  const body = {
    data_id: scNumber('sc-dataid', 0),
    molecule_path: $('#sc-molecule').value || null,
    gpu_id: $('#sc-gpu').value ? parseInt($('#sc-gpu').value) : null,
    auto_evaluate: $('#sc-auto-eval').checked,
    auto_extract: $('#sc-auto-extract').checked,
    remove_fragments: $('#sc-remove-fragments').checked,
    scaffold_mode: $('#sc-scaffold-mode').value,
    scaffold_source: $('#sc-source').value,
    scaffold_atom_indices: $('#sc-atom-indices').value || null,
    scaffold_smarts: $('#sc-smarts').value || null,
    fix_scaffold_pos: $('#sc-fix-pos').checked,
    fix_scaffold_type: $('#sc-fix-type').checked,
    scaffold_enable_refine: $('#sc-enable-refine').checked,
    qed_weight: scNumber('sc-qed-w', 1.0),
    sa_weight: scNumber('sc-sa-w', 1.0),
    diversity_weight: scNumber('sc-div-w', 0.5),
    min_qed: scNumber('sc-min-qed', 0.15),
    min_sa: scNumber('sc-min-sa', 0.15),
    max_tanimoto: scNumber('sc-max-tanimoto', 0.9),
    diversity_filter_enable: $('#sc-div-filter').checked,
    grow_num_samples: scNumber('sc-grow-samples', 200),
    grow_start_t: scNumber('sc-grow-start-t', 357),
    grow_stride: scNumber('sc-grow-stride', 15),
    grow_step_size: scNumber('sc-grow-step', 0.3262),
    grow_lambda_a: scNumber('sc-grow-lambda-a', 47),
    grow_lambda_b: scNumber('sc-grow-lambda-b', 11),
    grow_n_extra_mode: $('#sc-n-extra-mode').value,
    grow_n_extra_fixed: scNumber('sc-n-extra-fixed', 8),
    grow_n_extra_min: scNumber('sc-n-extra-min', 3),
    grow_n_extra_max: scNumber('sc-n-extra-max', 20),
    evolve_population_size: scNumber('sc-evo-pop', 50),
    evolve_n_generations: scNumber('sc-evo-gen', 5),
    evolve_children_per_parent: scNumber('sc-evo-children', 10),
    evolve_start_t_high: scNumber('sc-evo-t-high', 200),
    evolve_start_t_low: scNumber('sc-evo-t-low', 30),
    evolve_stride: scNumber('sc-evo-stride', 2),
    evolve_step_size: scNumber('sc-evo-step', 0.2),
    evolve_lambda_a: scNumber('sc-evo-lambda-a', 10),
    evolve_lambda_b: scNumber('sc-evo-lambda-b', 1),
    prudent_n_generations: scNumber('sc-prudent-gen', 5),
    prudent_n_chains_per_seed: scNumber('sc-prudent-chains', 4),
    prudent_advance_top_k: scNumber('sc-prudent-top-k', 2),
    prudent_renoise_t: scNumber('sc-prudent-renoise-t', 600),
    prudent_qed_weight: scNumber('sc-prudent-qed-w', 0.2),
    prudent_sa_weight: scNumber('sc-prudent-sa-w', 0.2),
    prudent_vina_weight: scNumber('sc-prudent-vina-w', 0.6),
    prudent_vina_exhaustiveness: scNumber('sc-prudent-vina-exh', 8),
    prudent_min_qed_for_docking: scNumber('sc-prudent-min-qed-dock', 0.15),
    prudent_min_sa_for_docking: scNumber('sc-prudent-min-sa-dock', 0.15),
    gate_mode: ($('#sc-gate-mode') || {}).value || 'adaptive',
    adaptive_gates_enable: (($('#sc-gate-mode') || {}).value || 'adaptive') === 'adaptive',
    manual_max_logp: scNumber('sc-manual-max-logp', 5.0),
    manual_max_molwt: scNumber('sc-manual-max-molwt', 850),
    manual_max_heavy_atoms: scNumber('sc-manual-max-heavy', 60),
    manual_max_rings: scNumber('sc-manual-max-rings', 7),
    // 任务原始参考配体：UI 的 molecule 即配体；同时写入 ligand_path 供后端自适应门控
    ligand_path: ($('#sc-molecule').value || '').trim() || null,
  };
  if (!body.molecule_path && body.ligand_path) {
    body.molecule_path = body.ligand_path;
  }
  return ScaffoldPlacement.request(body, contextOnly);
}

let scSubmissionActive = false;
function scNumber(id, fallback) {
  const value = document.getElementById(id).value;
  return value.trim() === "" || !Number.isFinite(Number(value)) ? fallback : Number(value);
}
function setScSubmissionActive(active) {
  scSubmissionActive = active;
  document.querySelector("#sc-form button[type=submit]").disabled = active || quotaLeft('generate') === 0;
}

async function handleScaffold(e) {
  e.preventDefault();
  if (scSubmissionActive) return;
  try {
    const body = buildScaffoldRequest();
    setScSubmissionActive(true);
    setLocalizedContent('sc-status', () => escHtml(t('common.submitting')));
    const data = await api('/api/scaffold', { method: 'POST', body: JSON.stringify(body) });
    scJobId = data.job_id;
    setJobStatus('sc-status', data);
    const scBar = $('#sc-progress-bar');
    scBar.style.display = 'block';
    setProgressRunning(scBar, true);
    $('#sc-cancel-btn').disabled = false;
    $('#sc-log').textContent = '';
    startScPolling();
  } catch (err) {
    setScSubmissionActive(false);
    setLocalizedContent('sc-status', () => escHtml(err.i18nKey ? t(err.i18nKey) : err.message));
  }
}

async function handleScaffoldCancel() {
  if (!scJobId) return;
  try { await api(`/api/jobs/${scJobId}`, { method: 'DELETE' }); }
  catch (err) { setLocalizedContent('sc-status', () => escHtml(err.i18nKey ? t(err.i18nKey) : err.message)); return; }
  setScSubmissionActive(false);
  stopScPolling();
  $('#sc-cancel-btn').disabled = true;
  setLocalizedContent('sc-status', () => `<span class="status-failed">${t('common.cancelled')}</span>`);
}

function startScPolling() {
  stopScPolling();
  scPollTimer = ddInterval(pollScaffold, 2000);
}
function stopScPolling() {
  if (scPollTimer) { clearInterval(scPollTimer); scPollTimer = null; }
}

async function pollScaffold() {
  if (!scJobId) return;
  try {
    const data = await api(`/api/scaffold/${scJobId}`);
    const pct = Math.round((data.progress || 0) * 100);
    setProgressFill($('#sc-progress-fill'), pct);
    setProgressRunning($('#sc-progress-bar'), !['completed', 'failed', 'cancelled'].includes(data.status));
    if (data.log_tail && data.log_tail.length) {
      $('#sc-log').textContent = data.log_tail.join('\n');
      $('#sc-log').scrollTop = $('#sc-log').scrollHeight;
    }
    if (['completed', 'failed', 'cancelled'].includes(data.status)) setScSubmissionActive(false);
    setJobStatus('sc-status', data);
    if (data.status === 'completed') {
      stopScPolling();
      setProgressRunning($('#sc-progress-bar'), false);
      $('#sc-cancel-btn').disabled = true;
      setLocalizedContent('sc-status', () => `<span class="status-completed flash">${t('hist.st_completed')} (${data.run_id})</span>`);
    } else if (data.status === 'failed') {
      stopScPolling();
      setProgressRunning($('#sc-progress-bar'), false);
      $('#sc-cancel-btn').disabled = true;
      setLocalizedContent('sc-status', () => `<span class="status-failed">${t('hist.st_failed')}: ${escHtml(data.error || '')}</span>`);
    } else if (data.status === 'cancelled') {
      stopScPolling();
      setProgressRunning($('#sc-progress-bar'), false);
      $('#sc-cancel-btn').disabled = true;
      setLocalizedContent('sc-status', () => `<span class="status-failed">${t('common.cancelled')}</span>`);
    }
  } catch {}
}

// ── Scaffold Mode Visual Selection ───────────────────────────────────

function selectScMode(mode) {
  const sel = document.getElementById('sc-scaffold-mode');
  sel.value = mode;
  document.querySelectorAll('.sc-diag-item').forEach(el => {
    el.classList.toggle('active', el.dataset.mode === mode);
  });
  toggleScaffoldMode();
  ScaffoldPlacement.invalidate();
}

function toggleScAdvanced(advanced) {
  document.getElementById('sc-basic-btn').classList.toggle('active', !advanced);
  document.getElementById('sc-advanced-btn').classList.toggle('active', advanced);
  document.querySelectorAll('.sc-advanced-only').forEach(el => {
    el.style.display = advanced ? '' : 'none';
  });
}

function toggleScSource() {
  const src = document.getElementById('sc-source').value;
  document.getElementById('sc-atom-indices-group').style.display = src === 'atom_indices' ? '' : 'none';
  document.getElementById('sc-smarts-group').style.display = src === 'smarts' ? '' : 'none';
}

