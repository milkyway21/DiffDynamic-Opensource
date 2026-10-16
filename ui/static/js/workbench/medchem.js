/* Medicinal chemistry workbench. Classic globals match the existing UI. */
Object.assign(T, {
  'mc.title': { zh: '药化检查与修复', en: 'Chemistry check & repair' },
  'mc.intro': { zh: '检查结构问题，采用可靠修复，比较可追溯的替代候选。', en: 'Inspect structural issues, apply reliable repairs, and compare traceable alternatives.' },
  'mc.traceable': { zh: '原始 → 主结果 → 候选', en: 'Original → Primary → Candidates' },
  'mc.source': { zh: '输入来源', en: 'Input source' },
  'mc.file': { zh: 'SDF / MOL 文件', en: 'SDF / MOL file' },
  'mc.smiles': { zh: '多行 SMILES', en: 'SMILES lines' },
  'mc.smiles_hint': { zh: '每行一个 SMILES，最多 100 条；每条结构独立处理，解析失败会保留原因。', en: 'One SMILES per line, up to 100 records. Each record is processed separately, including parse failures.' },
  'mc.run': { zh: '历史生成结果', en: 'Previous generation' },
  'mc.run_id': { zh: '生成 Run ID', en: 'Generation run ID' },
  'mc.recent_runs': { zh: '最近的生成任务', en: 'Recent generation runs' },
  'mc.select_run': { zh: '选择任务或填写 Run ID', en: 'Choose a run or enter its ID' },
  'mc.operation': { zh: '处理方式', en: 'Operation' },
  'mc.check': { zh: '仅检查', en: 'Check only' },
  'mc.repair': { zh: '可靠修复', en: 'Reliable repair' },
  'mc.propose': { zh: '修复并提出替代候选', en: 'Repair & propose alternatives' },
  'mc.profile': { zh: '候选改造范围', en: 'Candidate change scope' },
  'mc.local': { zh: '局部改造', en: 'Local changes' },
  'mc.aggressive': { zh: '激进改造', en: 'Aggressive changes' },
  'mc.aggressive_hint': { zh: '激进候选可能改变环系和骨架；作为独立设计建议保留，不自动替换主结果。', en: 'Aggressive candidates may change rings and scaffolds. They remain separate design proposals and do not replace the primary result.' },
  'mc.evaluation': { zh: '候选评估', en: 'Candidate evaluation' },
  'mc.chemistry_3d': { zh: '化学 + 局部 3D 检查', en: 'Chemistry + local 3D checks' },
  'mc.vina_dock': { zh: '额外 Vina 对接', en: 'Include Vina docking' },
  'mc.local_hint': { zh: '检查化学合法性与局部几何；仅对具有可信蛋白坐标的结构检查口袋碰撞。', en: 'Checks chemical validity and local geometry. Pocket clashes are checked only for structures in the protein coordinate frame.' },
  'mc.dock_hint': { zh: '对有效原始结构、主结果和候选使用相同受体与对接框重新对接；可能需要数分钟。', en: 'Redock valid originals, primary results, and candidates with the same receptor and box. This can take several minutes.' },
  'mc.advanced': { zh: '候选与时间参数', en: 'Candidate and time settings' },
  'mc.top_k': { zh: '每个分子的候选数', en: 'Candidates per molecule' },
  'mc.similarity': { zh: '与母体的最低相似度', en: 'Minimum parent similarity' },
  'mc.min_delta': { zh: '最少重原子变化', en: 'Minimum heavy-atom change' },
  'mc.max_delta': { zh: '最多重原子变化', en: 'Maximum heavy-atom change' },
  'mc.budget': { zh: '每分子改造时限（秒）', en: 'Edit budget per molecule (s)' },
  'mc.exhaustiveness': { zh: '对接搜索强度', en: 'Docking exhaustiveness' },
  'mc.timeout': { zh: '每次对接超时（秒）', en: 'Timeout per docking (s)' },
  'mc.seed': { zh: '随机种子', en: 'Random seed' },
  'mc.box': { zh: '对接框', en: 'Docking box' },
  'mc.box_reference': { zh: '从参考配体或历史口袋获得', en: 'Use reference ligand or saved pocket' },
  'mc.box_explicit': { zh: '输入中心与尺寸（Å）', en: 'Enter center and size (Å)' },
  'mc.center': { zh: '中心', en: 'Center' },
  'mc.size': { zh: '尺寸', en: 'Size' },
  'mc.pocket': { zh: '蛋白、参考配体与坐标', en: 'Protein, reference ligand & coordinates' },
  'mc.pocket_hint': { zh: '对接需要蛋白及参考配体或显式对接框。历史任务可继承已保存的口袋。', en: 'Docking requires a protein and either a reference ligand or explicit box. Previous runs can inherit their saved pocket.' },
  'mc.reference': { zh: '蛋白坐标系中的参考配体', en: 'Reference ligand in the protein frame' },
  'mc.aligned': { zh: '输入的 3D 文件已处于该蛋白的坐标系', en: 'The input 3D file is already in this protein’s coordinate frame' },
  'mc.coordinate_hint': { zh: '外部文件默认未对齐；SMILES 生成的坐标仅用于内部几何检查，不代表结合姿态。', en: 'External files are unaligned by default. Coordinates embedded from SMILES support internal geometry checks, not a binding pose.' },
  'mc.start': { zh: '开始药化处理', en: 'Start chemistry task' },
  'mc.gen_enable': { zh: '生成后进行药化修复并追加候选', en: 'Repair chemistry and append candidates after generation' },
  'mc.gen_hint': { zh: '保留原始生成文件；可靠修复成为主结果，替代候选单独保留。下方设置用于候选比较，上方自动评估针对主结果；评估和提取使用相同的主结构。', en: 'Keep the original generation. Reliable repairs become the primary result; alternatives remain separate. The settings below control candidate comparison; auto evaluation above applies to primary results. Evaluation and extraction share the same primary structures.' },
  'mc.report': { zh: '打开药化报告', en: 'Open chemistry report' },
  'mc.use_run': { zh: '药化处理', en: 'Process chemistry' },
  'mc.total': { zh: '输入', en: 'Inputs' },
  'mc.processed': { zh: '已处理', en: 'Processed' },
  'mc.valid': { zh: '有效', en: 'Valid' },
  'mc.repaired': { zh: '已修复', en: 'Repaired' },
  'mc.candidates': { zh: '候选', en: 'Candidates' },
  'mc.failed': { zh: '失败', en: 'Failed' },
  'mc.original': { zh: '原始结构', en: 'Original' },
  'mc.primary': { zh: '主结果', en: 'Primary result' },
  'mc.candidate': { zh: '替代候选', en: 'Alternative' },
  'mc.record': { zh: '输入记录', en: 'Input record' },
  'mc.no_variant': { zh: '此记录没有可用结构。', en: 'No structure is available for this record.' },
  'mc.no_candidates': { zh: '没有通过筛选的替代候选。', en: 'No alternative candidates passed the checks.' },
  'mc.findings': { zh: '检查发现', en: 'Findings' },
  'mc.changes': { zh: '结构变化', en: 'Changes' },
  'mc.geometry': { zh: '几何检查', en: 'Geometry' },
  'mc.coordinate_source': { zh: '坐标来源', en: 'Coordinate source' },
  'mc.coordinate_external': { zh: '外部输入', en: 'External input' },
  'mc.coordinate_generated_conformer': { zh: '生成的游离构象', en: 'Embedded free conformer' },
  'mc.coordinate_generated_pocket': { zh: '生成时的蛋白口袋坐标', en: 'Generated pocket coordinates' },
  'mc.pose': { zh: '查看对接姿态', en: 'View docked pose' },
  'mc.structure': { zh: '查看结构 3D', en: 'View structure in 3D' },
  'mc.download_pose': { zh: '对接姿态 PDBQT', en: 'Docked pose PDBQT' },
  'mc.download_structure': { zh: '结构 SDF', en: 'Structure SDF' },
  'mc.not_docked': { zh: 'SDF 为处理后的结构坐标；对接姿态单独保存在 PDBQT。', en: 'SDF contains the processed structure coordinates. Docked coordinates are stored separately as PDBQT.' },
  'mc.mapping': { zh: '原子映射与变换', en: 'Atom mapping & transform' },
  'mc.empty_report': { zh: '报告尚未就绪，完成后可在此查看。', en: 'The report is not ready yet. Results will appear here when available.' },
  'mc.batch_reports': { zh: '选择口袋报告', en: 'Choose a pocket report' },
  'mc.skipped': { zh: '未执行', en: 'Skipped' },
  'mc.timeout_state': { zh: '超时', en: 'Timed out' },
  'mc.partial': { zh: '部分完成', en: 'Partially completed' },
  'mc.passed': { zh: '通过', en: 'Passed' },
  'mc.unchanged': { zh: '保持原样', en: 'Unchanged' },
  'mc.rejected': { zh: '未采用', en: 'Rejected' },
  'mc.ambiguous': { zh: '存在歧义', en: 'Ambiguous' },
  'mc.not_applicable': { zh: '不适用', en: 'Not applicable' },
  'mc.pocket_check': { zh: '口袋检查', en: 'Pocket check' },
  'mc.main_evaluation': { zh: '生成主结果自动评估', en: 'Primary result auto evaluation' },
  'mc.score_only': { zh: 'Vina 原位打分', en: 'Vina pose scoring' },
  'mc.dock_eval': { zh: 'Vina 对接', en: 'Vina docking' },
  'mc.chemistry_eval': { zh: '化学指标', en: 'Chemical properties' },
  'mc.pockets': { zh: '口袋与对接框', en: 'Pockets and docking boxes' },
  'mc.saved_options': { zh: '本次处理设置', en: 'Settings used for this report' },
  'mc.rejected_counts': { zh: '候选排除原因', en: 'Candidate rejection counts' },
  'mc.too_many': { zh: '一次最多处理 100 条输入记录。', en: 'At most 100 input records can be processed at once.' },
  'mc.input_required': { zh: '请提供所选来源的分子输入。', en: 'Provide molecules for the selected input source.' },
  'mc.invalid_parameter': { zh: '参数 {name} 须在 {min}–{max} 之间。', en: '{name} must be between {min} and {max}.' },
  'mc.invalid_delta': { zh: '最少重原子变化不能大于最多重原子变化。', en: 'Minimum heavy-atom change must not exceed the maximum.' },
  'mc.dock_required': { zh: '请选择蛋白，并提供蛋白坐标系中的参考配体或显式对接框。', en: 'Choose a protein and provide a reference ligand in its coordinate frame or an explicit box.' },
  'mc.generating': { zh: '药化处理将在生成后启动。', en: 'Chemistry processing starts after generation.' },
  'mc.retry': { zh: '重新读取报告', en: 'Reload report' },
});

const medchemState = { jobId: null, runId: null, timer: null, report: null, record: 0, candidate: 0, revision: 0, viewers: [], images: [], pocketViewer: null, pocketRevision: 0, busy: false, polling: false };

function mcEl(id) { return document.getElementById(id); }
function mcRadio(prefix, field) { return document.querySelector(`input[name="${prefix}-${field}"]:checked`)?.value; }
function mcLabel(key) { return `<span data-i18n="${key}">${escHtml(t(key))}</span>`; }
function mcNumberInput(prefix, field, key, value, min, max, step = 1) {
  return `<div class="form-group"><label for="${prefix}-${field}" data-i18n="${key}">${escHtml(t(key))}</label><input id="${prefix}-${field}" type="number" value="${value}" min="${min}" max="${max}" step="${step}"></div>`;
}
function mcChoice(prefix, field, value, selected, key) {
  return `<label class="mc-choice"><input type="radio" name="${prefix}-${field}" value="${value}" ${selected ? 'checked' : ''}>${mcLabel(key)}</label>`;
}

function mountMedchemOptions(prefix, targetId, generation = false) {
  const target = mcEl(targetId);
  target.classList.add('mc-options');
  target.innerHTML = `${generation ? '' : `<fieldset><legend data-i18n="mc.operation">${t('mc.operation')}</legend><div class="mc-choices">${['check', 'repair', 'propose'].map(value => mcChoice(prefix, 'operation', value, value === 'propose', 'mc.' + value)).join('')}</div></fieldset>`}
    <fieldset id="${prefix}-profile-group"><legend data-i18n="mc.profile">${t('mc.profile')}</legend><div class="mc-choices">${mcChoice(prefix, 'profile', 'local', true, 'mc.local')}${mcChoice(prefix, 'profile', 'aggressive', false, 'mc.aggressive')}</div><p id="${prefix}-aggressive-note" class="mc-aggressive-note" hidden data-i18n="mc.aggressive_hint">${t('mc.aggressive_hint')}</p></fieldset>
    <fieldset><legend data-i18n="mc.evaluation">${t('mc.evaluation')}</legend><div class="mc-choices">${mcChoice(prefix, 'evaluation', 'chemistry_3d', true, 'mc.chemistry_3d')}${mcChoice(prefix, 'evaluation', 'vina_dock', false, 'mc.vina_dock')}</div><p class="mc-note" id="${prefix}-evaluation-hint" data-i18n="mc.local_hint">${t('mc.local_hint')}</p>
    <div id="${prefix}-docking-options" hidden><div class="form-row">${mcNumberInput(prefix, 'exhaustiveness', 'mc.exhaustiveness', 8, 1, 32)}${mcNumberInput(prefix, 'timeout', 'mc.timeout', 120, 5, 600)}${mcNumberInput(prefix, 'seed', 'mc.seed', 42, 0, 2147483647)}</div><div class="form-group"><label for="${prefix}-box-mode" data-i18n="mc.box">${t('mc.box')}</label><select id="${prefix}-box-mode"><option value="reference" data-i18n="mc.box_reference">${t('mc.box_reference')}</option><option value="explicit" data-i18n="mc.box_explicit">${t('mc.box_explicit')}</option></select></div><div id="${prefix}-box-fields" hidden>${['center', 'size'].map(field => `<div class="form-row">${['x', 'y', 'z'].map(axis => `<div class="form-group"><label for="${prefix}-${field}-${axis}">${mcLabel('mc.' + field)} ${axis.toUpperCase()} (Å)</label><input id="${prefix}-${field}-${axis}" type="number" step="any" ${field === 'size' ? 'min="10" max="40" value="20"' : ''}></div>`).join('')}</div>`).join('')}</div></div></fieldset>
    <details id="${prefix}-candidate-settings"><summary data-i18n="mc.advanced">${t('mc.advanced')}</summary><div class="form-row">${mcNumberInput(prefix, 'top-k', 'mc.top_k', 3, 1, 10)}${mcNumberInput(prefix, 'similarity', 'mc.similarity', .55, 0, 1, .01)}${mcNumberInput(prefix, 'budget', 'mc.budget', 30, 1, 120)}</div><div class="form-row">${mcNumberInput(prefix, 'min-delta', 'mc.min_delta', -3, -10, 0)}${mcNumberInput(prefix, 'max-delta', 'mc.max_delta', 4, 0, 10)}</div></details>`;
  target.addEventListener('change', () => syncMedchemOptions(prefix));
  syncMedchemOptions(prefix);
}

function syncMedchemOptions(prefix) {
  const dock = mcRadio(prefix, 'evaluation') === 'vina_dock';
  mcEl(prefix + '-docking-options').hidden = !dock;
  mcEl(prefix + '-box-fields').hidden = !dock || mcEl(prefix + '-box-mode').value !== 'explicit';
  const hint = mcEl(prefix + '-evaluation-hint');
  hint.dataset.i18n = dock ? 'mc.dock_hint' : 'mc.local_hint';
  hint.textContent = t(hint.dataset.i18n);
  const propose = !mcRadio(prefix, 'operation') || mcRadio(prefix, 'operation') === 'propose';
  mcEl(prefix + '-profile-group').hidden = !propose;
  mcEl(prefix + '-aggressive-note').hidden = mcRadio(prefix, 'profile') !== 'aggressive';
  if (prefix === 'mc' && dock) mcEl('mc-pocket-section').open = true;
  const target = mcEl(prefix === 'mc' ? 'mc-options' : 'gen-medchem-options');
  target.querySelectorAll('input, select').forEach(input => { input.disabled = Boolean(input.closest('[hidden]')); });
}

function mcReadNumber(id, key, min, max, integer = false) {
  const input = mcEl(id);
  const value = Number(input?.value);
  if (!input?.value.trim() || !Number.isFinite(value) || value < min || value > max || (integer && !Number.isInteger(value))) {
    throw new Error(tf('mc.invalid_parameter', { name: t(key), min, max }));
  }
  return value;
}

function readMedchemOptions(prefix, generation = false) {
  const evaluation = { mode: mcRadio(prefix, 'evaluation') || 'chemistry_3d', exhaustiveness: 8, timeout_seconds: 120, seed: 42 };
  if (evaluation.mode === 'vina_dock') {
    evaluation.exhaustiveness = mcReadNumber(prefix + '-exhaustiveness', 'mc.exhaustiveness', 1, 32, true);
    evaluation.timeout_seconds = mcReadNumber(prefix + '-timeout', 'mc.timeout', 5, 600, true);
    evaluation.seed = mcReadNumber(prefix + '-seed', 'mc.seed', 0, 2147483647, true);
    if (mcEl(prefix + '-box-mode').value === 'explicit') {
      evaluation.center_xyz = ['x', 'y', 'z'].map(axis => mcReadNumber(`${prefix}-center-${axis}`, 'mc.center', -100000, 100000));
      evaluation.size_xyz = ['x', 'y', 'z'].map(axis => mcReadNumber(`${prefix}-size-${axis}`, 'mc.size', 10, 40));
    }
  }
  const options = { operation: generation ? 'propose' : mcRadio(prefix, 'operation'), profile: mcRadio(prefix, 'profile'),
    top_k: mcReadNumber(prefix + '-top-k', 'mc.top_k', 1, 10, true),
    min_similarity: mcReadNumber(prefix + '-similarity', 'mc.similarity', 0, 1),
    min_heavy_delta: mcReadNumber(prefix + '-min-delta', 'mc.min_delta', -10, 0, true),
    max_heavy_delta: mcReadNumber(prefix + '-max-delta', 'mc.max_delta', 0, 10, true),
    time_budget_seconds: mcReadNumber(prefix + '-budget', 'mc.budget', 1, 120), evaluation };
  if (options.min_heavy_delta > options.max_heavy_delta) throw new Error(t('mc.invalid_delta'));
  return options;
}

function generationMedchemOptions() {
  if (!mcEl('gen-medchem-enable')?.checked) return null;
  const options = readMedchemOptions('gen-mc', true);
  if (options.evaluation.mode === 'vina_dock' && currentGenMode === 'custom' && !mcEl('custom-ligand').value.trim() && !options.evaluation.center_xyz) throw new Error(t('mc.dock_required'));
  return options;
}

function syncMedchemSource() {
  const source = mcEl('mc-source').value;
  ['file', 'smiles', 'run'].forEach(value => {
    const section = mcEl('mc-input-' + value);
    section.hidden = source !== value;
    section.querySelectorAll('input, select, textarea').forEach(input => { input.disabled = source !== value; });
  });
  mcEl('mc-aligned').disabled = source !== 'file';
  if (source !== 'file') mcEl('mc-aligned').checked = false;
  if (source === 'run') refreshMedchemRuns();
}

function buildMedchemRequest() {
  const body = { options: readMedchemOptions('mc'), input_aligned: false };
  const source = mcEl('mc-source').value;
  if (source === 'file') { body.input_path = mcEl('mc-input-path').value.trim(); body.input_aligned = mcEl('mc-aligned').checked; }
  if (source === 'smiles') body.smiles = mcEl('mc-smiles').value.trim();
  if (body.smiles && body.smiles.split(/\r?\n/).filter(line => line.trim()).length > 100) throw new Error(t('mc.too_many'));
  if (source === 'run') body.source_run_id = mcReadNumber('mc-source-run', 'mc.run_id', 1, 2147483647, true);
  if (!body.input_path && !body.smiles && !body.source_run_id) throw new Error(t('mc.input_required'));
  const protein = mcEl('mc-protein').value.trim();
  const reference = mcEl('mc-reference').value.trim();
  if (protein) body.protein_path = protein;
  if (reference) body.reference_ligand_path = reference;
  if (body.options.evaluation.mode === 'vina_dock' && source !== 'run' && (!protein || (!reference && !body.options.evaluation.center_xyz))) throw new Error(t('mc.dock_required'));
  return body;
}

function setMedchemBusy(busy) {
  medchemState.busy = busy;
  mcEl('mc-submit').disabled = busy;
  mcEl('mc-cancel').disabled = !busy;
}

async function handleMedchemSubmit(event) {
  event.preventDefault();
  if (medchemState.busy) return;
  try {
    const request = buildMedchemRequest();
    setMedchemBusy(true);
    const result = await api('/api/medchem', { method: 'POST', body: JSON.stringify(request) });
    beginMedchemJob(result);
  } catch (error) { setMedchemBusy(false); mcShowError(error); }
}

function beginMedchemJob(result) {
  stopMedchemPolling();
  medchemState.jobId = result.job_id;
  medchemState.runId = result.run_id;
  medchemState.report = null;
  clearMedchemViews();
  mcEl('mc-report').hidden = true;
  mcEl('mc-progress-bar').hidden = false;
  setProgressRunning(mcEl('mc-progress-bar'), true);
  setMedchemBusy(true);
  setJobStatus('mc-status', result);
  medchemState.timer = ddInterval(pollMedchemJob, 2000);
  pollMedchemJob();
}

function stopMedchemPolling() {
  if (medchemState.timer) clearInterval(medchemState.timer);
  medchemState.timer = null;
}

async function pollMedchemJob() {
  if (!medchemState.jobId || medchemState.polling) return;
  medchemState.polling = true;
  const jobId = medchemState.jobId;
  try {
    const job = await api('/api/jobs/' + encodeURIComponent(jobId));
    if (jobId !== medchemState.jobId) return;
    setJobStatus('mc-status', job);
    setProgressFill(mcEl('mc-progress-fill'), Math.round((job.progress || 0) * 100));
    if (['completed', 'failed', 'cancelled'].includes(job.status)) {
      stopMedchemPolling();
      setMedchemBusy(false);
      setProgressRunning(mcEl('mc-progress-bar'), false);
      if (['completed', 'failed'].includes(job.status) && (job.run_id || medchemState.runId)) await loadMedchemReport(job.run_id || medchemState.runId);
    }
  } catch (error) { mcShowError(error); }
  finally { medchemState.polling = false; }
}

async function cancelMedchemJob() {
  if (!medchemState.jobId) return;
  try {
    await api('/api/jobs/' + encodeURIComponent(medchemState.jobId), { method: 'DELETE' });
    stopMedchemPolling(); setMedchemBusy(false);
    medchemState.jobId = null;
    setProgressRunning(mcEl('mc-progress-bar'), false);
    setJobStatus('mc-status', { status: 'cancelled', run_id: medchemState.runId });
  } catch (error) { mcShowError(error); }
}

function mcShowError(error) {
  setLocalizedContent('mc-status', () => `<span class="status-failed">${escHtml(error.message || String(error))}</span>`);
}

async function refreshMedchemRuns() {
  try {
    const runs = await api('/api/runs?status=completed&limit=100');
    mcEl('mc-run-select').innerHTML = `<option value="">${escHtml(t('mc.select_run'))}</option>` + (Array.isArray(runs) ? runs : []).filter(run => String(run.run_type).startsWith('generate_')).map(run => `<option value="${Number(run.id)}">#${Number(run.id)} · ${escHtml(run.run_type)} · ${escHtml(fmtTime(run.created_at))}</option>`).join('');
  } catch (error) { mcShowError(error); }
}

function useRunForMedchem(runId) {
  switchToTab('medchem');
  mcEl('mc-source').value = 'run';
  mcEl('mc-source-run').value = String(Number(runId));
  syncMedchemSource();
  mcEl('mc-source-run').focus();
}

async function openMedchemReport(runId) {
  switchToTab('medchem');
  await loadMedchemReport(runId);
}

async function loadMedchemReport(runId) {
  try {
    const report = await api(`/api/medchem/runs/${Number(runId)}/report`);
    medchemState.report = report;
    medchemState.record = 0; medchemState.candidate = 0;
    renderMedchemReport();
  } catch (error) { mcShowError(error); }
}

function medchemArtifactURL(runId, artifactId) {
  return `${API}/api/medchem/runs/${Number(runId)}/artifacts/${encodeURIComponent(artifactId)}`;
}

async function downloadMedchemArtifact(runId, artifactId, filename) {
  try { await downloadBlob(medchemArtifactURL(runId, artifactId), filename); }
  catch (error) { mcShowError(error); }
}

function clearMedchemViews() {
  medchemState.revision += 1;
  medchemState.images.forEach(url => URL.revokeObjectURL(url));
  medchemState.images = [];
  medchemState.viewers.forEach(viewer => { try { viewer.clear(); } catch (_) {} });
  medchemState.viewers = [];
}

function mcFormat(value, digits = 2) { return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—'; }
function mcStateText(value) {
  const labels = { skipped: 'mc.skipped', timeout: 'mc.timeout_state', partial: 'mc.partial', passed: 'mc.passed', unchanged: 'mc.unchanged', repaired: 'mc.repaired', rejected: 'mc.rejected', ambiguous: 'mc.ambiguous', not_applicable: 'mc.not_applicable' };
  return labels[value] ? t(labels[value]) : jobStatusText(value || 'skipped');
}
function mcExplain(item) {
  if (typeof item === 'string') return item;
  if (!item || typeof item !== 'object') return String(item ?? '');
  return [item.rule_id || item.code || item.transform_id, item.message || item.description || item.reason || item.rationale || item.name].filter(Boolean).join(' · ') || JSON.stringify(item);
}
function mcList(items, title) {
  if (!Array.isArray(items) || !items.length) return '';
  return `<h5>${escHtml(t(title))}</h5><ul class="mc-changes">${items.map(item => `<li>${escHtml(mcExplain(item))}</li>`).join('')}</ul>`;
}

function mcReportOptions(options) {
  if (!options) return '';
  const evaluation = options.evaluation || {};
  const values = [['mc.operation', t('mc.' + options.operation)], ['mc.profile', t('mc.' + options.profile)],
    ['mc.top_k', options.top_k], ['mc.similarity', options.min_similarity],
    ['mc.min_delta', options.min_heavy_delta], ['mc.max_delta', options.max_heavy_delta],
    ['mc.budget', options.time_budget_seconds], ['mc.evaluation', t('mc.' + evaluation.mode)]];
  if (evaluation.mode === 'vina_dock') values.push(['mc.exhaustiveness', evaluation.exhaustiveness], ['mc.timeout', evaluation.timeout_seconds], ['mc.seed', evaluation.seed]);
  return `<details class="mc-detail"><summary>${escHtml(t('mc.saved_options'))}</summary><dl class="mc-properties">${values.map(([key, value]) => `<div><dt>${escHtml(t(key))}</dt><dd>${escHtml(value ?? '—')}</dd></div>`).join('')}</dl></details>`;
}

function mcVariantMarkup(variant, stage, runId, slot) {
  if (!variant) return `<article class="mc-variant" data-stage="${stage}"><h4>${escHtml(t('mc.' + stage))}</h4><p class="mc-empty">${escHtml(t(stage === 'candidate' ? 'mc.no_candidates' : 'mc.no_variant'))}</p></article>`;
  const properties = variant.properties || {}, geometry = variant.geometry || {}, evaluation = variant.evaluation || {};
  const mainEvaluation = variant.main_evaluation;
  const proteinArtifact = variant.protein_artifact_id || medchemState.report?.protein_artifact_id || '';
  const action = (artifactId, label, format) => artifactId ? `<button type="button" class="btn btn-secondary btn-sm" data-mc-download="${escapeAttr(artifactId)}" data-mc-format="${format}">${escHtml(label)}</button>` : '';
  return `<article class="mc-variant" data-stage="${stage}"><h4>${escHtml(t('mc.' + stage))}</h4>${variant.name ? `<p class="mc-label">${escHtml(variant.name)}</p>` : ''}
    ${variant.depiction_artifact_id ? `<img id="mc-image-${slot}" class="mc-depiction" alt="${escapeAttr(t('mc.' + stage))}" data-mc-image="${escapeAttr(variant.depiction_artifact_id)}">` : ''}
    <code class="mc-smiles">${escHtml(variant.smiles || '—')}</code>
    <dl class="mc-properties">${[['mw', 'MW'], ['qed', 'QED'], ['sa', 'SA'], ['logp', 'LogP'], ['tpsa', 'TPSA'], ['hbd', 'HBD'], ['hba', 'HBA'], ['rotb', 'RotB']].map(([key, name]) => `<div><dt>${name}</dt><dd>${mcFormat(properties[key], ['hbd', 'hba', 'rotb'].includes(key) ? 0 : 2)}</dd></div>`).join('')}${variant.similarity != null ? `<div><dt>${escHtml(t('mc.similarity'))}</dt><dd>${mcFormat(variant.similarity)}</dd></div>` : ''}</dl>
    ${variant.rationale ? `<p class="mc-state">${escHtml(variant.rationale)}</p>` : ''}${mcList(variant.findings, 'mc.findings')}${mcList(variant.changes, 'mc.changes')}
    <p class="mc-state"><strong>${escHtml(t('mc.geometry'))}:</strong> ${escHtml(mcStateText(geometry.status))}${geometry.reason ? ' · ' + escHtml(geometry.reason) : ''}</p>
    <p class="mc-label">${escHtml(t('mc.coordinate_source'))}: ${escHtml(T['mc.coordinate_' + geometry.coordinate_source] ? t('mc.coordinate_' + geometry.coordinate_source) : geometry.coordinate_source || '—')}</p>
    ${geometry.pocket_status ? `<p class="mc-state">${escHtml(t('mc.pocket_check'))}: ${escHtml(mcStateText(geometry.pocket_status))}${geometry.pocket_reason ? ' · ' + escHtml(geometry.pocket_reason) : ''}</p>` : ''}
    <p class="mc-state"><strong>Vina:</strong> ${escHtml(mcStateText(evaluation.status))}${evaluation.status === 'completed' && Number.isFinite(evaluation.affinity) ? ` · ${mcFormat(evaluation.affinity)} kcal/mol` : ''}${evaluation.reason ? ' · ' + escHtml(evaluation.reason) : ''}</p>
    ${mainEvaluation ? `<div class="mc-detail"><h5>${escHtml(t('mc.main_evaluation'))}</h5><p class="mc-state">${escHtml(t(({ score_only: 'mc.score_only', dock: 'mc.dock_eval', chemistry: 'mc.chemistry_eval' })[mainEvaluation.mode] || 'mc.chemistry_eval'))} · ${escHtml(mcStateText(mainEvaluation.status))}${mainEvaluation.status === 'completed' && Number.isFinite(mainEvaluation.affinity) ? ` · ${mcFormat(mainEvaluation.affinity)} kcal/mol` : ''}${mainEvaluation.reason ? ' · ' + escHtml(mainEvaluation.reason) : ''}</p></div>` : ''}
    <div class="mc-actions">${variant.structure_artifact_id ? `<button type="button" class="btn btn-secondary btn-sm" data-mc-view="${escapeAttr(variant.structure_artifact_id)}" data-mc-view-format="${escapeAttr(variant.structure_format || 'sdf')}" data-mc-slot="${slot}" data-mc-with-protein="${Boolean(geometry.aligned)}" data-mc-protein="${escapeAttr(proteinArtifact)}">${escHtml(t('mc.structure'))}</button>` : ''}${evaluation.pose_artifact_id ? `<button type="button" class="btn btn-secondary btn-sm" data-mc-view="${escapeAttr(evaluation.pose_artifact_id)}" data-mc-view-format="pdbqt" data-mc-slot="${slot}" data-mc-with-protein="true" data-mc-protein="${escapeAttr(proteinArtifact)}">${escHtml(t('mc.pose'))}</button>` : ''}${action(variant.structure_artifact_id, t('mc.download_structure'), 'sdf')}${action(evaluation.pose_artifact_id, t('mc.download_pose'), 'pdbqt')}</div>
    <div id="mc-view-${slot}" class="mc-structure" hidden role="img" aria-label="${escapeAttr(t('mc.' + stage))}"></div>${evaluation.pose_artifact_id ? `<p class="mc-label">${escHtml(t('mc.not_docked'))}</p>` : ''}
    ${variant.atom_mapping || variant.transform_id ? `<details class="mc-detail"><summary>${escHtml(t('mc.mapping'))}</summary><pre>${escHtml(JSON.stringify({ transform: variant.transform_id || null, atom_mapping: variant.atom_mapping || {} }, null, 2))}</pre></details>` : ''}</article>`;
}

function renderMedchemReport() {
  const report = medchemState.report;
  if (!report) return;
  const focusedId = document.activeElement?.id;
  clearMedchemViews();
  const revision = medchemState.revision, root = mcEl('mc-report');
  const runId = Number(report.run_id), records = report.records || [], summary = report.summary || {};
  const record = records[medchemState.record], candidate = record?.candidates?.[medchemState.candidate];
  root.hidden = false;
  root.innerHTML = `<div class="mc-heading"><h3>${escHtml(t('mc.title'))} · Run #${runId}</h3><span class="mc-state">${escHtml(mcStateText(report.status))}</span></div>
    ${mcReportOptions(report.options)}
    <div class="mc-actions"><button type="button" class="btn btn-secondary btn-sm" data-mc-reload="${runId}">${escHtml(t('mc.retry'))}</button><button type="button" class="btn btn-secondary btn-sm" data-mc-rerun="${runId}">${escHtml(t('common.rerun'))}</button>${(report.artifacts || []).filter(artifact => ['csv', 'zip', 'json'].includes(artifact.format) || String(artifact.artifact_id).startsWith('download_')).map(artifact => `<button type="button" class="btn btn-secondary btn-sm" data-mc-download="${escapeAttr(artifact.artifact_id)}" data-mc-format="${escapeAttr(artifact.format)}">${escHtml(artifact.label || artifact.format.toUpperCase())}</button>`).join('')}</div>
    ${report.pockets?.length ? `<details class="mc-detail"><summary>${escHtml(t('mc.pockets'))}</summary><ul class="mc-changes">${report.pockets.map(pocket => `<li><strong>${escHtml(pocket.label)}</strong> · ${escHtml(mcStateText(pocket.status))}${pocket.error ? ` · ${escHtml(pocket.error)}` : ''}${pocket.box ? `<pre>${escHtml(JSON.stringify(pocket.box))}</pre>` : ''}</li>`).join('')}</ul></details>` : ''}
    ${report.reports?.length ? `<label for="mc-batch-report">${escHtml(t('mc.batch_reports'))}</label><select id="mc-batch-report" class="mc-record-picker"><option value="">—</option>${report.reports.map(item => `<option value="${Number(item.run_id)}">${escHtml(item.label || String(item.run_id))} · ${escHtml(mcStateText(item.status))}</option>`).join('')}</select>` : ''}
    <div class="mc-summary">${['total', 'processed', 'valid', 'repaired', 'candidates', 'failed'].map(key => `<div><strong>${Number(summary[key]) || 0}</strong><span>${escHtml(t('mc.' + key))}</span></div>`).join('')}</div>
    ${records.length ? `<label for="mc-record-select">${escHtml(t('mc.record'))}</label><select id="mc-record-select" class="mc-record-picker">${records.map((item, index) => `<option value="${index}" ${index === medchemState.record ? 'selected' : ''}>${escHtml(item.name || item.record_id || String(index + 1))} · ${escHtml(mcStateText(item.status))}</option>`).join('')}</select>` : `<p class="mc-empty">${escHtml(t('mc.empty_report'))}</p>`}
    ${record ? `${record.error ? `<p class="mc-record-error">${escHtml(record.error)}</p>` : ''}${record.repair_reason ? `<p class="mc-note">${escHtml(record.repair_reason)}</p>` : ''}${mcList(record.findings, 'mc.findings')}${mcList(record.changes, 'mc.changes')}${Object.keys(record.rejected_counts || {}).length ? `<details class="mc-detail"><summary>${escHtml(t('mc.rejected_counts'))}</summary><ul class="mc-changes">${Object.entries(record.rejected_counts).map(([reason, count]) => `<li>${escHtml(reason)}: ${Number(count) || 0}</li>`).join('')}</ul></details>` : ''}${record.candidates?.length ? `<label for="mc-candidate-select">${escHtml(t('mc.candidate'))}</label><select id="mc-candidate-select" class="mc-candidate-select">${record.candidates.map((item, index) => `<option value="${index}" ${index === medchemState.candidate ? 'selected' : ''}>${escHtml(item.name || item.transform_id || item.variant_id || String(index + 1))}</option>`).join('')}</select>` : ''}<div class="mc-comparison">${mcVariantMarkup(record.original, 'original', runId, 0)}${mcVariantMarkup(record.primary, 'primary', runId, 1)}${mcVariantMarkup(candidate, 'candidate', runId, 2)}</div>` : ''}`;
  root.querySelectorAll('[data-mc-image]').forEach(image => loadMedchemImage(image, runId, revision));
  if (['mc-record-select', 'mc-candidate-select'].includes(focusedId)) mcEl(focusedId)?.focus();
}

async function loadMedchemImage(image, runId, revision) {
  try {
    const response = await fetch(medchemArtifactURL(runId, image.dataset.mcImage), { credentials: 'same-origin', headers: accountHeaders() });
    if (!response.ok) throw new Error(t('common.load_failed'));
    const blob = await response.blob();
    if (revision !== medchemState.revision) return;
    const url = URL.createObjectURL(blob); medchemState.images.push(url); image.src = url;
  } catch (_) { if (revision === medchemState.revision) image.alt = t('common.load_failed'); }
}

async function viewMedchemStructure(runId, artifactId, format, slot, withProtein = false, proteinArtifactId = null) {
  const revision = medchemState.revision;
  const element = mcEl('mc-view-' + slot);
  element.hidden = false;
  element.textContent = t('common.loading');
  try {
    const response = await fetch(medchemArtifactURL(runId, artifactId), { credentials: 'same-origin', headers: accountHeaders() });
    if (!response.ok) throw new Error(t('common.load_failed'));
    const structure = await response.text();
    if (revision !== medchemState.revision) return;
    if (!window.$3Dmol) throw new Error(t('common.load_failed'));
    element.textContent = '';
    const viewer = $3Dmol.createViewer(element, { backgroundColor: '#f7fafe' });
    const proteins = (medchemState.report?.artifacts || []).filter(item => item.format === 'pdb');
    const proteinId = proteinArtifactId || medchemState.report?.protein_artifact_id || (proteins.length === 1 ? proteins[0].artifact_id : null);
    if (withProtein && proteinId) {
      const proteinResponse = await fetch(medchemArtifactURL(runId, proteinId), { credentials: 'same-origin', headers: accountHeaders() });
      if (!proteinResponse.ok) throw new Error(t('common.load_failed'));
      const protein = await proteinResponse.text();
      if (revision !== medchemState.revision) { viewer.clear(); return; }
      const model = viewer.addModel(protein, 'pdb');
      model.setStyle({}, { cartoon: { color: '#79b5dc', opacity: .7 } });
    }
    const ligand = viewer.addModel(structure, format === 'pdbqt' ? 'pdbqt' : 'sdf');
    ligand.setStyle({}, { stick: { radius: .15 }, sphere: { scale: .22 } });
    viewer.zoomTo({ model: ligand.getID() }); viewer.render(); medchemState.viewers.push(viewer);
  } catch (error) { if (revision === medchemState.revision) element.textContent = error.message; }
}

function medchemHistoryActions(run) {
  const runId = Number(run.id);
  if (!Number.isInteger(runId) || runId < 1) return '';
  if (String(run.run_type).startsWith('medchem')) return `<button type="button" class="btn btn-secondary btn-sm" onclick="openMedchemReport(${runId})">${escHtml(t('mc.report'))}</button>`;
  if (String(run.run_type).startsWith('generate_')) return `<button type="button" class="btn btn-secondary btn-sm" onclick="useRunForMedchem(${runId})">${escHtml(t('mc.use_run'))}</button><button type="button" class="btn btn-secondary btn-sm" onclick="openMedchemReport(${runId})">${escHtml(t('mc.report'))}</button>`;
  return '';
}

function showGenerationMedchemLink(runId, enabled) {
  const element = mcEl('gen-medchem-link');
  element.hidden = !enabled || !runId;
  element.innerHTML = enabled && runId ? `<button type="button" class="btn btn-secondary" onclick="openMedchemReport(${Number(runId)})" data-i18n="mc.report">${escHtml(t('mc.report'))}</button>` : '';
}

async function previewMedchemPocket() {
  const button = mcEl('mc-preview-pocket');
  const revision = ++medchemState.pocketRevision;
  button.disabled = true;
  try {
    const protein = mcEl('mc-protein').value.trim();
    if (!protein) throw new Error(t('common.need_protein'));
    const context = await api('/api/generate/context', { method: 'POST', body: JSON.stringify({ protein_path: protein, ligand_path: mcEl('mc-reference').value.trim() || null }) });
    if (revision !== medchemState.pocketRevision) return;
    mcEl('mc-pocket-viewer').hidden = false;
    if (!medchemState.pocketViewer) medchemState.pocketViewer = MolecularViewer.create('mc-pocket-viewer');
    MolecularViewer.load(medchemState.pocketViewer, context);
    mcEl('mc-pocket-message').textContent = t('mc.coordinate_hint');
  } catch (error) { if (revision === medchemState.pocketRevision) mcEl('mc-pocket-message').textContent = error.message; }
  finally { button.disabled = false; }
}

function initMedchem() {
  mountMedchemOptions('mc', 'mc-options');
  mountMedchemOptions('gen-mc', 'gen-medchem-options', true);
  mcEl('gen-medchem-enable').addEventListener('change', event => {
    mcEl('gen-medchem-options').hidden = !event.target.checked;
    syncMedchemOptions('gen-mc');
  });
  mcEl('medchem-form').addEventListener('submit', handleMedchemSubmit);
  mcEl('mc-source').addEventListener('change', syncMedchemSource);
  mcEl('mc-cancel').addEventListener('click', cancelMedchemJob);
  mcEl('mc-preview-pocket').addEventListener('click', previewMedchemPocket);
  mcEl('medchem-form').addEventListener('input', event => {
    if (['mc-input-path', 'mc-protein'].includes(event.target.id)) mcEl('mc-aligned').checked = false;
    if (!['mc-protein', 'mc-reference'].includes(event.target.id)) return;
    medchemState.pocketRevision += 1;
    if (medchemState.pocketViewer) MolecularViewer.clear(medchemState.pocketViewer);
    mcEl('mc-pocket-viewer').hidden = true;
    mcEl('mc-pocket-message').textContent = t('gen.structure_changed');
  });
  mcEl('mc-refresh-runs').addEventListener('click', refreshMedchemRuns);
  mcEl('mc-run-select').addEventListener('change', event => { if (event.target.value) mcEl('mc-source-run').value = event.target.value; });
  mcEl('mc-report').addEventListener('click', event => {
    const button = event.target.closest('button');
    if (!button) return;
    const data = button.dataset, runId = Number(medchemState.report?.run_id);
    if (data.mcDownload) downloadMedchemArtifact(runId, data.mcDownload, `run_${runId}_${String(data.mcDownload).replace(/[^a-zA-Z0-9_-]/g, '_')}.${data.mcFormat}`);
    if (data.mcView) viewMedchemStructure(runId, data.mcView, data.mcViewFormat, data.mcSlot, data.mcWithProtein === 'true', data.mcProtein);
    if (data.mcReload) loadMedchemReport(data.mcReload);
    if (data.mcRerun) rerunRun(data.mcRerun);
  });
  mcEl('mc-report').addEventListener('change', event => {
    if (event.target.id === 'mc-record-select') { medchemState.record = Number(event.target.value); medchemState.candidate = 0; renderMedchemReport(); }
    if (event.target.id === 'mc-candidate-select') { medchemState.candidate = Number(event.target.value); renderMedchemReport(); }
    if (event.target.id === 'mc-batch-report' && event.target.value) loadMedchemReport(event.target.value);
  });
  window.addEventListener('dd:language', () => { if (medchemState.report) renderMedchemReport(); });
  window.addEventListener('resize', () => medchemState.viewers.forEach(viewer => viewer.resize()));
  syncMedchemSource();
}
