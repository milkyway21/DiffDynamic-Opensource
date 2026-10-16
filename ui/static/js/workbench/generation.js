let currentJobId = null;
let pollTimer = null;
let currentRunId = null;
let genFollow = null;
let currentGenMode = 'custom';  // UI 当前选中模式：custom | testset | batch
let currentJobMode = null;      // 正在轮询的任务实际所属模式
let currentGenerationMedchem = false;

// ── Generation ─────────────────────────────────────────────────────────────

function togglePrudentPanel() {
  const panel = document.getElementById('gen-prudent-panel');
  if (!panel) return;
  panel.style.display = ($('#gen-mode').value === 'prudent') ? '' : 'none';
}

// 一体化生成表单：按 #gen-source 下拉切换三个参数区块
function toggleGenSource() {
  const mode = ($('#gen-source') || {}).value || 'custom';
  currentGenMode = mode;
  const sections = {
    custom: 'gen-section-custom',
    testset: 'gen-section-testset',
    batch: 'gen-section-batch',
  };
  for (const [key, id] of Object.entries(sections)) {
    const el = document.getElementById(id);
    if (el) el.style.display = key === mode ? '' : 'none';
  }
  // 隐藏区块的必填字段不应阻止其他模式提交（display:none 本就不参与校验，显式处理更稳妥）
  const protein = document.getElementById('custom-protein');
  if (protein) protein.required = (mode === 'custom');
  togglePrudentPanel();
  if (window.GenerationViewer) GenerationViewer.syncSource();
}

// 统一提交入口：分派到 custom / testset / batch 三个构造器
async function handleGenSubmit(e) {
  e.preventDefault();
  if (currentGenMode === 'custom') return handleCustomGenerate();
  if (currentGenMode === 'testset') return handleGenerate();
  if (currentGenMode === 'batch') return handleBatchGenerate();
}

// 提交前的每日配额预检（生成 / 自动评估），三种模式共用
function genQuotaPrecheck(autoEval) {
  if (quotaLeft('generate') === 0) {
    setLocalizedContent('gen-status', () => `<span class="status-failed">${escHtml(t('quota.gen_empty'))}</span>`);
    return false;
  }
  if (autoEval && quotaLeft('eval') === 0) {
    setLocalizedContent('gen-status', () => `<span class="status-failed">${escHtml(t('quota.eval_empty'))}</span>`);
    return false;
  }
  return true;
}

async function handleGenerate() {
  const body = {
    mode: 'dynamic',
    data_id: clampInt($('#gen-dataid').value, 0, 10000000, 0),
    use_test_set: true,
    auto_evaluate: $('#gen-auto-eval').checked,
    auto_extract: $('#gen-auto-extract').checked,
    remove_fragments: $('#gen-remove-fragments').checked,
    max_samples: clampInt($('#gen-max-samples').value, 1, 100, 5),
    vina_timeout: clampInt($('#gen-vina-timeout').value, 5, 600, 20),
    batch_size: clampInt($('#gen-batch').value, 1, 100, 5),
  };
  const gpu = $('#gen-gpu').value;
  if (gpu) body.gpu_id = parseInt(gpu);

  if (!genQuotaPrecheck(body.auto_evaluate)) return;

  try {
    body.medchem = generationMedchemOptions();
    const result = await api('/api/generate', { method: 'POST', body: JSON.stringify(body) });
    currentGenerationMedchem = Boolean(body.medchem);
    showGenerationMedchemLink(result.run_id, currentGenerationMedchem);
    currentJobId = result.job_id;
    currentRunId = result.run_id || null;
    currentJobMode = 'testset';
    genFollow = null;

    setJobStatus('gen-status', { ...result, status: result.status === 'queued' ? 'pending' : result.status || 'running' });
    $('#gen-cancel-btn').disabled = false;
    const bar = $('#gen-progress-bar');
    bar.style.display = 'block';
    setProgressRunning(bar, true);
    startPolling();
    refreshQuota();
  } catch (e) {
    setLocalizedContent('gen-status', () => `<span class="status-failed">${t('common.error')}: ${escHtml(e.message)}</span>`);
  }
}

async function handleCancel() {
  if (!currentJobId) return;
  try {
    await api(`/api/jobs/${currentJobId}`, { method: 'DELETE' });
    setLocalizedContent('gen-status', () => `<span class="status-pending">${t('common.cancelled')}</span>`);
    setProgressRunning($('#gen-progress-bar'), false);
    genFollow = null;
    stopPolling();
  } catch (e) {
    setLocalizedContent('gen-status', () => `<span class="status-failed">${t('common.cancel_failed')}: ${escHtml(e.message)}</span>`);
  }
}

async function handleCustomGenerate() {
  const protein = $('#custom-protein').value.trim();
  if (!protein) { alert(t('common.need_protein')); return; }
  const body = {
    protein_path: protein,
    ligand_path: $('#custom-ligand').value.trim() || null,
    pocket_radius: parseFloat($('#custom-radius').value) || 10.0,
    num_samples: parseInt($('#custom-samples').value) || 5,
    auto_evaluate: $('#custom-auto-eval').checked,
    auto_extract: $('#custom-auto-extract').checked,
    remove_fragments: $('#custom-remove-fragments').checked,
  };
  const gpu = $('#custom-gpu').value;
  if (gpu) body.gpu_id = parseInt(gpu);
  if (!genQuotaPrecheck(body.auto_evaluate)) return;
  try {
    body.medchem = generationMedchemOptions();
    const result = await api('/api/generate/custom', { method: 'POST', body: JSON.stringify(body) });
    currentGenerationMedchem = Boolean(body.medchem);
    showGenerationMedchemLink(result.run_id, currentGenerationMedchem);
    currentJobId = result.job_id;
    currentRunId = result.run_id || null;
    currentJobMode = 'custom';
    genFollow = null;
    setJobStatus('gen-status', { ...result, status: result.status === 'queued' ? 'pending' : result.status || 'running' });
    $('#gen-cancel-btn').disabled = false;
    const bar = $('#gen-progress-bar');
    bar.style.display = 'block';
    setProgressRunning(bar, true);
    startPolling();
    refreshQuota();
  } catch (e) {
    setLocalizedContent('gen-status', () => `<span class="status-failed">${t('common.error')}: ${escHtml(e.message)}</span>`);
  }
}

function startPolling() {
  stopPolling();
  pollTimer = ddInterval(pollJob, 2000);
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

async function pollJob() {
  if (genFollow) {
    await pollExtractFollow();
    return;
  }
  if (!currentJobId) return;
  try {
    const job = await api(`/api/jobs/${currentJobId}`);
    const pct = Math.round((job.progress || 0) * 100);

    const bar = $('#gen-progress-bar');
    setProgressFill($('#gen-progress-fill'), pct);
    setProgressRunning(bar, job.status === 'running' || job.status === 'pending');
    setJobStatus('gen-status', job);
    if (job.status === 'completed') {
      const el = $('#gen-status').querySelector('.status-completed');
      if (el) el.classList.add('flash');
    }
    $('#gen-log').textContent = (job.log_tail || []).slice(-30).join('\n');
    if (['completed', 'failed', 'cancelled'].includes(job.status)) {
      setProgressRunning(bar, false);
      $('#gen-cancel-btn').disabled = true;
      if (job.status === 'completed') {
        currentRunId = job.run_id || currentRunId;
        // 按任务提交时的模式读取对应的“自动提取”勾选；批量模式不跟随提取
        const extractBoxId = currentJobMode === 'custom' ? 'custom-auto-extract'
          : currentJobMode === 'testset' ? 'gen-auto-extract' : null;
        const wantExtract = extractBoxId
          ? !!(document.getElementById(extractBoxId) || {}).checked
          : false;
        if (wantExtract && currentRunId) {
          // score_only 默认下评估约 10s/分子；提取还需排队处理，放宽到 10 分钟
          genFollow = { genRunId: currentRunId, until: Date.now() + 600000 };
          setLocalizedContent('gen-status', () => `<span class="status-running">${escHtml(t('gen.extracting'))}</span>`);
          return;
        }
        showGenDownloads(currentRunId);
      }
      stopPolling();
    }
  } catch (e) { /* ignore */ }
}

async function pollExtractFollow() {
  if (!genFollow) return;
  const runId = genFollow.genRunId;
  try {
    const data = await api(`/api/molecules?run_id=${runId}&limit=1`);
    if ((data.total || 0) > 0) {
      showGenDownloads(runId);
      genFollow = null;
      stopPolling();
      return;
    }
  } catch (e) { /* keep waiting */ }
  if (Date.now() > genFollow.until) {
    setLocalizedContent('gen-status', () => `<span class="status-failed">${escHtml(t('gen.extract_wait'))}</span>`);
    genFollow = null;
    stopPolling();
  }
}

function showGenDownloads(runId) {
  if (!runId) return;
  showGenerationMedchemLink(runId, currentGenerationMedchem);
  const rid = parseInt(runId, 10);
  const molRun = document.getElementById('mol-run');
  if (molRun) molRun.value = String(rid);
  $('#gen-status').innerHTML = `<span class="status-completed" data-i18n="gen.ready_dl">${escHtml(t('gen.ready_dl'))}</span>
    — Run <code>${rid}</code>
    <button type="button" class="btn btn-secondary btn-sm" onclick="downloadRunExport(${rid}, 'xlsx')" data-i18n="hist.dl_xlsx">${t('hist.dl_xlsx')}</button>
    <button type="button" class="btn btn-secondary btn-sm" onclick="downloadRunExport(${rid}, 'csv')" data-i18n="hist.dl_csv">${t('hist.dl_csv')}</button>
    <button type="button" class="btn btn-secondary btn-sm" onclick="downloadRunExport(${rid}, 'sdfzip')" data-i18n="hist.dl_zip">${t('hist.dl_zip')}</button>
    <button type="button" class="btn btn-primary btn-sm" onclick="openMoleculeRun(${rid})" data-i18n="gen.open_mols">${t('gen.open_mols')}</button>`;
  // 补充结果 .pt 下载入口（生成 run 的 output_path 即 result_*.pt）
  api(`/api/runs/${rid}`).then((d) => {
    if (d && (d.output_path || '').endsWith('.pt')) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'btn btn-secondary btn-sm';
      btn.dataset.i18n = 'hist.dl_pt';
      btn.textContent = t('hist.dl_pt');
      btn.onclick = () => downloadPtFile(d.output_path);
      $('#gen-status').appendChild(btn);
    }
  }).catch(() => {});
  handleMolSearch({ preventDefault() {} });
}

function openMoleculeRun(runId) {
  const molRun = document.getElementById('mol-run');
  if (molRun) molRun.value = String(runId);
  switchToTab('molecules');
  handleMolSearch({ preventDefault() {} });
}

async function handleBatchGenerate() {
  const body = {
    start_id: clampInt($('#batch-start').value, 0, 10000000, 0),
    end_id: clampInt($('#batch-end').value, 0, 10000000, 0),
    batch_size: clampInt($('#batch-size').value, 1, 100, 5),
    auto_evaluate: $('#batch-auto-eval').checked,
    sample_only: !$('#batch-auto-eval').checked,
  };
  const gpus = $('#batch-gpus').value.trim();
  if (gpus) body.gpus = gpus;
  if (!genQuotaPrecheck(false)) return;
  try {
    body.medchem = generationMedchemOptions();
    const r = await api('/api/generate/batch', { method: 'POST', body: JSON.stringify(body) });
    currentGenerationMedchem = Boolean(body.medchem);
    showGenerationMedchemLink(r.run_id, currentGenerationMedchem);
    setJobStatus('gen-status', { ...r, status: r.status === 'queued' ? 'pending' : r.status || 'running' });
    currentJobId = r.job_id;
    currentRunId = r.run_id || null;
    currentJobMode = 'batch';
    genFollow = null;
    $('#gen-cancel-btn').disabled = false;
    refreshQuota();
    const bar = $('#gen-progress-bar');
    bar.style.display = 'block';
    setProgressRunning(bar, true);
    startPolling();
  } catch (e) { setLocalizedContent('gen-status', () => `<span class="status-failed">${escHtml(e.message)}</span>`); }
}

