// ── Optimization Tab ────────────────────────────────────────────────────

let optJobId = null;
let optPollTimer = null;

async function handleOptimization(e) {
  e.preventDefault();
  const body = {
    data_id: parseInt($('#opt-dataid').value) || 0,
    molecule_path: $('#opt-molecule').value || null,
    gpu_id: $('#opt-gpu').value ? parseInt($('#opt-gpu').value) : null,
    auto_evaluate: $('#opt-auto-eval').checked,
    auto_extract: $('#opt-auto-extract').checked,
    remove_fragments: $('#opt-remove-fragments').checked,
    num_samples: parseInt($('#opt-num-samples').value) || 2,
    start_t: parseInt($('#opt-start-t').value) || 16,
    stride: parseInt($('#opt-stride').value) || 2,
    step_size: parseFloat($('#opt-step-size').value) || 0.2,
    cycles: parseInt($('#opt-cycles').value) || 5,
    schedule: $('#opt-schedule').value,
    min_qed: parseFloat($('#opt-min-qed').value) || 0.3,
    min_sa: parseFloat($('#opt-min-sa').value) || 0.3,
    max_survivors_per_cycle: parseInt($('#opt-max-survivors').value) || 10,
  };
  try {
    const data = await api('/api/optimization', { method: 'POST', body: JSON.stringify(body) });
    optJobId = data.job_id;
    setLocalizedContent('opt-status', () => `<span class="status-running">${t('hist.st_running')} (${data.job_id})</span>`);
    const optBar = $('#opt-progress-bar');
    optBar.style.display = 'block';
    setProgressRunning(optBar, true);
    $('#opt-cancel-btn').disabled = false;
    $('#opt-log').textContent = '';
    startOptPolling();
  } catch (err) {
    setLocalizedContent('opt-status', () => `<span class="status-failed">${escHtml(err.message)}</span>`);
  }
}

async function handleOptCancel() {
  if (!optJobId) return;
  try { await api(`/api/jobs/${optJobId}`, { method: 'DELETE' }); } catch {}
  stopOptPolling();
  $('#opt-cancel-btn').disabled = true;
  setLocalizedContent('opt-status', () => `<span class="status-failed">${t('common.cancelled')}</span>`);
}

function startOptPolling() {
  stopOptPolling();
  optPollTimer = ddInterval(pollOptimization, 2000);
}
function stopOptPolling() {
  if (optPollTimer) { clearInterval(optPollTimer); optPollTimer = null; }
}

async function pollOptimization() {
  if (!optJobId) return;
  try {
    const data = await api(`/api/optimization/${optJobId}`);
    const pct = Math.round((data.progress || 0) * 100);
    setProgressFill($('#opt-progress-fill'), pct);
    setProgressRunning($('#opt-progress-bar'), !['completed', 'failed', 'cancelled'].includes(data.status));
    if (data.log_tail && data.log_tail.length) {
      $('#opt-log').textContent = data.log_tail.join('\n');
      $('#opt-log').scrollTop = $('#opt-log').scrollHeight;
    }
    if (data.status === 'completed') {
      stopOptPolling();
      setProgressRunning($('#opt-progress-bar'), false);
      $('#opt-cancel-btn').disabled = true;
      setLocalizedContent('opt-status', () => `<span class="status-completed flash">${t('hist.st_completed')} (${data.run_id})</span>`);
    } else if (data.status === 'failed') {
      stopOptPolling();
      setProgressRunning($('#opt-progress-bar'), false);
      $('#opt-cancel-btn').disabled = true;
      setLocalizedContent('opt-status', () => `<span class="status-failed">${t('hist.st_failed')}: ${escHtml(data.error || '')}</span>`);
    } else if (data.status === 'cancelled') {
      stopOptPolling();
      setProgressRunning($('#opt-progress-bar'), false);
      $('#opt-cancel-btn').disabled = true;
      setLocalizedContent('opt-status', () => `<span class="status-failed">${t('common.cancelled')}</span>`);
    }
  } catch {}
}

