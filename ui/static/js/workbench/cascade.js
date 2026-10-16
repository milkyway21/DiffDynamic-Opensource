// ── Scaffold Cascade Tab ──────────────────────────────────────────────────

let cascadeJobId = null;
let cascadePollTimer = null;

async function handleCascade(e) {
  e.preventDefault();
  const body = {
    data_id: parseInt($('#cascade-dataid').value) || 0,
    gpu_id: $('#cascade-gpu').value ? parseInt($('#cascade-gpu').value) : null,
    samples_per_round: parseInt($('#cascade-samples').value) || 5,
    rounds: parseInt($('#cascade-rounds').value) || 2,
    auto_evaluate: $('#cascade-auto-eval').checked,
    auto_extract: $('#cascade-auto-extract').checked,
    remove_fragments: $('#cascade-remove-fragments').checked,
  };
  try {
    const data = await api('/api/scaffold-cascade', { method: 'POST', body: JSON.stringify(body) });
    cascadeJobId = data.job_id;
    setLocalizedContent('cascade-status', () => `<span class="status-running">${t('hist.st_running')} (${data.job_id})</span>`);
    const cascadeBar = $('#cascade-progress-bar');
    cascadeBar.style.display = 'block';
    setProgressRunning(cascadeBar, true);
    $('#cascade-cancel-btn').disabled = false;
    $('#cascade-log').textContent = '';
    startCascadePolling();
  } catch (err) {
    setLocalizedContent('cascade-status', () => `<span class="status-failed">${escHtml(err.message)}</span>`);
  }
}

async function handleCascadeCancel() {
  if (!cascadeJobId) return;
  try { await api(`/api/jobs/${cascadeJobId}`, { method: 'DELETE' }); } catch {}
  stopCascadePolling();
  $('#cascade-cancel-btn').disabled = true;
  setLocalizedContent('cascade-status', () => `<span class="status-failed">${t('common.cancelled')}</span>`);
}

function startCascadePolling() {
  stopCascadePolling();
  cascadePollTimer = ddInterval(pollCascade, 2000);
}
function stopCascadePolling() {
  if (cascadePollTimer) { clearInterval(cascadePollTimer); cascadePollTimer = null; }
}

async function pollCascade() {
  if (!cascadeJobId) return;
  try {
    const data = await api(`/api/scaffold-cascade/${cascadeJobId}`);
    const pct = Math.round((data.progress || 0) * 100);
    setProgressFill($('#cascade-progress-fill'), pct);
    setProgressRunning($('#cascade-progress-bar'), !['completed', 'failed', 'cancelled'].includes(data.status));
    if (data.log_tail && data.log_tail.length) {
      $('#cascade-log').textContent = data.log_tail.join('\n');
      $('#cascade-log').scrollTop = $('#cascade-log').scrollHeight;
    }
    if (data.status === 'completed') {
      stopCascadePolling();
      setProgressRunning($('#cascade-progress-bar'), false);
      $('#cascade-cancel-btn').disabled = true;
      setLocalizedContent('cascade-status', () => `<span class="status-completed flash">${t('hist.st_completed')} (${data.run_id})</span>`);
    } else if (data.status === 'failed') {
      stopCascadePolling();
      setProgressRunning($('#cascade-progress-bar'), false);
      $('#cascade-cancel-btn').disabled = true;
      setLocalizedContent('cascade-status', () => `<span class="status-failed">${t('hist.st_failed')}: ${escHtml(data.error || '')}</span>`);
    } else if (data.status === 'cancelled') {
      stopCascadePolling();
      setProgressRunning($('#cascade-progress-bar'), false);
      $('#cascade-cancel-btn').disabled = true;
      setLocalizedContent('cascade-status', () => `<span class="status-failed">${t('common.cancelled')}</span>`);
    }
  } catch {}
}

