// ── Evaluation ─────────────────────────────────────────────────────────────

async function handleEvaluate(e) {
  e.preventDefault();
  const pt = $('#eval-pt').value.trim();
  if (!pt) { alert(t('common.need_pt')); return; }
  const modes = [...$$('input[name="vina"]:checked')].map(c => c.value).join(',');
  try {
    const result = await api('/api/evaluate', {
      method: 'POST',
      body: JSON.stringify({
        pt_path: pt, protein_root: $('#eval-prot').value || null,
        vina_modes: modes || 'score_only',
        max_samples: parseInt($('#eval-max-samples').value) || 5,
        vina_timeout: parseInt($('#eval-vina-timeout').value) || 20,
      }),
    });
    setLocalizedContent('eval-status', () => `<span class="status-running">${t('common.started')}</span> — Job: <code>${result.job_id}</code>`);
    refreshQuota();
    pollEvalJob(result.job_id);
  } catch (e) {
    setLocalizedContent('eval-status', () => `<span class="status-failed">${t('common.error')}: ${escHtml(e.message)}</span>`);
  }
}

async function handleExtract(e) {
  e.preventDefault();
  const pt = $('#extract-pt').value.trim();
  if (!pt) { alert(t('common.need_pt')); return; }
  try {
    const result = await api('/api/extract', {
      method: 'POST',
      body: JSON.stringify({ pt_path: pt, protein_root: $('#extract-prot').value || null, remove_fragments: $('#extract-remove-fragments').checked }),
    });
    setLocalizedContent('eval-status', () => `<span class="status-running">${t('common.extract_started')}</span> — Job: <code>${result.job_id}</code>`);
    pollEvalJob(result.job_id);
  } catch (e) {
    setLocalizedContent('eval-status', () => `<span class="status-failed">${t('common.error')}: ${escHtml(e.message)}</span>`);
  }
}

async function pollEvalJob(jobId) {
  const poll = ddInterval(async () => {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      const pct = Math.round((job.progress || 0) * 100);
      setJobStatus('eval-status', job);
      $('#eval-log').textContent = (job.log_tail || []).slice(-20).join('\n');
      if (job.status !== 'running' && job.status !== 'pending') clearInterval(poll);
    } catch (e) { clearInterval(poll); }
  }, 3000);
}

