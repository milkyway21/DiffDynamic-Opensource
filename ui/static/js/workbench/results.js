// ── Result Summary Modal ─────────────────────────────────────────────

async function showResultSummary(runId) {
  try {
    const r = await api('/api/runs/' + runId);
    if (!r) throw new Error(t('result.run_missing'));
    setLocalizedContent('rs-title', () => escHtml(tf('result.title', { id: runId })));
    let mols = r.molecules || [];
    if (!mols.length && r.output_dir) {
      mols = [{ smiles: r.output_dir + ' (' + t('common.directory') + ')' }];
    }
    const evals = r.evaluations || [];
    const stats = computeSummaryStats(evals);
    let html = '<div class="rs-stats">';
    html += `<div class="rs-stat-card"><div class="rs-stat-label" data-i18n="result.generated">${t('result.generated')}</div><div class="rs-stat-value">${mols.length || '—'}</div></div>`;
    html += `<div class="rs-stat-card"><div class="rs-stat-label" data-i18n="result.valid">${t('result.valid')}</div><div class="rs-stat-value">${stats.valid}</div></div>`;
    html += `<div class="rs-stat-card"><div class="rs-stat-label" data-i18n="result.qed">${t('result.qed')}</div><div class="rs-stat-value">${stats.qed.toFixed(3)}</div></div>`;
    html += `<div class="rs-stat-card"><div class="rs-stat-label" data-i18n="result.sa">${t('result.sa')}</div><div class="rs-stat-value">${stats.sa.toFixed(3)}</div></div>`;
    html += `<div class="rs-stat-card"><div class="rs-stat-label" data-i18n="result.vina">${t('result.vina')}</div><div class="rs-stat-value">${stats.vina.toFixed(2)}</div></div>`;
    html += `<div class="rs-stat-card"><div class="rs-stat-label" data-i18n="result.lipinski">${t('result.lipinski')}</div><div class="rs-stat-value">${stats.lipinski}</div></div>`;
    html += '</div>';
    html += `<div class="form-actions" style="margin:0.75rem 0">
      ${(r.output_path || '').endsWith('.pt') ? `<button type="button" class="btn btn-secondary btn-sm" onclick='downloadPtFile(${escapeAttr(JSON.stringify(r.output_path))})' data-i18n="hist.dl_pt">${t('hist.dl_pt')}</button>` : ''}
      <button type="button" class="btn btn-secondary btn-sm" onclick="downloadRunExport(${runId}, 'xlsx')" data-i18n="hist.dl_xlsx">${t('hist.dl_xlsx')}</button>
      <button type="button" class="btn btn-secondary btn-sm" onclick="downloadRunExport(${runId}, 'csv')" data-i18n="hist.dl_csv">${t('hist.dl_csv')}</button>
      <button type="button" class="btn btn-secondary btn-sm" onclick="downloadRunExport(${runId}, 'sdfzip')" data-i18n="hist.dl_zip">${t('hist.dl_zip')}</button>
    </div>`;
    html += `<h4 style="margin:0.75rem 0 0.5rem" data-i18n="result.metrics">${t('result.metrics')}</h4><div class="rs-mol-list">`;
    if (evals.length === 0) html += `<p style="color:var(--text-secondary)" data-i18n="result.no_eval">${t('result.no_eval')}</p>`;
    evals.slice(0, 50).forEach((e, i) => {
      html += `<div class="rs-mol-item">
        <span style="font-weight:600;min-width:30px">#${i+1}</span>
        <span style="flex:1;font-family:monospace;font-size:0.8rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escHtml(e.metric || '')}</span>
        <span style="font-size:0.78rem;color:var(--text-secondary)">${escHtml(String(e.value ?? ''))}</span>
      </div>`;
    });
    html += '</div>';
    document.getElementById('rs-body').innerHTML = html;
    document.getElementById('result-summary-modal').style.display = 'flex';
  } catch (e) {
    Alpine.store('toasts').error(t('common.load_failed') + ': ' + e.message);
  }
}

function computeSummaryStats(evals) {
  const s = { valid: 0, qed: 0, sa: 0, vina: 0, lipinski: 0, n: evals.length };
  for (const e of evals) {
    if (e.valid) s.valid++;
    s.qed += e.qed || 0;
    s.sa += e.sa || 0;
    s.vina += e.vina_score || 0;
    if (e.lipinski_pass) s.lipinski++;
  }
  if (s.n) { s.qed /= s.n; s.sa /= s.n; s.vina /= s.n; }
  return s;
}

function closeResultSummary() {
  document.getElementById('result-summary-modal').style.display = 'none';
}

