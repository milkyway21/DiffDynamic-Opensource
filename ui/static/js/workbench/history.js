// ── History ────────────────────────────────────────────────────────────────

async function refreshHistory() {
  const action = $('#hist-action').value;
  const params = action ? `?action=${action}` : '';
  const tbody = $('#hist-table tbody');
  tbody.innerHTML = '<tr><td colspan="5"><div class="skeleton-text" style="width:100%"></div><div class="skeleton-text" style="width:95%"></div><div class="skeleton-text" style="width:90%"></div></td></tr>';
  try {
    const records = await api(`/api/history${params}`);
    tbody.innerHTML = records.map(r => `<tr>
      <td>${r.id}</td><td>${escHtml(r.action)}</td><td>${escHtml(r.user)}</td>
      <td>${fmtTime(r.created_at)}</td>
      <td title="${escHtml(JSON.stringify(r.details))}">${escHtml(JSON.stringify(r.details || '')).slice(0, 60)}</td>
    </tr>`).join('');
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="5" class="status-failed">${escHtml(e.message)}</td></tr>`;
  }
}

async function refreshRuns() {
  const params = new URLSearchParams();
  const type = $('#runs-type').value;
  if (type) params.set('run_type', type);
  const status = $('#runs-status').value;
  if (status) params.set('status', status);
  const tbody = $('#runs-table tbody');
  const colSpan = ddIsAdmin ? 8 : 7;
  tbody.innerHTML = `<tr><td colspan="${colSpan}"><div class="skeleton-text" style="width:100%"></div><div class="skeleton-text" style="width:95%"></div><div class="skeleton-text" style="width:90%"></div></td></tr>`;
  try {
    const runs = await api(`/api/runs?${params}`);
    document.querySelectorAll('.runs-owner-col').forEach((el) => {
      el.style.display = ddIsAdmin ? '' : 'none';
    });
    tbody.innerHTML = runs.map(r => `<tr data-run-id="${r.id}">
      <td><input type="checkbox" class="run-compare-cb" value="${r.id}"> ${r.id}</td>
      ${ddIsAdmin ? `<td>${escHtml(r.owner || r.triggered_by || '—')}</td>` : ''}
      <td>${escHtml(r.run_type)}</td>
      <td><span class="${statusClass(r.status)}" data-i18n="${({ cancelled: 'common.cancelled' })[r.status] || 'hist.st_' + r.status}">${escHtml(jobStatusText(r.status))}</span></td>
      <td>${fmtTime(r.created_at)}</td><td>${fmtTime(r.finished_at)}</td>
      <td>${r.progress != null ? Math.round(r.progress * 100) + '%' : '—'}</td>
      <td>
        ${medchemHistoryActions(r)}
        ${ddIsAdmin && r.has_log ? `<button class="btn btn-secondary btn-sm" onclick="viewLog(${r.id})"><span data-i18n="common.log">${t('common.log')}</span></button> ` : ''}
        <button class="btn btn-primary btn-sm" onclick="viewRunDetail(${r.id})"><span data-i18n="common.parameters">${t('common.parameters')}</span></button>
        ${!String(r.run_type).startsWith('medchem') ? `<button class="btn btn-secondary btn-sm" onclick="showResultSummary(${r.id})"><span data-i18n="common.results">${t('common.results')}</span></button>` : ''}
        ${(r.output_path || '').endsWith('.pt') ? `<button class="btn btn-secondary btn-sm" onclick='downloadPtFile(${escapeAttr(JSON.stringify(r.output_path))})' data-i18n="hist.dl_pt">${t('hist.dl_pt')}</button> ` : ''}
        ${!String(r.run_type).startsWith('medchem') ? `<button class="btn btn-secondary btn-sm" onclick="downloadRunExport(${r.id}, 'xlsx')" data-i18n="hist.dl_xlsx">${t('hist.dl_xlsx')}</button>
        <button class="btn btn-secondary btn-sm" onclick="downloadRunExport(${r.id}, 'csv')" data-i18n="hist.dl_csv">${t('hist.dl_csv')}</button>
        <button class="btn btn-secondary btn-sm" onclick="downloadRunExport(${r.id}, 'sdfzip')" data-i18n="hist.dl_zip">${t('hist.dl_zip')}</button>` : ''}
        <button class="btn btn-secondary btn-sm" onclick="rerunRun(${r.id})"><span data-i18n="common.rerun">${t('common.rerun')}</span></button>
      </td>
    </tr>`).join('');
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="${colSpan}" class="status-failed">${escHtml(e.message)}</td></tr>`;
  }
}

async function viewLog(runId) {
  if (!ddIsAdmin) return;
  try {
    const data = await api(`/api/runs/${runId}/log`);
    const modal = $('#log-modal');
    const content = $('#log-modal-content');
    content.textContent = data.log || t('common.no_log');
    modal.style.display = 'flex';
  } catch (e) {
    alert(t('common.load_failed') + ': ' + e.message);
  }
}

function closeLogModal() {
  $('#log-modal').style.display = 'none';
}

async function viewRunDetail(runId) {
  try {
    const d = await api(`/api/runs/${runId}`);
    const modal = $('#log-modal');
    $('#log-modal-content').textContent = JSON.stringify(d.parameters, null, 2);
    modal.style.display = 'flex';
  } catch (e) { alert(e.message); }
}

async function rerunRun(runId) {
  if (!confirm(tf('hist.rerun_confirm', { id: runId }))) return;
  try {
    const r = await api(`/api/runs/${runId}/rerun`, { method: 'POST', body: JSON.stringify({}) });
    if ((r.job_type || '').startsWith('medchem') || String(r.job_id).startsWith('medchem')) {
      switchToTab('medchem');
      beginMedchemJob(r);
    }
    alert(t('hist.submitted') + ': ' + r.job_id + (r.status === 'queued' ? ' (' + t('hist.st_pending') + ')' : ''));
    refreshRuns();
  } catch (e) { alert(e.message); }
}

async function compareSelectedRuns() {
  const ids = [...$$('.run-compare-cb:checked')].map(c => c.value);
  if (ids.length < 2) { alert(t('hist.need_two')); return; }
  try {
    const data = await api(`/api/runs/compare?ids=${ids.join(',')}`);
    const el = $('#compare-result');
    if (!el) return;
    el.innerHTML = data.map(r => `
      <div class="card" style="margin-bottom:0.8rem">
        <h3>Run #${r.id} — ${escHtml(r.run_type)} <span class="${statusClass(r.status)}" data-i18n="${({ cancelled: 'common.cancelled' })[r.status] || 'hist.st_' + r.status}">${escHtml(jobStatusText(r.status))}</span></h3>
        <p><span data-i18n="hist.molecules">${t('hist.molecules')}</span>: ${r.molecule_count} | Vina min: ${r.vina_min ?? '—'} | Vina avg: ${r.vina_avg?.toFixed(2) ?? '—'} | QED avg: ${r.qed_avg?.toFixed(3) ?? '—'} | <span data-i18n="hist.mean_score">${t('hist.mean_score')}</span>: ${r.score_avg?.toFixed(1) ?? '—'}</p>
      </div>`).join('');
  } catch (e) { alert(e.message); }
}

async function browseFiles(root, targetInputId) {
  try {
    const data = await api(`/api/browse/files?root=${encodeURIComponent(root)}`);
    const pick = (data.files || []).slice(0, 20);
    const name = pick.length ? pick.map((f, i) => `${i + 1}. ${f.name}`).join('\n') : t('common.no_files');
    const idx = prompt(t('common.file_prompt') + '\n' + name);
    if (idx && pick[parseInt(idx) - 1]) {
      $(targetInputId).value = pick[parseInt(idx) - 1].path;
    }
  } catch (e) { alert(e.message); }
}

