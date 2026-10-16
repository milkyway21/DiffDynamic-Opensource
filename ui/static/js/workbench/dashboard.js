// ── Dashboard ──────────────────────────────────────────────────────────────

async function refreshDashboard() {
  const statsBar = $('#dashboard-stats');
  const dimsEl = $('#dash-score-dims');
  // Skeleton while stats load
  if (statsBar) statsBar.innerHTML = `
    <div class="dash-stat-chip"><strong>&middot;</strong><span>&nbsp;</span></div>
    <div class="dash-stat-chip"><strong>&middot;</strong><span>&nbsp;</span></div>
    <div class="dash-stat-chip"><strong>&middot;</strong><span>&nbsp;</span></div>
    <div class="dash-stat-chip"><strong>&middot;</strong><span>&nbsp;</span></div>`;
  try {
    const stats = await api('/api/stats').catch(() => null);
    if (stats && statsBar) {
      statsBar.innerHTML = `
        <div class="dash-stat-chip"><strong>${stats.active_jobs || 0}</strong><span>${t('hero.live_running')}</span></div>
        <div class="dash-stat-chip"><strong>${stats.pending_jobs || 0}</strong><span>${t('hero.live_queued')}</span></div>
        <div class="dash-stat-chip"><strong>${stats.total_molecules || 0}</strong><span>${t('hero.live_mols')}</span></div>
        <div class="dash-stat-chip"><strong>${stats.completed_runs || 0}</strong><span>${t('hero.live_done')}</span></div>`;
      const active = (stats.active_jobs || 0) + (stats.pending_jobs || 0);
      const dot = document.querySelector('.dash-live-dot');
      const statusText = document.getElementById('dash-status-text');
      if (dot) dot.classList.toggle('is-active', active > 0);
      if (statusText) {
        statusText.dataset.i18n = active > 0 ? 'dash.status_active' : 'dash.status_idle';
        statusText.textContent = t(active > 0 ? 'dash.status_active' : 'dash.status_idle');
      }
    }
  } catch (e) {
    if (statsBar) statsBar.innerHTML = `<div class="dash-stat-chip"><strong>&mdash;</strong><span>${escHtml(e.message)}</span></div>`;
  }
  // Render eight-dimension labels (static, i18n aware)
  if (dimsEl) {
    const dims = [
      { l: 'A', k: 'pe.dim_a' }, { l: 'B', k: 'pe.dim_b' }, { l: 'C', k: 'pe.dim_c' }, { l: 'D', k: 'pe.dim_d' },
      { l: 'E', k: 'pe.dim_e' }, { l: 'F', k: 'pe.dim_f' }, { l: 'G', k: 'pe.dim_g' }, { l: 'H', k: 'pe.dim_h' },
    ];
    dimsEl.innerHTML = dims.map(d =>
      `<div class="dash-dim"><div class="dash-dim-letter">${d.l}</div><div class="dash-dim-name">${t(d.k)}</div></div>`
    ).join('');
  }
}

// Dashboard navigation: cards with data-goto open the matching tab
document.addEventListener('click', (e) => {
  const el = e.target.closest('[data-goto]');
  if (!el) return;
  const tab = el.getAttribute('data-goto');
  if (tab) switchToTab(tab);
});

