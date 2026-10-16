/* Workbench startup. Feature scripts load first in index.html.
 * Keep these as classic scripts: Alpine expressions and inline handlers
 * use the original global functions and shared lexical state.
 */

// ── Tab Navigation ─────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  // Init language
  setLang(currentLang);
  initDiffusionField();
  toggleGenSource();
  initMedchem();
  api('/api/me').then((me) => {
    applyMe(me);
  }).catch(() => {});

  // Close modals on backdrop click
  document.querySelectorAll('.modal').forEach(m => {
    m.addEventListener('click', e => { if (e.target === m) m.style.display = 'none'; });
  });

  const heroCta = $('#hero-cta-generate');
  if (heroCta) heroCta.addEventListener('click', () => switchToTab('generate'));

  $$('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      if (btn.dataset.tab === 'config' && !ddIsAdmin) return;
      $$('.nav-btn').forEach(b => b.classList.remove('active'));
      $$('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const panel = $(`#tab-${btn.dataset.tab}`);
      if (panel) {
        panel.classList.add('active');
      }
      // Auto-refresh on tab switch
      const tab = btn.dataset.tab;
      if (tab === 'dashboard') refreshDashboard();
      if (tab === 'history') { refreshHistory(); refreshRuns(); }
      if (tab === 'config') { loadConfig(); loadGPU(); }
      if (tab === 'molecules') handleMolSearch({ preventDefault() {} });
    });
  });

  // Form handlers
  $('#gen-form').addEventListener('submit', handleGenSubmit);
  $('#gen-cancel-btn').addEventListener('click', handleCancel);
  $('#eval-form').addEventListener('submit', handleEvaluate);
  $('#extract-form').addEventListener('submit', handleExtract);
  $('#mol-form').addEventListener('submit', handleMolSearch);
  const bindDl = (id, fmt) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', () => {
      const rid = parseInt(($('#mol-run') || {}).value, 10);
      downloadRunExport(rid, fmt);
    });
  };
  bindDl('mol-dl-xlsx', 'xlsx');
  bindDl('mol-dl-csv', 'csv');
  bindDl('mol-dl-zip', 'sdfzip');
  $('#hist-form').addEventListener('submit', e => { e.preventDefault(); refreshHistory(); });
  $('#runs-form').addEventListener('submit', e => { e.preventDefault(); refreshRuns(); });
  $('#cfg-load-btn').addEventListener('click', loadConfig);


  // Pocket eval handlers
  $('#pe-form').addEventListener('submit', handlePocketEval);
  $('#pe-cancel-btn').addEventListener('click', handlePocketEvalCancel);
  $('#pe-mode').addEventListener('change', togglePocketEvalMode);
  $('#opt-form').addEventListener('submit', handleOptimization);
  $('#opt-cancel-btn').addEventListener('click', handleOptCancel);
  $('#sc-form').addEventListener('submit', handleScaffold);
  ScaffoldPlacement.init();
  GenerationViewer.init();
  $('#sc-cancel-btn').addEventListener('click', handleScaffoldCancel);
  $('#sc-source').addEventListener('change', toggleScaffoldSource);
  const gateModeEl = $('#sc-gate-mode');
  if (gateModeEl) {
    gateModeEl.addEventListener('change', toggleGateMode);
    toggleGateMode();
  }
  toggleScaffoldMode();
  selectOptFunc('scaffold');
  $('#cascade-form').addEventListener('submit', handleCascade);
  $('#cascade-cancel-btn').addEventListener('click', handleCascadeCancel);
  const compareBtn = $('#compare-runs-btn');
  if (compareBtn) compareBtn.addEventListener('click', compareSelectedRuns);

  // Initial load
  refreshDashboard();
});
