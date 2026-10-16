/* De novo input preview shares the scaffold renderer, without scaffold locks. */
window.GenerationViewer = (() => {
  const state = { viewer: null, scene: null, context: null, revision: 0 };
  const el = id => document.getElementById(id);
  let status = { key: 'gen.structure_empty', vars: {}, literal: false };

  function message(key, vars = {}, literal = false) {
    status = { key, vars, literal };
    el('gen-structure-message').textContent = literal ? key : tf(key, vars);
  }

  function invalidate() {
    state.revision += 1;
    state.context = null;
    state.scene = null;
    el('gen-fit-context').disabled = true;
    if (state.viewer) { MolecularViewer.clear(state.viewer); state.viewer.render(); }
    message('gen.structure_changed');
  }

  function syncSource() {
    const batch = el('gen-source').value === 'batch';
    el('gen-preview-dataid-group').hidden = !batch;
    const input = el('gen-preview-dataid');
    const start = Number(el('batch-start').value), end = Number(el('batch-end').value);
    input.min = start;
    input.max = end;
    const current = Number(input.value);
    if (!Number.isInteger(current) || current < start || current > end) input.value = start;
    invalidate();
  }

  function request() {
    const source = el('gen-source').value;
    if (source === 'custom') {
      const protein = el('custom-protein').value.trim();
      if (!protein) throw new Error(t('common.need_protein'));
      const radius = Number(el('custom-radius').value);
      if (!Number.isFinite(radius) || radius < 4 || radius > 20) throw new Error(t('gen.radius_invalid'));
      return { protein_path: protein, ligand_path: el('custom-ligand').value.trim() || null,
        pocket_radius: radius };
    }
    const input = el(source === 'batch' ? 'gen-preview-dataid' : 'gen-dataid');
    const dataId = Number(input.value);
    if (!input.value.trim() || !Number.isInteger(dataId) || dataId < 0 || dataId > 10000000 ||
        (source === 'batch' && (dataId < Number(el('batch-start').value) || dataId > Number(el('batch-end').value)))) {
      throw new Error(t('gen.preview_id_invalid'));
    }
    return { data_id: dataId };
  }

  async function load() {
    const button = el('gen-load-context');
    button.disabled = true;
    invalidate();
    const revision = state.revision;
    try {
      const body = request();
      message('gen.structure_loading');
      const context = await api('/api/generate/context', { method: 'POST', body: JSON.stringify(body) });
      if (revision !== state.revision) return;
      if (!state.viewer) state.viewer = MolecularViewer.create('gen-structure-viewer');
      state.scene = MolecularViewer.load(state.viewer, context);
      state.context = context;
      el('gen-fit-context').disabled = false;
      message(context.ligand_sdf ? 'gen.structure_ready' : 'gen.structure_protein_only',
        { atoms: (context.atoms || []).length });
    } catch (error) {
      if (revision === state.revision) message(error.message, {}, true);
    } finally { button.disabled = false; }
  }

  function refreshLanguage() {
    message(status.key, status.vars, status.literal);
    if (state.viewer) state.viewer.render();
  }

  function init() {
    el('gen-load-context').addEventListener('click', load);
    el('gen-fit-context').addEventListener('click', () => { if (state.scene) state.scene.fit(); });
    const fields = ['custom-protein', 'custom-ligand', 'custom-radius', 'gen-dataid', 'gen-preview-dataid',
      'batch-start', 'batch-end'];
    const changed = event => {
      if (!fields.includes(event.target.id)) return;
      if (['batch-start', 'batch-end'].includes(event.target.id)) syncSource();
      else invalidate();
    };
    el('gen-form').addEventListener('input', changed);
    el('gen-form').addEventListener('change', changed);
    new ResizeObserver(() => { if (state.viewer) state.viewer.resize(); })
      .observe(el('gen-structure-viewer'));
  }

  return { init, syncSource, refreshLanguage, state };
})();
