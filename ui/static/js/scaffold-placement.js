/* Scaffold selection uses backend canonical heavy-atom indices and Å coordinates. */
window.ScaffoldPlacement = (() => {
  const colors = ['#ff9d4d', '#56c9e9', '#c49aff', '#8cd47a', '#f27baf'];
  const state = { context: null, viewer: null, regions: [], active: 0, pick: 'anchor', preview: null, revision: 0 };
  const el = id => document.getElementById(id);
  let currentMessage = { key: 'sc.placement_idle', vars: {} };
  const message = (key, vars = {}, literal = false) => {
    currentMessage = { key, vars, literal };
    el('sc-placement-message').textContent = literal ? key : tf(key, vars);
  };
  const invalid = key => Object.assign(new Error(t(key)), { i18nKey: key });
  const showError = error => message(error.i18nKey || error.message, {}, !error.i18nKey);
  const finite = value => Number.isFinite(Number(value)) && String(value).trim() !== '';
  const validCount = value => finite(value) && Number.isInteger(Number(value)) && Number(value) >= 1 && Number(value) <= 50;

  function countReady() {
    if (el('sc-placement-mode').value === 'custom') {
      return state.regions.length > 0 && state.regions.every(r => validCount(r.atom_count)) &&
        state.regions.reduce((total, r) => total + r.atom_count, 0) <= 50;
    }
    return el('sc-n-extra-mode').value === 'fixed' && validCount(el('sc-n-extra-fixed').value);
  }

  function invalidate(clearContext = false) {
    state.revision += 1;
    state.preview = null;
    if (clearContext) {
      state.context = null;
      state.regions = [];
      state.active = 0;
      state.pick = 'anchor';
      if (state.viewer) { state.viewer.removeAllModels(); state.viewer.removeAllShapes(); state.viewer.removeAllLabels(); state.viewer.render(); }
      renderRegions();
    } else draw();
    updateControls();
    message(clearContext ? 'sc.input_changed' : 'sc.preview_invalid');
  }

  function updateControls() {
    const mode = el('sc-scaffold-mode').value;
    const repaintAllowed = ['dynamic_locked', 'prudent'].includes(mode);
    el('sc-workflow').options[1].disabled = !repaintAllowed;
    if (!repaintAllowed) el('sc-workflow').value = 'legacy';
    el('sc-placement-mode').disabled = mode === 'evolve';
    if (mode === 'evolve') el('sc-placement-mode').value = 'auto';
    const custom = el('sc-placement-mode').value === 'custom' && mode !== 'evolve';
    el('sc-add-region').disabled = !state.context || !state.context.scaffold_indices.length || !custom;
    el('sc-preview-placement').disabled = !state.context || !countReady() || mode === 'evolve';
    el('sc-atom-count-group').hidden = custom || mode === 'evolve';
    el('sc-regions').hidden = !custom;
    const repaint = repaintAllowed && el('sc-workflow').value === 'repaint';
    if (repaintAllowed) { el('sc-fix-pos').checked = true; el('sc-fix-type').checked = repaint; }
    ['sc-n-extra-mode', 'sc-n-extra-min', 'sc-n-extra-max'].forEach(id => { el(id).disabled = custom; });
    el('sc-n-extra-fixed').disabled = custom || mode === 'evolve';
    updateNote();
    el('sc-workflow-hint').hidden = !repaint;
  }

  function updateNote() {
    const mode = el('sc-scaffold-mode').value;
    const repaintAllowed = ['dynamic_locked', 'prudent'].includes(mode);
    const repaint = repaintAllowed && el('sc-workflow').value === 'repaint';
    const custom = el('sc-placement-mode').value === 'custom' && mode !== 'evolve';
    const note = el('sc-mode-note');
    if (repaintAllowed) {
      note.textContent = t(repaint ? 'sc.repaint_note' : mode === 'prudent' ? 'sc.note_prudent' : 'sc.note_dynamic_locked');
      if (custom) note.textContent += ' ' + t('sc.custom_count_note');
    }
  }

  function refreshLanguage() {
    updateNote();
    translateElements(el('sc-regions'));
    el('sc-regions').querySelectorAll('[data-region] legend').forEach(legend => {
      legend.textContent = tf('sc.region', { n: Number(legend.closest('[data-region]').dataset.region) + 1 });
    });
    message(currentMessage.key, currentMessage.vars, currentMessage.literal);
    if (state.viewer && !state.context) state.viewer.render();
    draw();
  }

  function request(body, contextOnly = false) {
    const customInput = el('sc-input-mode').value === 'custom';
    body.data_id = customInput ? null : body.data_id;
    body.use_test_set = !customInput;
    body.protein_path = customInput ? el('sc-protein').value.trim() || null : null;
    if (customInput && !body.protein_path) throw invalid('common.need_protein');
    body.seed = Number(el('sc-seed').value);
    if (!finite(el('sc-seed').value) || !Number.isInteger(body.seed) || body.seed < 0 || body.seed > 4294967295) throw invalid('sc.seed_invalid');
    body.workflow = el('sc-workflow').value;
    body.placement = { mode: 'auto', regions: [] };
    if (!contextOnly && el('sc-placement-mode').value === 'auto' && el('sc-n-extra-fixed').value.trim()) {
      if (!validCount(el('sc-n-extra-fixed').value)) throw invalid('sc.need_atom_count');
    }
    if (!contextOnly && el('sc-placement-mode').value === 'custom') {
      if (!state.context || !state.regions.length) throw invalid('sc.need_regions');
      body.placement = { mode: 'custom', regions: state.regions.map(r => ({ ...r, center_xyz: [...r.center_xyz] })) };
      if (body.placement.regions.some(r => !Number.isInteger(r.anchor_atom_index) || !state.context.scaffold_indices.includes(r.anchor_atom_index) || r.center_xyz.some(v => !finite(v)) || !Number.isInteger(r.atom_count) || r.atom_count < 1 || !(r.sigma_axial > 0) || !(r.sigma_lateral > 0))) throw invalid('sc.region_invalid');
      if (state.regions.reduce((n, r) => n + r.atom_count, 0) > 50) throw invalid('sc.count_invalid');
      if (state.regions.some(r => r.sigma_axial > 10 || r.sigma_lateral > 10)) throw invalid('sc.width_invalid');
    }
    if (body.workflow === 'repaint') { body.fix_scaffold_pos = true; body.fix_scaffold_type = true; }
    return body;
  }

  async function loadContext() {
    const button = el('sc-load-context');
    button.disabled = true;
    try {
      const body = buildScaffoldRequest(true);
      invalidate(true);
      const revision = state.revision;
      message('sc.loading_context');
      const context = await api('/api/scaffold/context', { method: 'POST', body: JSON.stringify(body) });
      if (revision !== state.revision) return;
      state.context = context;
      if (!state.viewer) state.viewer = MolecularViewer.create('sc-placement-viewer');
      MolecularViewer.load(state.viewer, context, pickAtom);
      updateControls();
      message('sc.context_loaded', { atoms: context.atoms.length, scaffold: context.scaffold_indices.length });
    } catch (error) { showError(error); }
    finally { button.disabled = false; }
  }

  function addRegion() {
    const index = state.context.scaffold_indices[0];
    const xyz = state.context.atoms[index].xyz;
    state.regions.push({ anchor_atom_index: index, center_xyz: [xyz[0] + 3, xyz[1], xyz[2]], atom_count: NaN, method: 'gaussian', sigma_axial: 1, sigma_lateral: 1 });
    state.active = state.regions.length - 1;
    invalidate();
    renderRegions();
  }

  function pickAtom(atom, ligand) {
    const region = state.regions[state.active];
    if (!region || el('sc-placement-mode').value !== 'custom') { message('sc.pick_region_first'); return; }
    if (state.pick === 'anchor') {
      if (!ligand || !state.context.scaffold_indices.includes(atom.index)) { message('sc.pick_scaffold'); return; }
      region.anchor_atom_index = atom.index;
    } else region.center_xyz = [atom.x, atom.y, atom.z];
    invalidate();
    renderRegions();
    message(state.pick === 'anchor' ? 'sc.anchor_selected' : 'sc.center_selected', { region: state.active + 1, index: atom.index });
  }

  function renderRegions() {
    el('sc-regions').innerHTML = state.regions.map((r, i) => `
      <fieldset class="sc-region ${i === state.active ? 'active' : ''}" data-region="${i}">
        <legend>${tf('sc.region', { n: i + 1 })}</legend>
        <div class="sc-view-actions">
          <button type="button" data-action="anchor" data-i18n="sc.select_anchor" aria-pressed="${i === state.active && state.pick === 'anchor'}">${t('sc.select_anchor')}</button>
          <button type="button" data-action="center" data-i18n="sc.select_center" aria-pressed="${i === state.active && state.pick === 'center'}">${t('sc.select_center')}</button>
          <button type="button" data-action="delete" data-i18n="sc.delete">${t('sc.delete')}</button>
        </div>
        <label><span data-i18n="sc.anchor_index">${t('sc.anchor_index')}</span><input data-field="anchor_atom_index" type="number" min="0" step="1" value="${r.anchor_atom_index}"></label>
        <div class="sc-xyz">${['X', 'Y', 'Z'].map((axis, j) => `<label>${axis} (Å)<input data-axis="${j}" type="number" step="any" value="${Number(r.center_xyz[j]).toFixed(4)}"></label>`).join('')}</div>
        <label><span data-i18n="sc.atom_count">${t('sc.atom_count')}</span><input data-field="atom_count" type="number" min="1" max="50" step="1" value="${Number.isFinite(r.atom_count) ? r.atom_count : ''}" placeholder="${t('sc.atom_count_placeholder')}"></label>
        <label><span data-i18n="sc.method">${t('sc.method')}</span><select data-field="method"><option value="gaussian" data-i18n="sc.gaussian" ${r.method === 'gaussian' ? 'selected' : ''}>${t('sc.gaussian')}</option><option value="corridor" data-i18n="sc.corridor" ${r.method === 'corridor' ? 'selected' : ''}>${t('sc.corridor')}</option></select></label>
        <div class="sc-xyz"><label><span data-i18n="sc.axial_width">${t('sc.axial_width')}</span><input data-field="sigma_axial" type="number" min="0" max="10" step="any" value="${r.sigma_axial}"></label><label><span data-i18n="sc.lateral_width">${t('sc.lateral_width')}</span><input data-field="sigma_lateral" type="number" min="0" max="10" step="any" value="${r.sigma_lateral}"></label></div>
      </fieldset>`).join('');
  }

  function draw() {
    if (!state.viewer || !state.context) return;
    const viewer = state.viewer;
    viewer.removeAllShapes();
    viewer.removeAllLabels();
    if (el('sc-placement-mode').value === 'custom') state.regions.forEach((r, i) => {
      const atom = state.context.atoms[r.anchor_atom_index];
      if (!atom || r.center_xyz.some(v => !finite(v))) return;
      const color = colors[i % colors.length];
      const center = { x: r.center_xyz[0], y: r.center_xyz[1], z: r.center_xyz[2] };
      const anchor = { x: atom.xyz[0], y: atom.xyz[1], z: atom.xyz[2] };
      viewer.addSphere({ center, radius: 0.35, color, opacity: 0.8 });
      viewer.addArrow({ start: anchor, end: center, radius: 0.07, color });
      viewer.addLabel(tf('sc.anchor_label', { region: i + 1, index: r.anchor_atom_index }), { position: center, fontSize: 11, backgroundOpacity: 0.55 });
    });
    if (state.preview) {
      const allocations = state.preview.metadata.site_allocation || [];
      const pointColors = [];
      let offset = 0;
      allocations.forEach((a, i) => {
        const start = a.start !== undefined ? a.start : offset;
        for (let j = start; j < start + a.count; j++) pointColors[j] = colors[i % colors.length];
        offset = start + a.count;
      });
      state.preview.positions.forEach((p, i) => viewer.addSphere({
        center: { x: p[0], y: p[1], z: p[2] }, radius: 0.22, color: pointColors[i] || colors[0],
      }));
    }
    viewer.render();
  }

  async function preview() {
    const button = el('sc-preview-placement');
    button.disabled = true;
    try {
      if (!countReady()) throw invalid('sc.need_atom_count');
      const body = buildScaffoldRequest();
      const revision = state.revision;
      message('sc.preview_loading');
      const response = await api('/api/scaffold/preview', { method: 'POST', body: JSON.stringify(body) });
      if (revision !== state.revision) return;
      state.preview = response;
      draw();
      state.viewer.zoomTo({});
      state.viewer.zoom(0.8);
      state.viewer.render();
      message('sc.preview_ready', { atoms: response.positions.length, seed: response.seed });
    } catch (error) { showError(error); }
    finally { updateControls(); }
  }

  function init() {
    el('sc-load-context').addEventListener('click', loadContext);
    el('sc-add-region').addEventListener('click', addRegion);
    el('sc-preview-placement').addEventListener('click', preview);
    el('sc-input-mode').addEventListener('change', () => {
      el('sc-protein-group').hidden = el('sc-input-mode').value !== 'custom';
      el('sc-dataid-group').hidden = el('sc-input-mode').value === 'custom';
    });
    el('sc-workflow').addEventListener('change', () => { el('sc-enable-refine').checked = el('sc-workflow').value !== 'repaint'; updateControls(); });
    el('sc-form').addEventListener('input', event => {
      if (event.target.closest('#sc-regions')) return;
      if (event.target.id === 'sc-n-extra-fixed') {
        el('sc-n-extra-mode').value = event.target.value.trim() ? 'fixed' : 'pocket_prior';
      }
      invalidate(['sc-dataid', 'sc-protein', 'sc-molecule', 'sc-source', 'sc-atom-indices', 'sc-smarts', 'sc-input-mode'].includes(event.target.id));
    });
    el('sc-regions').addEventListener('click', event => {
      const button = event.target.closest('button[data-action]');
      if (!button) return;
      state.active = Number(button.closest('[data-region]').dataset.region);
      if (button.dataset.action === 'delete') { state.regions.splice(state.active, 1); state.active = Math.max(0, state.active - 1); invalidate(); }
      else state.pick = button.dataset.action;
      renderRegions();
      draw();
      if (button.dataset.action !== 'delete') message(state.pick === 'anchor' ? 'sc.pick_anchor_prompt' : 'sc.pick_center_prompt', { region: state.active + 1 });
    });
    el('sc-regions').addEventListener('input', event => {
      const input = event.target;
      const fieldset = input.closest('[data-region]');
      if (!fieldset) return;
      const region = state.regions[Number(fieldset.dataset.region)];
      if (input.dataset.axis !== undefined) region.center_xyz[Number(input.dataset.axis)] = input.value === '' ? NaN : Number(input.value);
      else if (input.dataset.field) region[input.dataset.field] = input.dataset.field === 'method' ? input.value : (input.value === '' ? NaN : Number(input.value));
      invalidate();
    });
    // Native select controls invalidate previews through change as well.
    el('sc-form').addEventListener('change', event => {
      if (event.target.id === 'sc-n-extra-mode' && event.target.value !== 'fixed') {
        el('sc-n-extra-fixed').value = '';
      }
      if (!event.target.closest('#sc-regions')) invalidate(['sc-dataid', 'sc-protein', 'sc-molecule', 'sc-source', 'sc-atom-indices', 'sc-smarts', 'sc-input-mode'].includes(event.target.id));
    });
    updateControls();
    new ResizeObserver(() => { if (state.viewer) state.viewer.resize(); })
      .observe(el('sc-placement-viewer'));
  }
  return { init, request, updateControls, invalidate, refreshLanguage, state };
})();
