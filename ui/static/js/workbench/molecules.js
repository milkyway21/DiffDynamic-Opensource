// ── Molecules ──────────────────────────────────────────────────────────────

function showToastError(msg) {
  if (window.Alpine && Alpine.store('toasts')) {
    Alpine.store('toasts').error(msg);
  } else {
    alert(msg);
  }
}

async function downloadBlob(url, filename) {
  // Platform downloads use the validated platform session.
  // 非 internal 用户目录下的文件（403 无权下载）。
  const res = await fetch(url, { credentials: 'same-origin', headers: accountHeaders() });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = typeof err.detail === 'string' ? err.detail : (err.message || res.statusText);
    throw new Error(detail);
  }
  const blob = await res.blob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename || 'download';
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

async function downloadSdfFile(sdfPath, filename) {
  if (!sdfPath) {
    showToastError(t('mol.dl_fail'));
    return;
  }
  try {
    await downloadBlob(
      `${API}/api/sdf?path=${encodeURIComponent(sdfPath)}`,
      filename || (sdfPath.split('/').pop() || 'molecule.sdf'),
    );
  } catch (e) {
    showToastError(`${t('mol.dl_fail')}: ${e.message}`);
  }
}

function downloadRunExport(runId, format) {
  const id = parseInt(runId, 10);
  if (!id) {
    showToastError(t('mol.need_run'));
    return;
  }
  const ext = format === 'xlsx' ? 'xlsx' : (format === 'sdfzip' ? 'zip' : 'csv');
  downloadBlob(
    `${API}/api/runs/${id}/export?format=${encodeURIComponent(format)}`,
    `run_${id}_molecules.${ext}`,
  ).catch((e) => showToastError(`${t('mol.dl_fail')}: ${e.message}`));
}

function downloadPtFile(ptPath) {
  if (!ptPath) return;
  downloadBlob(
    `${API}/api/file/download?path=${encodeURIComponent(ptPath)}`,
    ptPath.split('/').pop() || 'result.pt',
  ).catch((e) => showToastError(`${t('mol.dl_fail')}: ${e.message}`));
}

async function handleMolSearch(e) {
  e.preventDefault();
  const params = new URLSearchParams();
  const run = $('#mol-run').value;
  if (run) params.set('run_id', run);
  const prot = $('#mol-protein').value;
  if (prot) params.set('protein', prot);
  const smi = $('#mol-smiles').value;
  if (smi) params.set('smiles', smi);
  const minV = $('#mol-min-vina').value;
  if (minV) params.set('min_vina', minV);
  const maxV = $('#mol-max-vina').value;
  if (maxV) params.set('max_vina', maxV);
  if ($('#mol-lipinski').checked) params.set('lipinski', 'true');
  params.set('limit', '500');
  const sortBy = ($('#mol-sort') || {}).value || '';

  try {
    const data = await api(`/api/molecules?${params}`);
    let mols = data.items || data;
    const total = data.total != null ? data.total : mols.length;
    // Client-side sorting
    if (sortBy) {
      const [key, dir] = sortBy.split('_');
      const field = key === 'score' ? 'comprehensive_score' : key;
      mols.sort((a, b) => {
        const va = a[field] ?? (dir === 'asc' ? Infinity : -Infinity);
        const vb = b[field] ?? (dir === 'asc' ? Infinity : -Infinity);
        return dir === 'asc' ? va - vb : vb - va;
      });
    }
    setLocalizedContent('mol-count', () => escHtml(tf('mol.result_count', { n: mols.length, total })));
    const tbody = $('#mol-table tbody');
    tbody.innerHTML = mols.map(m => `<tr>
      <td>${m.id}</td>
      <td class="col-pocket" title="${escHtml(m.pocket_id)}">${escHtml(m.pocket_id||'')}</td>
      <td title="${escHtml(m.smiles)}">${escHtml((m.smiles||'').slice(0,30))}${(m.smiles||'').length>30?'...':''}</td>
      <td>${m.vina_score != null ? m.vina_score.toFixed(2) : '—'}</td>
      <td>${m.qed != null ? m.qed.toFixed(3) : '—'}</td>
      <td>${m.sa != null ? m.sa.toFixed(3) : '—'}</td>
      <td>${m.logp != null ? m.logp.toFixed(2) : '—'}</td>
      <td>${m.tpsa != null ? m.tpsa.toFixed(1) : '—'}</td>
      <td>${m.comprehensive_score != null ? m.comprehensive_score.toFixed(1) : '—'}</td>
      <td>${m.lipinski_pass != null ? m.lipinski_pass + '/5' : '—'}</td>
      <td>${m.lilly_passed != null ? (m.lilly_passed ? '✓' : '✗') : '—'}</td>
      <td>${m.lilly_demerit != null ? m.lilly_demerit : '—'}</td>
      <td>${m.conformer_energy != null ? m.conformer_energy.toFixed(1) : '—'}</td>
      <td>${m.molecule_stable != null ? (m.molecule_stable ? '✓' : '✗') : '—'}</td>
      <td>${m.tanimoto != null ? m.tanimoto.toFixed(3) : '—'}</td>
      <td>${m.sdf_path ? `<button type="button" class="btn btn-secondary btn-sm" onclick='downloadSdfFile(${escapeAttr(JSON.stringify(m.sdf_path))}, ${escapeAttr(JSON.stringify(m.sdf_path.split('/').pop() || 'molecule.sdf'))})'>SDF</button>` : '—'}</td>
      <td>${m.run_id ? `<button type="button" class="btn btn-secondary btn-sm" onclick='redockRun(${m.run_id})' title="${escHtml(t('mol.redock_hint'))}">${escHtml(t('mol.redock'))}</button>` : '—'}</td>
    </tr>`).join('');
  } catch (e) {
    setLocalizedContent('mol-count', () => escHtml(t('common.error') + ': ' + e.message));
  }
}

// 重新 Dock：对某个 run 的 .pt 重新提交评估（vina_modes=dock,score_only）。
// dock 是全局搜索，慢；默认自动链只跑 score_only，需要精确对接姿态时按此按钮。
async function redockRun(runId) {
  if (!confirm(t('mol.redock_confirm'))) return;
  try {
    const run = await api(`/api/runs/${runId}`);
    const pt = run.output_path;
    if (!pt || !pt.endsWith('.pt')) { alert(t('mol.redock_no_pt')); return; }
    const params = run.parameters || {};
    const result = await api('/api/evaluate', {
      method: 'POST',
      body: JSON.stringify({
        pt_path: pt,
        protein_root: params.protein_root || null,
        vina_modes: 'dock,score_only',
        max_samples: params.max_samples || 5,
        vina_timeout: params.vina_timeout || 20,
      }),
    });
    $('#mol-count').textContent = t('mol.redock_started').replace('{job}', result.job_id);
  } catch (e) {
    alert(`${t('mol.redock_failed')}: ${e.message}`);
  }
}

// ── 3Dmol.js Molecule Viewer ─────────────────────────────────────────

let _mol3dViewer = null;

function openMol3D(sdfPath, title) {
  setLocalizedContent('mol3d-title', () => escHtml(title || t('common.molecule_view')));
  document.getElementById('mol3d-modal').style.display = 'flex';
  if (!_mol3dViewer) {
    _mol3dViewer = $3Dmol.createViewer('mol3d-viewer', { backgroundColor: '#0a1628' });
  }
  _mol3dViewer.removeAllModels();
  _mol3dViewer.removeAllShapes();
  fetch(API + '/api/file/download?path=' + encodeURIComponent(sdfPath), { credentials: 'same-origin', headers: accountHeaders() })
    .then(r => { if (!r.ok) throw new Error(t('common.load_failed')); return r.text(); })
    .then(sdf => {
      _mol3dViewer.addModel(sdf, 'sdf');
      _mol3dViewer.setStyle({}, { stick: { radius: 0.15 }, sphere: { scale: 0.25 } });
      _mol3dViewer.zoomTo();
      _mol3dViewer.render();
      _mol3dViewer.zoom(1.2, 500);
    })
    .catch(e => {
      setLocalizedContent('mol3d-info', () => escHtml(t('common.load_failed') + ': ' + e.message));
    });
}

function closeMol3D() {
  document.getElementById('mol3d-modal').style.display = 'none';
}

