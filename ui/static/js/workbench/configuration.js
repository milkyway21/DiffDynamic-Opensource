// ── Config ─────────────────────────────────────────────────────────────────

async function loadConfig() {
  if (!ddIsAdmin) return;
  try {
    const cfg = await api('/api/config');
    $('#cfg-yaml').value = cfg.sampling_yml ? jsYamlDump(cfg.sampling_yml) : '# ' + t('common.load_failed');
    $('#cfg-runtime').textContent = JSON.stringify(cfg.runtime || {}, null, 2);
  } catch (e) {
    setLocalizedContent('cfg-status', () => escHtml(t('common.error') + ': ' + e.message));
  }
}

async function saveConfig() {
  try {
    // Simple YAML validation: just send as text to backend
    const text = $('#cfg-yaml').value;
    await api('/api/config', {
      method: 'PUT',
      body: JSON.stringify({ sampling_yml_text: text }),
    });
    setLocalizedContent('cfg-status', () => escHtml(t('cfg.saved')));
  } catch (e) {
    setLocalizedContent('cfg-status', () => escHtml(t('cfg.save_failed') + ': ' + e.message));
  }
}

async function loadGPU() {
  if (!ddIsAdmin) return;
  try {
    const gpus = await api('/api/gpus');
    if (gpus.length) {
      $('#cfg-gpu').innerHTML = '<table><thead><tr><th>GPU</th><th data-i18n="cfg.gpu_name">' + t('cfg.gpu_name') + '</th><th data-i18n="cfg.gpu_memory">' + t('cfg.gpu_memory') + '</th><th data-i18n="cfg.gpu_utilization">' + t('cfg.gpu_utilization') + '</th></tr></thead><tbody>' +
        gpus.map(g => `<tr><td>${g.index}</td><td>${g.name}</td><td>${g.memory_used_mb}/${g.memory_total_mb} MB</td><td>${g.utilization_pct}%</td></tr>`).join('') + '</tbody></table>';
    }
  } catch (e) { /* ignore */ }
}

// Minimal YAML-like dump (for display only)
function jsYamlDump(obj, indent = 0) {
  if (obj === null || obj === undefined) return 'null';
  if (typeof obj === 'string') return obj;
  if (typeof obj === 'number' || typeof obj === 'boolean') return String(obj);
  if (Array.isArray(obj)) {
    return obj.map(item => '  '.repeat(indent) + '- ' + jsYamlDump(item, indent + 1)).join('\n');
  }
  if (typeof obj === 'object') {
    return Object.entries(obj).map(([k, v]) => {
      if (typeof v === 'object' && v !== null) {
        return '  '.repeat(indent) + k + ':\n' + jsYamlDump(v, indent + 1);
      }
      return '  '.repeat(indent) + k + ': ' + jsYamlDump(v, indent + 1);
    }).join('\n');
  }
  return String(obj);
}

// ── Preset Config Selector ───────────────────────────────────────────

async function loadPresets(selectId) {
  try {
    const data = await api('/api/presets');
    const sel = document.getElementById(selectId);
    if (!sel) return;
    const current = sel.value;
    sel.innerHTML = `<option value="" data-i18n="common.preset_select">${t('common.preset_select')}</option>`;
    for (const p of data.presets) {
      const opt = document.createElement('option');
      opt.value = p.path;
      opt.textContent = p.name;
      sel.appendChild(opt);
    }
    sel.value = current;
  } catch (e) { /* presets optional */ }
}

function applyPreset(selectId, targetInputId) {
  const sel = document.getElementById(selectId);
  const path = sel.value;
  if (path) {
    document.getElementById(targetInputId).value = path;
    Alpine.store('toasts').info(t('common.preset_loaded') + ': ' + sel.options[sel.selectedIndex].text);
  }
}

