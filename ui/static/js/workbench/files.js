// ── Modal File Browser ────────────────────────────────────────────────

let _fbTargetInput = null;
let _fbCurrentPath = '';
let _fbExtensions = null;
let _fbRoot = 'outputs';
let _fbSubpath = '';

function browseFile(inputId, extensions) {
  _fbTargetInput = document.getElementById(inputId);
  _fbExtensions = extensions || null;
  const exts = (extensions || []).map(String);
  if (exts.includes('pdb')) _fbRoot = 'data';
  else if (exts.includes('yml') || exts.includes('yaml')) _fbRoot = 'configs';
  else _fbRoot = 'outputs';
  _fbSubpath = '';
  setLocalizedContent('fb-title', () => escHtml(t('common.select_file') + ' — ' + (extensions ? extensions.join(', ') : t('common.all_files'))));
  document.getElementById('file-browser-modal').style.display = 'flex';
  loadFileBrowser();
}

async function loadFileBrowser(subpath) {
  if (typeof subpath === 'string') _fbSubpath = subpath;
  const listEl = document.getElementById('fb-list');
  listEl.innerHTML = `<div style="padding:1rem;color:var(--text-secondary)" data-i18n="common.loading">${t('common.loading')}</div>`;
  try {
    const params = new URLSearchParams({ root: _fbRoot, subpath: _fbSubpath });
    if (_fbExtensions) params.set('ext', _fbExtensions.map(e => (String(e).startsWith('.') ? e : '.' + e)).join(','));
    const data = await api('/api/browse/files?' + params.toString());
    _fbSubpath = data.subpath || '';
    _fbCurrentPath = data.current_path;
    document.getElementById('fb-path').value = _fbRoot + '/' + _fbSubpath;
    let html = '';
    for (const d of (data.dirs || [])) {
      const next = _fbSubpath ? (_fbSubpath + '/' + d.name) : d.name;
      html += `<div class="fb-item fb-dir" data-subpath="${escapeAttr(next)}" style="padding:0.5rem 0.75rem;cursor:pointer;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:0.5rem">
        <span style="color:#f59e0b">&#128193;</span><span>${escHtml(d.name)}</span>
      </div>`;
    }
    for (const f of (data.files || [])) {
      html += `<div class="fb-item fb-file" data-name="${escapeAttr(f.name)}" data-path="${escapeAttr(f.path)}" style="padding:0.5rem 0.75rem;cursor:pointer;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:0.5rem">
        <span style="color:var(--primary)">&#128196;</span>
        <span style="flex:1">${escHtml(f.name)}</span>
        <span style="font-size:0.75rem;color:var(--text-secondary)">${formatSize((f.size_kb || 0) * 1024)}</span>
      </div>`;
    }
    if (!html) html = `<div style="padding:1.5rem;text-align:center;color:var(--text-secondary)" data-i18n="common.empty_dir">${t('common.empty_dir')}</div>`;
    listEl.innerHTML = html;
    if (!listEl.dataset.bound) {
      listEl.dataset.bound = '1';
      listEl.addEventListener('click', (ev) => {
        const dir = ev.target.closest('.fb-dir');
        if (dir) { loadFileBrowser(dir.dataset.subpath || ''); return; }
        const file = ev.target.closest('.fb-file');
        if (file) selectFile(file.dataset.path || '');
      });
    }
    const searchInput = document.getElementById('fb-search');
    searchInput.oninput = () => {
      const q = searchInput.value.toLowerCase();
      listEl.querySelectorAll('.fb-file').forEach(el => {
        el.style.display = (el.dataset.name || '').toLowerCase().includes(q) ? '' : 'none';
      });
    };
  } catch (e) {
    listEl.innerHTML = `<div style="padding:1rem;color:var(--danger)">${t('common.error')}: ${escHtml(e.message)}</div>`;
  }
}

function selectFile(path) {
  if (_fbTargetInput) {
    _fbTargetInput.value = path;
    _fbTargetInput.dispatchEvent(new Event('input', { bubbles: true }));
  }
  closeFileBrowser();
  Alpine.store('toasts').success(t('common.selected') + ': ' + path.split('/').pop());
}

function closeFileBrowser() {
  document.getElementById('file-browser-modal').style.display = 'none';
  _fbTargetInput = null;
}

document.getElementById('fb-up-btn').onclick = () => {
  if (!_fbSubpath) return;
  const parts = _fbSubpath.replace(/\/$/, '').split('/').filter(Boolean);
  parts.pop();
  loadFileBrowser(parts.join('/'));
};

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

// ── File Upload ───────────────────────────────────────────────────────

function uploadAndFill(inputId, accept) {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = accept;
  input.onchange = async () => {
    if (!input.files[0]) return;
    const file = input.files[0];
    if (file.size > ddMaxUploadBytes) {
      Alpine.store('toasts').error(
        tf('common.file_too_large', { name: file.name, size: formatSize(file.size), limit: formatSize(ddMaxUploadBytes) })
      );
      return;
    }
    const fd = new FormData();
    fd.append('file', file);
    const tid = Alpine.store('toasts').info(t('common.uploading') + ': ' + file.name, { timeout: 0 });
    try {
      // 上传与后续提交必须用同一身份（X-DD-User），否则文件落在
      // uploads/users/internal/ 而 web_ui 提交时会被路径 ACL 拒绝。
      const r = await fetch(API + '/api/upload', {
        method: 'POST',
        body: fd,
        headers: accountHeaders(),
        credentials: 'same-origin',
      });
      if (!r.ok) {
        const err = await r.json();
        throw new Error(err.detail || t('common.upload_failed'));
      }
      const data = await r.json();
      const target = document.getElementById(inputId);
      target.value = data.stored_path;
      target.dispatchEvent(new Event('input', { bubbles: true }));
      Alpine.store('toasts').remove(tid);
      Alpine.store('toasts').success(t('common.upload_done'), { subtitle: data.filename + ' (' + formatSize(data.size) + ')' });
    } catch (e) {
      Alpine.store('toasts').remove(tid);
      Alpine.store('toasts').error(t('common.upload_failed') + ': ' + e.message);
    }
  };
  input.click();
}

