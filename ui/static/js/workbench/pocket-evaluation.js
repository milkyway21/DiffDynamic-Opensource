// ── Pocket Evaluation ──────────────────────────────────────────────────

let peJobId = null;
let pePollTimer = null;

function togglePocketEvalMode() {
  const mode = $('#pe-mode').value;
  const ptGroup = $('#pe-pt-group');
  const genOpts = $('#pe-gen-options');
  if (mode === 'generate') {
    ptGroup.style.display = 'none';
    genOpts.style.display = 'block';
  } else {
    ptGroup.style.display = 'block';
    genOpts.style.display = 'none';
  }
}

async function handlePocketEval(e) {
  e.preventDefault();
  const protein = $('#pe-protein').value.trim();
  const ligand = $('#pe-ligand').value.trim();
  if (!protein) { alert(t('pe.protein')); return; }
  if (!ligand) { alert(t('pe.ligand')); return; }

  const mode = $('#pe-mode').value;
  const body = {
    protein_path: protein,
    ligand_path: ligand,
    generate_first: mode === 'generate',
  };

  if (mode === 'existing') {
    const pt = $('#pe-pt').value.trim();
    if (!pt) { alert(t('pe.pt_path')); return; }
    body.pt_path = pt;
  } else {
    const gpu = $('#pe-gpu').value;
    if (gpu) body.gpu_id = parseInt(gpu);
    body.batch_size = parseInt($('#pe-batch').value) || 5;
    body.pocket_radius = parseFloat($('#pe-radius').value) || 10.0;
  }

  try {
    const result = await api('/api/pocket-eval', { method: 'POST', body: JSON.stringify(body) });
    peJobId = result.job_id;
    setJobStatus('pe-status', { ...result, status: result.status === 'queued' ? 'pending' : result.status || 'running' });
    $('#pe-cancel-btn').disabled = false;
    const peBar = $('#pe-progress-bar');
    peBar.style.display = 'block';
    setProgressRunning(peBar, true);
    $('#pe-scores').innerHTML = '';
    $('#pe-gallery').innerHTML = '';
    startPePolling();
  } catch (e) {
    setLocalizedContent('pe-status', () => `<span class="status-failed">${t('common.error')}: ${escHtml(e.message)}</span>`);
  }
}

async function handlePocketEvalCancel() {
  if (!peJobId) return;
  try {
    await api(`/api/jobs/${peJobId}`, { method: 'DELETE' });
    setLocalizedContent('pe-status', () => `<span class="status-pending">${t('common.cancelled')}</span>`);
    stopPePolling();
  } catch (e) {
    setLocalizedContent('pe-status', () => `<span class="status-failed">${t('common.cancel_failed')}: ${escHtml(e.message)}</span>`);
  }
}

function startPePolling() {
  stopPePolling();
  pePollTimer = ddInterval(pollPocketEval, 3000);
}

function stopPePolling() {
  if (pePollTimer) { clearInterval(pePollTimer); pePollTimer = null; }
}

async function pollPocketEval() {
  if (!peJobId) return;
  try {
    const job = await api('/api/jobs/' + encodeURIComponent(peJobId));
    if (!job) return;

    const pct = Math.round((job.progress || 0) * 100);
    setProgressFill($('#pe-progress-fill'), pct);
    setProgressRunning($('#pe-progress-bar'), job.status === 'running' || job.status === 'pending');
    setJobStatus('pe-status', job);

    const log = (job.log_tail || []).slice(-25).join('\n');
    $('#pe-log').textContent = log;

    if (job.status === 'completed') {
      stopPePolling();
      setProgressRunning($('#pe-progress-bar'), false);
      $('#pe-cancel-btn').disabled = true;
      loadPocketEvalResults(peJobId);
    } else if (job.status === 'failed' || job.status === 'cancelled') {
      stopPePolling();
      setProgressRunning($('#pe-progress-bar'), false);
      $('#pe-cancel-btn').disabled = true;
    }
  } catch (e) { /* ignore poll errors */ }
}

async function loadPocketEvalResults(jobId) {
  try {
    const data = await api(`/api/pocket-eval/results/${jobId}`);
    renderPeScores(data.scores);
    renderPeGallery(data.images, data.dir_name);
  } catch (e) {
    $('#pe-scores').innerHTML = `<span class="status-failed">${escHtml(e.message)}</span>`;
  }
}

function renderPeScores(scores) {
  if (!scores || Object.keys(scores).length === 0) {
    $('#pe-scores').innerHTML = '<p style="color:var(--text-secondary)">No scores available</p>';
    return;
  }
  const dims = [
    { key: 'score_a', label: t('pe.dim_a') },
    { key: 'score_b', label: t('pe.dim_b') },
    { key: 'score_c', label: t('pe.dim_c') },
    { key: 'score_d', label: t('pe.dim_d') },
    { key: 'score_e', label: t('pe.dim_e') },
    { key: 'score_f', label: t('pe.dim_f') },
    { key: 'score_g', label: t('pe.dim_g') },
    { key: 'score_h', label: t('pe.dim_h') },
  ];
  let html = '';
  for (const d of dims) {
    const raw = scores[d.key];
    const val = raw === 'N/A' ? '—' : raw;
    const numVal = parseFloat(val);
    let cls = 'pe-score-na';
    let label = 'N/A';
    if (!isNaN(numVal)) {
      if (numVal >= 0.6) { cls = 'pe-score-high'; label = 'HIGH'; }
      else if (numVal >= 0.3) { cls = 'pe-score-medium'; label = 'MEDIUM'; }
      else { cls = 'pe-score-low'; label = 'LOW'; }
    }
    html += `<div class="pe-score-card ${cls}">
      <div class="pe-dim">${d.label}</div>
      <div class="pe-val">${val}</div>
      <div class="pe-label">${label}</div>
    </div>`;
  }
  // Overall
  const ov = scores.overall_score || '—';
  const ovLabel = scores.overall_label || '';
  let ovCls = 'pe-score-na';
  const ovNum = parseFloat(ov);
  if (!isNaN(ovNum)) {
    if (ovNum >= 0.6) ovCls = 'pe-score-high';
    else if (ovNum >= 0.3) ovCls = 'pe-score-medium';
    else ovCls = 'pe-score-low';
  }
  html += `<div class="pe-score-card ${ovCls}" style="grid-column: span 2">
    <div class="pe-dim">OVERALL</div>
    <div class="pe-val">${ov}</div>
    <div class="pe-label">${ovLabel.toUpperCase()}</div>
  </div>`;
  html += '<div class="radar-container"><canvas id="pe-radar" width="320" height="320"></canvas></div>';
  $('#pe-scores').innerHTML = html;
  const labels = dims.map(d => d.label);
  const values = dims.map(d => {
    const raw = scores[d.key];
    const v = parseFloat(raw);
    return isNaN(v) ? 0 : v;
  });
  renderRadarChart('pe-radar', labels, values);
}

function renderPeGallery(images, dirName) {
  if (!images || images.length === 0) {
    $('#pe-gallery').innerHTML = `<p style="color:var(--text-secondary)">${t('pe.no_images')}</p>`;
    return;
  }
  dirName = dirName || '';
  const html = images.map(img => {
    const src = `${API}/api/pocket-eval/images/${encodeURIComponent(dirName)}/${encodeURIComponent(img)}`;
    const label = img.replace('.png', '').replace(/_/g, ' ');
    return `<div class="pe-gallery-item">
      <img src="${escHtml(src)}" alt="${escHtml(img)}" data-fullsrc="${escHtml(src)}" loading="lazy">
      <div class="pe-img-label">${escHtml(label)}</div>
    </div>`;
  }).join('');
  $('#pe-gallery').innerHTML = html;
  const gallery = $('#pe-gallery');
  if (gallery && !gallery.dataset.bound) {
    gallery.dataset.bound = '1';
    gallery.addEventListener('click', (ev) => {
      const img = ev.target.closest('img');
      if (img && img.dataset.fullsrc) openPeLightbox(img.dataset.fullsrc);
    });
  }
}

function openPeLightbox(src) {
  const overlay = document.createElement('div');
  overlay.className = 'pe-lightbox';
  overlay.onclick = () => overlay.remove();
  const img = document.createElement('img');
  img.src = src;
  overlay.appendChild(img);
  document.body.appendChild(overlay);
}

// ── Radar Chart for Pocket Evaluation ────────────────────────────────

function renderRadarChart(canvasId, labels, values) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = 320;
  const h = canvas.height = 320;
  const cx = w / 2, cy = h / 2, r = 120;
  const n = labels.length;
  ctx.clearRect(0, 0, w, h);

  for (let ring = 1; ring <= 5; ring++) {
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const angle = (Math.PI * 2 * i / n) - Math.PI / 2;
      const rr = r * ring / 5;
      const x = cx + rr * Math.cos(angle);
      const y = cy + rr * Math.sin(angle);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = '#e5e7eb';
    ctx.stroke();
  }

  for (let i = 0; i < n; i++) {
    const angle = (Math.PI * 2 * i / n) - Math.PI / 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + r * Math.cos(angle), cy + r * Math.sin(angle));
    ctx.strokeStyle = '#d1d5db';
    ctx.stroke();
    ctx.fillStyle = '#374151';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const lx = cx + (r + 20) * Math.cos(angle);
    const ly = cy + (r + 20) * Math.sin(angle);
    ctx.fillText(labels[i], lx, ly);
  }

  ctx.beginPath();
  for (let i = 0; i <= n; i++) {
    const idx = i % n;
    const angle = (Math.PI * 2 * idx / n) - Math.PI / 2;
    const v = Math.max(0, Math.min(1, values[idx] || 0));
    const x = cx + r * v * Math.cos(angle);
    const y = cy + r * v * Math.sin(angle);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.fillStyle = 'rgba(37, 99, 235, 0.2)';
  ctx.fill();
  ctx.strokeStyle = '#2563eb';
  ctx.lineWidth = 2;
  ctx.stroke();

  for (let i = 0; i < n; i++) {
    const angle = (Math.PI * 2 * i / n) - Math.PI / 2;
    const v = Math.max(0, Math.min(1, values[i] || 0));
    const x = cx + r * v * Math.cos(angle);
    const y = cy + r * v * Math.sin(angle);
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#2563eb';
    ctx.fill();
  }
}

