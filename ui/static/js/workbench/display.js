function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function statusClass(s) {
  if (s === 'completed') return 'status-completed';
  if (s === 'running') return 'status-running';
  if (s === 'failed') return 'status-failed';
  return 'status-pending';
}

function setProgressRunning(barEl, running) {
  if (!barEl) return;
  barEl.classList.toggle('is-running', !!running);
}

function switchToTab(tabName) {
  const btn = document.querySelector(`.nav-btn[data-tab="${tabName}"]`);
  if (btn) btn.click();
}

function prefersReducedMotion() {
  return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** Signature: light particle field in the header (diffusion metaphor). */
function initDiffusionField() {
  const canvas = document.getElementById('diffusion-field');
  if (!canvas || prefersReducedMotion()) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  let particles = [];
  let raf = 0;
  let w = 0;
  let h = 0;

  function resize() {
    const rect = canvas.parentElement.getBoundingClientRect();
    w = Math.max(1, Math.floor(rect.width));
    h = Math.max(1, Math.floor(rect.height));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.min(40, Math.max(24, Math.floor(w / 28)));
    particles = Array.from({ length: n }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      r: 1.2 + Math.random() * 2.2,
      vx: (Math.random() - 0.5) * 0.35,
      vy: (Math.random() - 0.5) * 0.25,
      a: 0.15 + Math.random() * 0.35,
    }));
  }

  function tick() {
    ctx.clearRect(0, 0, w, h);
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < -10) p.x = w + 10;
      if (p.x > w + 10) p.x = -10;
      if (p.y < -10) p.y = h + 10;
      if (p.y > h + 10) p.y = -10;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(34, 211, 238, ${p.a})`;
      ctx.fill();
    }
    // soft links between nearby particles
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const a = particles[i];
        const b = particles[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const d2 = dx * dx + dy * dy;
        if (d2 < 90 * 90) {
          const alpha = 0.12 * (1 - Math.sqrt(d2) / 90);
          ctx.strokeStyle = `rgba(126, 176, 255, ${alpha})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }
    raf = requestAnimationFrame(tick);
  }

  resize();
  tick();
  window.addEventListener('resize', () => {
    cancelAnimationFrame(raf);
    resize();
    tick();
  });
}

function fmtTime(t) {
  if (!t) return '—';
  return new Date(t).toLocaleString('zh-CN');
}

function escHtml(s) {
  if (s == null || s === '') return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function escapeAttr(s) {
  return escHtml(s).replace(/'/g, '&#39;');
}

const escAttr = escapeAttr;

function setProgressFill(el, pct) {
  if (el) el.style.transform = `scaleX(${Math.max(0, Math.min(100, pct)) / 100})`;
}

