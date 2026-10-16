// ── Helpers ────────────────────────────────────────────────────────────────

async function api(path, opts = {}) {
  const headers = { 'Content-Type': 'application/json', ...accountHeaders(), ...(opts.headers || {}) };
  const res = await fetch(API + path, { ...opts, headers, credentials: 'same-origin' });
  if (res.status === 401) {
    redirectToLogin();
    throw new Error('Unauthorized');
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = err.detail;
    let msg = typeof detail === 'string' ? detail : (err.message || res.statusText);
    if (Array.isArray(detail)) {
      msg = detail.map(item => `${(item.loc || []).filter(part => part !== 'body').join('.')}: ${item.msg}`).join('; ');
    }
    if (res.status === 429) msg = t('common.limit_reached') + ': ' + msg;
    throw new Error(msg);
  }
  return res.json();
}

function redirectToLogin() {
  if (window.DD_AUTH_MODE === 'platform' && window.DD_LOGIN_URL) {
    const login = String(window.DD_LOGIN_URL);
    const sep = login.indexOf('?') >= 0 ? '&' : '?';
    window.location.href = login + sep + 'redirect=' + encodeURIComponent(window.location.href);
  }
}

// Visibility-aware interval: skips callbacks while the tab is hidden so
// background tabs stop hitting /api endpoints. On return to the tab the
// callback fires immediately (once) to refresh stale panels.
function ddInterval(fn, ms) {
  return setInterval(() => {
    if (document.hidden) return;
    fn();
  }, ms);
}

// Adaptive interval: runs at fastMs while condition() is true, slowMs
// otherwise; paused entirely while the tab is hidden.
function ddAdaptiveInterval(fn, condition, fastMs, slowMs) {
  let current = null;
  const tick = () => {
    if (document.hidden) return;
    fn().then(() => {
      const next = condition() ? fastMs : slowMs;
      if (next !== current) {
        current = next;
        clearInterval(timer);
        timer = setInterval(tick, current);
      }
    }).catch(() => {});
  };
  current = fastMs;
  let timer = setInterval(tick, current);
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) tick();
  });
  return timer;
}

function clampInt(v, min, max, dflt) {
  const n = parseInt(v, 10);
  if (isNaN(n)) return dflt;
  return Math.max(min, Math.min(max, n));
}

function getUser() { return localStorage.getItem('dd_user') || 'web_ui'; }
function setUser(name) { localStorage.setItem('dd_user', name || 'web_ui'); }
function accountHeaders() {
  return window.DD_AUTH_MODE === 'off' ? { 'X-DD-User': getUser() } : {};
}

let ddIsAdmin = false;
let ddMaxUploadBytes = 52428800;
let ddQuota = null;

function quotaLeft(kind) {
  if (ddIsAdmin || !ddQuota) return Infinity;
  const limit = kind === 'eval' ? ddQuota.eval_limit : ddQuota.generate_limit;
  if (!limit) return Infinity;
  const used = kind === 'eval' ? ddQuota.eval_used : ddQuota.generate_used;
  return Math.max(0, Number(limit) - Number(used || 0));
}

function applyDailyQuota() {
  const header = document.getElementById('header-quota');
  if (header) {
    if (ddIsAdmin) {
      header.style.display = 'inline-block';
      header.textContent = t('quota.unlimited');
    } else if (ddQuota && (ddQuota.generate_limit > 0 || ddQuota.eval_limit > 0)) {
      header.style.display = 'inline-block';
      const g = quotaLeft('generate');
      const e = quotaLeft('eval');
      header.textContent = tf('quota.header', {
        gen: g === Infinity ? '∞' : g,
        eval: e === Infinity ? '∞' : e,
      });
    } else {
      header.style.display = 'none';
    }
  }
  document.querySelectorAll('.js-needs-gen').forEach((btn) => {
    const left = quotaLeft('generate');
    btn.disabled = left === 0 || (btn.form && btn.form.id === 'sc-form' && scSubmissionActive);
    btn.title = left === 0 ? t('quota.gen_empty') : '';
  });
  document.querySelectorAll('.js-needs-eval').forEach((btn) => {
    const left = quotaLeft('eval');
    btn.disabled = left === 0;
    btn.title = left === 0 ? t('quota.eval_empty') : '';
  });
  document.querySelectorAll('.quota-hint[data-quota]').forEach((el) => {
    const kind = el.getAttribute('data-quota');
    const left = quotaLeft(kind);
    el.classList.toggle('is-empty', left === 0);
    if (ddIsAdmin) {
      el.textContent = t('quota.unlimited');
    } else if (left === Infinity) {
      el.textContent = '';
    } else if (left === 0) {
      el.textContent = t(kind === 'eval' ? 'quota.eval_empty' : 'quota.gen_empty');
    } else {
      el.textContent = tf(kind === 'eval' ? 'quota.eval_left' : 'quota.gen_left', { n: left });
    }
  });
}

function refreshQuota() {
  return api('/api/me').then((me) => {
    applyMe(me);
    return me;
  }).catch(() => null);
}

function applyMe(me) {
  if (!me) return;
  if (me.username) setUser(me.username);
  ddIsAdmin = !!me.is_admin;
  if (window.Alpine && Alpine.store('account')) Alpine.store('account').isAdmin = ddIsAdmin;
  document.querySelectorAll('[data-admin-only]').forEach(el => { el.hidden = !ddIsAdmin; });
  if (me.quota && me.quota.max_upload_bytes > 0) ddMaxUploadBytes = me.quota.max_upload_bytes;
  ddQuota = me.quota || null;
  const wrap = document.getElementById('header-user');
  const nameEl = document.getElementById('header-user-name');
  const badge = document.getElementById('header-admin-badge');
  if (wrap && nameEl) {
    wrap.style.display = 'flex';
    nameEl.textContent = me.username || '';
  }
  if (badge) badge.style.display = ddIsAdmin ? 'inline-block' : 'none';
  document.querySelectorAll('.runs-owner-col').forEach((el) => {
    el.style.display = ddIsAdmin ? '' : 'none';
  });
  applyDailyQuota();
}

