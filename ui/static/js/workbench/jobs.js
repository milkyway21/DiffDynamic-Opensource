// ── Alpine.js Global Stores & Components ──────────────────────────────────

// Toast notification system
document.addEventListener('alpine:init', () => {
  Alpine.store('account', { isAdmin: ddIsAdmin });
  Alpine.store('toasts', {
    list: [],
    _counter: 0,
    add(message, opts = {}) {
      const id = ++this._counter;
      const type = opts.type || 'info';
      const icons = {
        success: '&#10003;', error: '&#10007;', warning: '&#9888;', info: '&#8505;'
      };
      this.list.push({ id, message, type, icon: icons[type] || icons.info, subtitle: opts.subtitle || '' });
      if (opts.timeout !== 0) {
        setTimeout(() => this.remove(id), opts.timeout || 5000);
      }
      return id;
    },
    success(msg, opts) { return this.add(msg, { ...opts, type: 'success' }); },
    error(msg, opts) { return this.add(msg, { ...opts, type: 'error' }); },
    warning(msg, opts) { return this.add(msg, { ...opts, type: 'warning' }); },
    info(msg, opts) { return this.add(msg, { ...opts, type: 'info' }); },
    remove(id) { this.list = this.list.filter(t => t.id !== id); },
  });
});

function toasts() {
  return {
    init() { this.store = Alpine.store('toasts'); },
    get list() { return this.store.list; },
    remove(id) { this.store.remove(id); },
  };
}

// Track previously seen job IDs to detect completion
let _seenJobStatuses = {};

// Global job monitor
function globalJobs() {
  return {
    language: currentLang,
    open: false,
    jobs: [],
    timer: null,
    get runningCount() { return this.jobs.filter(j => j.status === 'running').length; },
    get pendingCount() { return this.jobs.filter(j => j.status === 'pending').length; },
    get activeCount() { return this.jobs.filter(j => j.status === 'running' || j.status === 'pending').length; },
    label(key) { return T[key][this.language] || T[key].zh; },
    get counts() { return this.label('global.counts').replace('{running}', this.runningCount).replace('{pending}', this.pendingCount); },
    start() {
      // Adaptive: 2s while jobs are active, 20s when idle; paused when tab hidden
      this.poll();
      this.timer = ddAdaptiveInterval(
        async () => { await this.poll(); },
        () => this.activeCount > 0,
        2000,
        20000
      );
    },
    async poll() {
      try {
        const data = await api('/api/jobs');
        const active = data.filter(j => j.status === 'running' || j.status === 'pending');
        // Detect status transitions
        for (const j of data) {
          const prev = _seenJobStatuses[j.job_id];
          if (prev && prev !== j.status) {
            if (j.status === 'completed') {
              Alpine.store('toasts').success(`${j.job_type} ${t('hist.st_completed')}`, { subtitle: `Run #${j.run_id}` });
            } else if (j.status === 'failed') {
              Alpine.store('toasts').error(`${j.job_type} ${t('hist.st_failed')}`, { subtitle: (j.error || '').slice(0, 80) });
            } else if (j.status === 'cancelled') {
              Alpine.store('toasts').warning(`${j.job_type} ${t('common.cancelled')}`);
            }
          }
          _seenJobStatuses[j.job_id] = j.status;
        }
        this.jobs = active;
      } catch (e) { /* server may be restarting */ }
    },
    async cancelJob(jobId) {
      if (!confirm(tf('global.cancel_confirm', { id: jobId }))) return;
      try {
        await api(`/api/jobs/${jobId}`, { method: 'DELETE' });
        Alpine.store('toasts').info(t('common.cancelled'));
        this.poll();
      } catch (e) {
        Alpine.store('toasts').error(t('common.cancel_failed') + ': ' + e.message);
      }
    },
  };
}

