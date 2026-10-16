/* DiffDynamic Web UI — API client + UI logic */

const API = (typeof window !== 'undefined' && window.DD_ROOT) ? String(window.DD_ROOT).replace(/\/$/, '') : '';

