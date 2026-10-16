/* CPU projection fallback for browsers without WebGL. The scene remains in 3D. */
window.createScaffoldCanvasViewer = function(containerId) {
  const host = document.getElementById(containerId);
  host.replaceChildren();
  const canvas = document.createElement('canvas');
  canvas.tabIndex = 0;
  canvas.setAttribute('aria-label', t('sc.canvas_label'));
  host.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  const scene = { models: [], spheres: [], arrows: [], labels: [], center: [0, 0, 0], radius: 12, yaw: 0.2, pitch: -0.3, picked: [] };
  let down = null;
  function project(point) {
    const x = point.x - scene.center[0], y = point.y - scene.center[1], z = point.z - scene.center[2];
    const cy = Math.cos(scene.yaw), sy = Math.sin(scene.yaw), cp = Math.cos(scene.pitch), sp = Math.sin(scene.pitch);
    const a = cy * x + sy * z, b = -sy * x + cy * z;
    const scale = Math.min(canvas.width, canvas.height) / (scene.radius * 2.6);
    return { x: canvas.width / 2 + a * scale, y: canvas.height / 2 - (cp * y - sp * b) * scale, z: sp * y + cp * b, scale };
  }
  function backboneTraces(atoms) {
    const residues = new Map();
    atoms.forEach(atom => {
      const key = `${atom.segment}:${atom.chain}:${atom.resi}:${atom.icode}`;
      if (!residues.has(key)) residues.set(key, {});
      const residue = residues.get(key);
      if (!residue[atom.atom]) residue[atom.atom] = atom;
    });
    const traces = [];
    let trace = [];
    for (const residue of residues.values()) {
      const atom = residue.CA;
      if (!atom) { if (trace.length > 1) traces.push(trace); trace = []; continue; }
      const previous = trace.length ? trace[trace.length - 1].atom : null;
      const distance = previous ? Math.hypot(atom.x - previous.x, atom.y - previous.y, atom.z - previous.z) : 0;
      const contiguous = previous && atom.chain === previous.chain && atom.segment === previous.segment &&
        (atom.resi === previous.resi + 1 || (atom.resi === previous.resi && atom.icode !== previous.icode)) && distance < 4.8;
      if (previous && !contiguous) { if (trace.length > 1) traces.push(trace); trace = []; }
      trace.push({ atom, oxygen: residue.O, carbon: residue.C });
    }
    if (trace.length > 1) traces.push(trace);
    return traces;
  }

  function drawRibbon(trace, style) {
    const samples = [];
    let previousNormal = null;
    const normalize = vector => {
      const length = Math.hypot(...vector) || 1;
      return vector.map(value => value / length);
    };
    for (let i = 0; i < trace.length - 1; i++) {
      const p = [trace[Math.max(0, i - 1)].atom, trace[i].atom,
        trace[i + 1].atom, trace[Math.min(trace.length - 1, i + 2)].atom];
      const residue = trace[i];
      const tangent = normalize(['x', 'y', 'z'].map(axis => p[2][axis] - p[1][axis]));
      let normal = residue.oxygen && residue.carbon
        ? ['x', 'y', 'z'].map(axis => residue.oxygen[axis] - residue.carbon[axis])
        : [-tangent[1], tangent[0], 0.2];
      const dot = normal.reduce((sum, value, axis) => sum + value * tangent[axis], 0);
      normal = normalize(normal.map((value, axis) => value - dot * tangent[axis]));
      if (previousNormal && normal.reduce((sum, value, axis) => sum + value * previousNormal[axis], 0) < 0) {
        normal = normal.map(value => -value);
      }
      const oldNormal = previousNormal || normal;
      const sampleCount = i === trace.length - 2 ? 9 : 8;
      for (let j = 0; j < sampleCount; j++) {
        const t = j / 8, t2 = t * t, t3 = t2 * t;
        const point = {};
        ['x', 'y', 'z'].forEach(axis => {
          point[axis] = 0.5 * ((2 * p[1][axis]) + (-p[0][axis] + p[2][axis]) * t +
            (2 * p[0][axis] - 5 * p[1][axis] + 4 * p[2][axis] - p[3][axis]) * t2 +
            (-p[0][axis] + 3 * p[1][axis] - 3 * p[2][axis] + p[3][axis]) * t3);
        });
        const side = normalize(normal.map((value, axis) => oldNormal[axis] * (1 - t) + value * t));
        const edge = sign => Object.fromEntries(['x', 'y', 'z'].map((axis, k) => [axis, point[axis] + sign * side[k] * 0.85]));
        samples.push({ left: project(edge(-1)), right: project(edge(1)) });
      }
      previousNormal = normal;
    }
    ctx.save();
    ctx.globalAlpha = style.opacity || 0.9;
    ctx.fillStyle = '#79b5dc';
    ctx.strokeStyle = '#aed6ee';
    ctx.lineWidth = 0.7;
    ctx.beginPath();
    samples.forEach((sample, index) => {
      if (index === 0) ctx.moveTo(sample.left.x, sample.left.y);
      else ctx.lineTo(sample.left.x, sample.left.y);
    });
    [...samples].reverse().forEach(sample => ctx.lineTo(sample.right.x, sample.right.y));
    ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.restore();
  }

  function render() {
    canvas.setAttribute('aria-label', t('sc.canvas_label'));
    const width = Math.max(host.clientWidth, 100), height = Math.max(host.clientHeight, 100);
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    ctx.fillStyle = '#0a1628'; ctx.fillRect(0, 0, width, height);
    scene.picked = [];
    const all = [];
    scene.models.forEach(model => {
      if (model.protein) {
        model.ribbons.forEach(trace => drawRibbon(trace, model.cartoon));
        model.atoms.filter(atom => atom.atom === 'CA').forEach(atom => {
          scene.picked.push({ ...project(atom), atom, model, radius: 8 });
        });
        return;
      }
      model.bonds.forEach(([i, j]) => {
        const a = project(model.atoms[i]), b = project(model.atoms[j]);
        ctx.strokeStyle = '#91b5b9'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      });
      model.atoms.forEach(atom => all.push({ ...project(atom), atom, model, color: atom.color || '#a9b8c7', radius: 5 }));
    });
    canvas.dataset.proteinRepresentation = 'ribbon';
    scene.spheres.forEach(s => all.push({ ...project(s.center), color: s.color, radius: s.radius < 0.3 ? 4 : 7 }));
    all.sort((a, b) => a.z - b.z).forEach(p => {
      ctx.fillStyle = p.color; ctx.beginPath(); ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2); ctx.fill();
      if (p.atom) scene.picked.push(p);
    });
    scene.arrows.forEach(a => {
      const start = project(a.start), end = project(a.end);
      ctx.strokeStyle = a.color; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(start.x, start.y); ctx.lineTo(end.x, end.y); ctx.stroke();
    });
    ctx.font = '12px sans-serif';
    scene.labels.forEach(l => { const p = project(l.style.position); ctx.fillStyle = '#fff'; ctx.fillText(l.text, p.x + 10, p.y - 8); });
    ctx.fillStyle = '#a8becf'; ctx.fillText(t('sc.canvas_hint'), 10, height - 12);
  }
  canvas.addEventListener('pointerdown', event => { down = { x: event.offsetX, y: event.offsetY, travel: 0 }; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener('pointermove', event => {
    if (!down) return;
    const dx = event.offsetX - down.x, dy = event.offsetY - down.y;
    down.travel += Math.abs(dx) + Math.abs(dy); down.x = event.offsetX; down.y = event.offsetY;
    scene.yaw += dx * 0.01; scene.pitch += dy * 0.01; render();
  });
  canvas.addEventListener('pointerup', event => {
    if (down && down.travel < 5) {
      const nearby = scene.picked.filter(p => Math.hypot(p.x - event.offsetX, p.y - event.offsetY) < 10);
      const distance = p => Math.hypot(p.x - event.offsetX, p.y - event.offsetY) / p.radius;
      const p = nearby.sort((a, b) => distance(a) - distance(b) || b.z - a.z)[0];
      if (p && p.model.callback) p.model.callback(p.atom);
    }
    down = null;
  });
  canvas.addEventListener('wheel', event => { event.preventDefault(); scene.radius = Math.max(2, Math.min(100, scene.radius * Math.exp(event.deltaY * 0.001))); render(); }, { passive: false });
  canvas.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '+', '-', '='].includes(event.key)) return;
    event.preventDefault();
    if (event.key === 'ArrowLeft') scene.yaw -= 0.1;
    if (event.key === 'ArrowRight') scene.yaw += 0.1;
    if (event.key === 'ArrowUp') scene.pitch -= 0.1;
    if (event.key === 'ArrowDown') scene.pitch += 0.1;
    if (event.key === '+' || event.key === '=') scene.radius = Math.max(2, scene.radius / 1.1);
    if (event.key === '-') scene.radius = Math.min(100, scene.radius * 1.1);
    render();
  });
  const viewer = {
    addModel(block, format) {
      const lines = block.split('\n'), atoms = [], bonds = [];
      if (format === 'sdf') {
        const n = Number(lines[3].slice(0, 3)), m = Number(lines[3].slice(3, 6));
        for (let i = 0; i < n; i++) { const l = lines[4 + i]; atoms.push({ index: i, x: Number(l.slice(0, 10)), y: Number(l.slice(10, 20)), z: Number(l.slice(20, 30)) }); }
        for (let i = 0; i < m; i++) { const l = lines[4 + n + i]; bonds.push([Number(l.slice(0, 3)) - 1, Number(l.slice(3, 6)) - 1]); }
      } else {
        let segment = 0;
        for (const line of lines) {
          if (line.startsWith('ENDMDL')) break;
          if (line.startsWith('TER')) { segment++; continue; }
          if (!/^(ATOM  |HETATM)/.test(line) || ![' ', 'A', ''].includes(line.slice(16, 17))) continue;
          const atom = { index: atoms.length, atom: line.slice(12, 16).trim(),
            chain: line.slice(21, 22), resi: Number(line.slice(22, 26)),
            icode: line.slice(26, 27), segment,
            x: Number(line.slice(30, 38)), y: Number(line.slice(38, 46)), z: Number(line.slice(46, 54)) };
          if ([atom.x, atom.y, atom.z].every(Number.isFinite)) atoms.push(atom);
        }
      }
      const id = scene.models.length;
      const model = { atoms, bonds, protein: format === 'pdb', callback: null,
        ribbons: format === 'pdb' ? backboneTraces(atoms) : [], cartoon: {},
        getID: () => id, setClickable: (selection, enabled, callback) => { model.callback = enabled ? callback : null; },
        selectedAtoms: selection => atoms.filter(atom => !selection.atom || atom.atom === selection.atom),
        setStyle(selection, style) {
          if (style.cartoon) model.cartoon = style.cartoon;
          const color = (style.stick || style.sphere || {}).color;
          atoms.forEach(atom => { if (!selection.index || selection.index.includes(atom.index)) atom.color = color; });
        },
      };
      scene.models.push(model); return model;
    },
    removeAllModels: () => { scene.models = []; }, removeAllShapes: () => { scene.spheres = []; scene.arrows = []; }, removeAllLabels: () => { scene.labels = []; },
    addSphere: sphere => { scene.spheres.push(sphere); }, addArrow: arrow => { scene.arrows.push(arrow); }, addLabel: (text, style) => { scene.labels.push({ text, style }); },
    zoomTo(selection = {}) {
      const atoms = selection.model === undefined
        ? [...scene.models.flatMap(model => model.atoms), ...scene.spheres.map(sphere => sphere.center)]
        : scene.models[selection.model].atoms;
      if (!atoms.length) return;
      scene.center = ['x', 'y', 'z'].map(key => atoms.reduce((n, a) => n + a[key], 0) / atoms.length);
      scene.radius = Math.max(4, ...atoms.map(a => Math.hypot(a.x - scene.center[0], a.y - scene.center[1], a.z - scene.center[2])));
    },
    resetOrientation: () => { scene.yaw = 0.2; scene.pitch = -0.3; },
    zoom: factor => { scene.radius /= factor; }, resize: render, render,
  };
  render();
  return viewer;
};
