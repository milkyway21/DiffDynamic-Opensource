import * as THREE from 'three';

// ======= ORIGINAL PANEL CODE (DIFFUSION PATHS) =======

/** 平面地面标记（贴 XY，BasicMaterial 均匀着色） */
function addGroundDiskMarker(scene, x, y, zTerrain, fillHex, ringHex, scale = 1, showRing = true, showFill = true) {
  const lift = 0.065;
  const z = zTerrain + lift;
  const common = {
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -4,
    polygonOffsetUnits: -4,
  };
  if (showRing) {
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.17 * scale, 0.275 * scale, 56),
      new THREE.MeshBasicMaterial({ ...common, color: ringHex, opacity: 0.95 })
    );
    ring.position.set(x, y, z);
    ring.renderOrder = 10;
    scene.add(ring);
  }
  if (showFill) {
    const disk = new THREE.Mesh(
      new THREE.CircleGeometry(0.155 * scale, 48),
      new THREE.MeshBasicMaterial({ ...common, color: fillHex, opacity: 0.9 })
    );
    disk.position.set(x, y, z + 0.004);
    disk.renderOrder = 11;
    scene.add(disk);
  }
}

/** 降低饱和度（仅用于等高线、采样球与起终点标记） */
function softenColor(hex, satMul = 0.92) {
  const c = new THREE.Color(hex);
  const hsl = { h: 0, s: 0, l: 0 };
  c.getHSL(hsl);
  c.setHSL(hsl.h, THREE.MathUtils.clamp(hsl.s * satMul, 0, 1), hsl.l);
  return '#' + c.getHexString();
}

// Terrain vertex shading; canvas background is pure white.
function terrainSurfaceRgb(normalizedT) {
  const t = Math.min(1, Math.max(0, normalizedT));
  return {
    r: 0.82 - t * 0.12,
    g: 0.88 - t * 0.10,
    b: 0.94 - t * 0.08,
  };
}
const VIEW_BG_HEX = 0xffffff;

// Path palette: soft scientific colors.
const VIEW_PALETTE = {
  pathBlue: '#76b6e8',
  pathGreen: '#82d2b6',
  pathPurple: '#aa92dc',
  pathPrudent: '#82d2b6',
  prudentLevelOne: '#a6dfcb',
  prudentLevelTwo: '#91d4bb',
  prudentTerminal: '#dbb688',
  prudentTerminalEmissive: '#b58448',
  prudentBest: '#e59aa3',
  prudentBestEmissive: '#b96b77',
  prudentWinnerStageOne: '#e59aa3',
  prudentWinnerStageOneEmissive: '#b96b77',
  prudentWinnerStageTwo: '#e59aa3',
  prudentWinnerStageTwoEmissive: '#b96b77',
  prudentBranchDeep: '#5aa889',
  repairOrange: '#dbb688',
  repairEmissive: '#b58448',
  startRose: '#e74c3c',
  startEmissive: '#c0392b',
  endMint: '#27ae60',
  endEmissive: '#1e8449',
};

// ╔═══════════════ TERRAIN ═══════════════╗/ Uses XY = horizontal plane, Z = height (natural for PlaneGeometry w/o rotation)
const SIZE = 10;
const N = 120;

function gauss(x, y, cx, cy, sx, sy, A) {
  const dx = x - cx, dy = y - cy;
  return A * Math.exp(-(dx*dx)/(2*sx*sx) - (dy*dy)/(2*sy*sy));
}

function height(x, y) {
  let h = gauss(x, y, 3.5, 3.5, 1.7, 1.7, 4.2);
  h += gauss(x, y, -1.5, 2.2, 1.8, 1.4, 2.3);
  h += gauss(x, y, 1.2, -2.3, 1.4, 1.8, 1.9);
  h += gauss(x, y, -3.2, -0.8, 1.1, 1.1, 1.4);
  h += gauss(x, y, 0.2, 0.6, 1.5, 1.3, 1.6);
  // deep valley bottom-left
  h -= 1.4 * gauss(x, y, -4.5, -4.5, 0.75, 0.75, 1.0);
  // gentle saddle center
  h += 0.5 * gauss(x, y, -0.3, -0.3, 2.4, 2.4, 0.7);
  return h;
}

const START = { x: -4.5, y: -4.5 };
const END   = { x: 3.5, y: 3.5 };
START.z = height(START.x, START.y);
END.z   = height(END.x, END.y);

/** DiffDynamic 桥接段分段数（与下方 loop bi≤COUNT 一致）顶点索引 0…COUNT */
const DIFF_DYNAMIC_BRIDGE_COUNT = 4;
/** 路径 schedule 与采样球疏密共用幂（大步→小步递减） */
const DD_MARKER_SCHEDULE_POW = 0.8;
/** DiffDynamic：绿色路线上恰好 N 颗球；相邻弧段 Δ_k=Δ_1·rho^(k-1)（沿弧长递减） */
const DD_GREEN_BALL_COUNT = 20;
const DD_GREEN_GAP_RHO = 0.87;
/** Prudent 每条叶分支末端的黄色采样点数量，与 DiffDynamic 可见末端点数量一致。 */
const PRUDENT_TERMINAL_BALL_COUNT = 4;
/** Prudent 三个路径阶段分别显示 6 / 8 / 10 个采样球。 */
const PRUDENT_STAGE_GREEN_BALL_COUNTS = [6, 8, 10];
/** 黄色修复段仅占每条叶路径的最后 14%，使其接近 DiffDynamic 的紧凑末端。 */
const PRUDENT_TERMINAL_START = 0.86;

// ╔═══════════════ PATH GENERATORS ═══════════════╗/ All return arrays of {x, y, z} with correct terrain height

function pathTargetDiff(n) {
  // Travel from START to END along a straight baseline,
  // with smooth sinusoidal undulation perpendicular to the travel direction.
  const dx = END.x - START.x, dy = END.y - START.y;
  const travel = Math.sqrt(dx*dx + dy*dy);
  // Unit perpendicular (rotate 90°)
  const perpX = -dy / travel, perpY = dx / travel;
  const pts = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const bx = START.x + dx * t;
    const by = START.y + dy * t;
    // Amplitude envelope: rises and falls, peaks at center
    const amp = 2.5 * Math.sin(t * Math.PI);
    // Multi-frequency sinusoidal undulation → smooth wave look with strong visible amplitude
    const wave = Math.sin(t * 5.5 * Math.PI) * 0.6
               + Math.sin(t * 11 * Math.PI) * 0.25
               + Math.cos(t * 17 * Math.PI) * 0.15;
    const offset = amp * wave;
    let px = bx + perpX * offset;
    let py = by + perpY * offset;
    px = Math.max(-SIZE/2+0.1, Math.min(SIZE/2-0.1, px));
    py = Math.max(-SIZE/2+0.1, Math.min(SIZE/2-0.1, py));
    pts.push({ x: px, y: py, z: height(px, py) });
  }
  return pts;
}

function pathDiffDynamic(n, orangeSteps) {
  // Two-phase schedule:
  //   Phase 1 (28 green steps): t_sched = (i/mainSteps)^0.8 → early large Δt, later smaller Δt
  //   Phase 2 (3 orange steps): ...
  const mainSteps = n - orangeSteps;
  const greenPhaseScale = 0.94;
  const pts = [];
  const dx = END.x - START.x, dy = END.y - START.y;
  const travel = Math.sqrt(dx*dx + dy*dy);
  const perpX = -dy / travel, perpY = dx / travel;

  // --- Phase 1: power schedule, strides gradually decreasing ---
  const schedule = [];
  for (let i = 1; i <= mainSteps; i++) {
    const raw = i / mainSteps;
    // pow(raw, DD) → 早期大步、末期步幅递减（与采样球疏密一致）
    schedule.push(Math.pow(raw, DD_MARKER_SCHEDULE_POW));
  }

  // Compute first scheduled point for interpolation
  const t0 = schedule[0] * greenPhaseScale;
  const bx0 = START.x + dx * t0;
  const by0 = START.y + dy * t0;
  const eps0 = 0.06;
  const dzdx0 = (height(bx0+eps0,by0)-height(bx0-eps0,by0))/(2*eps0);
  const dzdy0 = (height(bx0,by0+eps0)-height(bx0,by0-eps0))/(2*eps0);
  const lambda0 = 1 - Math.pow(t0, 2.2);
  const gs0 = 0.28;
  const globalDx0 = (END.x - bx0) * lambda0 * 0.09;
  const globalDy0 = (END.y - by0) * lambda0 * 0.09;
  const localDx0 = -gs0 * dzdx0 * (1 - lambda0) * 0.6;
  const localDy0 = -gs0 * dzdy0 * (1 - lambda0) * 0.6;
  const waveAmp0 = 0.35 * Math.sin(schedule[0] * Math.PI);
  const waveOff0 = Math.sin(schedule[0] * 5 * Math.PI) * waveAmp0;
  const px0 = Math.max(-SIZE/2+0.1, Math.min(SIZE/2-0.1, bx0 + globalDx0 + localDx0 + perpX * waveOff0));
  const py0 = Math.max(-SIZE/2+0.1, Math.min(SIZE/2-0.1, by0 + globalDy0 + localDy0 + perpY * waveOff0));

  // Insert START + bridging points 鈫?smooth from START to first green
  for (let bi = 0; bi <= DIFF_DYNAMIC_BRIDGE_COUNT; bi++) {
    const frac = bi / DIFF_DYNAMIC_BRIDGE_COUNT;
    const ease = frac * frac; // quadratic ease-out
    const px = START.x + (px0 - START.x) * ease;
    const py = START.y + (py0 - START.y) * ease;
    pts.push({ x: px, y: py, z: height(px, py), t: frac * t0, i: -DIFF_DYNAMIC_BRIDGE_COUNT + bi });
  }

  for (let j = 1; j < schedule.length; j++) {
    const t = schedule[j] * greenPhaseScale; // Phase 1 covers 0..greenPhaseScale
    const bx = START.x + dx * t;
    const by = START.y + dy * t;
    const eps = 0.06;
    const dzdx = (height(bx+eps,by)-height(bx-eps,by))/(2*eps);
    const dzdy = (height(bx,by+eps)-height(bx,by-eps))/(2*eps);
    const lambda = 1 - Math.pow(t, 2.2);
    const gs = 0.28;
    const globalDx = (END.x - bx) * lambda * 0.09;
    const globalDy = (END.y - by) * lambda * 0.09;
    const localDx = -gs * dzdx * (1 - lambda) * 0.6;
    const localDy = -gs * dzdy * (1 - lambda) * 0.6;
    // Mild perpendicular wave (visible but subtle)
    const waveAmp = 0.35 * Math.sin(schedule[j] * Math.PI);
    const waveOff = Math.sin(schedule[j] * 5 * Math.PI) * waveAmp;
    let px = bx + globalDx + localDx + perpX * waveOff;
    let py = by + globalDy + localDy + perpY * waveOff;
    px = Math.max(-SIZE/2+0.1, Math.min(SIZE/2-0.1, px));
    py = Math.max(-SIZE/2+0.1, Math.min(SIZE/2-0.1, py));
    pts.push({ x: px, y: py, z: height(px, py), t: schedule[j], i: j });
  }

  // --- Phase 2: orange segment continues in t right after last green (same formulas, no blend gap)
  const lastGreenT = schedule[schedule.length - 1] * greenPhaseScale;
  const tailSpan = 1 - lastGreenT;
  for (let i = 1; i <= orangeSteps; i++) {
    // Spread orange samples across (lastGreenT, 1): farther apart than fixed 0.97+delta
    const t = lastGreenT + (i / (orangeSteps + 1)) * tailSpan;
    const bx = START.x + dx * t;
    const by = START.y + dy * t;
    const eps = 0.06;
    const dzdx = (height(bx+eps,by)-height(bx-eps,by))/(2*eps);
    const dzdy = (height(bx,by+eps)-height(bx,by-eps))/(2*eps);
    const lambda = 1 - Math.pow(t, 2.2);
    const gs = 0.22;
    const globalDx = (END.x - bx) * lambda * 0.04;
    const globalDy = (END.y - by) * lambda * 0.04;
    const localDx = -gs * dzdx * (1 - lambda) * 0.35;
    const localDy = -gs * dzdy * (1 - lambda) * 0.35;
    let px = bx + globalDx + localDx;
    let py = by + globalDy + localDy;
    px = Math.max(-SIZE/2+0.1, Math.min(SIZE/2-0.1, px));
    py = Math.max(-SIZE/2+0.1, Math.min(SIZE/2-0.1, py));
    pts.push({ x: px, y: py, z: height(px, py), t, i: mainSteps + i });
  }
  // Ensure final steps end near END
  pts.push({ x: END.x, y: END.y, z: height(END.x, END.y), t: 1, i: mainSteps + orangeSteps });

  return pts;
}

function pathGlintDM(n) {
  const pts = [];
  const dx = END.x - START.x, dy = END.y - START.y;
  const travel = Math.sqrt(dx*dx + dy*dy);
  const perpX = -dy / travel, perpY = dx / travel;
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const bx = START.x + dx * t;
    const by = START.y + dy * t;
    const lambda = t;
    const eps = 0.06;
    const dzdx = (height(bx+eps,by)-height(bx-eps,by))/(2*eps);
    const dzdy = (height(bx,by+eps)-height(bx,by-eps))/(2*eps);
    const sigmaSq = 0.12 * t * (1 - t) * 3 + 0.015;
    const gs = 0.18;
    const globalDx = (END.x - bx) * lambda * 0.04 * sigmaSq * 2.5;
    const globalDy = (END.y - by) * lambda * 0.04 * sigmaSq * 2.5;
    const localDx = -gs * dzdx * (1 - lambda) * sigmaSq * 1.5;
    const localDy = -gs * dzdy * (1 - lambda) * sigmaSq * 1.5;
    const gumbelPhase = t * 12;
    const gumbelAmp = 0.03 * sigmaSq * (1 - t);
    const gumbelDx = gumbelAmp * Math.cos(gumbelPhase);
    const gumbelDy = gumbelAmp * Math.sin(gumbelPhase);
    // Pronounced sinusoidal undulation along the path
    const waveAmp = 1.4 * Math.sin(t * Math.PI);
    const wave = Math.sin(t * 3.5 * Math.PI) * waveAmp;
    let px = bx + globalDx + localDx + gumbelDx + perpX * wave;
    let py = by + globalDy + localDy + gumbelDy + perpY * wave;
    px = Math.max(-SIZE/2+0.1, Math.min(SIZE/2-0.1, px));
    py = Math.max(-SIZE/2+0.1, Math.min(SIZE/2-0.1, py));
    pts.push({ x: px, y: py, z: height(px, py) });
  }
  return pts;
}

// 在两点之间生成贴合地形的平滑路径段。结合 TargetDiff 的垂直波动与 DiffDynamic 的局部地形梯度，避免树枝成为直线。
function terrainBranchSegment(from, to, samples = 16, bend = 0, terrainStrength = 0.28, wavePhase = 0) {
  const dx = to.x - from.x, dy = to.y - from.y;
  const length = Math.hypot(dx, dy) || 1;
  const perpX = -dy / length, perpY = dx / length;
  const pts = [];
  for (let i = 0; i <= samples; i++) {
    const t = i / samples;
    const eased = t * t * (3 - 2 * t);
    const envelope = Math.sin(Math.PI * t);
    const baseX = from.x + dx * eased;
    const baseY = from.y + dy * eased;

    const eps = 0.07;
    const dzdx = (height(baseX + eps, baseY) - height(baseX - eps, baseY)) / (2 * eps);
    const dzdy = (height(baseX, baseY + eps) - height(baseX, baseY - eps)) / (2 * eps);
    const lateralSlope = dzdx * perpX + dzdy * perpY;
    const terrainOffset = THREE.MathUtils.clamp(
      -lateralSlope * terrainStrength,
      -0.42,
      0.42
    ) * envelope;
    const waveAmplitude = Math.min(0.21, 0.1 + length * 0.014);
    const waveOffset = waveAmplitude * Math.sin(t * Math.PI * 2.2 + wavePhase) * envelope;
    const bow = bend * envelope + terrainOffset + waveOffset;

    const x = THREE.MathUtils.clamp(baseX + perpX * bow, -SIZE / 2 + 0.1, SIZE / 2 - 0.1);
    const y = THREE.MathUtils.clamp(baseY + perpY * bow, -SIZE / 2 + 0.1, SIZE / 2 - 0.1);
    pts.push({ x, y, z: height(x, y) });
  }
  return pts;
}

/** 按三维弧长比例切分割折线，并在切点处插入贴合地形的新顶点。 */
function splitPolylineByFractions(polyPts, fractions) {
  if (polyPts.length < 2) return [polyPts.slice()];
  const segmentLengths = [];
  let totalLength = 0;
  for (let i = 0; i < polyPts.length - 1; i++) {
    const a = polyPts[i], b = polyPts[i + 1];
    const length = Math.hypot(b.x - a.x, b.y - a.y, b.z - a.z);
    segmentLengths.push(length);
    totalLength += length;
  }
  const targets = fractions.map((fraction) => totalLength * fraction);
  const result = [[]];
  const appendUnique = (array, point) => {
    const last = array[array.length - 1];
    if (!last || Math.hypot(point.x - last.x, point.y - last.y, point.z - last.z) > 1e-8) {
      array.push(point);
    }
  };
  appendUnique(result[0], { ...polyPts[0] });
  let accumulated = 0;
  let targetIndex = 0;
  for (let i = 0; i < segmentLengths.length; i++) {
    const a = polyPts[i], b = polyPts[i + 1];
    const length = segmentLengths[i];
    while (targetIndex < targets.length && targets[targetIndex] <= accumulated + length + 1e-9) {
      const localT = length > 1e-10 ? (targets[targetIndex] - accumulated) / length : 0;
      const x = a.x + (b.x - a.x) * localT;
      const y = a.y + (b.y - a.y) * localT;
      const cut = { x, y, z: height(x, y) };
      appendUnique(result[result.length - 1], cut);
      result.push([{ ...cut }]);
      targetIndex++;
    }
    appendUnique(result[result.length - 1], { ...b });
    accumulated += length;
  }
  return result;
}

// Prudent 模式：单一路径先抵达第一分叉点，分为三路；三路分别再次三分，最终得到包含原 END 在内的 9 个候选终点。
function pathPrudentTree() {
  const dx = END.x - START.x, dy = END.y - START.y;
  const travel = Math.hypot(dx, dy);
  const ux = dx / travel, uy = dy / travel;
  const perpX = -uy, perpY = ux;

  const terrainPoint = (x, y) => ({ x, y, z: height(x, y) });
  const pointOnAxis = (progress, along = 0, lateral = 0) => {
    const baseX = START.x + dx * progress;
    const baseY = START.y + dy * progress;
    return terrainPoint(baseX + ux * along + perpX * lateral, baseY + uy * along + perpY * lateral);
  };

  // Core path comes from DiffDynamic and is cut into 26% / 28% / 46% stages.
  const corePath = pathDiffDynamic(28, 3);
  const coreSegments = splitPolylineByFractions(corePath, [0.26, 0.54]);
  const firstFork = coreSegments[0][coreSegments[0].length - 1];
  const coreSecondFork = coreSegments[1][coreSegments[1].length - 1];

  // Spread the nine endpoints along a jittered wave curve near the upper-right target.
  const curveTangentX = Math.SQRT1_2;
  const curveTangentY = -Math.SQRT1_2;
  const curveNormalX = Math.SQRT1_2;
  const curveNormalY = Math.SQRT1_2;
  const fixedJitter = (seed) => {
    const value = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
    return value - Math.floor(value) - 0.5;
  };
  const endpointSpreadScale = 1.5;
  const endpointOnCurve = (t, index) => {
    if (index === 4) return terrainPoint(END.x, END.y);
    const along = 2.2 * t + fixedJitter(index + 11) * 0.15;
    const wave = 0.52 * Math.sin(t * Math.PI * 2)
      + 0.16 * Math.sin(t * Math.PI * 5)
      + fixedJitter(index + 37) * 0.22;
    const rawX = END.x + curveTangentX * along + curveNormalX * wave;
    const rawY = END.y + curveTangentY * along + curveNormalY * wave;
    const x = END.x + (rawX - END.x) * endpointSpreadScale;
    const y = END.y + (rawY - END.y) * endpointSpreadScale;
    return terrainPoint(
      THREE.MathUtils.clamp(x, -SIZE / 2 + 0.25, SIZE / 2 - 0.25),
      THREE.MathUtils.clamp(y, -SIZE / 2 + 0.25, SIZE / 2 - 0.25)
    );
  };
  const endpoints = Array.from({ length: 9 }, (_, index) => endpointOnCurve(-1 + index / 4, index));

  // Three second-level forks share the 54% core progress point and spread toward the terrain edges.
  const secondForkSpread = 1.05;
  const secondForks = [1, 0, -1].map((lane) => terrainPoint(
    coreSecondFork.x + perpX * lane * secondForkSpread,
    coreSecondFork.y + perpY * lane * secondForkSpread
  ));

  const segments = [coreSegments[0]];
  // Outer first-level branches bend toward the terrain edges.
  const levelOneBends = [0.72, 0, -0.72];
  secondForks.forEach((fork, branchIndex) => {
    if (branchIndex === 1) {
      segments.push(coreSegments[1]);
    } else {
      segments.push(terrainBranchSegment(
        firstFork,
        fork,
        26,
        levelOneBends[branchIndex],
        0.3,
        branchIndex * 1.25 + 0.4
      ));
    }
  });
  const leafBends = [-0.42, 0, 0.42];
  const centerLeafBends = [-0.72, 0, 0.72];
  secondForks.forEach((fork, branchIndex) => {
    for (let childIndex = 0; childIndex < 3; childIndex++) {
      const endpoint = endpoints[branchIndex * 3 + childIndex];
      if (branchIndex === 1 && childIndex === 1) {
        segments.push(coreSegments[2]);
      } else {
        segments.push(terrainBranchSegment(
          fork,
          endpoint,
          30,
          (branchIndex === 1 ? centerLeafBends[childIndex] : leafBends[childIndex])
            + (branchIndex - 1) * 0.04,
          0.28,
          (branchIndex * 3 + childIndex) * 0.72 + 0.25
        ));
      }
    }
  });

  return {
    segments,
    branchPoints: [firstFork, ...secondForks],
    endpoints,
    bestEndpointIndex: 4,
    bestSegmentIndex: 8,
    stageRatios: [0.26, 0.28, 0.46],
    coreStageTwoSegmentIndex: 2,
    corePathMode: 'DiffDynamic',
    topSegmentIndices: [0, 2, 8],
    endpointSpreadScale,
  };
}

// Terrain contour generation.
function makeContours(levels) {
  const G = 80;
  const data = new Float32Array(G*G);
  const hs = SIZE/2;
  for (let iy = 0; iy < G; iy++) {
    for (let ix = 0; ix < G; ix++) {
      data[iy*G+ix] = height((ix/(G-1)-0.5)*SIZE, (iy/(G-1)-0.5)*SIZE);
    }
  }
  const results = [];
  levels.forEach(level => {
    const segs = [];
    for (let iy = 0; iy < G-1; iy++) {
      for (let ix = 0; ix < G-1; ix++) {
        const v00=data[iy*G+ix], v10=data[iy*G+ix+1], v01=data[(iy+1)*G+ix], v11=data[(iy+1)*G+ix+1];
        const x0=(ix/(G-1)-0.5)*SIZE, x1=((ix+1)/(G-1)-0.5)*SIZE;
        const y0=(iy/(G-1)-0.5)*SIZE, y1=((iy+1)/(G-1)-0.5)*SIZE;
        const iso=level;
        const pts=[];
        if((v00<iso)!==(v10<iso)){const t=(iso-v00)/(v10-v00); pts.push({x:x0+t*(x1-x0),y:y0});}
        if((v00<iso)!==(v01<iso)){const t=(iso-v00)/(v01-v00); pts.push({x:x0,y:y0+t*(y1-y0)});}
        if((v10<iso)!==(v11<iso)){const t=(iso-v10)/(v11-v10); pts.push({x:x1,y:y0+t*(y1-y0)});}
        if((v01<iso)!==(v11<iso)){const t=(iso-v01)/(v11-v01); pts.push({x:x0+t*(x1-x0),y:y1});}
        if(pts.length>=2)segs.push([pts[0],pts[1]]);
        if(pts.length===4)segs.push([pts[2],pts[3]]);
      }
    }
    results.push({level,segs});
  });
  return results;
}

// 沿 3D 折线按弧长取 numBalls 个点：相邻两段弧长 Δ_k = Δ_1·rho^(k-1)，rho<1 则愈往后愈密。
function samplePolylineDecreasingGaps(polyPts, numBalls, rho) {
  const out = [];
  if (!polyPts.length || numBalls < 1) return out;
  if (polyPts.length === 1 || numBalls === 1) {
    const p = polyPts[0];
    for (let i = 0; i < numBalls; i++) out.push({ x: p.x, y: p.y, z: p.z });
    return out;
  }
  const segLen = [];
  let total = 0;
  for (let i = 0; i < polyPts.length - 1; i++) {
    const a = polyPts[i], b = polyPts[i + 1];
    const len = Math.hypot(b.x - a.x, b.y - a.y, b.z - a.z);
    segLen.push(len);
    total += len;
  }
  if (total < 1e-10) {
    const p = polyPts[0];
    for (let i = 0; i < numBalls; i++) out.push({ x: p.x, y: p.y, z: p.z });
    return out;
  }
  const nGap = numBalls - 1;
  const geomSum = (1 - Math.pow(rho, nGap)) / (1 - rho);
  const d1 = total / geomSum;
  const targets = [0];
  let acc = 0;
  for (let k = 1; k < numBalls; k++) {
    acc += d1 * Math.pow(rho, k - 1);
    targets.push(acc);
  }
  targets[numBalls - 1] = total;

  const atDist = (dist) => {
    const d = Math.min(total, Math.max(0, dist));
    let cum = 0;
    for (let i = 0; i < segLen.length; i++) {
      if (cum + segLen[i] >= d - 1e-10) {
        const p0 = polyPts[i], p1 = polyPts[i + 1];
        const sl = segLen[i];
        const t = sl > 1e-12 ? (d - cum) / sl : 0;
        return {
          x: p0.x + t * (p1.x - p0.x),
          y: p0.y + t * (p1.y - p0.y),
          z: p0.z + t * (p1.z - p0.z),
        };
      }
      cum += segLen[i];
    }
    const last = polyPts[polyPts.length - 1];
    return { x: last.x, y: last.y, z: last.z };
  };

  for (let i = 0; i < numBalls; i++) out.push(atDist(targets[i]));
  return out;
}


export {
  DD_GREEN_BALL_COUNT,
  DD_GREEN_GAP_RHO,
  END,
  N,
  PRUDENT_STAGE_GREEN_BALL_COUNTS,
  PRUDENT_TERMINAL_BALL_COUNT,
  PRUDENT_TERMINAL_START,
  SIZE,
  START,
  VIEW_BG_HEX,
  VIEW_PALETTE,
  addGroundDiskMarker,
  height,
  makeContours,
  pathDiffDynamic,
  pathGlintDM,
  pathPrudentTree,
  pathTargetDiff,
  samplePolylineDecreasingGaps,
  softenColor,
  terrainSurfaceRgb
};
