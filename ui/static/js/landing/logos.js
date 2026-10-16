

// DiffDynamic logo.
function drawLogo() {
  const canvas = document.getElementById('logoCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = 170;
  const R = 148;
  let tick = 0;

  // Draw the logo sphere body.
  function drawSphere() {
    const breathe = 1 + 0.02 * Math.sin(tick * 0.03);
    // 主体渐变
    const grad = ctx.createRadialGradient(cx - 30, cy - 55, R * 0.05, cx, cy, R * breathe);
    grad.addColorStop(0, '#f5faff');
    grad.addColorStop(0.15, '#d6e9fa');
    grad.addColorStop(0.5, '#a8cdee');
    grad.addColorStop(0.8, '#7dadd8');
    grad.addColorStop(1, '#4a80b0');

    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();

    // 光泽高光
    const highlight = ctx.createRadialGradient(cx - 50, cy - 70, R * 0.02, cx - 25, cy - 45, R * 0.75);
    highlight.addColorStop(0, 'rgba(255,255,255,0.6)');
    highlight.addColorStop(0.4, 'rgba(220,240,255,0.25)');
    highlight.addColorStop(0.8, 'rgba(140,190,230,0.06)');
    highlight.addColorStop(1, 'rgba(50,90,150,0)');
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.fillStyle = highlight;
    ctx.fill();

    // Soft sphere rim.
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(130,195,240,0.4)';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // 黄金螺旋轨迹 — 等角螺线，严格两圈  // r(胃) = a · e^(b胃)，从 胃_start 到 胃_start + 4蟺（两圈）
  // 黄金比例体现在增长因子 b：选 b 使 r 一周后增长 蠁 倍 → e^(b·2蟺) = 蠁
  // 即 b = ln(蠁) / (2蟺)
  const PHI = 1.61803398875;
  const B = Math.log(PHI) / (2 * Math.PI);
  const THETA_START = Math.PI * 0.55;
  const THETA_END = THETA_START + 4 * Math.PI; // 两圈
  const A = 0.30; // 起始半径（占球面半径的比例）

  const TRAIL_COUNT = 16;
  const trailPoints = [];
  for (let i = 0; i < TRAIL_COUNT; i++) {
    const t = i / (TRAIL_COUNT - 1);
    const theta = THETA_START + t * (THETA_END - THETA_START);
    const r = A * Math.exp(B * theta);
    const xRaw = r * Math.cos(theta);
    const yRaw = r * Math.sin(theta);

    const dist = Math.sqrt(xRaw * xRaw + yRaw * yRaw);
    const maxDist = 0.85;
    const scale = dist > maxDist ? maxDist / dist : 1;
    let gx = xRaw * scale;
    let gy = yRaw * scale;

    const rx = gy;
    const ry = -gx;

    trailPoints.push({ x: rx, y: ry });
  }

  // 跳跃高度统一
  function jumpSize(i) {
    return 40; // 所有弧线统一腾空高度
  }

  // Project a normalized point onto the sphere.
  function sphereSurfaceXY(x, y) {
    const sx = cx + x * R;
    const sy = cy + y * R;
    const len = Math.sqrt(x * x + y * y);
    const zOnSphere = Math.sqrt(Math.max(0, 1 - Math.min(len, 0.98) * Math.min(len, 0.98)));
    return { sx, sy, zOnSphere };
  }

  function drawTrail() {
    // Global motion factor.
    const dyn = 1 + 0.04 * Math.sin(tick * 0.05);
    const arcDyn = arcOffset => arcOffset * dyn;

    for (let i = 0; i < trailPoints.length - 1; i++) {
      const a = trailPoints[i];
      const b = trailPoints[i + 1];

      const pa = sphereSurfaceXY(a.x, a.y);
      const pb = sphereSurfaceXY(b.x, b.y);

      const offset = arcDyn(jumpSize(i));

      const q0x = (a.x * 3 + b.x) / 4;
      const q0y = (a.y * 3 + b.y) / 4;
      const q1x = (a.x + b.x * 3) / 4;
      const q1y = (a.y + b.y * 3) / 4;

      const cp0 = sphereSurfaceXY(q0x, q0y);
      const cp1 = sphereSurfaceXY(q1x, q1y);
      const len0 = Math.sqrt(q0x * q0x + q0y * q0y) || 0.01;
      const len1 = Math.sqrt(q1x * q1x + q1y * q1y) || 0.01;
      const cp0sx = cp0.sx + (q0x / len0) * offset;
      const cp0sy = cp0.sy + (q0y / len0) * offset;
      const cp1sx = cp1.sx + (q1x / len1) * offset;
      const cp1sy = cp1.sy + (q1y / len1) * offset;

      // 发光底层
      ctx.beginPath();
      ctx.moveTo(pa.sx, pa.sy);
      ctx.bezierCurveTo(cp0sx, cp0sy, cp1sx, cp1sy, pb.sx, pb.sy);
      ctx.strokeStyle = 'rgba(140, 210, 255, 0.35)';
      ctx.lineWidth = 8;
      ctx.stroke();

      // 轨迹主线
      ctx.beginPath();
      ctx.moveTo(pa.sx, pa.sy);
      ctx.bezierCurveTo(cp0sx, cp0sy, cp1sx, cp1sy, pb.sx, pb.sy);
      ctx.strokeStyle = 'rgba(190, 235, 255, 0.55)';
      ctx.lineWidth = 2.5;
      ctx.stroke();

      // 娌胯矾寰勬粦鍔ㄧ殑鍏夌偣
      const t0 = ((-tick * 0.012 + i * 0.4) % 1 + 1) % 1;
      const t1 = t0;
      const mt = 1 - t1;
      const px = mt*mt*mt*pa.sx + 3*mt*mt*t1*cp0sx + 3*mt*t1*t1*cp1sx + t1*t1*t1*pb.sx;
      const py = mt*mt*mt*pa.sy + 3*mt*mt*t1*cp0sy + 3*mt*t1*t1*cp1sy + t1*t1*t1*pb.sy;
      const glowR = 7;
      const spotGlow = ctx.createRadialGradient(px, py, 0, px, py, glowR);
      spotGlow.addColorStop(0, 'rgba(255, 255, 255, 0.95)');
      spotGlow.addColorStop(0.4, 'rgba(210, 245, 255, 0.6)');
      spotGlow.addColorStop(1, 'rgba(100, 180, 255, 0)');
      ctx.beginPath();
      ctx.arc(px, py, glowR, 0, Math.PI * 2);
      ctx.fillStyle = spotGlow;
      ctx.fill();
    }

    // Draw fixed anchor balls along the route.
    for (let i = 0; i < trailPoints.length; i++) {
      const tp = trailPoints[i];
      const p = sphereSurfaceXY(tp.x, tp.y);
      const pulse = 1 + 0.06 * Math.sin(tick * 0.05 + i * 0.8);
      const baseRad = 5.5 * (1 + p.zOnSphere * 0.4) * pulse;

      // Soft outer glow.
      const glow = ctx.createRadialGradient(p.sx, p.sy, baseRad * 0.3, p.sx, p.sy, baseRad * 2.2);
      glow.addColorStop(0, 'rgba(210, 245, 255, 0.85)');
      glow.addColorStop(0.4, 'rgba(120, 200, 255, 0.4)');
      glow.addColorStop(0.8, 'rgba(40, 120, 220, 0.06)');
      glow.addColorStop(1, 'rgba(30, 90, 200, 0)');
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, baseRad * 2.2, 0, Math.PI * 2);
      ctx.fillStyle = glow;
      ctx.fill();

      // Lit inner core.
      const core = ctx.createRadialGradient(p.sx - baseRad * 0.2, p.sy - baseRad * 0.3, baseRad * 0.05, p.sx, p.sy, baseRad);
      core.addColorStop(0, '#ffffff');
      core.addColorStop(0.35, '#d4edff');
      core.addColorStop(0.75, '#5aa8e8');
      core.addColorStop(1, '#2870b8');
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, baseRad, 0, Math.PI * 2);
      ctx.fillStyle = core;
      ctx.fill();
    }
  }

  // （粒子环已移除）

  // 底部文字（球体下方）
  function drawText() {
    const textY = cy + R + 38;
    ctx.save();
    ctx.font = 'bold 28px "Noto Sans SC", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    ctx.fillStyle = 'rgba(30, 60, 120, 0.25)';
    ctx.fillText('DiffDynamic', cx + 1.5, textY + 1.5);

    const textGrad = ctx.createLinearGradient(cx - 80, textY, cx + 80, textY);
    textGrad.addColorStop(0, '#68b8f0');
    textGrad.addColorStop(0.5, '#b0dfff');
    textGrad.addColorStop(1, '#68b8f0');
    ctx.fillStyle = textGrad;
    ctx.fillText('DiffDynamic', cx, textY);
    ctx.restore();
  }

  function frame() {
    ctx.clearRect(0, 0, W, H);
    tick++;
    drawSphere();
    drawTrail();
    drawText();
    requestAnimationFrame(frame);
  }

  frame();
}

// DiffDynamic orange logo.
function drawLogoOrange() {
  const canvas = document.getElementById('logoCanvas3');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = 170;
  const R = 148;
  let tick = 0;

  function drawSphere() {
    const breathe = 1 + 0.02 * Math.sin(tick * 0.03);
    const grad = ctx.createRadialGradient(cx - 30, cy - 55, R * 0.05, cx, cy, R * breathe);
    grad.addColorStop(0, '#fef9f4');
    grad.addColorStop(0.15, '#fae4ce');
    grad.addColorStop(0.5, '#f0c498');
    grad.addColorStop(0.8, '#e0a068');
    grad.addColorStop(1, '#b87040');

    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();

    const highlight = ctx.createRadialGradient(cx - 50, cy - 70, R * 0.02, cx - 25, cy - 45, R * 0.75);
    highlight.addColorStop(0, 'rgba(255,255,255,0.6)');
    highlight.addColorStop(0.4, 'rgba(255,235,210,0.25)');
    highlight.addColorStop(0.8, 'rgba(230,180,140,0.06)');
    highlight.addColorStop(1, 'rgba(180,110,60,0)');
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.fillStyle = highlight;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(210,160,120,0.4)';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  const PHI = 1.61803398875;
  const B = Math.log(PHI) / (2 * Math.PI);
  const THETA_START = Math.PI * 0.55;
  const THETA_END = THETA_START + 4 * Math.PI;
  const A = 0.30;

  const TRAIL_COUNT = 16;
  const trailPoints = [];
  for (let i = 0; i < TRAIL_COUNT; i++) {
    const t = i / (TRAIL_COUNT - 1);
    const theta = THETA_START + t * (THETA_END - THETA_START);
    const r = A * Math.exp(B * theta);
    const xRaw = r * Math.cos(theta);
    const yRaw = r * Math.sin(theta);
    const dist = Math.sqrt(xRaw * xRaw + yRaw * yRaw);
    const maxDist = 0.85;
    const scale = dist > maxDist ? maxDist / dist : 1;
    let gx = xRaw * scale;
    let gy = yRaw * scale;
    const rx = gy;
    const ry = -gx;
    trailPoints.push({ x: rx, y: ry });
  }

  function jumpSize(i) { return 40; }

  function sphereSurfaceXY(x, y) {
    const sx = cx + x * R;
    const sy = cy + y * R;
    const len = Math.sqrt(x * x + y * y);
    const zOnSphere = Math.sqrt(Math.max(0, 1 - Math.min(len, 0.98) * Math.min(len, 0.98)));
    return { sx, sy, zOnSphere };
  }

  function drawTrail() {
    const dyn = 1 + 0.04 * Math.sin(tick * 0.05);
    const arcDyn = arcOffset => arcOffset * dyn;

    for (let i = 0; i < trailPoints.length - 1; i++) {
      const a = trailPoints[i];
      const b = trailPoints[i + 1];
      const pa = sphereSurfaceXY(a.x, a.y);
      const pb = sphereSurfaceXY(b.x, b.y);
      const offset = arcDyn(jumpSize(i));

      const q0x = (a.x * 3 + b.x) / 4;
      const q0y = (a.y * 3 + b.y) / 4;
      const q1x = (a.x + b.x * 3) / 4;
      const q1y = (a.y + b.y * 3) / 4;

      const cp0 = sphereSurfaceXY(q0x, q0y);
      const cp1 = sphereSurfaceXY(q1x, q1y);
      const len0 = Math.sqrt(q0x * q0x + q0y * q0y) || 0.01;
      const len1 = Math.sqrt(q1x * q1x + q1y * q1y) || 0.01;
      const cp0sx = cp0.sx + (q0x / len0) * offset;
      const cp0sy = cp0.sy + (q0y / len0) * offset;
      const cp1sx = cp1.sx + (q1x / len1) * offset;
      const cp1sy = cp1.sy + (q1y / len1) * offset;

      ctx.beginPath();
      ctx.moveTo(pa.sx, pa.sy);
      ctx.bezierCurveTo(cp0sx, cp0sy, cp1sx, cp1sy, pb.sx, pb.sy);
      ctx.strokeStyle = 'rgba(230, 180, 130, 0.35)';
      ctx.lineWidth = 8;
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(pa.sx, pa.sy);
      ctx.bezierCurveTo(cp0sx, cp0sy, cp1sx, cp1sy, pb.sx, pb.sy);
      ctx.strokeStyle = 'rgba(245, 200, 145, 0.55)';
      ctx.lineWidth = 2.5;
      ctx.stroke();

      const t0 = ((-tick * 0.012 + i * 0.4) % 1 + 1) % 1;
      const t1 = t0;
      const mt = 1 - t1;
      const px = mt*mt*mt*pa.sx + 3*mt*mt*t1*cp0sx + 3*mt*t1*t1*cp1sx + t1*t1*t1*pb.sx;
      const py = mt*mt*mt*pa.sy + 3*mt*mt*t1*cp0sy + 3*mt*t1*t1*cp1sy + t1*t1*t1*pb.sy;
      const glowR = 7;
      const spotGlow = ctx.createRadialGradient(px, py, 0, px, py, glowR);
      spotGlow.addColorStop(0, 'rgba(255, 255, 255, 0.95)');
      spotGlow.addColorStop(0.4, 'rgba(255, 225, 190, 0.6)');
      spotGlow.addColorStop(1, 'rgba(230, 160, 100, 0)');
      ctx.beginPath();
      ctx.arc(px, py, glowR, 0, Math.PI * 2);
      ctx.fillStyle = spotGlow;
      ctx.fill();
    }

    for (let i = 0; i < trailPoints.length; i++) {
      const tp = trailPoints[i];
      const p = sphereSurfaceXY(tp.x, tp.y);
      const pulse = 1 + 0.06 * Math.sin(tick * 0.05 + i * 0.8);
      const baseRad = 5.5 * (1 + p.zOnSphere * 0.4) * pulse;

      const glow = ctx.createRadialGradient(p.sx, p.sy, baseRad * 0.3, p.sx, p.sy, baseRad * 2.2);
      glow.addColorStop(0, 'rgba(255, 235, 210, 0.85)');
      glow.addColorStop(0.4, 'rgba(240, 190, 140, 0.4)');
      glow.addColorStop(0.8, 'rgba(200, 140, 80, 0.06)');
      glow.addColorStop(1, 'rgba(180, 110, 50, 0)');
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, baseRad * 2.2, 0, Math.PI * 2);
      ctx.fillStyle = glow;
      ctx.fill();

      const core = ctx.createRadialGradient(p.sx - baseRad * 0.2, p.sy - baseRad * 0.3, baseRad * 0.05, p.sx, p.sy, baseRad);
      core.addColorStop(0, '#ffffff');
      core.addColorStop(0.35, '#fde8cc');
      core.addColorStop(0.75, '#e8a860');
      core.addColorStop(1, '#bf7030');
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, baseRad, 0, Math.PI * 2);
      ctx.fillStyle = core;
      ctx.fill();
    }
  }

  function drawText() {
    const textY = cy + R + 38;
    ctx.save();
    ctx.font = 'bold 28px "Noto Sans SC", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    ctx.fillStyle = 'rgba(140, 80, 40, 0.25)';
    ctx.fillText('DiffDynamic', cx + 1.5, textY + 1.5);

    const textGrad = ctx.createLinearGradient(cx - 80, textY, cx + 80, textY);
    textGrad.addColorStop(0, '#e8a060');
    textGrad.addColorStop(0.5, '#fcd9b0');
    textGrad.addColorStop(1, '#e8a060');
    ctx.fillStyle = textGrad;
    ctx.fillText('DiffDynamic', cx, textY);
    ctx.restore();
  }

  function frame() {
    ctx.clearRect(0, 0, W, H);
    tick++;
    drawSphere();
    drawTrail();
    drawText();
    requestAnimationFrame(frame);
  }

  frame();
}

// DiffDynamic sunflower logo.
function drawLogo2() {
  const canvas = document.getElementById('logoCanvas2');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = 185;
  const R = 105;
  let tick = 0;

  const PETAL_COUNT = 18;
  const CORE_RATIO = 0.30;

  // 固定花瓣长度（伪随机但确定）
  function pseudoRand(seed) {
    const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453;
    return x - Math.floor(x);
  }
  const petalLengths = [];
  for (let i = 0; i < PETAL_COUNT; i++) {
    petalLengths.push(0.72 + pseudoRand(i * 17 + 53) * 0.14);
  }

  function drawSunflower() {
    // 花瓣
    for (let i = 0; i < PETAL_COUNT; i++) {
      const angle = (i / PETAL_COUNT) * Math.PI * 2 - Math.PI / 2;
      const halfSpread = Math.PI / 12 * 1.15;
      const innerR = R * CORE_RATIO;
      const outerR = R * petalLengths[i];

      const p1x = cx + Math.cos(angle - halfSpread) * innerR;
      const p1y = cy + Math.sin(angle - halfSpread) * innerR;
      const p2x = cx + Math.cos(angle + halfSpread) * innerR;
      const p2y = cy + Math.sin(angle + halfSpread) * innerR;
      const tipX = cx + Math.cos(angle) * outerR;
      const tipY = cy + Math.sin(angle) * outerR;

      const bulge = outerR * 0.88;
      const cpLx = cx + Math.cos(angle - halfSpread * 0.3) * bulge;
      const cpLy = cy + Math.sin(angle - halfSpread * 0.3) * bulge;
      const cpRx = cx + Math.cos(angle + halfSpread * 0.3) * bulge;
      const cpRy = cy + Math.sin(angle + halfSpread * 0.3) * bulge;

      ctx.beginPath();
      ctx.moveTo(p1x, p1y);
      ctx.quadraticCurveTo(cpLx, cpLy, tipX, tipY);
      ctx.quadraticCurveTo(cpRx, cpRy, p2x, p2y);
      ctx.closePath();

      ctx.fillStyle = 'rgba(250, 165, 85, 1.0)';
      ctx.fill();
    }

    // Deeper orange center with radial shading.
    const coreR = R * CORE_RATIO;
    const coreGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreR);
    coreGrad.addColorStop(0, '#fde4c8');
    coreGrad.addColorStop(0.5, '#f2b878');
    coreGrad.addColorStop(1, '#e09650');
    ctx.beginPath();
    ctx.arc(cx, cy, coreR, 0, Math.PI * 2);
    ctx.fillStyle = coreGrad;
    ctx.fill();
  }

  function drawText2() {
    const textY = cy + R + 48;
    ctx.save();
    ctx.font = 'bold 28px "Noto Sans SC", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    ctx.fillStyle = 'rgba(140, 70, 30, 0.25)';
    ctx.fillText('DiffDynamic', cx + 1.5, textY + 1.5);

    const textGrad = ctx.createLinearGradient(cx - 80, textY, cx + 80, textY);
    textGrad.addColorStop(0, '#e8a060');
    textGrad.addColorStop(0.5, '#fcd9b0');
    textGrad.addColorStop(1, '#e8a060');
    ctx.fillStyle = textGrad;
    ctx.fillText('DiffDynamic', cx, textY);
    ctx.restore();
  }

  function frame2() {
    ctx.clearRect(0, 0, W, H);
    tick++;
    drawSunflower();
    drawText2();
    requestAnimationFrame(frame2);
  }

  frame2();
}

// Prudent logo.
function drawLogo4() {
  const canvas = document.getElementById('logoCanvas4');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = 175;
  const R = 105;
  let tick = 0;

  const CORE_RATIO = 0.30;

  // 14 鐡ｏ，98.2% 递减
  const petalLengths = [];
  for (let i = 0; i < 18; i++) {
    petalLengths.push(0.98 * Math.pow(0.982, i));
  }

  function drawStem() {
    const stemTop = cy + R * CORE_RATIO * 0.5;
    const stemBot = cy + R + 48;

    const stemGrad = ctx.createLinearGradient(cx - 4, 0, cx + 8, 0);
    stemGrad.addColorStop(0, '#3d6b2e');
    stemGrad.addColorStop(0.3, '#6b9a48');
    stemGrad.addColorStop(0.5, '#7dad55');
    stemGrad.addColorStop(0.7, '#6b9a48');
    stemGrad.addColorStop(1, '#3d6b2e');

    ctx.beginPath();
    ctx.moveTo(cx + 0, stemTop);
    ctx.quadraticCurveTo(cx + 5, (stemTop + stemBot) / 2, cx + 4, stemBot);
    ctx.lineTo(cx + 2, stemBot);
    ctx.quadraticCurveTo(cx - 1, (stemTop + stemBot) / 2, cx + 0, stemTop);
    ctx.closePath();
    ctx.fillStyle = stemGrad;
    ctx.fill();

    ctx.strokeStyle = 'rgba(40, 70, 20, 0.4)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  function drawLeaves() {
    const stemTop = cy + R * CORE_RATIO * 0.5;

    const leafY1 = stemTop + 85;
    const lx = cx + 0;
    const ly = leafY1;
    const tipLx = cx - 61;
    const tipLy = leafY1 - 30;
    ctx.beginPath();
    ctx.moveTo(lx, ly);
    ctx.quadraticCurveTo(cx - 36, leafY1 - 42, tipLx, tipLy);
    ctx.quadraticCurveTo(cx - 34, leafY1 + 10, lx, ly + 8);
    ctx.closePath();
    ctx.fillStyle = 'rgba(245, 190, 195, 0.75)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(200, 135, 145, 0.5)';
    ctx.lineWidth = 1.2;
    ctx.stroke();
    // 叶脉
    ctx.beginPath();
    ctx.moveTo(lx, ly + 4);
    ctx.lineTo(tipLx + 6, tipLy + 4);
    ctx.strokeStyle = 'rgba(210, 150, 155, 0.5)';
    ctx.lineWidth = 0.8;
    ctx.stroke();

    // Right pale-blue leaf.
    const leafY2 = stemTop + 92;
    const rlx = cx + 3;
    const rly = leafY2;
    const tipRx = cx + 55;
    const tipRy = leafY2 - 26;
    ctx.beginPath();
    ctx.moveTo(rlx, rly);
    ctx.quadraticCurveTo(cx + 32, leafY2 - 38, tipRx, tipRy);
    ctx.quadraticCurveTo(cx + 32, leafY2 + 10, rlx, rly + 8);
    ctx.closePath();
    ctx.fillStyle = 'rgba(175, 210, 235, 0.75)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(130, 175, 210, 0.5)';
    ctx.lineWidth = 1.2;
    ctx.stroke();
    // 叶脉
    ctx.beginPath();
    ctx.moveTo(rlx, rly + 4);
    ctx.lineTo(tipRx - 6, tipRy + 4);
    ctx.strokeStyle = 'rgba(155, 195, 225, 0.5)';
    ctx.lineWidth = 0.8;
    ctx.stroke();
  }

  function drawSunflower() {
    drawLeaves();
    drawStem();

    for (let i = 0; i < 18; i++) {
      const angle = (i / 18) * Math.PI * 2 - Math.PI * 1.25;
      const halfSpread = Math.PI / 12 * 1.15;
      const innerR = R * CORE_RATIO;
      const outerR = R * petalLengths[i];

      const p1x = cx + Math.cos(angle - halfSpread) * innerR;
      const p1y = cy + Math.sin(angle - halfSpread) * innerR;
      const p2x = cx + Math.cos(angle + halfSpread) * innerR;
      const p2y = cy + Math.sin(angle + halfSpread) * innerR;
      const tipX = cx + Math.cos(angle) * outerR;
      const tipY = cy + Math.sin(angle) * outerR;

      const bulge = outerR * 0.88;
      const cpLx = cx + Math.cos(angle - halfSpread * 0.3) * bulge;
      const cpLy = cy + Math.sin(angle - halfSpread * 0.3) * bulge;
      const cpRx = cx + Math.cos(angle + halfSpread * 0.3) * bulge;
      const cpRy = cy + Math.sin(angle + halfSpread * 0.3) * bulge;

      ctx.beginPath();
      ctx.moveTo(p1x, p1y);
      ctx.quadraticCurveTo(cpLx, cpLy, tipX, tipY);
      ctx.quadraticCurveTo(cpRx, cpRy, p2x, p2y);
      ctx.closePath();

      ctx.fillStyle = 'rgba(250, 165, 85, 1.0)';
      ctx.fill();
    }

    const coreR = R * CORE_RATIO;
    const coreGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreR);
    coreGrad.addColorStop(0, '#fde4c8');
    coreGrad.addColorStop(0.5, '#f2b878');
    coreGrad.addColorStop(1, '#e09650');
    ctx.beginPath();
    ctx.arc(cx, cy, coreR, 0, Math.PI * 2);
    ctx.fillStyle = coreGrad;
    ctx.fill();
  }

  function drawText4() {
    const textY = cy + R + 70;
    ctx.save();
    ctx.font = 'bold 28px "Noto Sans SC", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    ctx.fillStyle = 'rgba(140, 70, 30, 0.25)';
    ctx.fillText('DiffDynamic', cx + 1.5, textY + 1.5);

    const textGrad = ctx.createLinearGradient(cx - 80, textY, cx + 80, textY);
    textGrad.addColorStop(0, '#e8a060');
    textGrad.addColorStop(0.5, '#fcd9b0');
    textGrad.addColorStop(1, '#e8a060');
    ctx.fillStyle = textGrad;
    ctx.fillText('DiffDynamic', cx, textY);
    ctx.restore();
  }

  function frame4() {
    ctx.clearRect(0, 0, W, H);
    tick++;
    drawSunflower();
    drawText4();
    requestAnimationFrame(frame4);
  }

  frame4();
}


export {
  drawLogo,
  drawLogo2,
  drawLogo4,
  drawLogoOrange
};
