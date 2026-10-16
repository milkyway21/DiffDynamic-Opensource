import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { DD_GREEN_BALL_COUNT, DD_GREEN_GAP_RHO, END, N, PRUDENT_STAGE_GREEN_BALL_COUNTS, PRUDENT_TERMINAL_BALL_COUNT, PRUDENT_TERMINAL_START, SIZE, START, VIEW_BG_HEX, VIEW_PALETTE, addGroundDiskMarker, height, makeContours, pathDiffDynamic, pathGlintDM, pathPrudentTree, pathTargetDiff, samplePolylineDecreasingGaps, softenColor, terrainSurfaceRgb } from './diffusion-paths.js';

// ╔═══════════════ SCENE BUILDER ═══════════════╗/ CRITICAL: terrain uses XY plane, Z=height. Path points map directly.
// orangeSteps > 0: split last N path points into orange tube (Transformer repair)
function buildScene(containerId, path, color, orangeSteps = 0) {
  try {
  const tree = !Array.isArray(path) && Array.isArray(path?.segments) ? path : null;
  const primaryPath = tree ? tree.segments[0] : path;
  const container = document.getElementById(containerId);
  if (!container) { console.warn('[DIAG] Container not found:', containerId); return; }
  console.log('[DIAG] buildScene:', containerId, 'primaryPath length:', primaryPath.length, 'isTree:', !!tree);
  // Clear previous canvas if any
  while (container.firstChild) container.removeChild(container.firstChild);

  const cw = container.clientWidth || 600;
  const ch = container.clientHeight || 600;
  const side = Math.min(cw, ch) || 600;
  const w = side;
  const h = side;

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(w, h, false);
  renderer.setClearColor(VIEW_BG_HEX, 1);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(VIEW_BG_HEX);

  const cam = new THREE.PerspectiveCamera(42, w/h, 0.5, 50);
  // Camera position: Z轴垂直朝上，XY平面水平，视角稍远
  cam.position.set(0, 0, 18);
  cam.up.set(0, 0, 1);
  cam.lookAt(0, 0, 1.5);

  const ctrl = new OrbitControls(cam, renderer.domElement);
  ctrl.target.set(0, 0, 1.5);
  ctrl.enableDamping = true; ctrl.dampingFactor = 0.08;
  ctrl.minDistance = 4; ctrl.maxDistance = 25;
  ctrl.update();

  // Lighting
  const amb = new THREE.AmbientLight('#f5f7fa', 2.0);
  scene.add(amb);
  const key = new THREE.DirectionalLight('#ffffff', 2.2);
  key.position.set(10, -6, 14);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  key.shadow.camera.left=-14; key.shadow.camera.right=14;
  key.shadow.camera.top=14; key.shadow.camera.bottom=-14;
  key.shadow.bias=-0.00015;
  scene.add(key);
  const fill = new THREE.DirectionalLight('#d0dce8', 1.2);
  fill.position.set(-8, 8, 5); scene.add(fill);

  // ---- Terrain (XY plane, Z=height) ----
  const tGeo = new THREE.PlaneGeometry(SIZE, SIZE, N, N);
  // PlaneGeometry default: XY plane. Z is height. Perfect.
  const tPos = tGeo.attributes.position;
  const col = new Float32Array(tPos.count * 3);
  let minH = Infinity, maxH = -Infinity;
  const hsStore = [];
  for (let i = 0; i < tPos.count; i++) {
    const x = tPos.getX(i), y = tPos.getY(i);
    const h = height(x, y);
    tPos.setZ(i, h);
    hsStore.push(h);
    if (h < minH) minH = h; if (h > maxH) maxH = h;
  }
  const hRange = maxH - minH || 1;
  for (let i = 0; i < tPos.count; i++) {
    const t = (hsStore[i] - minH) / hRange;
    const rgb = terrainSurfaceRgb(t);
    col[i*3]=rgb.r; col[i*3+1]=rgb.g; col[i*3+2]=rgb.b;
  }
  tGeo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  tGeo.computeVertexNormals();
  const tMat = new THREE.MeshStandardMaterial({
    vertexColors: true, roughness: 0.55, metalness: 0.03,
    flatShading: true, side: THREE.DoubleSide
  });
  const terrain = new THREE.Mesh(tGeo, tMat);
  terrain.castShadow = true; terrain.receiveShadow = true;
  scene.add(terrain);

  // ---- Contours (very light) ----
  const contourData = makeContours([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
  contourData.forEach(({level, segs}) => {
    segs.forEach(seg => {
      if (seg.length < 2) return;
      const pts = seg.map(p => {
        const z = height(p.x, p.y) + 0.06;
        return new THREE.Vector3(p.x, p.y, z);
      });
      const lg = new THREE.BufferGeometry().setFromPoints(pts);
      const lm = new THREE.LineBasicMaterial({
        color: softenColor('#9aabbf', 0.93), transparent: true, opacity: 0.32, depthTest: true
      });
      scene.add(new THREE.Line(lg, lm));
    });
  });

  // ---- Path tube ----
  // Map path {x,y,z} -> THREE.Vector3(x, y, z) (XY plane, Z=height)
  const curvePts = primaryPath.map(p => new THREE.Vector3(p.x, p.y, p.z + 0.12));
  if (curvePts.length >= 2) {
    const tubeRadius = tree ? 0.075 : 0.09;
    if (tree) {
      const treeBranchColor = (segmentIndex) => {
        if (segmentIndex === 0) return VIEW_PALETTE.prudentWinnerStageOne;
        if (segmentIndex === tree.coreStageTwoSegmentIndex) return VIEW_PALETTE.prudentWinnerStageTwo;
        if (segmentIndex <= 3) return VIEW_PALETTE.prudentLevelOne;
        if (segmentIndex === tree.bestSegmentIndex) return VIEW_PALETTE.prudentBest;
        return VIEW_PALETTE.prudentLevelTwo;
      };
      const addTreeTube = (
        points,
        tubeColor,
        emissiveColor,
        emissiveIntensity,
        radius = tubeRadius,
        onTop = false
      ) => {
        if (points.length < 2) return;
        const curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.5);
        const geometry = new THREE.TubeGeometry(curve, Math.max(8, points.length), radius, 7, false);
        const material = new THREE.MeshStandardMaterial({
          color: tubeColor,
          roughness: 0.34,
          metalness: 0.08,
          emissive: emissiveColor,
          emissiveIntensity,
          depthTest: !onTop,
          depthWrite: !onTop
        });
        const tube = new THREE.Mesh(geometry, material);
        tube.castShadow = true;
        tube.renderOrder = onTop ? 20 : 3;
        scene.add(tube);
      };
      tree.segments.forEach((segment, segmentIndex) => {
        const points = segment.map(p => new THREE.Vector3(p.x, p.y, p.z + 0.12));
        if (points.length < 2) return;
        const branchColor = treeBranchColor(segmentIndex);
        const onTop = tree.topSegmentIndices.includes(segmentIndex);
        const branchEmissive = onTop ? VIEW_PALETTE.prudentBestEmissive : branchColor;
        if (segmentIndex >= 4) {
          const terminalStart = Math.max(2, Math.floor(points.length * PRUDENT_TERMINAL_START));
          addTreeTube(points.slice(0, terminalStart + 1), branchColor, branchEmissive, 0.22, tubeRadius, onTop);
          const terminalColor = segmentIndex === tree.bestSegmentIndex
            ? VIEW_PALETTE.prudentBest
            : VIEW_PALETTE.prudentTerminal;
          const terminalEmissive = segmentIndex === tree.bestSegmentIndex
            ? VIEW_PALETTE.prudentBestEmissive
            : VIEW_PALETTE.prudentTerminalEmissive;
          addTreeTube(
            points.slice(terminalStart),
            terminalColor,
            terminalEmissive,
            0.34,
            tubeRadius + 0.01,
            onTop
          );
        } else {
          addTreeTube(
            points,
            branchColor,
            branchEmissive,
            segmentIndex === 0 ? 0.16 : 0.22,
            tubeRadius,
            onTop
          );
        }
      });
    } else if (orangeSteps > 0 && orangeSteps < primaryPath.length) {
      // Split: main path + last orangeSteps as orange tube (Transformer repair)
      const splitIdx = curvePts.length - orangeSteps - 1;
      // Main tube (green): ends at last green vertex
      if (splitIdx >= 1) {
        const mainPts = curvePts.slice(0, splitIdx);
        const mainCurve = new THREE.CatmullRomCurve3(mainPts, false, 'catmullrom', 0.5);
        const mainSegments = Math.max(8, Math.floor(mainPts.length * 1.2));
        const mainGeo = new THREE.TubeGeometry(mainCurve, mainSegments, tubeRadius, 6, false);
        const mainMat = new THREE.MeshStandardMaterial({
          color, roughness: 0.35, metalness: 0.08, emissive: color, emissiveIntensity: 0.14, depthTest: true
        });
        const mainTube = new THREE.Mesh(mainGeo, mainMat);
        mainTube.castShadow = true; mainTube.renderOrder = 3;
        scene.add(mainTube);
      }
      // Orange tube: duplicate last green at start so TubeGeometry meets green tube with no seam gap
      const orangePts = splitIdx >= 1
        ? [curvePts[splitIdx - 1]].concat(curvePts.slice(splitIdx))
        : curvePts.slice(splitIdx);
      const orangeCurve = new THREE.CatmullRomCurve3(orangePts, false, 'catmullrom', 0.5);
      const orangeSegments = Math.max(8, Math.floor(orangePts.length * 2));
      const orangeGeo = new THREE.TubeGeometry(orangeCurve, orangeSegments, tubeRadius + 0.02, 8, false);
      const orangeMat = new THREE.MeshStandardMaterial({
        color: VIEW_PALETTE.repairOrange,
        roughness: 0.32,
        metalness: 0.1,
        emissive: VIEW_PALETTE.repairEmissive,
        emissiveIntensity: 0.42,
        depthTest: true
      });
      const orangeTube = new THREE.Mesh(orangeGeo, orangeMat);
      orangeTube.castShadow = true; orangeTube.renderOrder = 4;
      scene.add(orangeTube);
    } else {
      const curve = new THREE.CatmullRomCurve3(curvePts, false, 'catmullrom', 0.5);
      const segments = Math.max(2, Math.floor(primaryPath.length * 0.8));
      const tubeGeo = new THREE.TubeGeometry(curve, segments, tubeRadius, 6, false);
      const tubeMat = new THREE.MeshStandardMaterial({
        color, roughness: 0.35, metalness: 0.08, emissive: color, emissiveIntensity: 0.14, depthTest: true
      });
      const tube = new THREE.Mesh(tubeGeo, tubeMat);
      tube.castShadow = true; tube.renderOrder = 3;
      scene.add(tube);
    }
  }

  // ---- Sampled marker balls ----
  const lastIdx = primaryPath.length - 1;
  const orangeStartIdx = orangeSteps > 0 ? primaryPath.length - orangeSteps - 1 : lastIdx;

  function addSphereMarkerAt(
    p,
    isOrange,
    radius = 0.15,
    colorOverride = null,
    emissiveOverride = null,
    onTop = false,
    highDetail = false,
    colorSatMul = 0.92
  ) {
    const baseCol = colorOverride || (isOrange ? VIEW_PALETTE.repairOrange : color);
    const baseEmit = emissiveOverride || (isOrange ? VIEW_PALETTE.repairEmissive : baseCol);
    const ballColor = softenColor(baseCol, colorSatMul);
    const emissiveColor = softenColor(baseEmit, colorSatMul);
    const emissiveIntensity = highDetail ? 0.1 : (isOrange ? 0.46 : 0.26);
    const sg = new THREE.SphereGeometry(radius, highDetail ? 28 : 10, highDetail ? 18 : 8);
    const sm = new THREE.MeshStandardMaterial({
      color: ballColor,
      roughness: highDetail ? 0.16 : 0.25,
      metalness: highDetail ? 0.22 : 0.15,
      emissive: emissiveColor, emissiveIntensity,
      depthTest: !onTop,
      depthWrite: !onTop
    });
    const sphere = new THREE.Mesh(sg, sm);
    sphere.position.set(p.x, p.y, p.z + radius);
    sphere.castShadow = true;
    sphere.receiveShadow = highDetail;
    sphere.renderOrder = onTop ? 22 : 5;
    scene.add(sphere);
  }

  if (tree) {
    tree.segments.forEach((segment, segmentIndex) => {
      if (segmentIndex >= 4) {
        const onTop = tree.topSegmentIndices.includes(segmentIndex);
        const terminalStart = Math.max(2, Math.floor(segment.length * PRUDENT_TERMINAL_START));
        const branchPart = segment.slice(0, terminalStart + 1);
        const branchColor = segmentIndex === tree.bestSegmentIndex
          ? VIEW_PALETTE.prudentBest
          : VIEW_PALETTE.prudentLevelTwo;
        const branchEmissive = segmentIndex === tree.bestSegmentIndex
          ? VIEW_PALETTE.prudentBestEmissive
          : VIEW_PALETTE.prudentLevelTwo;
        const branchSamples = samplePolylineDecreasingGaps(
          branchPart,
          PRUDENT_STAGE_GREEN_BALL_COUNTS[2],
          DD_GREEN_GAP_RHO
        );
        branchSamples.forEach((p, sampleIndex) => {
          if (sampleIndex === 0 || sampleIndex === branchSamples.length - 1) return;
          addSphereMarkerAt(p, false, 0.11, branchColor, branchEmissive, onTop, false, onTop ? 1 : 0.92);
        });

        // Match DiffDynamic: each leaf shows four compact terminal sample balls.
        const terminalPart = segment.slice(terminalStart);
        const terminalSamples = samplePolylineDecreasingGaps(
          terminalPart,
          PRUDENT_TERMINAL_BALL_COUNT + 1,
          DD_GREEN_GAP_RHO
        ).slice(0, PRUDENT_TERMINAL_BALL_COUNT);
        const terminalMarkerColor = segmentIndex === tree.bestSegmentIndex
          ? VIEW_PALETTE.prudentBest
          : VIEW_PALETTE.prudentTerminal;
        const terminalMarkerEmissive = segmentIndex === tree.bestSegmentIndex
          ? VIEW_PALETTE.prudentBestEmissive
          : VIEW_PALETTE.prudentTerminalEmissive;
        terminalSamples.forEach((p) => addSphereMarkerAt(
          p,
          false,
          0.12,
          terminalMarkerColor,
          terminalMarkerEmissive,
          onTop,
          false,
          onTop ? 1 : 0.92
        ));
        return;
      }

      const stageIndex = segmentIndex === 0 ? 0 : 1;
      const sampleCount = PRUDENT_STAGE_GREEN_BALL_COUNTS[stageIndex];
      const samples = samplePolylineDecreasingGaps(segment, sampleCount, DD_GREEN_GAP_RHO);
      const markerColor = segmentIndex === 0
        ? VIEW_PALETTE.prudentWinnerStageOne
        : (segmentIndex === tree.coreStageTwoSegmentIndex
          ? VIEW_PALETTE.prudentWinnerStageTwo
          : VIEW_PALETTE.prudentLevelOne);
      const onTop = tree.topSegmentIndices.includes(segmentIndex);
      samples.forEach((p, sampleIndex) => {
        if (segmentIndex > 0 && sampleIndex === 0) return;
        if (sampleIndex === samples.length - 1) return;
        addSphereMarkerAt(
          p,
          false,
          0.11,
          markerColor,
          onTop ? VIEW_PALETTE.prudentBestEmissive : markerColor,
          onTop,
          false,
          onTop ? 1 : 0.92
        );
      });
    });
    tree.branchPoints.forEach((p) => {
      addSphereMarkerAt(
        p,
        false,
        0.16,
        VIEW_PALETTE.prudentBranchDeep,
        VIEW_PALETTE.prudentBranchDeep,
        true,
        true
      );
    });
  } else if (orangeSteps > 0 && orangeSteps < primaryPath.length) {
    const splitIdx = orangeStartIdx;
    const greenPoly = primaryPath.slice(0, splitIdx);
    const greenSamples = samplePolylineDecreasingGaps(greenPoly, DD_GREEN_BALL_COUNT, DD_GREEN_GAP_RHO);
    greenSamples.forEach((p) => addSphereMarkerAt(p, false));
    for (let i = splitIdx; i <= lastIdx; i++) addSphereMarkerAt(primaryPath[i], true);
  } else {
    const stride = Math.max(1, Math.floor(primaryPath.length / 50));
    for (let idx = 0; idx <= lastIdx; idx++) {
      if (idx % stride !== 0 && idx !== lastIdx) continue;
      addSphereMarkerAt(primaryPath[idx], false);
    }
  }

  // ---- Start & End markers（平面圆盘 + 圆环，贴 XY，无光照立体感） ----
  if (tree) {
    // Start uses winner-path color (pale red) for visual consistency
    addGroundDiskMarker(
      scene,
      START.x,
      START.y,
      START.z,
      softenColor(VIEW_PALETTE.prudentBest),
      softenColor(VIEW_PALETTE.prudentBestEmissive)
    );
  } else {
    addGroundDiskMarker(
      scene,
      START.x,
      START.y,
      START.z,
      softenColor(VIEW_PALETTE.startRose),
      softenColor(VIEW_PALETTE.startEmissive)
    );
  }
  const endpoints = tree ? tree.endpoints : [END];
  endpoints.forEach((endpoint, endpointIndex) => {
    const isBestEndpoint = tree && endpointIndex === tree.bestEndpointIndex;
    if (tree && !isBestEndpoint) return;
    addGroundDiskMarker(
      scene,
      endpoint.x,
      endpoint.y,
      endpoint.z,
      softenColor(tree ? VIEW_PALETTE.prudentBest : VIEW_PALETTE.startRose),
      softenColor(tree ? VIEW_PALETTE.endEmissive : VIEW_PALETTE.endEmissive),
      1,
      true,
      !tree
    );
    if (tree) {
      addSphereMarkerAt(
        endpoint,
        false,
        0.19,
        VIEW_PALETTE.prudentBest,
        VIEW_PALETTE.prudentBestEmissive,
        true,
        true,
        1
      );
    }
  });
  container.dataset.pathSegmentCount = String(tree ? tree.segments.length : 1);
  container.dataset.endpointCount = String(endpoints.length);
  if (tree) {
    container.dataset.stageRatios = tree.stageRatios.join('/');
    container.dataset.greenPointCounts = PRUDENT_STAGE_GREEN_BALL_COUNTS.join('/');
    container.dataset.terminalBallCount = String(PRUDENT_TERMINAL_BALL_COUNT);
    container.dataset.terminalFraction = String(1 - PRUDENT_TERMINAL_START);
    container.dataset.corePathMode = tree.corePathMode;
    container.dataset.endpointRingCount = '1';
    container.dataset.endpointFillCount = '0';
    container.dataset.branchNodeCount = String(tree.branchPoints.length);
    container.dataset.redEndpointCount = '1';
    container.dataset.winningPathTopLayer = 'true';
    container.dataset.highDetailNodeCount = String(tree.branchPoints.length + 1);
    container.dataset.startMarkerRadius = '0.11';
    container.dataset.startGroundMarker = 'ring-and-disk';
    container.dataset.endpointSpreadScale = String(tree.endpointSpreadScale);
  }

  // ---- Render loop ----
  function resize() {
    const cw = container.clientWidth, ch = container.clientHeight;
    if (cw===0||ch===0) return;
    const side = Math.min(cw, ch);
    renderer.setSize(side, side, false);
    cam.aspect = 1;
    cam.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);

  function animate() {
    requestAnimationFrame(animate);
    ctrl.update();
    renderer.render(scene, cam);
  }
  animate();
  } catch(e) {
    console.error('[DIAG] buildScene crashed for', containerId, ':', e.message, e.stack);
    if (diag) diag.textContent += 'buildScene(' + containerId + ') CRASH: ' + e.message + '\n';
  }
}

// Boot all canvases.
function initAll() {
  try {
  if (diag) diag.textContent += 'initAll() started\n';
  console.log('[DIAG] initAll() started');
  // 1) TargetDiff
  const td = pathTargetDiff(1000);
  buildScene('wrap1', td, VIEW_PALETTE.pathBlue);
  if (diag) diag.textContent += 'TargetDiff done\n';

  // 2) DiffDynamic: ~37 points (bridge 5 + 28 green + 3 orange + END)
  const dd = pathDiffDynamic(28, 3);
  buildScene('wrap2', dd, VIEW_PALETTE.pathGreen, 3);

  // 3) GlintDM
  const gd = pathGlintDM(50);
  buildScene('wrap3', gd, VIEW_PALETTE.pathPurple);

  // 4) Prudent: 1 → 3 → 9 tree search; the center leaf is the original END.
  const prudent = pathPrudentTree();
  buildScene('wrap4', prudent, VIEW_PALETTE.pathPrudent);
  if (diag) diag.textContent += 'Prudent done\n';
  if (diag) diag.textContent += 'All 4 scenes built OK!\n';
  } catch(e) {
    if (diag) diag.textContent += 'CRASH in initAll: ' + e.message + '\n' + e.stack + '\n';
    console.error('[DIAG] initAll crashed:', e);
  }
}


export {
  initAll
};
