import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PDB_DATA, REF_LIGAND_SDF, SDF_TRAJECTORY } from './molecular-data.js';
import { LIGAND_SAMPLING_STEPS, computeBonds, computeCentroid, getElementRadius, getLigandSamplingPhase, getLigandSamplingStep, parsePDB, parseSDFModel, parseSDFTrajectory, resampleTrajectoryFrames, stabilizeTrajectoryAtomIdentity, translateAtoms } from './molecule-parser.js';
import { createProteinSketchGroup, disposeObject3D, disposeTrajectoryPlayer, setProteinMaterialOpacity, updateProteinWaveColors } from './protein-renderer.js';
import { PROTEIN_LIGHT_BLUE, PROTEIN_LIGHT_BLUE_DEEP, smoothstep01 } from './protein-geometry.js';
import { createBallStickGroup, isolateGroupMaterials } from './molecule-renderer.js';
import { MoleculeAnimator, getHoldBreathScale } from './molecule-animator.js';

// ======= HERO SCENE =======

const HERO_TRAJECTORY_ASSETS = [
  { id: '004', url: new URL('../../data/hero-trajectories/004_trajectory_pymol.sdf', import.meta.url) },
  { id: '090', url: new URL('../../data/hero-trajectories/090_trajectory_pymol.sdf', import.meta.url) },
  { id: '097', url: new URL('../../data/hero-trajectories/097_trajectory_pymol.sdf', import.meta.url) },
];

async function loadHeroTrajectorySources() {
  const additionalSources = await Promise.all(HERO_TRAJECTORY_ASSETS.map(async source => {
    const response = await fetch(source.url);
    if (!response.ok) throw new Error(`Unable to load trajectory ${source.id}: HTTP ${response.status}`);
    return { id: source.id, sdf: await response.text() };
  }));
  return [{ id: 'default', sdf: SDF_TRAJECTORY }, ...additionalSources];
}

function pickNextTrajectoryIndex(currentIndex, trajectoryCount, random = Math.random) {
  if (trajectoryCount <= 1) return 0;
  const candidate = Math.floor(random() * (trajectoryCount - 1));
  return candidate >= currentIndex ? candidate + 1 : candidate;
}

function getWaterMoleculeSpriteMetrics() {
  const canvasSize = 96;
  const carbonPixelRadius = 16;
  const oxygenPixelRadius = carbonPixelRadius * (getElementRadius('O') / getElementRadius('C'));
  const hydrogenPixelRadius = carbonPixelRadius * (getElementRadius('H') / getElementRadius('C'));
  const oxygenCenter = { x: 48, y: 30 };
  const bondLength = 32;
  const halfBondAngle = THREE.MathUtils.degToRad(104.5 * 0.5);
  const horizontalOffset = Math.sin(halfBondAngle) * bondLength;
  const verticalOffset = Math.cos(halfBondAngle) * bondLength;
  return {
    canvasSize,
    carbonPixelRadius,
    oxygenPixelRadius,
    hydrogenPixelRadius,
    oxygenCenter,
    leftHydrogenCenter: {
      x: oxygenCenter.x - horizontalOffset,
      y: oxygenCenter.y + verticalOffset,
    },
    rightHydrogenCenter: {
      x: oxygenCenter.x + horizontalOffset,
      y: oxygenCenter.y + verticalOffset,
    },
  };
}

function createWaterMoleculeSpriteTexture(metrics = getWaterMoleculeSpriteMetrics()) {
  const spriteCanvas = document.createElement('canvas');
  spriteCanvas.width = metrics.canvasSize;
  spriteCanvas.height = metrics.canvasSize;
  const context = spriteCanvas.getContext('2d');

  // Draw both O-H bonds first so the shaded spheres sit cleanly on top.
  context.lineCap = 'round';
  context.lineWidth = metrics.hydrogenPixelRadius * 0.56;
  context.strokeStyle = 'rgba(178, 178, 178, 0.78)';
  context.beginPath();
  context.moveTo(metrics.leftHydrogenCenter.x, metrics.leftHydrogenCenter.y);
  context.lineTo(48, 30);
  context.moveTo(metrics.rightHydrogenCenter.x, metrics.rightHydrogenCenter.y);
  context.lineTo(48, 30);
  context.stroke();

  // Bent H–O–H silhouette. A shared radial highlight gives all three circles
  // spherical volume while the PointsMaterial keeps O and H the same grey.
  for (const circle of [
    { ...metrics.oxygenCenter, radius: metrics.oxygenPixelRadius },
    { ...metrics.leftHydrogenCenter, radius: metrics.hydrogenPixelRadius },
    { ...metrics.rightHydrogenCenter, radius: metrics.hydrogenPixelRadius },
  ]) {
    const highlight = context.createRadialGradient(
      circle.x - circle.radius * 0.34,
      circle.y - circle.radius * 0.38,
      circle.radius * 0.08,
      circle.x,
      circle.y,
      circle.radius
    );
    highlight.addColorStop(0, 'rgba(240, 240, 240, 0.96)');
    highlight.addColorStop(0.42, 'rgba(222, 222, 222, 0.94)');
    highlight.addColorStop(0.82, 'rgba(196, 196, 196, 0.82)');
    highlight.addColorStop(1, 'rgba(178, 178, 178, 0.04)');
    context.fillStyle = highlight;
    context.beginPath();
    context.arc(circle.x, circle.y, circle.radius, 0, Math.PI * 2);
    context.fill();
  }

  const texture = new THREE.CanvasTexture(spriteCanvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.needsUpdate = true;
  return texture;
}

function buildHeroScene(canvasId, trajectorySources = [{ id: 'default', sdf: SDF_TRAJECTORY }]) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) { console.warn('Hero canvas not found'); return null; }
  const heroPhaseLabel = document.getElementById('heroPhaseLabel');
  const heroPhaseIndex = document.getElementById('heroPhaseIndex');
  const heroPhaseText = document.getElementById('heroPhaseText');
  const ligandPhaseLabels = Object.freeze({
    gaussian: { index: '01', text: '放入呈高斯离散态的原子' },
    denoise: { index: '02', text: '进行采样生成分子 · 33 步' },
    repair: { index: '03', text: '进行结构修复 · 10 步' },
    hold: { index: '04', text: '分子生成完成' },
  });
  const setHeroPhaseLabel = phase => {
    if (!heroPhaseLabel || !heroPhaseIndex || !heroPhaseText) return;
    const visiblePhase = phase === 'fade-in' ? 'gaussian' : phase;
    const copy = ligandPhaseLabels[visiblePhase];
    if (!copy) {
      heroPhaseLabel.classList.remove('is-visible');
      heroPhaseLabel.dataset.phase = 'waiting';
      return;
    }
    if (heroPhaseLabel.dataset.phase !== visiblePhase) {
      heroPhaseIndex.textContent = copy.index;
      heroPhaseText.textContent = copy.text;
      heroPhaseLabel.dataset.phase = visiblePhase;
    }
    heroPhaseLabel.classList.add('is-visible');
  };

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);
  // White depth haze preserves opacity while separating near and far content.
  scene.fog = new THREE.FogExp2(0xffffff, 0.0065);

  // Camera
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 500);

  // Renderer
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.95;

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Controls
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.autoRotate = false;
  controls.enableRotate = true;
  controls.enableZoom = true;
  controls.enablePan = false;
  controls.minDistance = 20;
  controls.maxDistance = 100;

  // Soft white-theme lighting keeps the broad cartoon readable without a
  // plastic toon edge or a blown-out far side.
  const hemisphere = new THREE.HemisphereLight(0xf7fbff, 0xc7d7e4, 0.72);
  scene.add(hemisphere);

  const keyLight = new THREE.DirectionalLight(0xffffff, 1.15);
  keyLight.position.set(15, 25, 30);
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0xbcdcf2, 0.32);
  fillLight.position.set(-18, 8, 16);
  scene.add(fillLight);

  const rimLight = new THREE.DirectionalLight(0x9fc7e4, 0.24);
  rimLight.position.set(-15, -5, -20);
  scene.add(rimLight);

  // Each point is a physically scaled H2O sprite. Per-molecule drift, spin and
  // lifetime stay on the GPU, keeping 960 molecules to one draw call.
  const WATER_VISIBLE_DURATION = 10.0;
  const WATER_FADE_DURATION = 2.0;
  const WATER_HIDDEN_DURATION = 1.0;
  const waterSpriteMetrics = getWaterMoleculeSpriteMetrics();
  const waterSpriteWorldSize = getElementRadius('C') * (
    waterSpriteMetrics.canvasSize / waterSpriteMetrics.carbonPixelRadius
  );
  const waterMoleculeCount = 280;
  const particleGeo = new THREE.BufferGeometry();
  const positions = new Float32Array(waterMoleculeCount * 3);
  const waterSeeds = new Float32Array(waterMoleculeCount);
  const waterPhases = new Float32Array(waterMoleculeCount);
  const waterSpins = new Float32Array(waterMoleculeCount);
  const waterSizes = new Float32Array(waterMoleculeCount);
  for (let i = 0; i < waterMoleculeCount; i++) {
    positions[i*3] = (Math.random() - 0.5) * 100;
    positions[i*3+1] = (Math.random() - 0.5) * 60;
    positions[i*3+2] = (Math.random() - 0.5) * 100;
    waterSeeds[i] = Math.random() * 1000;
    waterPhases[i] = Math.random() * (
      WATER_VISIBLE_DURATION + WATER_FADE_DURATION + WATER_HIDDEN_DURATION
    );
    waterSpins[i] = (Math.random() < 0.5 ? -1 : 1) * (0.16 + Math.random() * 0.34);
    waterSizes[i] = waterSpriteWorldSize;
  }
  particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  particleGeo.setAttribute('aSeed', new THREE.BufferAttribute(waterSeeds, 1));
  particleGeo.setAttribute('aPhase', new THREE.BufferAttribute(waterPhases, 1));
  particleGeo.setAttribute('aSpin', new THREE.BufferAttribute(waterSpins, 1));
  particleGeo.setAttribute('aSize', new THREE.BufferAttribute(waterSizes, 1));

  const waterMoleculeTexture = createWaterMoleculeSpriteTexture(waterSpriteMetrics);
  const particleMat = new THREE.ShaderMaterial({
    uniforms: {
      uSprite: { value: waterMoleculeTexture },
      uTime: { value: 0 },
      uMotion: { value: reducedMotion ? 0 : 1 },
      uPixelRatio: { value: renderer.getPixelRatio() },
      uViewportHeight: { value: Math.max(1, canvas.clientHeight) },
      uVisibleDuration: { value: WATER_VISIBLE_DURATION },
      uFadeDuration: { value: WATER_FADE_DURATION },
      uCycleDuration: {
        value: WATER_VISIBLE_DURATION + WATER_FADE_DURATION + WATER_HIDDEN_DURATION,
      },
    },
    vertexShader: `
      uniform float uTime;
      uniform float uMotion;
      uniform float uPixelRatio;
      uniform float uViewportHeight;
      uniform float uVisibleDuration;
      uniform float uFadeDuration;
      uniform float uCycleDuration;
      attribute float aSeed;
      attribute float aPhase;
      attribute float aSpin;
      attribute float aSize;
      varying float vOpacity;
      varying float vAngle;

      float waterHash(float value) {
        return fract(sin(value * 91.731) * 43758.5453);
      }

      void main() {
        float elapsed = mod(uTime + aPhase, uCycleDuration);
        float cycleIndex = floor((uTime + aPhase) / uCycleDuration);
        float fadeIn = smoothstep(0.0, 0.7, elapsed);
        float fadeOut = 1.0 - smoothstep(
          uVisibleDuration,
          uVisibleDuration + uFadeDuration,
          elapsed
        );
        float visibleEnvelope = fadeIn * fadeOut;
        vOpacity = 0.08 * mix(1.0, visibleEnvelope, uMotion);
        vAngle = aSpin * uTime * uMotion + aSeed * 6.2831853;

        vec3 cycleShift = vec3(
          waterHash(aSeed + cycleIndex * 7.1) - 0.5,
          waterHash(aSeed + cycleIndex * 11.7) - 0.5,
          waterHash(aSeed + cycleIndex * 17.3) - 0.5
        ) * vec3(14.0, 8.0, 14.0);
        vec3 driftDirection = normalize(vec3(
          waterHash(aSeed + cycleIndex * 5.3) - 0.5,
          waterHash(aSeed + cycleIndex * 9.9) - 0.5,
          waterHash(aSeed + cycleIndex * 13.1) - 0.5
        ) + vec3(0.001));
        vec3 drift = driftDirection * min(elapsed, uVisibleDuration) * 0.2;
        drift += vec3(
          sin(uTime * 0.31 + aSeed) * 0.42,
          sin(uTime * 0.43 + aSeed * 1.7) * 0.58,
          cos(uTime * 0.27 + aSeed * 0.8) * 0.38
        );
        float waterWave = sin(position.x * 0.11 + position.z * 0.045 - uTime * 1.25) * 0.32;

        vec3 animatedPosition = position + (cycleShift + drift) * uMotion;
        animatedPosition.y += waterWave * uMotion;
        vec4 mvPosition = modelViewMatrix * vec4(animatedPosition, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        float projectedPointSize = aSize * uPixelRatio * (
          uViewportHeight * 0.5 * projectionMatrix[1][1] / max(1.0, -mvPosition.z)
        );
        gl_PointSize = min(projectedPointSize, 96.0 * uPixelRatio);
      }
    `,
    fragmentShader: `
      uniform sampler2D uSprite;
      varying float vOpacity;
      varying float vAngle;

      void main() {
        vec2 centered = gl_PointCoord - 0.5;
        float sineAngle = sin(vAngle);
        float cosineAngle = cos(vAngle);
        vec2 rotated = mat2(cosineAngle, -sineAngle, sineAngle, cosineAngle) * centered + 0.5;
        if (rotated.x < 0.0 || rotated.x > 1.0 || rotated.y < 0.0 || rotated.y > 1.0) discard;
        vec4 sprite = texture2D(uSprite, rotated);
        if (sprite.a < 0.015) discard;
        vec3 neutralGrey = vec3(0.60, 0.62, 0.64);
        gl_FragColor = vec4(neutralGrey * sprite.rgb, sprite.a * vOpacity);
      }
    `,
    transparent: true,
    depthWrite: false,
  });
  const particles = new THREE.Points(particleGeo, particleMat);
  particles.name = 'hero-water-molecule-field';
  scene.add(particles);
  canvas.dataset.waterMoleculeCount = String(waterMoleculeCount);
  canvas.dataset.waterMoleculeLifetime = `${WATER_VISIBLE_DURATION}s`;
  canvas.dataset.waterRadiusRatio = [
    getElementRadius('H') / getElementRadius('C'),
    1,
    getElementRadius('O') / getElementRadius('C'),
  ].map(value => value.toFixed(3)).join(':');
  canvas.dataset.waterSpriteWorldSize = waterSpriteWorldSize.toFixed(3);

  // Parse data
  const pdbData = parsePDB(PDB_DATA);
  const refLigand = parseSDFModel(REF_LIGAND_SDF);
  const parsedTrajectories = trajectorySources.map(source => {
    const stabilizedFrames = stabilizeTrajectoryAtomIdentity(parseSDFTrajectory(source.sdf));
    return {
      id: source.id,
      frames: resampleTrajectoryFrames(stabilizedFrames),
    };
  }).filter(source => source.frames.length > 0);
  if (parsedTrajectories.length === 0) throw new Error('No valid hero trajectories were loaded');

  log(`PDB: ${pdbData.atoms.length} atoms`);
  log(`Hero trajectories: ${parsedTrajectories.map(source => `${source.id}:${source.frames.length}`).join(', ')}`);
  log(`Ref ligand: ${refLigand.atoms.length} atoms`);

  // ---- Protein (cartoon ribbon / helix style) ----
  const proteinGroup = createProteinSketchGroup(pdbData.atoms, {
    helixColor: PROTEIN_LIGHT_BLUE_DEEP,
    sheetColor: PROTEIN_LIGHT_BLUE,
    loopColor: PROTEIN_LIGHT_BLUE_DEEP,
    outlineColor: PROTEIN_LIGHT_BLUE_DEEP,
    ribbonWidthScale: 1.5,
  });
  log(`Secondary structure: ${JSON.stringify(proteinGroup.userData.secondaryStructure || {})}`);
  log(`Protein geometry: ${JSON.stringify(proteinGroup.userData.geometryDiagnostics || [])}`);
  scene.add(proteinGroup);
  proteinGroup.userData.targetOpacity = 1;

  // Offset protein slightly left to avoid blocking the title
  proteinGroup.position.x = -4;

  // Pre-collect protein materials with their base opacity for efficient fade-in
  const proteinMaterials = [];
  proteinGroup.traverse((obj) => {
    if (obj.isMesh && obj.material) {
      proteinMaterials.push({
        mat: obj.material,
        baseOpacity: obj.material.opacity,
      });
      const isProteinBody = obj.material.userData.isProteinBody === true;
      obj.material.transparent = true;
      obj.material.depthTest = true;
      obj.material.depthWrite = isProteinBody;
      setProteinMaterialOpacity(obj.material, 0);
    }
  });

  // Compute pocket center from reference ligand
  const pocketCenter = computeCentroid(refLigand.atoms);
  log(`Pocket center: (${pocketCenter.x.toFixed(1)}, ${pocketCenter.y.toFixed(1)}, ${pocketCenter.z.toFixed(1)})`);

  // ---- Ligand trajectory players ----
  // Each trajectory has a different atom/bond topology, so each owns its own
  // mesh group and animator. Only the active group is visible.
  const ligandGroup = new THREE.Group();
  ligandGroup.name = 'hero-ligand-trajectories';
  scene.add(ligandGroup);
  const trajectoryPlayers = parsedTrajectories.map(source => {
    const finalFrame = source.frames[source.frames.length - 1];
    const finalCenter = computeCentroid(finalFrame.atoms);
    // Keep trajectory coordinates local to the finished molecule so scaling and
    // breathing happen around the ligand itself, not the scene origin.
    const offsetX = -finalCenter.x;
    const offsetY = -finalCenter.y;
    const offsetZ = -finalCenter.z;
    for (const frame of source.frames) {
      frame.atoms = translateAtoms(frame.atoms, offsetX, offsetY, offsetZ);
    }

    const alignedFinalFrame = source.frames[source.frames.length - 1];
    const bonds = alignedFinalFrame.bonds && alignedFinalFrame.bonds.length > 0
      ? alignedFinalFrame.bonds.map(bond => ({ ...bond, length: 0 }))
      : computeBonds(alignedFinalFrame.atoms, 2.0);
    const group = createBallStickGroup(alignedFinalFrame.atoms, bonds, {
      atomScale: 1.0,
      bondRadius: 0.15,
      sketchOutline: true,
      atomOutline: false,
      bondOutline: true,
      outlineColor: 0x1a3a5c,
      outlineThickness: 0.08,
      atomDetail: 10,
      toonShading: false,
    });
    group.name = `hero-ligand-${source.id}`;
    isolateGroupMaterials(group);
    group.position.set(pocketCenter.x, pocketCenter.y, pocketCenter.z);
    group.visible = false;
    ligandGroup.add(group);
    const animator = new MoleculeAnimator(source.frames, group);
    animator.setProgress(0);
    return {
      id: source.id,
      frames: source.frames,
      group,
      animator,
      driftPhase: Math.random() * Math.PI * 2,
    };
  });

  let activeTrajectoryIndex = Math.floor(Math.random() * trajectoryPlayers.length);
  const getActiveTrajectory = () => trajectoryPlayers[activeTrajectoryIndex];
  const setActiveTrajectory = (nextIndex, initialVisibility = 1) => {
    trajectoryPlayers.forEach((player, index) => {
      player.group.visible = index === nextIndex;
      player.group.scale.setScalar(1);
      player.group.position.set(pocketCenter.x, pocketCenter.y, pocketCenter.z);
      player.group.rotation.set(0, 0, 0);
    });
    activeTrajectoryIndex = nextIndex;
    const active = getActiveTrajectory();
    active.animator.setProgress(0);
    active.animator.setVisibility(initialVisibility);
    canvas.dataset.activeTrajectory = active.id;
  };

  // Once the ligand is nearly formed, let it drift subtly inside the pocket.
  // The reduced envelope preserves visible water motion without allowing the
  // molecule to sweep into the surrounding protein ribbons.
  const LIGAND_BOUND_DRIFT_SCALE = 0.32;
  function updateBoundLigandDrift(player, timeMs, reducedMotion) {
    const boundEnvelope = smoothstep01((player.animator.progress - 0.82) / 0.18);
    const phase = player.driftPhase;
    const motion = reducedMotion ? 0 : 1;
    const driftX = boundEnvelope * motion * LIGAND_BOUND_DRIFT_SCALE * (
      Math.sin(timeMs * 0.00031 + phase) * 0.52 +
      Math.sin(timeMs * 0.00011 + phase * 2.1) * 0.12
    );
    const driftY = boundEnvelope * motion * LIGAND_BOUND_DRIFT_SCALE * (
      Math.sin(timeMs * 0.00027 + phase * 1.43) * 0.42 +
      Math.sin(timeMs * 0.00015 + phase * 0.61) * 0.10
    );
    const driftZ = boundEnvelope * motion * LIGAND_BOUND_DRIFT_SCALE * (
      Math.sin(timeMs * 0.00023 + phase * 2.11) * 0.36 +
      Math.sin(timeMs * 0.00017 + phase * 0.37) * 0.10
    );

    player.group.position.set(
      pocketCenter.x + driftX,
      pocketCenter.y + driftY,
      pocketCenter.z + driftZ
    );
    player.group.rotation.set(
      boundEnvelope * motion * LIGAND_BOUND_DRIFT_SCALE * Math.sin(timeMs * 0.00019 + phase) * 0.035,
      boundEnvelope * motion * LIGAND_BOUND_DRIFT_SCALE * Math.sin(timeMs * 0.00016 + phase * 1.71) * 0.050,
      boundEnvelope * motion * LIGAND_BOUND_DRIFT_SCALE * Math.sin(timeMs * 0.00021 + phase * 0.83) * 0.028
    );
  }

  setActiveTrajectory(activeTrajectoryIndex);
  ligandGroup.visible = false;
  canvas.dataset.heroStage = 'protein';
  canvas.dataset.ligandPhase = 'waiting';
  canvas.dataset.trajectoryCount = String(trajectoryPlayers.length);
  canvas.dataset.completedMolecules = '0';
  canvas.dataset.samplingSteps = String(LIGAND_SAMPLING_STEPS);
  canvas.dataset.samplingStage = 'gaussian';

  // ---- Camera setup: a bounded, front-facing 180-degree ping-pong sweep ----
  const CAMERA_SWEEP_HALF_ANGLE = Math.PI / 2;
  const CAMERA_SWEEP_PERIOD = 720000;
  const CAMERA_OFFSET_X = 30;
      const CAMERA_OFFSET_Y = 15;
  const CAMERA_OFFSET_Z = 25;
  const CAMERA_ORBIT_RADIUS = Math.hypot(CAMERA_OFFSET_X, CAMERA_OFFSET_Z);
  const CAMERA_START_AZIMUTH_OFFSET = THREE.MathUtils.degToRad(-50);
  const CAMERA_SWEEP_CENTER_AZIMUTH =
    Math.atan2(CAMERA_OFFSET_X, CAMERA_OFFSET_Z) + CAMERA_START_AZIMUTH_OFFSET;
  const pocketVec = new THREE.Vector3(pocketCenter.x, pocketCenter.y, pocketCenter.z);
  camera.position.set(
    pocketCenter.x + Math.sin(CAMERA_SWEEP_CENTER_AZIMUTH) * CAMERA_ORBIT_RADIUS,
    pocketCenter.y + CAMERA_OFFSET_Y,
    pocketCenter.z + Math.cos(CAMERA_SWEEP_CENTER_AZIMUTH) * CAMERA_ORBIT_RADIUS
  );
  controls.target.copy(pocketVec);
  controls.minAzimuthAngle = CAMERA_SWEEP_CENTER_AZIMUTH - CAMERA_SWEEP_HALF_ANGLE;
  controls.maxAzimuthAngle = CAMERA_SWEEP_CENTER_AZIMUTH + CAMERA_SWEEP_HALF_ANGLE;
  controls.minPolarAngle = Math.PI * 0.28;
  controls.maxPolarAngle = Math.PI * 0.72;
  controls.update();
  canvas.dataset.cameraOrbitMode = 'bounded-ping-pong';
  canvas.dataset.cameraSweepDegrees = '180';
  canvas.dataset.cameraMode = 'auto';

  // ---- Resize ----
  function resize() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (w === 0 || h === 0) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    particleMat.uniforms.uPixelRatio.value = renderer.getPixelRatio();
    particleMat.uniforms.uViewportHeight.value = h;
  }
  window.addEventListener('resize', resize);
  resize();

  // ---- Animation state ----
  const HERO_TIMELINE = Object.freeze({
    introDuration: 33333,
    proteinStart: 0.02,
    proteinSettleEnd: 0.28,
    ligandRevealStart: 0.24,
    ligandStart: 0.30,
    ligandEnd: 0.90,
  });
  const LIGAND_CYCLE = Object.freeze({
    denoiseDuration: 20000,
    holdDuration: 4000,
    fadeOutDuration: 3000,
    fadeInDuration: 2000,
  });
  const state = {
    time: 0,
    proteinFadeIn: 0,
    breathPhase: 0,
    lastTimestamp: null,
    heroStage: 'protein',
    ligandPhase: 'waiting',
    ligandPhaseTime: 0,
    completedMolecules: 0,
    introCompletionRecorded: false,
    cameraManualOverride: false,
  };

  const enterManualCameraMode = () => {
    state.cameraManualOverride = true;
    canvas.dataset.cameraMode = 'manual';
  };
  controls.addEventListener('start', enterManualCameraMode);

  const setLigandPhase = phase => {
    state.ligandPhase = phase;
    if (canvas.dataset.ligandPhase !== phase) canvas.dataset.ligandPhase = phase;
    setHeroPhaseLabel(phase);
  };
  const setLigandSamplingStep = value => {
    const samplingStep = Math.max(1, Math.min(LIGAND_SAMPLING_STEPS, value));
    canvas.dataset.samplingStep = String(samplingStep);
    canvas.dataset.samplingStage = getLigandSamplingPhase(samplingStep);
    setHeroPhaseLabel(getLigandSamplingPhase(samplingStep));
  };
  setHeroPhaseLabel('waiting');

  const recordCompletedMolecule = () => {
    state.completedMolecules++;
    canvas.dataset.completedMolecules = String(state.completedMolecules);
  };

  const updateLigandCycle = dt => {
    state.ligandPhaseTime += dt;
    let active = getActiveTrajectory();

    if (state.ligandPhase === 'hold') {
      if (active.animator.progress !== 1) active.animator.setProgress(1);
      canvas.dataset.samplingStep = String(LIGAND_SAMPLING_STEPS);
      const breath = getHoldBreathScale(
        state.ligandPhaseTime,
        LIGAND_CYCLE.holdDuration,
        state.breathPhase,
        reducedMotion
      );
      active.group.scale.setScalar(breath);
      if (state.ligandPhaseTime >= LIGAND_CYCLE.holdDuration) {
        state.ligandPhaseTime -= LIGAND_CYCLE.holdDuration;
        setLigandPhase('fade-out');
      } else {
        return;
      }
    }

    if (state.ligandPhase === 'fade-out') {
      if (active.animator.progress !== 1) active.animator.setProgress(1);
      const progress = Math.min(1, state.ligandPhaseTime / LIGAND_CYCLE.fadeOutDuration);
      active.animator.setVisibility(1 - smoothstep01(progress));
      if (state.ligandPhaseTime >= LIGAND_CYCLE.fadeOutDuration) {
        state.ligandPhaseTime -= LIGAND_CYCLE.fadeOutDuration;
        active.animator.setVisibility(0);
        const nextIndex = pickNextTrajectoryIndex(activeTrajectoryIndex, trajectoryPlayers.length);
        setActiveTrajectory(nextIndex, 0);
        active = getActiveTrajectory();
        setLigandPhase('fade-in');
      } else {
        return;
      }
    }

    if (state.ligandPhase === 'fade-in') {
      if (active.animator.progress !== 0) active.animator.setProgress(0);
      canvas.dataset.samplingStep = '1';
      const progress = Math.min(1, state.ligandPhaseTime / LIGAND_CYCLE.fadeInDuration);
      active.animator.setVisibility(smoothstep01(progress));
      if (state.ligandPhaseTime >= LIGAND_CYCLE.fadeInDuration) {
        state.ligandPhaseTime -= LIGAND_CYCLE.fadeInDuration;
        active.animator.setVisibility(1);
        setLigandPhase('denoise');
      } else {
        return;
      }
    }

    const progress = Math.min(1, state.ligandPhaseTime / LIGAND_CYCLE.denoiseDuration);
    active.animator.setProgress(progress);
    setLigandSamplingStep(getLigandSamplingStep(progress));
    if (state.ligandPhaseTime >= LIGAND_CYCLE.denoiseDuration) {
      state.ligandPhaseTime -= LIGAND_CYCLE.denoiseDuration;
      active.animator.setProgress(1);
      recordCompletedMolecule();
      setLigandPhase('hold');
    }
  };

  function updateHeroCamera() {
    const focusTarget = getActiveTrajectory().group.position;
    controls.target.lerp(focusTarget, 0.08);
    if (!reducedMotion && !state.cameraManualOverride) {
      const cameraSweepPhase = state.time / CAMERA_SWEEP_PERIOD * Math.PI * 2;
      const cameraAzimuth = CAMERA_SWEEP_CENTER_AZIMUTH +
        Math.sin(cameraSweepPhase) * CAMERA_SWEEP_HALF_ANGLE;
      camera.position.set(
        controls.target.x + Math.sin(cameraAzimuth) * CAMERA_ORBIT_RADIUS,
        controls.target.y + CAMERA_OFFSET_Y + Math.sin(cameraSweepPhase * 0.5) * 1.15,
        controls.target.z + Math.cos(cameraAzimuth) * CAMERA_ORBIT_RADIUS
      );
    }
    controls.update();
  }

  // ---- Animation loop ----
  let running = true;
  function animate(timestamp) {
    if (!running) return;
    requestAnimationFrame(animate);

    const dt = state.lastTimestamp === null ? 16.67 : Math.min(100, timestamp - state.lastTimestamp);
    state.lastTimestamp = timestamp;
    state.time += dt;
    state.breathPhase += dt * 0.000625;

    // Intro timeline
    if (state.time < HERO_TIMELINE.introDuration) {
      const t = state.time / HERO_TIMELINE.introDuration;

      // Establish the binding environment first, then let the molecule emerge.
      const proteinT = (t - HERO_TIMELINE.proteinStart) /
        (HERO_TIMELINE.proteinSettleEnd - HERO_TIMELINE.proteinStart);
      state.proteinFadeIn = smoothstep01(proteinT);
      for (const pm of proteinMaterials) {
        setProteinMaterialOpacity(pm.mat, pm.baseOpacity * state.proteinFadeIn);
      }

      // Let the Gaussian atoms emerge during the last part of the protein
      // fade. The actual denoising clock still starts at ligandStart, so the
      // total intro duration and twenty-second molecular growth stay intact.
      const active = getActiveTrajectory();
      if (t >= HERO_TIMELINE.ligandRevealStart && t < HERO_TIMELINE.ligandStart) {
        ligandGroup.visible = true;
        setHeroPhaseLabel('gaussian');
        canvas.dataset.samplingStep = '1';
        const ligandRevealT = (t - HERO_TIMELINE.ligandRevealStart) /
          (HERO_TIMELINE.ligandStart - HERO_TIMELINE.ligandRevealStart);
        const ligandReveal = smoothstep01(ligandRevealT);
        active.animator.setVisibility(ligandReveal);
      }

      if (t >= HERO_TIMELINE.ligandStart) {
        ligandGroup.visible = true;
        if (state.heroStage !== 'ligand') {
          state.heroStage = 'ligand';
          canvas.dataset.heroStage = 'ligand';
        }
        const ligT = (t - HERO_TIMELINE.ligandStart) /
          (HERO_TIMELINE.ligandEnd - HERO_TIMELINE.ligandStart);
        const samplingProgress = Math.max(0, Math.min(1, ligT));
        active.animator.setVisibility(1);
        active.animator.setProgress(samplingProgress);
        const samplingStep = getLigandSamplingStep(samplingProgress);
        if (ligT < 1) {
          setLigandPhase('denoise');
          setLigandSamplingStep(samplingStep);
          state.ligandPhaseTime = Math.max(0, ligT) * LIGAND_CYCLE.denoiseDuration;
        } else {
          canvas.dataset.samplingStep = String(LIGAND_SAMPLING_STEPS);
          canvas.dataset.samplingStage = 'repair';
          active.animator.setProgress(1);
          if (!state.introCompletionRecorded) {
            state.introCompletionRecorded = true;
            recordCompletedMolecule();
          }
          setLigandPhase('hold');
          state.ligandPhaseTime = (t - HERO_TIMELINE.ligandEnd) * HERO_TIMELINE.introDuration;
          const breath = getHoldBreathScale(
            state.ligandPhaseTime,
            LIGAND_CYCLE.holdDuration,
            state.breathPhase,
            reducedMotion
          );
          active.group.scale.setScalar(breath);
        }
      }
    } else {
      ligandGroup.visible = true;
      if (state.heroStage !== 'loop') {
        state.heroStage = 'loop';
        canvas.dataset.heroStage = 'loop';
      }
      updateLigandCycle(dt);
    }

    updateBoundLigandDrift(getActiveTrajectory(), state.time, reducedMotion);

    // Layer two slow frequencies so the complex drifts instead of bobbing on a rail.
    if (!reducedMotion) {
      proteinGroup.position.y = Math.sin(state.time * 0.00042) * 1.15 + Math.sin(state.time * 0.00013) * 0.35;
      proteinGroup.position.x = -4 + Math.sin(state.time * 0.00018) * 0.65 + Math.sin(state.time * 0.000047) * 0.25;
      proteinGroup.position.z = Math.sin(state.time * 0.00023) * 0.5;
      proteinGroup.rotation.x = Math.sin(state.time * 0.00017) * 0.018;
      proteinGroup.rotation.z = Math.sin(state.time * 0.00029) * 0.035;
    }
    updateProteinWaveColors(proteinGroup.userData.bodyMaterials, state.time, reducedMotion);

    // Auto-resize check (handles cases where canvas starts at 0 size)
    const targetCanvasWidth = Math.floor(canvas.clientWidth * renderer.getPixelRatio());
    const targetCanvasHeight = Math.floor(canvas.clientHeight * renderer.getPixelRatio());
    if (canvas.width !== targetCanvasWidth || canvas.height !== targetCanvasHeight) {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (w > 0 && h > 0) {
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      }
    }

    // Water lifetimes, per-particle drift and rotation are evaluated on the GPU.
    particleMat.uniforms.uTime.value = state.time * 0.001;

    updateHeroCamera();
    renderer.render(scene, camera);
  }
  requestAnimationFrame(animate);

  log('Hero scene built OK');

  return {
    scene, camera, renderer, controls,
    proteinGroup, ligandGroup, trajectoryPlayers,
    pocketCenter,
    resize,
    destroy: () => {
      running = false;
      window.removeEventListener('resize', resize);
      disposeObject3D(proteinGroup);
      trajectoryPlayers.forEach(disposeTrajectoryPlayer);
      particleGeo.dispose();
      particleMat.dispose();
      waterMoleculeTexture.dispose();
      controls.removeEventListener('start', enterManualCameraMode);
      controls.dispose();
      renderer.dispose();
    },
  };
}



export {
  buildHeroScene,
  loadHeroTrajectorySources
};
