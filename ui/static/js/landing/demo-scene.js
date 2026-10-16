import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { computeBonds, computeCentroid, parseSDFTrajectory, translateAtoms } from './molecule-parser.js';
import { SDF_TRAJECTORY } from './molecular-data.js';
import { createBallStickGroup, isolateGroupMaterials } from './molecule-renderer.js';
import { MoleculeAnimator } from './molecule-animator.js';

// ======= MOLECULE DEMO SCENE =======

function buildDemoScene(canvasId, onReady) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) { console.warn('Demo canvas not found'); return null; }

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf0f6ff);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);

  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: false,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.5;
  controls.minDistance = 3;
  controls.maxDistance = 20;

  // Lighting - bright/clean
  const ambient = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambient);
  const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
  keyLight.position.set(5, 10, 7);
  scene.add(keyLight);
  const fillLight = new THREE.DirectionalLight(0xb0d0ff, 0.5);
  fillLight.position.set(-5, -3, -5);
  scene.add(fillLight);

  // Parse trajectory
  const trajFrames = parseSDFTrajectory(SDF_TRAJECTORY);

  // Use bond topology from SDF file (more accurate than distance-based)
  const lastFrame = trajFrames[trajFrames.length - 1];
  const demoBonds = lastFrame.bonds && lastFrame.bonds.length > 0
    ? lastFrame.bonds.map(b => ({ ...b, length: 0 }))
    : computeBonds(lastFrame.atoms, 2.0);

  // Center all trajectory frames on the last frame's centroid
  const trajCenter = computeCentroid(lastFrame.atoms);
  for (const frame of trajFrames) {
    frame.atoms = translateAtoms(frame.atoms, -trajCenter.x, -trajCenter.y, -trajCenter.z);
  }

  // Create molecule from final frame (correct bond topology)
  const molGroup = createBallStickGroup(
    lastFrame.atoms,
    demoBonds,
    {
      atomScale: 1.2,
      bondRadius: 0.18,
      sketchOutline: true,
      outlineColor: 0x2c3e50,
      outlineThickness: 0.06,
      atomDetail: 12,
      toonShading: true,
    }
  );
  isolateGroupMaterials(molGroup);
  scene.add(molGroup);

  // Create animator
  const animator = new MoleculeAnimator(trajFrames, molGroup);
  animator.setProgress(0);

  // Camera
  const bboxSize = 8;
  camera.position.set(0, 2, bboxSize * 1.5);
  controls.target.set(0, 0, 0);
  controls.update();

  // Resize
  function resize() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (w === 0 || h === 0) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);
  resize();

  // Animation
  let isPlaying = false;
  let progress = 0;
  let playSpeed = 0.0015;

  function setProgress(p) {
    progress = Math.max(0, Math.min(1, p));
    animator.setProgress(progress);
    // Update UI if callback
    if (onProgressChange) onProgressChange(progress);
  }

  let onProgressChange = null;
  function setProgressCallback(cb) { onProgressChange = cb; }

  let running = true;
  function animate() {
    if (!running) return;
    requestAnimationFrame(animate);

    if (isPlaying) {
      progress += playSpeed;
      if (progress >= 1) {
        progress = 1;
        isPlaying = false;
      }
      animator.setProgress(progress);
      if (onProgressChange) onProgressChange(progress);
    }

    // Auto-resize check (handles cases where canvas starts at 0 size)
    if (canvas.width !== canvas.clientWidth * renderer.getPixelRatio() ||
        canvas.height !== canvas.clientHeight * renderer.getPixelRatio()) {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (w > 0 && h > 0) {
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      }
    }

    controls.update();
    renderer.render(scene, camera);
  }
  requestAnimationFrame(animate);

  log('Demo scene built OK');

  return {
    scene, camera, renderer, controls,
    molGroup, animator,
    setProgress,
    setProgressCallback,
    play: () => { isPlaying = true; },
    pause: () => { isPlaying = false; },
    toggle: () => { isPlaying = !isPlaying; return isPlaying; },
    isPlaying: () => isPlaying,
    reset: () => { progress = 0; isPlaying = false; animator.setProgress(0); if (onProgressChange) onProgressChange(0); },
    getProgress: () => progress,
    resize,
    destroy: () => {
      running = false;
      controls.dispose();
      renderer.dispose();
    },
  };
}



export {
  buildDemoScene
};
