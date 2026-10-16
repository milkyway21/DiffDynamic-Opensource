import * as THREE from 'three';
import { getElementColor, getElementRadius } from './molecule-parser.js';

// ======= BALL-STICK RENDERING =======

// Shared geometries for performance
const _sphereGeoCache = {};
function getSphereGeo(radius, detail = 8) {
  const key = `${radius}_${detail}`;
  if (!_sphereGeoCache[key]) {
    _sphereGeoCache[key] = new THREE.SphereGeometry(radius, detail, detail);
  }
  return _sphereGeoCache[key];
}

// Cached toon material by color (shared across atoms of same element)
const _toonMatCache = {};
function getCachedToonMat(color) {
  const key = 'toon_' + color;
  if (!_toonMatCache[key]) {
    const gradientTex = createToonGradient(color);
    _toonMatCache[key] = new THREE.MeshToonMaterial({
      color: color,
      gradientMap: gradientTex,
      transparent: true,
      opacity: 0.9,
    });
  }
  return _toonMatCache[key];
}

// Create ball-and-stick molecule group
function createBallStickGroup(atoms, bonds, options = {}) {
  const group = new THREE.Group();
  const {
    atomScale = 1.0,
    bondRadius = 0.12,
    sketchOutline = false,
    atomOutline = sketchOutline,
    bondOutline = sketchOutline,
    outlineColor = 0x1a2a40,
    outlineThickness = 0.08,
    atomDetail = 8,
    toonShading = true,
  } = options;

  // ---- Atoms ----
  const atomMeshes = [];
  for (let i = 0; i < atoms.length; i++) {
    const atom = atoms[i];
    const radius = getElementRadius(atom.element) * atomScale;
    const color = getElementColor(atom.element);

    // Use cached material by element color
    let material;
    if (toonShading) {
      material = getCachedToonMat(color);
    } else {
      material = new THREE.MeshPhysicalMaterial({
        color: color,
        roughness: 0.48,
        metalness: 0.01,
        clearcoat: 0.08,
        clearcoatRoughness: 0.78,
        transparent: true,
        opacity: 0.9,
      });
    }

    const geo = getSphereGeo(radius, atomDetail);
    const mesh = new THREE.Mesh(geo, material);
    mesh.position.set(atom.x, atom.y, atom.z);
    mesh.userData = { atomIndex: i, baseRadius: radius };
    group.add(mesh);
    atomMeshes.push(mesh);

    // Sketch outline (backface method)
    if (atomOutline) {
      const outlineMat = new THREE.MeshBasicMaterial({
        color: outlineColor,
        side: THREE.BackSide,
        transparent: true,
        opacity: 0.6,
      });
      const outlineMesh = new THREE.Mesh(geo, outlineMat);
      outlineMesh.position.copy(mesh.position);
      outlineMesh.scale.setScalar(1 + outlineThickness / radius);
      group.add(outlineMesh);
      mesh.userData.outlineMesh = outlineMesh;
    }
  }

  // ---- Bonds ----
  const bondMeshes = [];
  for (let i = 0; i < bonds.length; i++) {
    const bond = bonds[i];
    const a1 = atoms[bond.a1];
    const a2 = atoms[bond.a2];
    if (!a1 || !a2) continue;

    const start = new THREE.Vector3(a1.x, a1.y, a1.z);
    const end = new THREE.Vector3(a2.x, a2.y, a2.z);
    const direction = new THREE.Vector3().subVectors(end, start);
    const length = direction.length();

    if (length < 0.1) continue;

    // Render the SDF bond order explicitly: one centered cylinder for a single
    // bond, two parallel slimmer cylinders for a double bond (and three for a
    // rare triple bond). Keeping lane metadata lets the animator preserve the
    // order while atom coordinates move between trajectory frames.
    const laneCount = Math.min(3, Math.max(1, Math.round(bond.order || 1)));
    const segmentRadius = laneCount === 1 ? bondRadius : bondRadius * 0.66;
    const laneSpacing = bondRadius * 1.9;
    const cylGeo = new THREE.CylinderGeometry(segmentRadius, segmentRadius, length, 8);
    let material;
    if (toonShading) {
      // Average color of two atoms
      const c1 = getElementColor(a1.element);
      const c2 = getElementColor(a2.element);
      const avgColor = averageColors(c1, c2);
      const gradientTex = createToonGradient(avgColor);
      material = new THREE.MeshToonMaterial({
        color: avgColor,
        gradientMap: gradientTex,
        transparent: true,
        opacity: 0.85,
      });
    } else {
      material = new THREE.MeshStandardMaterial({
        color: 0x9ab8d0,
        roughness: 0.4,
        metalness: 0.1,
        transparent: true,
        opacity: 0.85,
      });
    }

    const outlineMat = bondOutline
      ? new THREE.MeshBasicMaterial({
        color: outlineColor,
        side: THREE.BackSide,
        transparent: true,
        opacity: 0.5,
      })
      : null;

    const unitDirection = direction.clone().normalize();
    const referenceAxis = Math.abs(unitDirection.y) < 0.9
      ? new THREE.Vector3(0, 1, 0)
      : new THREE.Vector3(1, 0, 0);
    const lateral = new THREE.Vector3().crossVectors(unitDirection, referenceAxis).normalize();

    for (let laneIndex = 0; laneIndex < laneCount; laneIndex++) {
      const laneOffset = (laneIndex - (laneCount - 1) * 0.5) * laneSpacing;
      const cyl = new THREE.Mesh(cylGeo, material);
      cyl.position.copy(start).add(end).multiplyScalar(0.5).addScaledVector(lateral, laneOffset);
      cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), unitDirection);
      cyl.userData = {
        bondIndex: i,
        laneIndex,
        laneCount,
        laneSpacing,
      };

      group.add(cyl);
      bondMeshes.push(cyl);

      if (outlineMat) {
        const outlineCyl = new THREE.Mesh(cylGeo, outlineMat);
        outlineCyl.position.copy(cyl.position);
        outlineCyl.quaternion.copy(cyl.quaternion);
        outlineCyl.scale.set(
          1 + outlineThickness / segmentRadius,
          1,
          1 + outlineThickness / segmentRadius
        );
        group.add(outlineCyl);
        cyl.userData.outlineMesh = outlineCyl;
      }
    }
  }

  group.userData = { atomMeshes, bondMeshes, atoms: atoms.map(a => ({...a})), bonds: bonds.map(b => ({...b})) };
  return group;
}

function isolateGroupMaterials(group) {
  const clones = new Map();
  group.traverse(object => {
    if (!object.isMesh || !object.material) return;
    if (!clones.has(object.material)) {
      clones.set(object.material, object.material.clone());
    }
    object.material = clones.get(object.material);
  });
}

// Helper: average two hex colors
function averageColors(c1, c2) {
  const r1 = (c1 >> 16) & 255, g1 = (c1 >> 8) & 255, b1 = c1 & 255;
  const r2 = (c2 >> 16) & 255, g2 = (c2 >> 8) & 255, b2 = c2 & 255;
  return ((r1 + r2) / 2) << 16 | ((g1 + g2) / 2) << 8 | ((b1 + b2) / 2);
}

// Create toon gradient texture (3 steps)
const _gradientCache = {};
function createToonGradient(baseColorHex) {
  const key = baseColorHex;
  if (_gradientCache[key]) return _gradientCache[key];

  const canvas = document.createElement('canvas');
  canvas.width = 4;
  canvas.height = 1;
  const ctx = canvas.getContext('2d');

  const base = new THREE.Color(baseColorHex);
  // Dark, medium, light steps
  const steps = [
    base.clone().multiplyScalar(0.55),
    base.clone().multiplyScalar(0.8),
    base.clone().multiplyScalar(1.0),
    base.clone().multiplyScalar(1.15),
  ];

  for (let i = 0; i < 4; i++) {
    ctx.fillStyle = '#' + steps[i].getHexString();
    ctx.fillRect(i, 0, 1, 1);
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  _gradientCache[key] = tex;
  return tex;
}


export {
  createBallStickGroup,
  isolateGroupMaterials
};
