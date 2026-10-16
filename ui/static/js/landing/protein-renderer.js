import * as THREE from 'three';
import { PROTEIN_LIGHT_BLUE, PROTEIN_LIGHT_BLUE_DEEP, buildStructureRuns, countStructureTypes, createContinuousCartoonGeometry, createVisualStructureTypes, inferSecondaryStructure, splitContinuousResidueChains, validateProteinGeometry } from './protein-geometry.js';

// Keep the approved structure colours physically matte. The animated colour
// tide is injected into Three's standard material, preserving the existing
// palette without the hard metallic highlights of the former custom shader.
const PYMOL_CARTOON_OPACITY = 0.50;

function makeCartoonMaterial() {
  const waveUniforms = {
    time: { value: 0 },
    strength: { value: 1 },
    color: { value: new THREE.Color(PROTEIN_LIGHT_BLUE) },
  };
  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.82,
    metalness: 0.0,
    transparent: true,
    opacity: PYMOL_CARTOON_OPACITY,
    depthWrite: true,
    side: THREE.DoubleSide,
    fog: true,
    dithering: true,
  });
  material.onBeforeCompile = shader => {
    shader.uniforms.uProteinWaveTime = waveUniforms.time;
    shader.uniforms.uProteinWaveStrength = waveUniforms.strength;
    shader.uniforms.uProteinWaveColor = waveUniforms.color;
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        `#include <common>
        attribute float waveCoord;
        varying float vProteinWave;
        uniform float uProteinWaveTime;
        uniform float uProteinWaveStrength;`
      )
      .replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
        float proteinWavePhase = fract(waveCoord - uProteinWaveTime);
        float proteinWaveDistance = (proteinWavePhase - 0.5) * 8.0;
        vProteinWave = exp(-proteinWaveDistance * proteinWaveDistance) * uProteinWaveStrength;`
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>
        varying float vProteinWave;
        uniform vec3 uProteinWaveColor;`
      )
      .replace(
        '#include <color_fragment>',
        `#include <color_fragment>
        diffuseColor.rgb = mix(diffuseColor.rgb, uProteinWaveColor, vProteinWave * 0.58);`
      );
  };
  material.customProgramCacheKey = () => 'matte-protein-wave-v1';
  material.userData.isProteinBody = true;
  material.userData.waveUniforms = waveUniforms;
  return material;
}

function setProteinMaterialOpacity(material, opacity) {
  material.opacity = opacity;
  if (material.userData.waveUniforms?.opacity) {
    material.userData.waveUniforms.opacity.value = opacity;
  }
}

function updateProteinWaveColors(materials, elapsedMs, reducedMotion = false) {
  for (const material of materials) {
    const uniforms = material.userData.waveUniforms;
    if (!uniforms) continue;
    uniforms.time.value = reducedMotion ? 0.5 : (elapsedMs * 0.000065) % 1;
    uniforms.strength.value = reducedMotion ? 0 : 1;
  }
}

function disposeObject3D(root) {
  const geometries = new Set();
  const materials = new Set();
  root.traverse(object => {
    if (object.geometry) geometries.add(object.geometry);
    if (Array.isArray(object.material)) object.material.forEach(material => materials.add(material));
    else if (object.material) materials.add(object.material);
  });
  geometries.forEach(geometry => geometry.dispose());
  materials.forEach(material => material.dispose());
}

function disposeTrajectoryPlayer(player) {
  const geometries = new Set();
  const materials = new Set();
  for (const bond of player.group.userData.bondMeshes || []) {
    if (bond.geometry) geometries.add(bond.geometry);
    if (bond.material) materials.add(bond.material);
    if (bond.userData.outlineMesh?.material) materials.add(bond.userData.outlineMesh.material);
  }
  for (const atom of player.group.userData.atomMeshes || []) {
    if (atom.userData.outlineMesh?.material) materials.add(atom.userData.outlineMesh.material);
  }
  geometries.forEach(geometry => geometry.dispose());
  materials.forEach(material => material.dispose());
}

// Create a continuous PyMOL-style cartoon. Each physical chain is one body mesh
// plus one restrained back-face rim; loops no longer run beneath ribbons.
function createProteinSketchGroup(pdbAtoms, options = {}) {
  const group = new THREE.Group();
  const {
    helixColor = PROTEIN_LIGHT_BLUE_DEEP,
    sheetColor = PROTEIN_LIGHT_BLUE,
    loopColor = PROTEIN_LIGHT_BLUE_DEEP,
    outlineColor = PROTEIN_LIGHT_BLUE_DEEP,
    ribbonWidthScale = 1,
  } = options;

  // Gather N/CA/C/O atoms by residue so peptide-plane orientation is preserved.
  const residueMap = new Map();
  for (const atom of pdbAtoms) {
    if (!['N', 'CA', 'C', 'O'].includes(atom.name)) continue;
    const key = `${atom.chainId}:${atom.resSeq}`;
    if (!residueMap.has(key)) {
      residueMap.set(key, { chainId: atom.chainId, resSeq: atom.resSeq, resName: atom.resName });
    }
    residueMap.get(key)[atom.name] = new THREE.Vector3(atom.x, atom.y, atom.z);
  }

  const residues = [...residueMap.values()]
    .filter(residue => residue.CA)
    .sort((a, b) => a.chainId.localeCompare(b.chainId) || a.resSeq - b.resSeq);

  if (residues.length < 2) {
    console.warn('Not enough CA atoms for backbone trace');
    return group;
  }

  const rawTypes = inferSecondaryStructure(residues);
  const visualTypes = createVisualStructureTypes(residues, rawTypes);
  const rawRuns = buildStructureRuns(residues, rawTypes);
  const visualRuns = buildStructureRuns(residues, visualTypes);
  const bodyMaterials = [];
  const rimMaterials = [];
  const geometryDiagnostics = [];

  const continuousChains = splitContinuousResidueChains(residues);
  continuousChains.forEach((chain, chainIndex) => {
    const chainResidues = residues.slice(chain.start, chain.end + 1);
    if (chainResidues.length < 2) return;
    const chainTypes = visualTypes.slice(chain.start, chain.end + 1);
    const geometry = createContinuousCartoonGeometry(chainResidues, chainTypes, {
      segmentsPerResidue: 12,
      crossSectionSides: 24,
      helixColor,
      sheetColor,
      loopColor,
      ribbonWidthScale,
      waveOffset: chainIndex * 0.17,
    });
    geometryDiagnostics.push(validateProteinGeometry(geometry));

    const bodyMaterial = makeCartoonMaterial();
    const bodyMesh = new THREE.Mesh(geometry, bodyMaterial);
    bodyMesh.name = `protein-cartoon-body-${chainIndex}`;
    group.add(bodyMesh);
    bodyMaterials.push(bodyMaterial);

    const rimMaterial = new THREE.MeshBasicMaterial({
      color: outlineColor,
      transparent: true,
      opacity: 0.06,
      side: THREE.BackSide,
      depthWrite: false,
    });
    rimMaterial.userData.isProteinRim = true;
    const rimMesh = new THREE.Mesh(geometry, rimMaterial);
    rimMesh.name = `protein-cartoon-rim-${chainIndex}`;
    rimMesh.scale.setScalar(1.025);
    rimMesh.position.copy(geometry.boundingSphere.center).multiplyScalar(1 - 1.025);
    rimMesh.renderOrder = -1;
    group.add(rimMesh);
    rimMaterials.push(rimMaterial);
  });

  const runSummary = runs => runs.map(run => ({
    type: run.type,
    startResidue: residues[run.start].resSeq,
    endResidue: residues[run.end].resSeq,
    chainId: residues[run.start].chainId,
  }));
  const rawCounts = countStructureTypes(rawTypes);
  const visualCounts = countStructureTypes(visualTypes);
  group.userData = {
    caAtoms: residues.map(residue => ({
      chainId: residue.chainId,
      resSeq: residue.resSeq,
      resName: residue.resName,
      x: residue.CA.x,
      y: residue.CA.y,
      z: residue.CA.z,
    })),
    atomCount: pdbAtoms.length,
    residueCount: residues.length,
    secondaryStructure: visualCounts,
    rawSecondaryStructure: rawCounts,
    visualSecondaryStructure: visualCounts,
    rawStructureRuns: runSummary(rawRuns),
    structureRuns: runSummary(visualRuns),
    continuousChainCount: continuousChains.length,
    geometryDiagnostics,
    bodyMaterials,
    rimMaterials,
  };

  console.info('Secondary structure (raw → visual):', rawCounts, visualCounts);
  console.info('Secondary structure runs (raw → visual):', rawRuns.length, visualRuns.length);
  return group;
}


export {
  createProteinSketchGroup,
  disposeObject3D,
  disposeTrajectoryPlayer,
  setProteinMaterialOpacity,
  updateProteinWaveColors
};
