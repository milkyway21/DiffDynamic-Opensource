import * as THREE from 'three';

// ======= MOLECULAR PARSING & RENDERING CORE =======

// Element colors (CPK-like, adjusted for sketch/blue theme)
const ELEMENT_COLORS = {
  C: 0x33ff33,   // Carbon - PyMOL default "carbon" (0.2, 1.0, 0.2)
  N: 0x6ec6ff,   // Nitrogen - bright cyan-blue
  O: 0xff8a8a,   // Oxygen - soft red
  S: 0xffd27a,   // Sulfur - gold
  P: 0xffa94d,   // Phosphorus - orange
  CL: 0x7fdc7f,  // Chlorine - green
  F: 0x90e0e0,   // Fluorine - cyan
  H: 0xa9b7c5,   // Hydrogen - cool grey, kept clear of white background
  BR: 0x8b4513,  // Bromine
  I: 0x9370db,   // Iodine
};

const ELEMENT_RADII = {
  C: 0.45, N: 0.40, O: 0.38, S: 0.55, P: 0.60,
  CL: 0.55, F: 0.35, H: 0.25, BR: 0.60, I: 0.65,
};

function getElementColor(el) {
  const key = el.toUpperCase().trim();
  return ELEMENT_COLORS[key] !== undefined ? ELEMENT_COLORS[key] : 0xb0b0b0;
}

function getElementRadius(el) {
  const key = el.toUpperCase().trim();
  return ELEMENT_RADII[key] !== undefined ? ELEMENT_RADII[key] : 0.4;
}

// ---- PDB Parser ----
function parsePDB(pdbText) {
  const atoms = [];
  const lines = pdbText.split('\n');
  for (const line of lines) {
    if (!line.startsWith('ATOM') && !line.startsWith('HETATM')) continue;
    // PDB fixed-width format
    const serial = parseInt(line.substring(6, 11).trim(), 10);
    const name = line.substring(12, 16).trim();
    const resName = line.substring(17, 20).trim();
    const chainId = line.substring(21, 22).trim();
    const resSeq = parseInt(line.substring(22, 26).trim(), 10);
    const x = parseFloat(line.substring(30, 38).trim());
    const y = parseFloat(line.substring(38, 46).trim());
    const z = parseFloat(line.substring(46, 54).trim());
    let element = line.substring(76, 78).trim();
    if (!element) {
      // Fallback: guess from atom name
      element = name.replace(/[0-9]/g, '').charAt(0) || 'C';
    }
    atoms.push({
      serial, name, resName, chainId, resSeq,
      x, y, z, element: element.toUpperCase()
    });
  }
  return { atoms };
}

// ---- SDF Parser (single model) ----
function parseSDFModel(sdfBlock) {
  const lines = sdfBlock.split('\n');
  if (lines.length < 4) return { atoms: [], bonds: [], name: '' };
  const name = lines[0].trim();
  // Counts line
  const countsLine = lines[3];
  const atomCount = parseInt(countsLine.substring(0, 3).trim(), 10);
  const bondCount = parseInt(countsLine.substring(3, 6).trim(), 10);

  const atoms = [];
  for (let i = 0; i < atomCount; i++) {
    const line = lines[4 + i];
    if (!line) continue;
    const x = parseFloat(line.substring(0, 10).trim());
    const y = parseFloat(line.substring(10, 20).trim());
    const z = parseFloat(line.substring(20, 30).trim());
    const element = line.substring(31, 34).trim().toUpperCase();
    atoms.push({ x, y, z, element, index: i });
  }

  const bonds = [];
  const bondStart = 4 + atomCount;
  for (let i = 0; i < bondCount; i++) {
    const line = lines[bondStart + i];
    if (!line) continue;
    const a1 = parseInt(line.substring(0, 3).trim(), 10) - 1;
    const a2 = parseInt(line.substring(3, 6).trim(), 10) - 1;
    const order = parseInt(line.substring(6, 9).trim(), 10);
    bonds.push({ a1, a2, order });
  }

  return { name, atoms, bonds };
}

// ---- SDF Trajectory Parser (multi-model) ----
function parseSDFTrajectory(sdfText) {
  // Split by $$$$ delimiter
  const blocks = sdfText.split(/\$\$\$\$/).map(b => b.trim()).filter(b => b.length > 5);
  const frames = [];
  for (const block of blocks) {
    const model = parseSDFModel(block);
    if (model.atoms.length > 0) {
      frames.push(model);
    }
  }
  return frames;
}

// Hungarian minimum-cost assignment. Diffusion exports can insert/remove an
// atom in an intermediate SDF model, which shifts every later array index even
// though the spatial atom identities remain continuous.
function solveMinimumCostAssignment(costMatrix) {
  const rowCount = costMatrix.length;
  const columnCount = costMatrix[0]?.length || 0;
  if (!rowCount || rowCount > columnCount) return [];
  const rowPotential = new Array(rowCount + 1).fill(0);
  const columnPotential = new Array(columnCount + 1).fill(0);
  const matchedRow = new Array(columnCount + 1).fill(0);
  const previousColumn = new Array(columnCount + 1).fill(0);

  for (let row = 1; row <= rowCount; row++) {
    matchedRow[0] = row;
    let column0 = 0;
    const minimum = new Array(columnCount + 1).fill(Infinity);
    const used = new Array(columnCount + 1).fill(false);
    do {
      used[column0] = true;
      const activeRow = matchedRow[column0];
      let delta = Infinity;
      let column1 = 0;
      for (let column = 1; column <= columnCount; column++) {
        if (used[column]) continue;
        const reducedCost = costMatrix[activeRow - 1][column - 1]
          - rowPotential[activeRow] - columnPotential[column];
        if (reducedCost < minimum[column]) {
          minimum[column] = reducedCost;
          previousColumn[column] = column0;
        }
        if (minimum[column] < delta) {
          delta = minimum[column];
          column1 = column;
        }
      }
      for (let column = 0; column <= columnCount; column++) {
        if (used[column]) {
          rowPotential[matchedRow[column]] += delta;
          columnPotential[column] -= delta;
        } else {
          minimum[column] -= delta;
        }
      }
      column0 = column1;
    } while (matchedRow[column0] !== 0);

    do {
      const column1 = previousColumn[column0];
      matchedRow[column0] = matchedRow[column1];
      column0 = column1;
    } while (column0 !== 0);
  }

  const assignment = new Array(rowCount).fill(-1);
  for (let column = 1; column <= columnCount; column++) {
    if (matchedRow[column]) assignment[matchedRow[column] - 1] = column - 1;
  }
  return assignment;
}

function stabilizeTrajectoryAtomIdentity(frames) {
  if (frames.length < 2) {
    return frames.map(frame => ({
      ...frame,
      atoms: frame.atoms.map((atom, index) => ({ ...atom, index, present: 1 })),
    }));
  }

  const slotCount = frames[frames.length - 1].atoms.length;
  if (frames.some(frame => frame.atoms.length > slotCount)) {
    console.warn('Trajectory has more atoms before its final frame; stable-slot matching skipped');
    return frames;
  }

  const stabilized = new Array(frames.length);
  const finalFrame = frames[frames.length - 1];
  stabilized[frames.length - 1] = {
    ...finalFrame,
    atoms: finalFrame.atoms.map((atom, index) => ({ ...atom, index, present: 1 })),
    bonds: finalFrame.bonds.map(bond => ({ ...bond })),
  };

  for (let frameIndex = frames.length - 2; frameIndex >= 0; frameIndex--) {
    const sourceFrame = frames[frameIndex];
    const targetAtoms = stabilized[frameIndex + 1].atoms;
    const costMatrix = sourceFrame.atoms.map((sourceAtom, sourceIndex) => targetAtoms.map((targetAtom, slotIndex) => {
      const dx = sourceAtom.x - targetAtom.x;
      const dy = sourceAtom.y - targetAtom.y;
      const dz = sourceAtom.z - targetAtom.z;
      return dx * dx + dy * dy + dz * dz + slotIndex * 1e-9 + sourceIndex * 1e-12;
    }));
    const sourceToSlot = solveMinimumCostAssignment(costMatrix);
    const oldIndexToSlot = new Map();
    const atoms = targetAtoms.map((targetAtom, slotIndex) => ({
      ...targetAtom,
      index: slotIndex,
      present: 0,
    }));

    sourceFrame.atoms.forEach((sourceAtom, sourceIndex) => {
      const slotIndex = sourceToSlot[sourceIndex];
      if (slotIndex < 0) return;
      atoms[slotIndex] = { ...sourceAtom, index: slotIndex, present: 1 };
      oldIndexToSlot.set(sourceIndex, slotIndex);
    });
    const bonds = sourceFrame.bonds.flatMap(bond => {
      const a1 = oldIndexToSlot.get(bond.a1);
      const a2 = oldIndexToSlot.get(bond.a2);
      return a1 === undefined || a2 === undefined ? [] : [{ ...bond, a1, a2 }];
    });
    stabilized[frameIndex] = { ...sourceFrame, atoms, bonds };
  }
  return stabilized;
}

const LIGAND_GENERATION_STEPS = 33;
const LIGAND_REPAIR_STEPS = 10;
const LIGAND_SAMPLING_STEPS =
  LIGAND_GENERATION_STEPS + LIGAND_REPAIR_STEPS;

function getLigandSamplingPhase(samplingStep) {
  return samplingStep <= LIGAND_GENERATION_STEPS ? 'denoise' : 'repair';
}

function getLigandSamplingStep(progress) {
  const normalizedProgress = Math.max(0, Math.min(1, progress));
  return Math.min(
    LIGAND_SAMPLING_STEPS,
    Math.floor(normalizedProgress * (LIGAND_SAMPLING_STEPS - 1)) + 1
  );
}

// Normalize every source trajectory to the same 43-step diffusion schedule.
// The first Gaussian state and the final molecule are preserved exactly; only
// the intermediate frames are evenly interpolated.
function resampleTrajectoryFrames(frames, targetCount = LIGAND_SAMPLING_STEPS) {
  if (!frames.length || targetCount <= 0) return [];
  if (frames.length === targetCount) return frames;
  if (frames.length === 1 || targetCount === 1) return [frames[0]];

  return Array.from({ length: targetCount }, (_, targetIndex) => {
    const framePosition = targetIndex * (frames.length - 1) / (targetCount - 1);
    const frameIndexA = Math.floor(framePosition);
    const frameIndexB = Math.min(frameIndexA + 1, frames.length - 1);
    const blend = framePosition - frameIndexA;
    const frameA = frames[frameIndexA];
    const frameB = frames[frameIndexB];
    const referenceFrame = blend < 0.5 ? frameA : frameB;
    const atomCount = Math.max(frameA.atoms.length, frameB.atoms.length);
    const atoms = Array.from({ length: atomCount }, (_, atomIndex) => {
      const atomA = frameA.atoms[atomIndex] || frameB.atoms[atomIndex];
      const atomB = frameB.atoms[atomIndex] || atomA;
      return {
        ...atomA,
        index: atomIndex,
        x: THREE.MathUtils.lerp(atomA.x, atomB.x, blend),
        y: THREE.MathUtils.lerp(atomA.y, atomB.y, blend),
        z: THREE.MathUtils.lerp(atomA.z, atomB.z, blend),
        element: blend < 0.5 ? atomA.element : atomB.element,
        present: THREE.MathUtils.lerp(atomA.present ?? 1, atomB.present ?? 1, blend),
      };
    });
    return {
      ...referenceFrame,
      atoms,
      bonds: (referenceFrame.bonds || []).map(bond => ({ ...bond })),
    };
  });
}

// ---- Compute bonds from atom positions (distance-based) ----
function computeBonds(atoms, maxDist = 1.7) {
  const bonds = [];
  for (let i = 0; i < atoms.length; i++) {
    for (let j = i + 1; j < atoms.length; j++) {
      const dx = atoms[i].x - atoms[j].x;
      const dy = atoms[i].y - atoms[j].y;
      const dz = atoms[i].z - atoms[j].z;
      const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
      if (dist < maxDist && dist > 0.3) {
        bonds.push({ a1: i, a2: j, order: 1, length: dist });
      }
    }
  }
  return bonds;
}

// ---- Compute centroid of atoms ----
function computeCentroid(atoms) {
  let cx = 0, cy = 0, cz = 0;
  for (const a of atoms) {
    cx += a.x; cy += a.y; cz += a.z;
  }
  const n = atoms.length;
  return { x: cx/n, y: cy/n, z: cz/n };
}

// ---- Translate all atoms by offset ----
function translateAtoms(atoms, dx, dy, dz) {
  return atoms.map(a => ({ ...a, x: a.x + dx, y: a.y + dy, z: a.z + dz }));
}


export {
  LIGAND_SAMPLING_STEPS,
  computeBonds,
  computeCentroid,
  getElementColor,
  getElementRadius,
  getLigandSamplingPhase,
  getLigandSamplingStep,
  parsePDB,
  parseSDFModel,
  parseSDFTrajectory,
  resampleTrajectoryFrames,
  stabilizeTrajectoryAtomIdentity,
  translateAtoms
};
