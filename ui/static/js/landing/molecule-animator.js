import * as THREE from 'three';
import { smoothstep01 } from './protein-geometry.js';
import { computeBonds, getElementColor, getElementRadius } from './molecule-parser.js';

// ======= MOLECULE ANIMATOR =======

const BOND_HIDDEN_FRAME_COUNT = 25;
const BOND_FADE_FRAME_COUNT = 8;

function getBondReveal(framePosition) {
  if (framePosition < BOND_HIDDEN_FRAME_COUNT) return 0;
  return smoothstep01(
    (framePosition - BOND_HIDDEN_FRAME_COUNT) / BOND_FADE_FRAME_COUNT
  );
}

function getHoldBreathScale(elapsed, duration, phase, motionReduced = false) {
  if (motionReduced || duration <= 0) return 1;
  const holdProgress = Math.max(0, Math.min(1, elapsed / duration));
  const boundaryEnvelope = Math.sin(Math.PI * holdProgress);
  return 1 + Math.sin(phase) * 0.03 * boundaryEnvelope;
}

function findSmallestRingForBond(bonds, bondIndex, maxRingSize = 7) {
  const targetBond = bonds[bondIndex];
  if (!targetBond) return null;

  // Remove this bond, then find the shortest alternate path between its two
  // atoms. The path plus the removed bond is the smallest ring containing it.
  const adjacency = new Map();
  const connect = (from, to) => {
    if (!adjacency.has(from)) adjacency.set(from, []);
    adjacency.get(from).push(to);
  };
  bonds.forEach((bond, index) => {
    if (index === bondIndex) return;
    connect(bond.a1, bond.a2);
    connect(bond.a2, bond.a1);
  });

  const queue = [[targetBond.a1]];
  while (queue.length > 0) {
    const path = queue.shift();
    const current = path[path.length - 1];
    if (path.length >= maxRingSize) continue;
    for (const neighbor of adjacency.get(current) || []) {
      if (neighbor === targetBond.a2) {
        const ring = [...path, neighbor];
        return ring.length >= 3 && ring.length <= maxRingSize ? ring : null;
      }
      if (!path.includes(neighbor)) queue.push([...path, neighbor]);
    }
  }
  return null;
}

function createStableBondLaneSides(frames, bonds) {
  const worldUp = new THREE.Vector3(0, 1, 0);
  const worldRight = new THREE.Vector3(1, 0, 0);
  const ringAtomsByBond = bonds.map((bond, bondIndex) =>
    Math.round(bond.order || 1) > 1
      ? findSmallestRingForBond(bonds, bondIndex)
      : null
  );

  return bonds.map((bond, bondIndex) => {
    const sides = new Array(frames.length);
    let nextSide = null;
    const ringAtoms = ringAtomsByBond[bondIndex];

    // Ring multiple bonds use the vector from the bond midpoint to the shared
    // ring centre. Projecting that vector off the bond axis puts every lane in
    // the same aromatic plane. Parallel transport then damps frame-to-frame
    // rotation without losing the chemically meaningful plane.
    for (let frameIndex = frames.length - 1; frameIndex >= 0; frameIndex--) {
      const frameAtoms = frames[frameIndex].atoms;
      const atomA = frameAtoms[bond.a1];
      const atomB = frameAtoms[bond.a2];
      if (!atomA || !atomB) {
        sides[frameIndex] = nextSide ? nextSide.clone() : worldRight.clone();
        continue;
      }

      const direction = new THREE.Vector3(
        atomB.x - atomA.x,
        atomB.y - atomA.y,
        atomB.z - atomA.z
      ).normalize();
      let transportedSide = null;
      let ringSide = null;

      if (nextSide) {
        transportedSide = nextSide.clone()
          .addScaledVector(direction, -nextSide.dot(direction));
        if (transportedSide.lengthSq() > 1e-8) transportedSide.normalize();
      }

      if (ringAtoms) {
        const ringCenter = new THREE.Vector3();
        let ringAtomCount = 0;
        for (const atomIndex of ringAtoms) {
          const atom = frameAtoms[atomIndex];
          if (!atom) continue;
          ringCenter.add(new THREE.Vector3(atom.x, atom.y, atom.z));
          ringAtomCount++;
        }
        if (ringAtomCount >= 3) {
          ringCenter.multiplyScalar(1 / ringAtomCount);
          const bondMidpoint = new THREE.Vector3(
            (atomA.x + atomB.x) * 0.5,
            (atomA.y + atomB.y) * 0.5,
            (atomA.z + atomB.z) * 0.5
          );
          ringSide = new THREE.Vector3();
          ringSide.copy(ringCenter).sub(bondMidpoint);
          ringSide.addScaledVector(direction, -ringSide.dot(direction));
          if (ringSide.lengthSq() > 1e-8) ringSide.normalize();
          else ringSide = null;
        }
      }

      let side;
      if (ringSide && transportedSide) {
        if (ringSide.dot(transportedSide) < 0) ringSide.negate();
        side = transportedSide.lerp(ringSide, 0.55);
        side.addScaledVector(direction, -side.dot(direction));
      } else if (ringSide) {
        side = ringSide;
      } else if (transportedSide) {
        side = transportedSide;
      } else {
        const anchorAxis = Math.abs(direction.y) < 0.9 ? worldUp : worldRight;
        side = new THREE.Vector3().crossVectors(direction, anchorAxis);
      }

      if (side.lengthSq() < 1e-8) {
        const fallbackAxis = Math.abs(direction.x) < 0.9 ? worldRight : worldUp;
        side.crossVectors(direction, fallbackAxis);
      }
      side.normalize();
      if (nextSide && side.dot(nextSide) < 0) side.negate();
      sides[frameIndex] = side;
      nextSide = side;
    }
    return sides;
  });
}

class MoleculeAnimator {
  constructor(trajectoryFrames, moleculeGroup) {
    this.frames = trajectoryFrames;
    this.molGroup = moleculeGroup;
    this.progress = 0; // 0 = first frame (noise), 1 = last frame (final)
    this.framePosition = 0;
    this.currentFrameIndex = 0;
    this.frameInterp = 0; // 0..1 between current and next
    this.visibility = 1;

    // Store base positions for interpolation
    this.atomMeshes = moleculeGroup.userData.atomMeshes;
    this.bondMeshes = moleculeGroup.userData.bondMeshes;

    // Pre-allocate reusable Vector3 objects (avoid GC pressure)
    this._vStart = new THREE.Vector3();
    this._vEnd = new THREE.Vector3();
    this._vDir = new THREE.Vector3();
    this._vSide = new THREE.Vector3();
    this._vUp = new THREE.Vector3(0, 1, 0);
    this._vRight = new THREE.Vector3(1, 0, 0);
    this._finishWhite = new THREE.Color(0xffffff);
    this._atomColorA = new THREE.Color();
    this._atomColorB = new THREE.Color();
    this._atomColorCurrent = new THREE.Color();

    // Use bond topology from SDF file (more accurate than distance-based)
    const lastFrame = this.frames[this.frames.length - 1];
    this.refBonds = lastFrame.bonds && lastFrame.bonds.length > 0
      ? lastFrame.bonds.map(b => ({ ...b, length: 0 }))
      : computeBonds(lastFrame.atoms, 2.0);
    this._bondLaneSides = createStableBondLaneSides(this.frames, this.refBonds);

    // Pre-compute original cylinder heights
    for (const cyl of this.bondMeshes) {
      cyl.geometry.computeBoundingBox();
      cyl.userData.origHeight = cyl.geometry.boundingBox.max.y - cyl.geometry.boundingBox.min.y;
      cyl.userData.baseOpacity = cyl.material?.opacity ?? 0.85;
      if (cyl.material?.color) cyl.userData.baseColor = cyl.material.color.clone();
      if (cyl.userData.outlineMesh) {
        cyl.userData.outlineMesh.userData.baseOpacity =
          cyl.userData.outlineMesh.material?.opacity ?? 0.5;
      }
    }
    for (let atomIndex = 0; atomIndex < this.atomMeshes.length; atomIndex++) {
      const atom = this.atomMeshes[atomIndex];
      const finalAtom = lastFrame.atoms[atomIndex];
      atom.userData.baseElementRadius = getElementRadius(finalAtom?.element || 'C');
      if (atom.material?.color) atom.userData.baseColor = atom.material.color.clone();
      if (atom.userData.outlineMesh) {
        atom.userData.outlineMesh.userData.baseScale = atom.userData.outlineMesh.scale.x;
        atom.userData.outlineMesh.userData.baseOpacity =
          atom.userData.outlineMesh.material?.opacity ?? 0.6;
      }
    }

    console.log('[MoleculeAnimator]', this.frames.length, 'frames,', this.refBonds.length, 'bonds');
  }

  setProgress(t) {
    this.progress = Math.max(0, Math.min(1, t));
    // Map progress to frame index
    this.framePosition = this.progress * (this.frames.length - 1);
    this.currentFrameIndex = Math.floor(this.framePosition);
    this.frameInterp = this.framePosition - this.currentFrameIndex;
    this.update();
  }

  setVisibility(value) {
    this.visibility = Math.max(0, Math.min(1, value));
    this.update();
  }

  update() {
    const i = this.currentFrameIndex;
    const j = Math.min(i + 1, this.frames.length - 1);
    const t = this.frameInterp;

    const frameA = this.frames[i];
    const frameB = this.frames[j];
    const atomsA = frameA.atoms;
    const atomsB = frameB.atoms;

    // Diffusion changes geometry, not the physical appearance of an atom.
    // Keep every present atom at its final element opacity throughout the
    // trajectory; visibility is reserved for whole-molecule cross-fades.
    const atomOpacity = this.visibility;
    // Transparent meshes must stop writing depth as soon as a cross-fade
    // starts. Otherwise they keep punching sphere-shaped holes through the
    // translucent protein and reveal the white page background.
    const ligandWritesDepth = this.visibility >= 0.999;
    const completionLighten = smoothstep01((this.progress - 0.75) / 0.25) * 0.2;

    // Update atom positions via interpolation
    for (let ai = 0; ai < this.atomMeshes.length && ai < atomsA.length && ai < atomsB.length; ai++) {
      const mesh = this.atomMeshes[ai];
      const aA = atomsA[ai];
      const aB = atomsB[ai];
      mesh.position.x = aA.x + (aB.x - aA.x) * t;
      mesh.position.y = aA.y + (aB.y - aA.y) * t;
      mesh.position.z = aA.z + (aB.z - aA.z) * t;

      // Atom identity is the stable SDF index. When its predicted element
      // changes between frames, morph color and radius on that same mesh;
      // never remap it to another atom position just to match an element.
      const elementBlend = smoothstep01(t);
      const presence = THREE.MathUtils.lerp(aA.present ?? 1, aB.present ?? 1, elementBlend);
      const radiusA = getElementRadius(aA.element);
      const radiusB = getElementRadius(aB.element);
      const baseElementRadius = mesh.userData.baseElementRadius || radiusB || 0.4;
      const interpolatedRadius = THREE.MathUtils.lerp(radiusA, radiusB, elementBlend);
      mesh.scale.setScalar(interpolatedRadius / baseElementRadius);
      mesh.userData.currentElement = elementBlend < 0.5 ? aA.element : aB.element;

      // Update outline position too
      if (mesh.userData.outlineMesh) {
        mesh.userData.outlineMesh.position.copy(mesh.position);
        const outlineBaseScale = mesh.userData.outlineMesh.userData.baseScale || 1;
        mesh.userData.outlineMesh.scale.setScalar(
          outlineBaseScale * interpolatedRadius / baseElementRadius
        );
        if (mesh.userData.outlineMesh.material) {
          const renderedOutlineOpacity =
            mesh.userData.outlineMesh.userData.baseOpacity * atomOpacity *
            presence;
          mesh.userData.outlineMesh.visible = renderedOutlineOpacity > 0.001;
          mesh.userData.outlineMesh.material.opacity = renderedOutlineOpacity;
          mesh.userData.outlineMesh.material.depthWrite = false;
        }
      }

      // The atom stays opaque and element-coloured at every diffusion step.
      // `presence` only handles a genuinely added/removed atom slot.
      if (mesh.material) {
        const renderedAtomOpacity = atomOpacity * presence;
        mesh.visible = renderedAtomOpacity > 0.001;
        mesh.material.opacity = renderedAtomOpacity;
        mesh.material.depthWrite = ligandWritesDepth && presence >= 0.999;
        if (mesh.material.color) {
          this._atomColorA.setHex(getElementColor(aA.element));
          this._atomColorB.setHex(getElementColor(aB.element));
          this._atomColorCurrent.copy(this._atomColorA)
            .lerp(this._atomColorB, elementBlend);
          mesh.material.color.copy(this._atomColorCurrent);
        }
      }
    }

    // Keep the first 25 trajectory frames atom-only, then fade bonds in.
    const bondReveal = this.framePosition < BOND_HIDDEN_FRAME_COUNT
      ? 0
      : getBondReveal(this.framePosition);
    const bondsVisible = bondReveal > 0.001 && this.visibility > 0.001;

    // Update bond positions and sizes
    const vStart = this._vStart;
    const vEnd = this._vEnd;
    const vDir = this._vDir;
    const vSide = this._vSide;
    const vUp = this._vUp;
    const vRight = this._vRight;

    for (let bi = 0; bi < this.bondMeshes.length; bi++) {
      const cyl = this.bondMeshes[bi];
      const bondIndex = cyl.userData.bondIndex ?? bi;
      const bond = this.refBonds[bondIndex];
      if (!bond) continue;
      const a1 = bond.a1;
      const a2 = bond.a2;
      if (a1 >= atomsA.length || a2 >= atomsA.length) continue;
      if (!atomsA[a1] || !atomsA[a2] || !atomsB[a1] || !atomsB[a2]) continue;

      // Interpolate positions using pre-allocated vectors
      vStart.set(
        atomsA[a1].x + (atomsB[a1].x - atomsA[a1].x) * t,
        atomsA[a1].y + (atomsB[a1].y - atomsA[a1].y) * t,
        atomsA[a1].z + (atomsB[a1].z - atomsA[a1].z) * t
      );
      vEnd.set(
        atomsA[a2].x + (atomsB[a2].x - atomsA[a2].x) * t,
        atomsA[a2].y + (atomsB[a2].y - atomsA[a2].y) * t,
        atomsA[a2].z + (atomsB[a2].z - atomsA[a2].z) * t
      );
      vDir.subVectors(vEnd, vStart);
      const length = vDir.length();

      if (length < 0.1 || !bondsVisible) {
        cyl.visible = false;
        if (cyl.userData.outlineMesh) cyl.userData.outlineMesh.visible = false;
        continue;
      }

      // Keep double/triple-bond lanes parallel in a stable local plane.
      vDir.normalize();
      const laneCount = cyl.userData.laneCount || 1;
      if (laneCount > 1) {
        const sideA = this._bondLaneSides[bondIndex]?.[i];
        const sideB = this._bondLaneSides[bondIndex]?.[j] || sideA;
        if (sideA && sideB) {
          vSide.copy(sideA).lerp(sideB, t);
          vSide.addScaledVector(vDir, -vSide.dot(vDir));
          if (vSide.lengthSq() < 1e-8) vSide.copy(sideA);
          vSide.normalize();
        } else {
          const fallbackAxis = Math.abs(vDir.y) < 0.9 ? vUp : vRight;
          vSide.crossVectors(vDir, fallbackAxis).normalize();
        }
        const laneOffset = (
          (cyl.userData.laneIndex || 0) - (laneCount - 1) * 0.5
        ) * (cyl.userData.laneSpacing || 0);
        vStart.addScaledVector(vSide, laneOffset);
        vEnd.addScaledVector(vSide, laneOffset);
      }

      cyl.visible = true;
      if (cyl.userData.outlineMesh) cyl.userData.outlineMesh.visible = true;

      cyl.position.copy(vStart).add(vEnd).multiplyScalar(0.5);
      cyl.quaternion.setFromUnitVectors(vUp, vDir);

      // Scale Y to match bond length
      const origHeight = cyl.userData.origHeight || 1;
      cyl.scale.y = length / origHeight;

      // Update outline
      if (cyl.userData.outlineMesh) {
        const ol = cyl.userData.outlineMesh;
        ol.position.copy(cyl.position);
        ol.quaternion.copy(cyl.quaternion);
        ol.scale.x = cyl.scale.x * 1.2;
        ol.scale.y = cyl.scale.y;
        ol.scale.z = cyl.scale.z * 1.2;
      }

      // Bond opacity grows smoothly after the atom-only opening.
      if (cyl.material) {
        cyl.material.opacity = cyl.userData.baseOpacity * bondReveal * this.visibility;
        cyl.material.depthWrite = ligandWritesDepth && bondReveal >= 0.999;
        if (cyl.material.color && cyl.userData.baseColor) {
          cyl.material.color.copy(cyl.userData.baseColor)
            .lerp(this._finishWhite, completionLighten);
        }
      }
      if (cyl.userData.outlineMesh?.material) {
        cyl.userData.outlineMesh.material.opacity =
          cyl.userData.outlineMesh.userData.baseOpacity * bondReveal * this.visibility;
        cyl.userData.outlineMesh.material.depthWrite = false;
      }
    }
  }
}





export {
  MoleculeAnimator,
  getHoldBreathScale
};
