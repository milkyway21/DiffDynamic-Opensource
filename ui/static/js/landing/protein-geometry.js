import * as THREE from 'three';

// ======= PROTEIN CARTOON RIBBON RENDERING =======

// Backbone dihedral angle in degrees (used by the simplified DSSP classifier).
function backboneDihedral(p0, p1, p2, p3) {
  const b0 = p0.clone().sub(p1);
  const b1 = p2.clone().sub(p1).normalize();
  const b2 = p3.clone().sub(p2);
  const v = b0.clone().addScaledVector(b1, -b0.dot(b1));
  const w = b2.clone().addScaledVector(b1, -b2.dot(b1));
  return THREE.MathUtils.radToDeg(Math.atan2(new THREE.Vector3().crossVectors(b1, v).dot(w), v.dot(w)));
}

function residuesAreContinuous(a, b) {
  return Boolean(
    a && b &&
    a.chainId === b.chainId &&
    b.resSeq - a.resSeq === 1 &&
    a.CA.distanceTo(b.CA) < 5.0
  );
}

// Infer the raw scientific labels. Visual smoothing is deliberately separate so
// the original φ/ψ classification remains available for diagnostics.
function inferSecondaryStructure(residues) {
  const types = new Array(residues.length).fill('loop');

  for (let i = 1; i < residues.length - 1; i++) {
    const prev = residues[i - 1];
    const curr = residues[i];
    const next = residues[i + 1];
    if (prev.chainId !== curr.chainId || curr.chainId !== next.chainId) continue;
    if (curr.resSeq - prev.resSeq !== 1 || next.resSeq - curr.resSeq !== 1) continue;
    if (!prev.C || !curr.N || !curr.CA || !curr.C || !next.N) continue;

    const phi = backboneDihedral(prev.C, curr.N, curr.CA, curr.C);
    const psi = backboneDihedral(curr.N, curr.CA, curr.C, next.N);

    if (phi >= -100 && phi <= -30 && psi >= -85 && psi <= 10) {
      types[i] = 'helix';
    } else if (phi >= -180 && phi <= -60 && (psi >= 70 || psi <= -120)) {
      types[i] = 'sheet';
    }
  }

  return types;
}

function buildStructureRuns(residues, types, offset = 0) {
  const runs = [];
  let start = 0;
  while (start < types.length) {
    let end = start;
    while (
      end + 1 < types.length &&
      types[end + 1] === types[start] &&
      residuesAreContinuous(residues[end], residues[end + 1])
    ) end++;
    runs.push({ type: types[start], start: start + offset, end: end + offset });
    start = end + 1;
  }
  return runs;
}

function createVisualStructureTypes(residues, rawTypes) {
  const visualTypes = rawTypes.slice();

  // One conservative pass: merge only 1–2 loop residues enclosed by the same
  // structure, and never cross a chain, residue gap, or CA-distance break.
  for (let i = 1; i < visualTypes.length - 1; i++) {
    if (rawTypes[i] !== 'loop') continue;
    let gapEnd = i;
    while (gapEnd + 1 < rawTypes.length && rawTypes[gapEnd + 1] === 'loop' && gapEnd - i < 1) gapEnd++;
    const left = i - 1;
    const right = gapEnd + 1;
    if (
      right < rawTypes.length &&
      rawTypes[left] === rawTypes[right] &&
      rawTypes[left] !== 'loop'
    ) {
      let continuous = true;
      for (let k = left; k < right; k++) {
        if (!residuesAreContinuous(residues[k], residues[k + 1])) continuous = false;
      }
      if (continuous) {
        for (let k = i; k <= gapEnd; k++) visualTypes[k] = rawTypes[left];
      }
    }
    i = gapEnd;
  }

  // Enforce minimum PyMOL-like visual lengths after the single merge pass.
  for (const [type, minLength] of [['helix', 4], ['sheet', 3]]) {
    for (const run of buildStructureRuns(residues, visualTypes)) {
      if (run.type === type && run.end - run.start + 1 < minLength) {
        for (let i = run.start; i <= run.end; i++) visualTypes[i] = 'loop';
      }
    }
  }

  return visualTypes;
}

function splitContinuousResidueChains(residues) {
  const chains = [];
  let start = 0;
  while (start < residues.length) {
    let end = start;
    while (end + 1 < residues.length && residuesAreContinuous(residues[end], residues[end + 1])) end++;
    chains.push({ start, end });
    start = end + 1;
  }
  return chains;
}

function countStructureTypes(types) {
  return types.reduce((counts, type) => {
    counts[type] = (counts[type] || 0) + 1;
    return counts;
  }, {});
}

function smoothstep01(value) {
  const t = THREE.MathUtils.clamp(value, 0, 1);
  return t * t * (3 - 2 * t);
}

function smootherstep01(value) {
  const t = THREE.MathUtils.clamp(value, 0, 1);
  return t * t * t * (t * (t * 6 - 15) + 10);
}

// PyMOL 3.x cartoon defaults and interpolation. The web renderer keeps a
// higher 12-sample density, but uses PyMOL's actual curve and helix-frame math.
// Sources:
// https://github.com/schrodinger/pymol-open-source/blob/master/layer1/SettingInfo.h
// https://github.com/schrodinger/pymol-open-source/blob/master/layer2/RepCartoon.cpp#L2158-L2289
// https://github.com/schrodinger/pymol-open-source/blob/master/layer2/RepCartoon.cpp#L3404-L3476
// https://github.com/schrodinger/pymol-open-source/blob/master/layer2/RepCartoon.cpp#L3648-L3755
const PYMOL_CARTOON_POWER = 2.0;
const PYMOL_CARTOON_POWER_B = 0.52;
const PYMOL_CARTOON_THROW = 1.35;
const PYMOL_CARTOON_REFINE = 5;

function pymolSmooth(value, power) {
  if (value <= 0) return 0;
  if (value >= 1) return 1;
  if (value <= 0.5) return 0.5 * Math.pow(2 * value, power);
  return 1 - 0.5 * Math.pow(2 * (1 - value), power);
}

function createPyMolResidueTangents(points) {
  const directions = [];
  for (let index = 0; index < points.length - 1; index++) {
    const direction = points[index + 1].clone().sub(points[index]);
    if (direction.lengthSq() < 1e-10 && directions.length) direction.copy(directions[directions.length - 1]);
    else direction.normalize();
    directions.push(direction);
  }
  return points.map((_, index) => {
    if (index === 0) return directions[0].clone();
    if (index === points.length - 1) return directions[directions.length - 1].clone();
    const tangent = directions[index - 1].clone().add(directions[index]);
    return tangent.lengthSq() > 1e-10 ? tangent.normalize() : directions[index].clone();
  });
}

function createPeptideOrientations(residues, tangents, structureTypes) {
  let parity = 1;
  return residues.map((residue, index) => {
    if (structureTypes[index] === 'helix') parity = 0;
    else if (structureTypes[index] === 'sheet') parity = parity ? 0 : 1;
    else parity = 0;

    let orientation = null;
    if (residue.N && residue.C && residue.O) {
      // PyMOL PASS1: normalize(N-C) x normalize(N-O), including the beta-sheet
      // parity flip before normal refinement.
      const carbonDirection = residue.N.clone().sub(residue.C).normalize();
      const oxygenDirection = residue.N.clone().sub(residue.O).normalize();
      orientation = new THREE.Vector3().crossVectors(carbonDirection, oxygenDirection);
      if (parity) orientation.negate();
    }
    if (!orientation || orientation.lengthSq() < 1e-10) {
      const reference = Math.abs(tangents[index].y) < 0.92
        ? new THREE.Vector3(0, 1, 0)
        : new THREE.Vector3(1, 0, 0);
      orientation = new THREE.Vector3().crossVectors(tangents[index], reference);
    }
    orientation.addScaledVector(tangents[index], -orientation.dot(tangents[index]));
    return orientation.lengthSq() > 1e-10 ? orientation.normalize() : new THREE.Vector3(1, 0, 0);
  });
}

function setPyMolHelixOrientation(orientations, tangents, index, helixAxis) {
  if (index < 0 || index >= orientations.length) return;
  const orientation = new THREE.Vector3().crossVectors(helixAxis, tangents[index]);
  if (orientation.lengthSq() > 1e-10) orientations[index].copy(orientation.normalize());
}

function applyPyMolRoundHelixOrientations(points, structureTypes, tangents, orientations) {
  let history = [];
  let previousAxisCenter = null;
  let axisCenterCount = 0;

  for (let index = 0; index < points.length; index++) {
    if (structureTypes[index] !== 'helix') {
      // This is PyMOL's short-helix fallback at early termination: average up
      // to four backward directions, then cross that axis with each tangent.
      if (axisCenterCount < 2 && history.length >= 2) {
        const helixAxis = history[0].clone().sub(points[index]).normalize();
        for (let offset = 1; offset < Math.min(4, history.length); offset++) {
          helixAxis.add(history[offset].clone().sub(history[offset - 1]).normalize());
        }
        if (helixAxis.lengthSq() > 1e-10) {
          helixAxis.normalize();
          for (let offset = 1; offset <= Math.min(4, history.length); offset++) {
            setPyMolHelixOrientation(orientations, tangents, index - offset, helixAxis);
          }
          if (history.length >= 4 && orientations[index - 3].dot(orientations[index - 4]) < -0.8) {
            orientations[index - 4].negate();
          }
        }
      }
      history = [];
      previousAxisCenter = null;
      axisCenterCount = 0;
      continue;
    }

    history.unshift(points[index]);
    if (history.length > 5) history.length = 5;
    if (history.length < 4) continue;

    // RepCartoonComputeRoundHelices uses these exact four-CA coefficients.
    const axisCenter = new THREE.Vector3()
      .addScaledVector(history[0], 0.2130)
      .addScaledVector(history[3], 0.2130)
      .addScaledVector(history[1], 0.2870)
      .addScaledVector(history[2], 0.2870);

    if (axisCenterCount > 0) {
      const helixAxis = previousAxisCenter.clone().sub(axisCenter);
      if (helixAxis.lengthSq() > 1e-10) {
        helixAxis.normalize();
        setPyMolHelixOrientation(orientations, tangents, index, helixAxis);
        setPyMolHelixOrientation(orientations, tangents, index - 1, helixAxis);
        setPyMolHelixOrientation(orientations, tangents, index - 2, helixAxis);
        if (axisCenterCount === 1) {
          setPyMolHelixOrientation(orientations, tangents, index - 3, helixAxis);
          setPyMolHelixOrientation(orientations, tangents, index - 4, helixAxis);
        }
      }
    }
    previousAxisCenter = axisCenter;
    axisCenterCount++;
  }
}

function createPyMolSegmentDirections(points) {
  const directions = [];
  for (let index = 0; index < points.length - 1; index++) {
    const direction = points[index + 1].clone().sub(points[index]);
    if (direction.lengthSq() < 1e-10 && directions.length) direction.copy(directions[directions.length - 1]);
    else direction.normalize();
    directions.push(direction);
  }
  return directions;
}

function refinePyMolOrientationVectors(orientations, structureTypes, tangents, points) {
  const directions = createPyMolSegmentDirections(points);

  // RepCartoonRefineNormals first makes every interior orientation orthogonal
  // to its trace tangent.
  for (let index = 1; index < orientations.length - 1; index++) {
    orientations[index].addScaledVector(tangents[index], -orientations[index].dot(tangents[index]));
    if (orientations[index].lengthSq() > 1e-10) orientations[index].normalize();
  }

  // PyMOL tests original/inverted candidates for non-helices only. Helices are
  // never inverted because that would swap their inside and outside faces.
  const candidates = orientations.map((orientation, index) => [
    orientation.clone(),
    structureTypes[index] === 'helix' ? orientation.clone() : orientation.clone().negate(),
  ]);
  for (let index = 1; index < orientations.length - 1; index++) {
    const chainDirection = directions[index - 1];
    const previous = orientations[index - 1].clone()
      .addScaledVector(chainDirection, -orientations[index - 1].dot(chainDirection))
      .normalize();
    const original = candidates[index][0].clone()
      .addScaledVector(chainDirection, -candidates[index][0].dot(chainDirection))
      .normalize();
    const inverted = candidates[index][1].clone()
      .addScaledVector(chainDirection, -candidates[index][1].dot(chainDirection))
      .normalize();
    orientations[index].copy(previous.dot(inverted) > previous.dot(original)
      ? candidates[index][1]
      : candidates[index][0]);
  }

  // Then soften only genuine orientation kinks (the official -0.10 threshold)
  // instead of globally damping every ribbon angle.
  const softened = orientations.map(orientation => orientation.clone());
  for (let index = 1; index < orientations.length - 1; index++) {
    const kink = orientations[index].dot(orientations[index + 1])
      * orientations[index].dot(orientations[index - 1]);
    if (kink >= -0.10) continue;
    const neighborAverage = orientations[index + 1].clone()
      .add(orientations[index - 1])
      .addScaledVector(orientations[index], 0.001)
      .addScaledVector(tangents[index], -tangents[index].dot(
        orientations[index + 1].clone()
          .add(orientations[index - 1])
          .addScaledVector(orientations[index], 0.001)
      ));
    if (neighborAverage.lengthSq() < 1e-10) continue;
    neighborAverage.normalize();
    const target = orientations[index].clone();
    if (orientations[index].dot(neighborAverage) < 0) target.sub(neighborAverage);
    else target.add(neighborAverage);
    target.normalize();
    softened[index].lerp(target, Math.min(1, 2 * (-0.10 - kink))).normalize();
  }
  for (let index = 1; index < orientations.length - 1; index++) {
    orientations[index].copy(softened[index]);
  }
}

function flattenPyMolSheets(points, orientations, structureTypes) {
  const PYMOL_CARTOON_FLAT_CYCLES = 4;
  let runStart = 0;
  while (runStart < structureTypes.length) {
    if (structureTypes[runStart] !== 'sheet') {
      runStart++;
      continue;
    }
    let runEnd = runStart;
    while (runEnd + 1 < structureTypes.length && structureTypes[runEnd + 1] === 'sheet') runEnd++;
    for (let cycle = 0; cycle < PYMOL_CARTOON_FLAT_CYCLES; cycle++) {
      const nextPoints = points.map(point => point.clone());
      const nextOrientations = orientations.map(orientation => orientation.clone());
      for (let index = runStart + 1; index < runEnd; index++) {
        nextPoints[index].copy(points[index - 1]).add(points[index]).add(points[index + 1]).multiplyScalar(1 / 3);
        nextOrientations[index].copy(orientations[index - 1])
          .add(orientations[index]).add(orientations[index + 1]).multiplyScalar(1 / 3);
      }
      for (let index = runStart + 1; index < runEnd; index++) {
        points[index].copy(nextPoints[index]);
        orientations[index].copy(nextOrientations[index]);
        const tangent = points[index + 1].clone().sub(points[index - 1]).normalize();
        orientations[index].addScaledVector(tangent, -orientations[index].dot(tangent));
        if (orientations[index].lengthSq() > 1e-10) orientations[index].normalize();
      }
    }
    runStart = runEnd + 1;
  }
}

function refinePyMolCartoonSegment(points, startIndex, endIndex, startOrientation, endOrientation) {
  const refinementAxis = new THREE.Vector3().crossVectors(startOrientation, endOrientation);
  if (refinementAxis.lengthSq() < 1e-10 || endIndex - startIndex < 2) return;
  refinementAxis.normalize();

  for (let cycle = 0; cycle < PYMOL_CARTOON_REFINE; cycle++) {
    const refined = [];
    for (let index = startIndex + 1; index < endIndex; index++) {
      const previousProjection = refinementAxis.dot(points[index - 1]);
      const currentProjection = refinementAxis.dot(points[index]);
      const nextProjection = refinementAxis.dot(points[index + 1]);
      refined.push(points[index].clone().addScaledVector(
        refinementAxis,
        (previousProjection + nextProjection) * 0.5 - currentProjection
      ));
    }
    for (let offset = 0; offset < refined.length; offset++) points[startIndex + 1 + offset].copy(refined[offset]);
  }
}

function createPyMolCartoonSamples(residues, structureTypes, samplesPerResidue) {
  const residuePoints = residues.map(residue => residue.CA.clone());
  let residueTangents = createPyMolResidueTangents(residuePoints);
  const residueOrientations = createPeptideOrientations(residues, residueTangents, structureTypes);
  applyPyMolRoundHelixOrientations(residuePoints, structureTypes, residueTangents, residueOrientations);
  refinePyMolOrientationVectors(residueOrientations, structureTypes, residueTangents, residuePoints);
  flattenPyMolSheets(residuePoints, residueOrientations, structureTypes);
  residueTangents = createPyMolResidueTangents(residuePoints);

  const points = [residuePoints[0].clone()];
  const orientations = [residueOrientations[0].clone()];
  for (let residueIndex = 0; residueIndex < residuePoints.length - 1; residueIndex++) {
    const startIndex = points.length - 1;
    const startPoint = residuePoints[residueIndex];
    const endPoint = residuePoints[residueIndex + 1];
    const startTangent = residueTangents[residueIndex];
    const endTangent = residueTangents[residueIndex + 1];
    const startOrientation = residueOrientations[residueIndex];
    const endOrientation = residueOrientations[residueIndex + 1];
    const deviation = PYMOL_CARTOON_THROW * startPoint.distanceTo(endPoint);

    for (let sampleIndex = 1; sampleIndex <= samplesPerResidue; sampleIndex++) {
      const rawFraction = sampleIndex / samplesPerResidue;
      const fraction = pymolSmooth(rawFraction, PYMOL_CARTOON_POWER);
      const inverse = 1 - fraction;
      const forwardBias = pymolSmooth(fraction, PYMOL_CARTOON_POWER_B);
      const backwardBias = pymolSmooth(inverse, PYMOL_CARTOON_POWER_B);
      const displacement = deviation * forwardBias * backwardBias;
      const point = startPoint.clone().multiplyScalar(inverse)
        .addScaledVector(endPoint, fraction)
        .addScaledVector(startTangent, displacement * backwardBias)
        .addScaledVector(endTangent, -displacement * forwardBias);
      points.push(point);

      if (sampleIndex === samplesPerResidue) {
        orientations.push(endOrientation.clone());
      } else {
        const orientation = startOrientation.clone().multiplyScalar(inverse * forwardBias)
          .addScaledVector(endOrientation, fraction * backwardBias);
        orientations.push(orientation.lengthSq() > 1e-10 ? orientation.normalize() : startOrientation.clone());
      }
    }
    refinePyMolCartoonSegment(
      points,
      startIndex,
      points.length - 1,
      startOrientation,
      endOrientation
    );
  }
  return { points, orientations };
}

function createPyMolCartoonFrames(samples) {
  const { points, orientations } = samples;
  const tangents = createPyMolResidueTangents(points);
  const frameNormals = [];
  const frameBinormals = [];

  for (let index = 0; index < points.length; index++) {
    const tangent = tangents[index];
    const normal = orientations[index].clone()
      .addScaledVector(tangent, -orientations[index].dot(tangent));
    if (normal.lengthSq() < 1e-10) {
      const reference = Math.abs(tangent.y) < 0.92
        ? new THREE.Vector3(0, 1, 0)
        : new THREE.Vector3(1, 0, 0);
      normal.crossVectors(tangent, reference);
    }
    normal.normalize();
    const binormal = new THREE.Vector3().crossVectors(tangent, normal).normalize();
    frameNormals.push(normal);
    frameBinormals.push(binormal);
  }
  return { points, tangents, frameNormals, frameBinormals };
}

// Default PyMOL cartoon cross-section settings (SettingInfo.h). ExtrudeOval
// treats oval length/width as radii; ExtrudeRectangle places vertices at the
// settings times cos/sin(pi/4), hence the sqrt(2) full extents below.
const PYMOL_CARTOON_OVAL_LENGTH = 1.35;
const PYMOL_CARTOON_OVAL_WIDTH = 0.25;
const PYMOL_CARTOON_RECT_LENGTH = 1.40;
const PYMOL_CARTOON_RECT_WIDTH = 0.40;
const PYMOL_CARTOON_LOOP_RADIUS = 0.20;
const PROTEIN_LIGHT_BLUE = 0xaed6ee;
const PROTEIN_LIGHT_BLUE_DEEP = 0x79b5dc;

function getStructureProfile(run, residuePosition, palette) {
  if (run.type === 'helix') {
    return {
      width: PYMOL_CARTOON_OVAL_LENGTH * 2,
      thickness: PYMOL_CARTOON_OVAL_WIDTH * 2,
      exponent: 2,
      color: palette.helix.clone(),
    };
  }
  if (run.type === 'sheet') {
    const runLength = Math.max(1, run.end - run.start);
    const t = THREE.MathUtils.clamp((residuePosition - run.start) / runLength, 0, 1);
    let widthScale = 1;
    if (t > 0.82) widthScale = THREE.MathUtils.lerp(1.48, 0.08, smoothstep01((t - 0.82) / 0.18));
    else if (t > 0.70) widthScale = THREE.MathUtils.lerp(1, 1.48, smoothstep01((t - 0.70) / 0.12));
    const tipBlend = t > 0.82 ? smoothstep01((t - 0.82) / 0.18) : 0;
    return {
      width: Math.SQRT2 * PYMOL_CARTOON_RECT_LENGTH * widthScale,
      thickness: Math.SQRT2 * PYMOL_CARTOON_RECT_WIDTH,
      exponent: THREE.MathUtils.lerp(12, 4, tipBlend),
      color: palette.sheet.clone(),
    };
  }
  const loopDiameter = PYMOL_CARTOON_LOOP_RADIUS * 2;
  return { width: loopDiameter, thickness: loopDiameter, exponent: 2, color: palette.loop.clone() };
}

function interpolateProfiles(a, b, amount) {
  return {
    width: THREE.MathUtils.lerp(a.width, b.width, amount),
    thickness: THREE.MathUtils.lerp(a.thickness, b.thickness, amount),
    exponent: THREE.MathUtils.lerp(a.exponent, b.exponent, amount),
    color: a.color.clone().lerp(b.color, amount),
  };
}

function createContinuousCartoonGeometry(residues, structureTypes, options = {}) {
  const segmentsPerResidue = options.segmentsPerResidue || 12;
  const ribbonWidthScale = options.ribbonWidthScale || 1;
  const crossSectionSides = options.crossSectionSides || 24;
  const transitionWidth = 1.75;
  const HELIX_LOOP_TRANSITION_HALF_WIDTH = 2.75;
  const ringVertexCount = crossSectionSides + 1;
  const frames = createPyMolCartoonFrames(
    createPyMolCartoonSamples(residues, structureTypes, segmentsPerResidue)
  );
  const segments = frames.points.length - 1;

  const palette = {
    helix: new THREE.Color(options.helixColor || 0x4f95cf),
    sheet: new THREE.Color(options.sheetColor || 0x78b4de),
    loop: new THREE.Color(options.loopColor || 0x3e7fb5),
  };
  const runs = buildStructureRuns(residues, structureTypes);
  const boundaries = runs.slice(0, -1).map((run, index) => ({
    position: (run.end + runs[index + 1].start) * 0.5,
    left: run,
    right: runs[index + 1],
  }));

  const positions = [];
  const normals = [];
  const colors = [];
  const uvs = [];
  const waveCoordinates = [];
  const profiles = [];
  let runIndex = 0;

  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const residuePosition = t * (residues.length - 1);
    while (runIndex + 1 < runs.length && residuePosition > runs[runIndex].end + 0.5) runIndex++;
    let profile = getStructureProfile(runs[runIndex], residuePosition, palette);

    let closestBoundary = null;
    let closestDistance = Infinity;
    for (const boundary of boundaries) {
      const distance = Math.abs(residuePosition - boundary.position);
      const boundaryConnectsHelixAndLoop =
        (boundary.left.type === 'helix' && boundary.right.type === 'loop') ||
        (boundary.left.type === 'loop' && boundary.right.type === 'helix');
      const boundaryTransitionWidth = boundaryConnectsHelixAndLoop
        ? HELIX_LOOP_TRANSITION_HALF_WIDTH
        : transitionWidth;
      if (distance <= boundaryTransitionWidth && distance < closestDistance) {
        closestBoundary = { ...boundary, transitionWidth: boundaryTransitionWidth };
        closestDistance = distance;
      }
    }
    if (closestBoundary) {
      const localTransitionWidth = closestBoundary.transitionWidth;
      const blend = smootherstep01(
        (residuePosition - (closestBoundary.position - localTransitionWidth)) /
        (localTransitionWidth * 2)
      );
      profile = interpolateProfiles(
        getStructureProfile(closestBoundary.left, residuePosition, palette),
        getStructureProfile(closestBoundary.right, residuePosition, palette),
        blend
      );
    }
    profile.width *= ribbonWidthScale;
    profiles.push(profile);

    const center = frames.points[i];
    const frameNormal = frames.frameNormals[i];
    const frameBinormal = frames.frameBinormals[i];
    const halfWidth = profile.width * 0.5;
    const halfThickness = profile.thickness * 0.5;
    for (let side = 0; side <= crossSectionSides; side++) {
      const theta = (side / crossSectionSides) * Math.PI * 2;
      const cos = Math.cos(theta);
      const sin = Math.sin(theta);
      const x = Math.sign(cos) * Math.pow(Math.abs(cos), 2 / profile.exponent);
      const y = Math.sign(sin) * Math.pow(Math.abs(sin), 2 / profile.exponent);
      const vertex = center.clone()
        .addScaledVector(frameNormal, x * halfWidth)
        .addScaledVector(frameBinormal, y * halfThickness);
      const normalX = Math.sign(x) * Math.pow(Math.abs(x), profile.exponent - 1) / halfWidth;
      const normalY = Math.sign(y) * Math.pow(Math.abs(y), profile.exponent - 1) / halfThickness;
      const surfaceNormal = frameNormal.clone().multiplyScalar(normalX)
        .addScaledVector(frameBinormal, normalY).normalize();
      positions.push(vertex.x, vertex.y, vertex.z);
      normals.push(surfaceNormal.x, surfaceNormal.y, surfaceNormal.z);
      colors.push(profile.color.r, profile.color.g, profile.color.b);
      uvs.push(t, side / crossSectionSides);
      waveCoordinates.push((t + (options.waveOffset || 0)) % 1);
    }
  }

  const indices = [];
  for (let ring = 0; ring < segments; ring++) {
    for (let side = 0; side < crossSectionSides; side++) {
      const a = ring * ringVertexCount + side;
      const b = (ring + 1) * ringVertexCount + side;
      const c = (ring + 1) * ringVertexCount + side + 1;
      const d = ring * ringVertexCount + side + 1;
      indices.push(a, d, b, d, c, b);
    }
  }

  // Cap only the true chain ends; secondary-structure transitions remain seamless.
  const addCap = (ring, outwardNormal, reverse) => {
    const centerIndex = positions.length / 3;
    const center = frames.points[ring];
    const profile = profiles[ring];
    positions.push(center.x, center.y, center.z);
    normals.push(outwardNormal.x, outwardNormal.y, outwardNormal.z);
    colors.push(profile.color.r, profile.color.g, profile.color.b);
    uvs.push(ring === 0 ? 0 : 1, 0.5);
    waveCoordinates.push(((ring / segments) + (options.waveOffset || 0)) % 1);
    for (let side = 0; side < crossSectionSides; side++) {
      const current = ring * ringVertexCount + side;
      const next = ring * ringVertexCount + side + 1;
      if (reverse) indices.push(centerIndex, next, current);
      else indices.push(centerIndex, current, next);
    }
  };
  addCap(0, frames.tangents[0].clone().negate(), true);
  addCap(segments, frames.tangents[segments].clone(), false);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setAttribute('waveCoord', new THREE.Float32BufferAttribute(waveCoordinates, 1));
  geometry.setIndex(indices);
  geometry.computeBoundingSphere();
  geometry.userData = {
    segments,
    crossSectionSides,
    triangleCount: indices.length / 3,
  };
  return geometry;
}

function validateProteinGeometry(geometry) {
  const position = geometry.getAttribute('position');
  const normal = geometry.getAttribute('normal');
  const index = geometry.getIndex();
  let finite = true;
  for (const attribute of [position, normal]) {
    for (let i = 0; i < attribute.array.length; i++) {
      if (!Number.isFinite(attribute.array[i])) {
        finite = false;
        break;
      }
    }
  }
  let indexInRange = true;
  for (let i = 0; i < index.array.length; i++) {
    if (index.array[i] < 0 || index.array[i] >= position.count) {
      indexInRange = false;
      break;
    }
  }
  return {
    finite,
    indexInRange,
    vertexCount: position.count,
    triangleCount: index.count / 3,
    boundingRadius: geometry.boundingSphere ? geometry.boundingSphere.radius : 0,
  };
}


export {
  PROTEIN_LIGHT_BLUE,
  PROTEIN_LIGHT_BLUE_DEEP,
  buildStructureRuns,
  countStructureTypes,
  createContinuousCartoonGeometry,
  createVisualStructureTypes,
  inferSecondaryStructure,
  smoothstep01,
  splitContinuousResidueChains,
  validateProteinGeometry
};
