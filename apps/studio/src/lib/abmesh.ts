import type { ParsedAbMesh, StepFrameMetadata } from "../types/contracts";

export const ABMESH_MAGIC = "ABMESH01";
export const ABMESH_HEADER_BYTES = 96;

export const ABMESH_FLAGS = {
  DISPLACEMENT: 0x01,
  BLUEPRINT: 0x02,
  POLICY: 0x04,
  MCTS: 0x08,
  DIFF: 0x10,
  RHO_CONT: 0x20,
} as const;

export interface PolicyPair {
  flat_idx: number;
  prob: number;
}

export interface MctsTriple {
  flat_idx: number;
  visits: number;
  q_value: number;
}

/**
 * Zero-copy view over the ArrayBuffer returned by `get_frame_mesh`.
 * Every typed array aliases the IPC buffer (no per-vertex JSON, no copies).
 */
export function parseAbMesh01(buf: ArrayBuffer): ParsedAbMesh {
  if (buf.byteLength < ABMESH_HEADER_BYTES) {
    throw new Error("ABMESH01: buffer menor que header (96 bytes)");
  }
  const dv = new DataView(buf);
  const magic = new TextDecoder("ascii").decode(new Uint8Array(buf, 0, 8));
  if (magic !== ABMESH_MAGIC) {
    throw new Error(`ABMESH01: magic inválido (${magic})`);
  }

  const totalByteLen = dv.getUint32(92, true);
  if (totalByteLen !== buf.byteLength) {
    throw new Error(`ABMESH01: tamanho divergente (${buf.byteLength} != ${totalByteLen})`);
  }

  const flags = dv.getUint32(8, true);
  const stepIndex = dv.getUint32(12, true);
  const dims: [number, number, number] = [
    dv.getUint32(16, true),
    dv.getUint32(20, true),
    dv.getUint32(24, true),
  ];
  const V = dv.getUint32(28, true);
  const I = dv.getUint32(32, true);
  const nPa = dv.getUint32(36, true);
  const nPr = dv.getUint32(40, true);
  const nMa = dv.getUint32(44, true);
  const nMr = dv.getUint32(48, true);
  const Vb = dv.getUint32(52, true);
  const Ib = dv.getUint32(56, true);
  const nDa = dv.getUint32(60, true);
  const nDr = dv.getUint32(64, true);
  const dispP99 = dv.getFloat32(68, true);
  const strainP99 = dv.getFloat32(72, true);
  const metaLen = dv.getUint32(76, true);
  const metaPaddedLen = dv.getUint32(80, true);

  if (V > 100_000_000 || I > 300_000_000 || Vb > 100_000_000) {
    throw new Error("ABMESH01: contagens de seção irrealistas");
  }

  let off = ABMESH_HEADER_BYTES;
  const takeF32 = (n: number): Float32Array => {
    const aligned = off % 4 === 0;
    if (!aligned) throw new Error("ABMESH01: seção f32 desalinhada");
    const value = new Float32Array(buf, off, n);
    off += n * 4;
    return value;
  };
  const takeU32 = (n: number): Uint32Array => {
    if (off % 4 !== 0) throw new Error("ABMESH01: seção u32 desalinhada");
    const value = new Uint32Array(buf, off, n);
    off += n * 4;
    return value;
  };

  const positions = takeF32(V * 3);
  const normals = takeF32(V * 3);
  const aoAndScalars = takeF32(V * 4);
  const voxelIds = takeU32(V);
  const indices = takeU32(I);
  const blueprintPositions = takeF32(Vb * 3);
  const blueprintNormals = takeF32(Vb * 3);
  const blueprintIndices = takeU32(Ib);
  const policyAddRaw = new DataView(buf, off, nPa * 8);
  off += nPa * 8;
  const policyRemRaw = new DataView(buf, off, nPr * 8);
  off += nPr * 8;
  const mctsAddRaw = new DataView(buf, off, nMa * 12);
  off += nMa * 12;
  const mctsRemRaw = new DataView(buf, off, nMr * 12);
  off += nMr * 12;
  const diffAdded = takeU32(nDa);
  const diffRemoved = takeU32(nDr);
  const metaJson = new TextDecoder("utf-8").decode(new Uint8Array(buf, off, metaLen));
  off += metaPaddedLen;
  if (off !== totalByteLen) {
    throw new Error(`ABMESH01: cursor final ${off} != ${totalByteLen}`);
  }

  return {
    flags,
    stepIndex,
    dims,
    dispP99,
    strainP99,
    positions,
    normals,
    aoAndScalars,
    voxelIds,
    indices,
    blueprintPositions,
    blueprintNormals,
    blueprintIndices,
    policyAddRaw,
    policyAddCount: nPa,
    policyRemRaw,
    policyRemCount: nPr,
    mctsAddRaw,
    mctsAddCount: nMa,
    mctsRemRaw,
    mctsRemCount: nMr,
    diffAdded,
    diffRemoved,
    metadata: JSON.parse(metaJson) as StepFrameMetadata,
  };
}

export function readPolicyPairs(view: DataView, count: number): PolicyPair[] {
  const pairs: PolicyPair[] = [];
  for (let i = 0; i < count; i += 1) {
    const offset = i * 8;
    pairs.push({
      flat_idx: view.getUint32(offset, true),
      prob: view.getFloat32(offset + 4, true),
    });
  }
  return pairs;
}

export function readMctsTriples(view: DataView, count: number): MctsTriple[] {
  const triples: MctsTriple[] = [];
  for (let i = 0; i < count; i += 1) {
    const offset = i * 12;
    triples.push({
      flat_idx: view.getUint32(offset, true),
      visits: view.getFloat32(offset + 4, true),
      q_value: view.getFloat32(offset + 8, true),
    });
  }
  return triples;
}

export function flatToCoord(
  flatIdx: number,
  dims: [number, number, number],
): [number, number, number] {
  const [d, h, w] = dims;
  const plane = h * w;
  const x = Math.floor(flatIdx / plane);
  const rem = flatIdx % plane;
  const y = Math.floor(rem / w);
  const z = rem % w;
  return [Math.min(d - 1, x), Math.min(h - 1, y), Math.min(w - 1, z)];
}

export function coordToWorld(
  x: number,
  y: number,
  z: number,
  dims: [number, number, number],
): [number, number, number] {
  return [x - dims[0] / 2 + 0.5, y + 0.5, z - dims[2] / 2 + 0.5];
}

export interface VoxelInspection {
  x: number;
  y: number;
  z: number;
  voxel_id: number;
  ao: number;
  density: number;
  disp_norm: number;
  strain_norm: number;
  policy_add: number;
  policy_remove: number;
  mcts_add: number;
  mcts_remove: number;
  q_add: number;
  q_remove: number;
}

export function inspectVoxel(frame: ParsedAbMesh, voxelId: number): VoxelInspection | null {
  const [x, y, z] = flatToCoord(voxelId, frame.dims);
  let vertex = -1;
  for (let i = 0; i < frame.voxelIds.length; i += 1) {
    if (frame.voxelIds[i] === voxelId) {
      vertex = i;
      break;
    }
  }
  if (vertex < 0) return null;
  const base = vertex * 4;
  const ao = frame.aoAndScalars[base] ?? 1;
  const density = frame.aoAndScalars[base + 1] ?? 0;
  const dispNorm = frame.aoAndScalars[base + 2] ?? 0;
  const strainNorm = frame.aoAndScalars[base + 3] ?? 0;
  const add = readPolicyPairs(frame.policyAddRaw, frame.policyAddCount).find(
    (pair) => pair.flat_idx === voxelId,
  );
  const rem = readPolicyPairs(frame.policyRemRaw, frame.policyRemCount).find(
    (pair) => pair.flat_idx === voxelId,
  );
  const mAdd = readMctsTriples(frame.mctsAddRaw, frame.mctsAddCount).find(
    (pair) => pair.flat_idx === voxelId,
  );
  const mRem = readMctsTriples(frame.mctsRemRaw, frame.mctsRemCount).find(
    (pair) => pair.flat_idx === voxelId,
  );
  return {
    x,
    y,
    z,
    voxel_id: voxelId,
    ao,
    density,
    disp_norm: dispNorm,
    strain_norm: strainNorm,
    policy_add: add?.prob ?? 0,
    policy_remove: rem?.prob ?? 0,
    mcts_add: mAdd?.visits ?? 0,
    mcts_remove: mRem?.visits ?? 0,
    q_add: mAdd?.q_value ?? 0,
    q_remove: mRem?.q_value ?? 0,
  };
}
