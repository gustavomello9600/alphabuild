import * as THREE from "three";
import type { LensMode, ParsedAbMesh } from "../types/contracts";
import { readMctsTriples, readPolicyPairs } from "../lib/abmesh";

type Vec3 = [number, number, number];

interface Pool {
  mesh: THREE.InstancedMesh;
  capacity: number;
  key: string;
}

function nextPowerOfTwo(value: number): number {
  let capacity = 1;
  while (capacity < value) capacity *= 2;
  return capacity;
}

function worldCenter(flatIdx: number, dims: [number, number, number]): Vec3 {
  const [d, h, w] = dims;
  const plane = h * w;
  const x = Math.floor(flatIdx / plane);
  const rem = flatIdx % plane;
  const y = Math.floor(rem / w);
  const z = rem % w;
  return [x - d / 2 + 0.5, y + 0.5, z - w / 2 + 0.5];
}

export class OverlayLayer {
  readonly group = new THREE.Group();
  private dims: [number, number, number] = [1, 1, 1];
  private lens: LensMode = "structure";
  private pools = new Map<string, Pool>();
  private labels = new THREE.Group();
  private box = new THREE.BoxGeometry(0.94, 0.94, 0.94);
  private matrix = new THREE.Matrix4();
  private position = new THREE.Vector3();
  private scale = new THREE.Vector3();
  private quaternion = new THREE.Quaternion();
  private materials = {
    policyAdd: new THREE.MeshBasicMaterial({
      color: "#00E599",
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      depthTest: false,
    }),
    policyRem: new THREE.MeshBasicMaterial({
      color: "#FF2466",
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      depthTest: false,
    }),
    policyWire: new THREE.MeshBasicMaterial({
      color: "#F5B700",
      wireframe: true,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      depthTest: false,
    }),
    mctsAdd: new THREE.MeshBasicMaterial({
      color: "#00F0FF",
      transparent: true,
      opacity: 0.42,
      depthWrite: false,
      depthTest: false,
    }),
    mctsRem: new THREE.MeshBasicMaterial({
      color: "#F5B700",
      transparent: true,
      opacity: 0.42,
      depthWrite: false,
      depthTest: false,
    }),
    diffAdd: new THREE.MeshBasicMaterial({
      color: "#00E599",
      transparent: true,
      opacity: 0.8,
      depthWrite: false,
      depthTest: false,
    }),
    diffRem: new THREE.MeshBasicMaterial({
      color: "#FF2466",
      transparent: true,
      opacity: 0.8,
      depthWrite: false,
      depthTest: false,
    }),
    selected: new THREE.MeshBasicMaterial({
      color: "#EEF2F8",
      transparent: true,
      opacity: 0.92,
      depthWrite: false,
      depthTest: false,
    }),
  };

  constructor() {
    this.group.name = "overlays";
    this.labels.name = "pv-labels";
    this.group.add(this.labels);
  }

  private ensure(key: string, material: THREE.Material, count: number): THREE.InstancedMesh {
    let pool = this.pools.get(key);
    if (!pool || pool.capacity < count) {
      if (pool) {
        this.group.remove(pool.mesh);
        pool.mesh.dispose();
      }
      const capacity = Math.max(4096, nextPowerOfTwo(Math.max(1, count)));
      const mesh = new THREE.InstancedMesh(this.box, material, capacity);
      mesh.count = 0;
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.frustumCulled = false;
      // Analytic overlays (especially REMOVE candidates inside solid voxels)
      // must remain visible through the opaque structure.
      mesh.renderOrder = 20;
      mesh.name = key;
      this.group.add(mesh);
      pool = { mesh, capacity, key };
      this.pools.set(key, pool);
    }
    pool.mesh.count = count;
    return pool.mesh;
  }

  private setInstance(
    mesh: THREE.InstancedMesh,
    index: number,
    center: Vec3,
    size: number,
    color?: THREE.Color,
  ): void {
    this.position.set(center[0], center[1], center[2]);
    this.scale.setScalar(size);
    this.matrix.compose(this.position, this.quaternion, this.scale);
    mesh.setMatrixAt(index, this.matrix);
    if (color) mesh.setColorAt(index, color);
  }

  private clearLabels(): void {
    for (const child of [...this.labels.children]) {
      this.labels.remove(child);
      if (child instanceof THREE.Sprite) {
        child.material.map?.dispose();
        child.material.dispose();
      }
    }
  }

  private makeLabel(text: string, color: string): THREE.Sprite {
    const canvas = document.createElement("canvas");
    canvas.width = 128;
    canvas.height = 64;
    const context = canvas.getContext("2d");
    if (context) {
      context.clearRect(0, 0, 128, 64);
      context.fillStyle = "rgba(5, 6, 8, 0.78)";
      context.beginPath();
      context.roundRect(8, 8, 112, 48, 12);
      context.fill();
      context.strokeStyle = color;
      context.lineWidth = 2;
      context.stroke();
      context.fillStyle = color;
      context.font = "bold 28px 'JetBrains Mono', monospace";
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(text, 64, 33);
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
    });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(1.8, 0.9, 1);
    // Keep PV labels above the x-ray overlay boxes.
    sprite.renderOrder = 30;
    return sprite;
  }

  update(frame: ParsedAbMesh): void {
    this.dims = frame.dims;
    this.clearLabels();

    const add = readPolicyPairs(frame.policyAddRaw, frame.policyAddCount);
    const rem = readPolicyPairs(frame.policyRemRaw, frame.policyRemCount);
    const maxAdd = add.reduce((max, pair) => Math.max(max, pair.prob), 1e-9);
    const maxRem = rem.reduce((max, pair) => Math.max(max, pair.prob), 1e-9);
    const addMesh = this.ensure("policy-add", this.materials.policyAdd, add.length);
    add.forEach((pair, index) => {
      const intensity = Math.sqrt(Math.max(0, pair.prob) / maxAdd);
      const color = new THREE.Color("#00E599").multiplyScalar(0.35 + 0.65 * intensity);
      this.setInstance(
        addMesh,
        index,
        worldCenter(pair.flat_idx, this.dims),
        0.62 + 0.34 * intensity,
        color,
      );
    });
    addMesh.instanceMatrix.needsUpdate = true;
    if (addMesh.instanceColor) addMesh.instanceColor.needsUpdate = true;

    const remMesh = this.ensure("policy-rem", this.materials.policyRem, rem.length);
    rem.forEach((pair, index) => {
      const intensity = Math.sqrt(Math.max(0, pair.prob) / maxRem);
      const color = new THREE.Color("#FF2466").multiplyScalar(0.35 + 0.65 * intensity);
      this.setInstance(
        remMesh,
        index,
        worldCenter(pair.flat_idx, this.dims),
        0.62 + 0.34 * intensity,
        color,
      );
    });
    remMesh.instanceMatrix.needsUpdate = true;
    if (remMesh.instanceColor) remMesh.instanceColor.needsUpdate = true;

    const refMesh = this.ensure(
      "refutation-policy",
      this.materials.policyWire,
      add.length + rem.length,
    );
    [...add, ...rem].forEach((pair, index) => {
      this.setInstance(refMesh, index, worldCenter(pair.flat_idx, this.dims), 1.02);
    });
    refMesh.instanceMatrix.needsUpdate = true;

    const visitsAdd = readMctsTriples(frame.mctsAddRaw, frame.mctsAddCount);
    const visitsRem = readMctsTriples(frame.mctsRemRaw, frame.mctsRemCount);
    const maxVisits = Math.max(
      1,
      ...visitsAdd.map((entry) => entry.visits),
      ...visitsRem.map((entry) => entry.visits),
    );
    const mctsAdd = this.ensure("mcts-add", this.materials.mctsAdd, visitsAdd.length);
    visitsAdd.forEach((entry, index) => {
      const intensity = Math.log1p(entry.visits) / Math.log1p(maxVisits);
      const color = new THREE.Color("#00F0FF").multiplyScalar(0.25 + 0.75 * intensity);
      this.setInstance(
        mctsAdd,
        index,
        worldCenter(entry.flat_idx, this.dims),
        0.4 + 0.55 * intensity,
        color,
      );
    });
    mctsAdd.instanceMatrix.needsUpdate = true;
    if (mctsAdd.instanceColor) mctsAdd.instanceColor.needsUpdate = true;

    const mctsRem = this.ensure("mcts-rem", this.materials.mctsRem, visitsRem.length);
    visitsRem.forEach((entry, index) => {
      const intensity = Math.log1p(entry.visits) / Math.log1p(maxVisits);
      const color = new THREE.Color("#F5B700").multiplyScalar(0.25 + 0.75 * intensity);
      this.setInstance(
        mctsRem,
        index,
        worldCenter(entry.flat_idx, this.dims),
        0.4 + 0.55 * intensity,
        color,
      );
    });
    mctsRem.instanceMatrix.needsUpdate = true;
    if (mctsRem.instanceColor) mctsRem.instanceColor.needsUpdate = true;

    const diffAdd = this.ensure("diff-add", this.materials.diffAdd, frame.diffAdded.length);
    frame.diffAdded.forEach((flatIdx, index) => {
      this.setInstance(diffAdd, index, worldCenter(flatIdx, this.dims), 1.02);
    });
    diffAdd.instanceMatrix.needsUpdate = true;
    const diffRem = this.ensure("diff-rem", this.materials.diffRem, frame.diffRemoved.length);
    frame.diffRemoved.forEach((flatIdx, index) => {
      this.setInstance(diffRem, index, worldCenter(flatIdx, this.dims), 1.02);
    });
    diffRem.instanceMatrix.needsUpdate = true;

    const selected = frame.metadata.selected_actions ?? [];
    const selectedList =
      selected.length > 0
        ? selected.slice(0, 8).map((action) => ({
            flat_idx:
              action.x * this.dims[1] * this.dims[2] + action.y * this.dims[2] + action.z,
            channel: action.channel,
          }))
        : [...visitsAdd, ...visitsRem]
            .sort((a, b) => b.visits - a.visits)
            .slice(0, 8)
            .map((entry) => ({ flat_idx: entry.flat_idx, channel: 0 }));
    const selectedMesh = this.ensure("selected-pv", this.materials.selected, selectedList.length);
    selectedList.forEach((action, index) => {
      const [x, y, z] = worldCenter(action.flat_idx, this.dims);
      this.setInstance(selectedMesh, index, [x, y, z], 0.78);
      const label = this.makeLabel(`#${index + 1}`, index === 0 ? "#F5B700" : "#00F0FF");
      label.position.set(x, y + 1.1, z);
      this.labels.add(label);
    });
    selectedMesh.instanceMatrix.needsUpdate = true;

    this.applyVisibility();
  }

  setLens(lens: LensMode): void {
    this.lens = lens;
    this.applyVisibility();
  }

  private applyVisibility(): void {
    const policy = this.lens === "policy";
    const mcts = this.lens === "mcts";
    const refutation = this.lens === "refutation";
    const decision = this.lens === "decision";
    for (const [key, pool] of this.pools) {
      if (key === "policy-add" || key === "policy-rem") pool.mesh.visible = policy;
      else if (key === "mcts-add" || key === "mcts-rem") pool.mesh.visible = mcts;
      else if (key === "refutation-policy" || key === "selected-pv") pool.mesh.visible = refutation;
      else if (key === "diff-add" || key === "diff-rem") pool.mesh.visible = decision;
    }
    this.labels.visible = refutation;
  }

  tick(timeSeconds: number): void {
    const pulse = 0.55 + Math.sin(timeSeconds * 3.2) * 0.12;
    this.materials.policyAdd.opacity = pulse;
    this.materials.policyRem.opacity = pulse;
  }

  clear(): void {
    this.clearLabels();
    for (const pool of this.pools.values()) pool.mesh.count = 0;
  }

  dispose(): void {
    this.clearLabels();
    for (const pool of this.pools.values()) {
      this.group.remove(pool.mesh);
      pool.mesh.dispose();
    }
    this.pools.clear();
    this.box.dispose();
    Object.values(this.materials).forEach((material) => material.dispose());
  }
}
