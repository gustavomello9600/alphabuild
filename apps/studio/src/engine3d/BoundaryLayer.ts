import * as THREE from "three";
import type { BcType, ForceVector, SupportVoxel } from "../types/contracts";

function worldCenter(
  x: number,
  y: number,
  z: number,
  dims: [number, number, number],
): [number, number, number] {
  return [x - dims[0] / 2 + 0.5, y + 0.5, z - dims[2] / 2 + 0.5];
}

export class BoundaryLayer {
  readonly group = new THREE.Group();
  private supports: SupportVoxel[] = [];
  private forces: ForceVector[] = [];
  private dims: [number, number, number] = [64, 32, 8];
  private bcType: BcType = "FULL_CLAMP";
  private supportMesh: THREE.InstancedMesh | null = null;
  private forceGroup = new THREE.Group();
  private box = new THREE.BoxGeometry(1.04, 1.04, 1.04);
  private cylinder = new THREE.CylinderGeometry(0.045, 0.045, 1, 10);
  private cone = new THREE.ConeGeometry(0.14, 0.36, 12);
  private ringGeo = new THREE.TorusGeometry(0.38, 0.025, 8, 24);
  private ringMaterial = new THREE.MeshBasicMaterial({
    color: "#FF7A1A",
    transparent: true,
    opacity: 0.65,
  });
  private supportMaterial = new THREE.MeshBasicMaterial({
    color: "#00F0FF",
    transparent: true,
    opacity: 0.28,
    depthWrite: false,
  });
  private railMaterial = new THREE.MeshBasicMaterial({
    color: "#8B5CF6",
    transparent: true,
    opacity: 0.34,
    depthWrite: false,
  });
  private forceMaterial = new THREE.MeshStandardMaterial({
    color: "#FF7A1A",
    emissive: "#FF7A1A",
    emissiveIntensity: 0.35,
    metalness: 0.2,
    roughness: 0.4,
  });
  private handleMaterial = new THREE.MeshStandardMaterial({
    color: "#FFB066",
    emissive: "#FF7A1A",
    emissiveIntensity: 0.5,
    metalness: 0.3,
    roughness: 0.3,
  });

  constructor() {
    this.group.name = "boundaries";
    this.group.add(this.forceGroup);
  }

  update(
    supports: SupportVoxel[],
    forces: ForceVector[],
    dims: [number, number, number],
    bcType: BcType,
  ): void {
    this.supports = supports;
    this.forces = forces;
    this.dims = dims;
    this.bcType = bcType;
    this.rebuildSupports();
    this.rebuildForces();
  }

  private rebuildSupports(): void {
    if (this.supportMesh) {
      this.group.remove(this.supportMesh);
      this.supportMesh.dispose();
      this.supportMesh = null;
    }
    if (this.supports.length === 0) return;
    const clampCount = this.supports.filter((support) => support.mask_z).length;
    const railCount = this.supports.length - clampCount;
    const material = clampCount >= railCount ? this.supportMaterial : this.railMaterial;
    const mesh = new THREE.InstancedMesh(this.box, material, Math.max(4096, this.supports.length));
    const matrix = new THREE.Matrix4();
    const colorClamp = new THREE.Color("#00F0FF");
    const colorRail = new THREE.Color("#8B5CF6");
    this.supports.forEach((support, index) => {
      const [x, y, z] = worldCenter(support.x, support.y, support.z, this.dims);
      matrix.makeTranslation(x, y, z);
      mesh.setMatrixAt(index, matrix);
      mesh.setColorAt(index, support.mask_z ? colorClamp : colorRail);
    });
    mesh.count = this.supports.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.renderOrder = 3;
    this.supportMesh = mesh;
    this.group.add(mesh);
  }

  private rebuildForces(): void {
    for (const child of [...this.forceGroup.children]) {
      this.forceGroup.remove(child);

    }
    for (const force of this.forces) {
      const [x, y, z] = worldCenter(force.x, force.y, force.z, this.dims);
      const direction = new THREE.Vector3(force.fx, force.fy, force.fz);
      if (direction.lengthSq() < 1e-12) continue;
      direction.normalize();
      const length = 1.8;
      const quaternion = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        direction,
      );
      const shaft = new THREE.Mesh(this.cylinder, this.forceMaterial);
      shaft.scale.set(1, length, 1);
      shaft.position.set(x, y, z).addScaledVector(direction, length * 0.5);
      shaft.quaternion.copy(quaternion);
      this.forceGroup.add(shaft);
      const head = new THREE.Mesh(this.cone, this.forceMaterial);
      head.position.set(x, y, z).addScaledVector(direction, length + 0.18);
      head.quaternion.copy(quaternion);
      this.forceGroup.add(head);
      const ring = new THREE.Mesh(this.ringGeo, this.ringMaterial);
      ring.position.set(x, y - direction.y * 0.42, z);
      ring.rotation.x = Math.PI / 2;
      this.forceGroup.add(ring);
    }
    if (this.forces.length > 0) {
      const first = this.forces[0];
      if (first) {
        const [x, y, z] = worldCenter(first.x, first.y, first.z, this.dims);
        const handle = new THREE.Mesh(
          new THREE.SphereGeometry(Math.max(0.34, Math.min(...this.dims) * 0.09), 20, 16),
          this.handleMaterial,
        );
        handle.position.set(x, y + 0.65, z);
        handle.name = "force-handle";
        handle.userData.isForceHandle = true;
        this.forceGroup.add(handle);
      }
    }
    this.forceGroup.visible = this.bcType !== "CUSTOM" || this.forces.length > 0;
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  dispose(): void {
    if (this.supportMesh) {
      this.group.remove(this.supportMesh);
      this.supportMesh.dispose();
    }
    for (const child of [...this.forceGroup.children]) {
      this.forceGroup.remove(child);
    }
    this.box.dispose();
    this.cylinder.dispose();
    this.cone.dispose();
    this.ringGeo.dispose();
    this.ringMaterial.dispose();
    this.supportMaterial.dispose();
    this.railMaterial.dispose();
    this.forceMaterial.dispose();
    this.handleMaterial.dispose();
  }
}
