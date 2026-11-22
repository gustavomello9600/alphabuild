import * as THREE from "three";

export interface VoxelPick {
  voxelId: number;
  point: THREE.Vector3;
  normal: THREE.Vector3;
}

export class VoxelPicker {
  private raycaster = new THREE.Raycaster();
  private pointer = new THREE.Vector2();

  pick(
    event: { clientX: number; clientY: number },
    element: HTMLElement,
    camera: THREE.Camera,
    mesh: THREE.Mesh,
    voxelIds: Uint32Array,
  ): VoxelPick | null {
    const rect = element.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return null;
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, camera);
    const hits = this.raycaster.intersectObject(mesh, false);
    const hit = hits[0];
    if (!hit || !hit.face) return null;
    const geometry = mesh.geometry as THREE.BufferGeometry;
    const index = geometry.getIndex();
    let vertexIndex = hit.face.a;
    if (index) {
      vertexIndex = index.getX(hit.face.a);
    }
    const voxelId = voxelIds[vertexIndex];
    if (voxelId === undefined) return null;
    const normal = hit.face.normal.clone().transformDirection(mesh.matrixWorld);
    return { voxelId, point: hit.point.clone(), normal };
  }

  pickPlane(
    event: { clientX: number; clientY: number },
    element: HTMLElement,
    camera: THREE.Camera,
    plane: THREE.Plane,
  ): THREE.Vector3 | null {
    const rect = element.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return null;
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, camera);
    const point = new THREE.Vector3();
    return this.raycaster.ray.intersectPlane(plane, point) ? point : null;
  }
}
