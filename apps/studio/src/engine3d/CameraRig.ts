import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export type CameraPreset = "iso" | "xy" | "xz" | "yz";

export class CameraRig {
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  private dims: [number, number, number] = [64, 32, 8];
  private syncing = false;
  private changeHandlers: Array<(state: { position: THREE.Vector3; target: THREE.Vector3 }) => void> =
    [];

  constructor(camera: THREE.PerspectiveCamera, element: HTMLElement) {
    this.camera = camera;
    this.controls = new OrbitControls(camera, element);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.rotateSpeed = 0.72;
    this.controls.zoomSpeed = 0.9;
    this.controls.panSpeed = 0.8;
    this.controls.minDistance = 4;
    this.controls.maxDistance = 800;
    this.controls.addEventListener("change", () => {
      if (this.syncing) return;
      const state = this.state();
      for (const handler of this.changeHandlers) handler(state);
    });
  }

  setDims(dims: [number, number, number]): void {
    this.dims = dims;
  }

  state(): { position: THREE.Vector3; target: THREE.Vector3 } {
    return {
      position: this.camera.position.clone(),
      target: this.controls.target.clone(),
    };
  }

  applyState(state: { position: THREE.Vector3; target: THREE.Vector3 }): void {
    this.syncing = true;
    this.camera.position.copy(state.position);
    this.controls.target.copy(state.target);
    this.controls.update();
    this.syncing = false;
  }

  onChange(handler: (state: { position: THREE.Vector3; target: THREE.Vector3 }) => void): () => void {
    this.changeHandlers.push(handler);
    return () => {
      this.changeHandlers = this.changeHandlers.filter((item) => item !== handler);
    };
  }

  setPreset(preset: CameraPreset): void {
    const [d, h, w] = this.dims;
    const center = new THREE.Vector3(0, h / 2, 0);
    const radius = Math.max(d, h, w);
    const distance = Math.max(8, radius * 1.6);
    if (preset === "xy") {
      this.camera.position.set(center.x, center.y, center.z + distance);
      this.camera.up.set(0, 1, 0);
    } else if (preset === "xz") {
      this.camera.position.set(center.x, center.y + distance, center.z + 0.001);
      this.camera.up.set(0, 0, -1);
    } else if (preset === "yz") {
      this.camera.position.set(center.x + distance, center.y, center.z);
      this.camera.up.set(0, 1, 0);
    } else {
      this.camera.position.set(
        center.x + radius * 0.78,
        center.y + radius * 0.55,
        center.z + radius * 0.92,
      );
      this.camera.up.set(0, 1, 0);
    }
    this.controls.target.copy(center);
    this.camera.lookAt(center);
    this.controls.update();
  }

  cycle(current: CameraPreset): CameraPreset {
    const order: CameraPreset[] = ["iso", "xy", "xz", "yz"];
    const index = order.indexOf(current);
    return order[(index + 1) % order.length] ?? "iso";
  }

  dispose(): void {
    this.controls.dispose();
    this.changeHandlers = [];
  }
}
