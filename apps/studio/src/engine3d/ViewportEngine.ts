import * as THREE from "three";
import type {
  BcType,
  ForceVector,
  LensMode,
  ParsedAbMesh,
  SupportVoxel,
} from "../types/contracts";
import { CameraRig, type CameraPreset } from "./CameraRig";
import { VoxelMeshLayer } from "./VoxelMeshLayer";
import { BlueprintLayer } from "./BlueprintLayer";
import { OverlayLayer } from "./OverlayLayer";
import { BoundaryLayer } from "./BoundaryLayer";
import { VoxelPicker, type VoxelPick } from "./VoxelPicker";

export class ViewportEngine {
  readonly canvas: HTMLCanvasElement;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly rig: CameraRig;
  private renderer: THREE.WebGLRenderer;
  private voxels: VoxelMeshLayer;
  private blueprint: BlueprintLayer;
  private overlays: OverlayLayer;
  private boundaries: BoundaryLayer;
  private picker = new VoxelPicker();
  private grid: THREE.GridHelper;
  private resizeObserver: ResizeObserver;
  private animationFrame = 0;
  private clock = new THREE.Clock();
  private frame: ParsedAbMesh | null = null;
  private hoverHandler: ((pick: VoxelPick | null) => void) | null = null;
  private lens: LensMode = "structure";
  private rhoContinuous = false;
  private clip: [number, number, number] = [0, 0, 0];
  private dims: [number, number, number] = [64, 32, 8];
  private background = new THREE.Color("#050608");
  private disposed = false;
  private cameraPreset: CameraPreset = "iso";
  private blueprintEnabled = false;
  private figureEnabled = false;
  private unbindSync: (() => void) | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: true,
      powerPreference: "high-performance",
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.setClearColor(this.background, 1);

    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 2000);
    this.camera.position.set(58, 40, 62);
    this.rig = new CameraRig(this.camera, canvas);

    this.scene.add(new THREE.HemisphereLight("#9fc8ff", "#0b0e14", 1.1));
    const key = new THREE.DirectionalLight("#ffffff", 1.4);
    key.position.set(1, 2, 1.5);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight("#00F0FF", 0.6);
    rim.position.set(-1, 0.4, -1.2);
    this.scene.add(rim);

    this.voxels = new VoxelMeshLayer();
    this.blueprint = new BlueprintLayer();
    this.overlays = new OverlayLayer();
    this.boundaries = new BoundaryLayer();
    this.scene.add(this.voxels.mesh, this.blueprint.mesh, this.overlays.group, this.boundaries.group);

    this.grid = new THREE.GridHelper(240, 60, "#1c2533", "#10151d");
    this.grid.position.set(0, -0.02, 0);
    const gridMaterial = this.grid.material as THREE.Material;
    gridMaterial.transparent = true;
    gridMaterial.opacity = 0.42;
    this.scene.add(this.grid);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    this.resize();
    canvas.addEventListener("pointermove", this.handlePointerMove);
    canvas.addEventListener("pointerleave", this.handlePointerLeave);
    this.animate();
  }

  private resize(): void {
    if (this.disposed) return;
    const width = this.canvas.clientWidth || 1;
    const height = this.canvas.clientHeight || 1;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  private animate = (): void => {
    if (this.disposed) return;
    this.animationFrame = requestAnimationFrame(this.animate);
    this.rig.controls.update();
    this.overlays.tick(this.clock.getElapsedTime());
    this.renderer.render(this.scene, this.camera);
  };

  private handlePointerMove = (event: PointerEvent): void => {
    if (!this.frame || !this.hoverHandler) return;
    const pick = this.picker.pick(
      event,
      this.canvas,
      this.camera,
      this.voxels.mesh,
      this.frame.voxelIds,
    );
    this.hoverHandler(pick);
  };

  private handlePointerLeave = (): void => {
    this.hoverHandler?.(null);
  };

  setHoverHandler(handler: ((pick: VoxelPick | null) => void) | null): void {
    this.hoverHandler = handler;
  }

  applyFrame(frame: ParsedAbMesh): void {
    const firstFrame = this.frame === null;
    const changedDims =
      !this.frame ||
      this.frame.dims[0] !== frame.dims[0] ||
      this.frame.dims[1] !== frame.dims[1] ||
      this.frame.dims[2] !== frame.dims[2];
    this.frame = frame;
    this.dims = frame.dims;
    this.voxels.update(frame);
    this.blueprint.update(frame);
    this.overlays.update(frame);
    this.rig.setDims(frame.dims);
    if (firstFrame || changedDims) {
      this.clip = [frame.dims[0], frame.dims[1], frame.dims[2]];
      this.applyClip();
      this.rig.setPreset(this.cameraPreset);
    }
    this.applyColorMode();
  }

  setBoundaries(
    supports: SupportVoxel[],
    forces: ForceVector[],
    bcType: BcType,
  ): void {
    this.boundaries.update(supports, forces, this.dims, bcType);
  }

  setLens(lens: LensMode): void {
    this.lens = lens;
    this.overlays.setLens(lens);
    this.applyColorMode();
  }

  setRhoContinuous(value: boolean): void {
    this.rhoContinuous = value;
    this.applyColorMode();
  }

  private applyColorMode(): void {
    if (this.rhoContinuous) {
      this.voxels.setColorMode(1);
      return;
    }
    if (this.lens === "deformation") this.voxels.setColorMode(2);
    else if (this.lens === "strain") this.voxels.setColorMode(3);
    else this.voxels.setColorMode(0);
  }

  setDeformScale(scale: number): void {
    this.voxels.setDeformScale(scale);
  }

  setBlueprintVisible(visible: boolean): void {
    this.blueprintEnabled = visible;
    this.blueprint.setVisible(visible);
  }

  setClipPlanes(cx: number, cy: number, cz: number): void {
    this.clip = [cx, cy, cz];
    this.applyClip();
  }

  private applyClip(): void {
    const [d, h, w] = this.dims;
    const min = new THREE.Vector3(-d / 2 - 0.5, -0.5, -w / 2 - 0.5);
    const max = new THREE.Vector3(
      -d / 2 + Math.max(0, this.clip[0]) + 0.5,
      Math.min(h, Math.max(0, this.clip[1])) + 0.5,
      -w / 2 + Math.max(0, this.clip[2]) + 0.5,
    );
    this.voxels.setClip(min, max);
  }

  setFigureMode(enabled: boolean, light = false): void {
    this.figureEnabled = enabled;
    this.background = new THREE.Color(enabled ? (light ? "#ffffff" : "#f4f6f9") : "#050608");
    this.renderer.setClearColor(this.background, 1);
    this.grid.visible = !enabled;
    this.boundaries.setVisible(!enabled);
    this.overlays.group.visible = !enabled;
    this.blueprint.setVisible(this.blueprintEnabled);
  }

  setTheme(light: boolean): void {
    if (this.figureEnabled) return;
    this.background = new THREE.Color(light ? "#ffffff" : "#050608");
    this.renderer.setClearColor(this.background, 1);
    const gridMaterial = this.grid.material as THREE.Material;
    gridMaterial.opacity = light ? 0.25 : 0.42;
  }

  clearFrame(): void {
    this.frame = null;
    this.voxels.clear();
    this.blueprint.clear();
    this.overlays.clear();
  }

  resetCamera(): void {
    this.cameraPreset = "iso";
    this.rig.setPreset("iso");
  }

  cycleView(): CameraPreset {
    this.cameraPreset = this.rig.cycle(this.cameraPreset);
    this.rig.setPreset(this.cameraPreset);
    return this.cameraPreset;
  }

  captureScreenshot(width = 3840, height = 2160, transparentBg = false): string {
    const previousSize = new THREE.Vector2();
    this.renderer.getSize(previousSize);
    const previousRatio = this.renderer.getPixelRatio();
    const previousClear = this.renderer.getClearColor(new THREE.Color());
    const previousAlpha = this.renderer.getClearAlpha();
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(width, height, false);
    this.renderer.setClearColor(transparentBg ? 0x000000 : this.background, transparentBg ? 0 : 1);
    this.renderer.render(this.scene, this.camera);
    const data = this.canvas.toDataURL("image/png");
    this.renderer.setPixelRatio(previousRatio);
    this.renderer.setSize(previousSize.x, previousSize.y, false);
    this.renderer.setClearColor(previousClear, previousAlpha);
    return data;
  }

  syncWith(other: ViewportEngine): void {
    this.unbindSync?.();
    this.unbindSync = this.rig.onChange((state) => {
      other.rig.applyState(state);
    });
  }

  unsync(): void {
    this.unbindSync?.();
    this.unbindSync = null;
  }

  getFrame(): ParsedAbMesh | null {
    return this.frame;
  }

  dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.animationFrame);
    this.resizeObserver.disconnect();
    this.canvas.removeEventListener("pointermove", this.handlePointerMove);
    this.canvas.removeEventListener("pointerleave", this.handlePointerLeave);
    this.unsync();
    this.voxels.dispose();
    this.blueprint.dispose();
    this.overlays.dispose();
    this.boundaries.dispose();
    this.grid.geometry.dispose();
    (this.grid.material as THREE.Material).dispose();
    this.rig.dispose();
    this.renderer.dispose();
  }
}
