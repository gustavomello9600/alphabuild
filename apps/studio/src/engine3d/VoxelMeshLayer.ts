import * as THREE from "three";
import type { ParsedAbMesh } from "../types/contracts";

const VERTEX_SHADER = /* glsl */ `
  uniform float uColorMode;
  uniform float uDeformScale;
  uniform vec3 uLoadDir;
  attribute vec4 aVoxelAttr;
  varying vec3 vWorldPos;
  varying vec3 vNormal;
  varying vec4 vAttr;

  void main() {
    vec3 pos = position;
    if (uColorMode > 1.5 && uColorMode < 2.5 && uDeformScale > 0.0) {
      pos += uLoadDir * (aVoxelAttr.z * uDeformScale * 6.0);
    }
    vec4 world = modelMatrix * vec4(pos, 1.0);
    vWorldPos = world.xyz;
    vNormal = normalize(mat3(modelMatrix) * normal);
    vAttr = aVoxelAttr;
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const FRAGMENT_SHADER = /* glsl */ `
  uniform vec3 uBaseColor;
  uniform vec3 uClipMin;
  uniform vec3 uClipMax;
  uniform vec3 uLightDir1;
  uniform vec3 uLightDir2;
  uniform float uColorMode;
  varying vec3 vWorldPos;
  varying vec3 vNormal;
  varying vec4 vAttr;

  vec3 turboColormap(float x) {
    const vec4 kRedVec4   = vec4( 0.13572138,  4.61539260, -42.66032258, 132.13108234);
    const vec4 kGreenVec4 = vec4( 0.09140261,  2.19418839,   4.84296658, -14.18503333);
    const vec4 kBlueVec4  = vec4( 0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 kRedVec2   = vec2(-152.94239396,  59.28637943);
    const vec2 kGreenVec2 = vec2(   4.27729857,   2.82956604);
    const vec2 kBlueVec2  = vec2( -89.90310912,  27.34824973);
    x = clamp(x, 0.0, 1.0);
    vec4 v4 = vec4(1.0, x, x * x, x * x * x);
    vec2 v2 = v4.zw * v4.z;
    return clamp(vec3(
      dot(v4, kRedVec4)   + dot(v2, kRedVec2),
      dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
      dot(v4, kBlueVec4)  + dot(v2, kBlueVec2)
    ), 0.0, 1.0);
  }

  void main() {
    if (vWorldPos.x < uClipMin.x || vWorldPos.y < uClipMin.y || vWorldPos.z < uClipMin.z ||
        vWorldPos.x > uClipMax.x || vWorldPos.y > uClipMax.y || vWorldPos.z > uClipMax.z) {
      discard;
    }
    vec3 n = normalize(vNormal);
    if (!gl_FrontFacing) n = -n;
    vec3 viewDir = normalize(cameraPosition - vWorldPos);
    float hemi = 0.5 + 0.5 * n.y;
    float d1 = max(dot(n, normalize(uLightDir1)), 0.0);
    float d2 = max(dot(n, normalize(uLightDir2)), 0.0);
    float rim = pow(1.0 - abs(dot(n, viewDir)), 2.0);
    float shading = 0.28 * hemi + 0.52 * d1 + 0.25 * d2 + 0.12 * rim;

    vec3 cell = abs(fract(vWorldPos) - 0.5);
    float edge = max(cell.x, max(cell.y, cell.z));
    float bevel = smoothstep(0.47, 0.5, edge) * 0.16;
    shading = clamp(shading - bevel, 0.0, 1.0);

    vec3 base = uBaseColor;
    if (uColorMode > 0.5 && uColorMode < 1.5) {
      base = mix(vec3(0.10, 0.13, 0.18), vec3(0.78, 0.95, 1.0), clamp(vAttr.y, 0.0, 1.0));
    } else if (uColorMode > 1.5 && uColorMode < 2.5) {
      base = turboColormap(vAttr.z);
    } else if (uColorMode > 2.5) {
      base = turboColormap(vAttr.w);
    }
    float ao = mix(0.40, 1.0, clamp(vAttr.x, 0.0, 1.0));
    vec3 color = base * (0.44 + 0.56 * shading) * ao;
    gl_FragColor = vec4(color, 1.0);
  }
`;

export type VoxelColorMode = 0 | 1 | 2 | 3;

export class VoxelMeshLayer {
  readonly mesh: THREE.Mesh;
  private geometry: THREE.BufferGeometry;
  private material: THREE.ShaderMaterial;

  constructor() {
    this.geometry = new THREE.BufferGeometry();
    this.material = new THREE.ShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      side: THREE.DoubleSide,
      uniforms: {
        uColorMode: { value: 0 },
        uDeformScale: { value: 0 },
        uLoadDir: { value: new THREE.Vector3(0, -1, 0) },
        uBaseColor: { value: new THREE.Color("#dfe9f5") },
        uClipMin: { value: new THREE.Vector3(-1e6, -1e6, -1e6) },
        uClipMax: { value: new THREE.Vector3(1e6, 1e6, 1e6) },
        uLightDir1: { value: new THREE.Vector3(0.55, 1.0, 0.6).normalize() },
        uLightDir2: { value: new THREE.Vector3(-0.5, 0.4, -0.8).normalize() },
      },
    });
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = true;
    this.mesh.name = "voxel-mesh";
  }

  update(frame: ParsedAbMesh): void {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(frame.positions, 3));
    geometry.setAttribute("normal", new THREE.BufferAttribute(frame.normals, 3));
    geometry.setAttribute("aVoxelAttr", new THREE.BufferAttribute(frame.aoAndScalars, 4));
    geometry.setIndex(new THREE.BufferAttribute(frame.indices, 1));
    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();
    this.geometry.dispose();
    this.geometry = geometry;
    this.mesh.geometry = geometry;
  }

  clear(): void {
    const empty = new THREE.BufferGeometry();
    this.geometry.dispose();
    this.geometry = empty;
    this.mesh.geometry = empty;
  }

  setColorMode(mode: VoxelColorMode): void {
    this.material.uniforms.uColorMode!.value = mode;
  }

  setDeformScale(scale: number): void {
    this.material.uniforms.uDeformScale!.value = scale;
  }

  setClip(min: THREE.Vector3, max: THREE.Vector3): void {
    this.material.uniforms.uClipMin!.value.copy(min);
    this.material.uniforms.uClipMax!.value.copy(max);
  }

  setVisible(visible: boolean): void {
    this.mesh.visible = visible;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}
