import * as THREE from "three";
import type { ParsedAbMesh } from "../types/contracts";

const VERTEX_SHADER = /* glsl */ `
  varying vec3 vWorldPos;
  varying vec3 vNormal;
  void main() {
    vec4 world = modelMatrix * vec4(position, 1.0);
    vWorldPos = world.xyz;
    vNormal = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const FRAGMENT_SHADER = /* glsl */ `
  varying vec3 vWorldPos;
  varying vec3 vNormal;
  void main() {
    float fresnel = pow(1.0 - abs(dot(normalize(vNormal), normalize(cameraPosition - vWorldPos))), 2.2);
    vec3 color = mix(vec3(0.486, 0.361, 0.965), vec3(0.0, 0.941, 1.0), fresnel);
    float alpha = mix(0.08, 0.65, fresnel);
    gl_FragColor = vec4(color, alpha);
  }
`;

export class BlueprintLayer {
  readonly mesh: THREE.Mesh;
  private geometry: THREE.BufferGeometry;
  private material: THREE.ShaderMaterial;

  constructor() {
    this.geometry = new THREE.BufferGeometry();
    this.material = new THREE.ShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.NormalBlending,
    });
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.renderOrder = 5;
    this.mesh.name = "blueprint-ghost";
  }

  update(frame: ParsedAbMesh): void {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(frame.blueprintPositions, 3));
    geometry.setAttribute("normal", new THREE.BufferAttribute(frame.blueprintNormals, 3));
    geometry.setIndex(new THREE.BufferAttribute(frame.blueprintIndices, 1));
    geometry.computeBoundingSphere();
    this.geometry.dispose();
    this.geometry = geometry;
    this.mesh.geometry = geometry;
    this.mesh.visible = frame.blueprintIndices.length > 0;
  }

  clear(): void {
    const empty = new THREE.BufferGeometry();
    this.geometry.dispose();
    this.geometry = empty;
    this.mesh.geometry = empty;
    this.mesh.visible = false;
  }

  setVisible(visible: boolean): void {
    this.mesh.visible = visible;
  }

  setOpacity(opacity: number): void {
    this.material.opacity = opacity;
    this.material.transparent = true;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}
