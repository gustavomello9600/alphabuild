import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { parseAbMesh01 } from "../lib/abmesh";
import type { ParsedAbMesh } from "../types/contracts";
import { OverlayLayer } from "./OverlayLayer";

function fixtureFrame(): ParsedAbMesh {
  const here = dirname(fileURLToPath(import.meta.url));
  const bytes = readFileSync(resolve(here, "../lib/__fixtures__/abmesh_fixture.bin"));
  const frame = parseAbMesh01(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer,
  );
  // Keep the test free of canvas-backed labels; policy and diff stay populated.
  return {
    ...frame,
    mctsAddCount: 0,
    mctsRemCount: 0,
    metadata: { ...frame.metadata, selected_actions: [] },
  };
}

function overlays(layer: OverlayLayer): THREE.InstancedMesh[] {
  return layer.group.children.filter(
    (child): child is THREE.InstancedMesh => child instanceof THREE.InstancedMesh,
  );
}

describe("OverlayLayer", () => {
  it("draws analytic overlays after opaque geometry without depth occlusion", () => {
    const layer = new OverlayLayer();
    layer.update(fixtureFrame());

    const meshes = overlays(layer);
    expect(meshes.length).toBeGreaterThan(0);
    for (const mesh of meshes) {
      expect(mesh.renderOrder).toBe(20);
      const material = mesh.material as THREE.MeshBasicMaterial;
      expect(material.depthTest).toBe(false);
      expect(material.depthWrite).toBe(false);
    }
  });

  it("keeps REMOVE overlays inside solid voxels addressable by their lenses", () => {
    const layer = new OverlayLayer();
    const frame = fixtureFrame();
    layer.update(frame);

    const byName = new Map(overlays(layer).map((mesh) => [mesh.name, mesh]));
    const policyRem = byName.get("policy-rem");
    const diffRem = byName.get("diff-rem");
    expect(policyRem?.count).toBe(frame.policyRemCount);
    expect(diffRem?.count).toBe(frame.diffRemoved.length);
    expect(policyRem?.count).toBeGreaterThan(0);
    expect(diffRem?.count).toBeGreaterThan(0);

    layer.setLens("structure");
    expect(policyRem?.visible).toBe(false);
    expect(diffRem?.visible).toBe(false);

    layer.setLens("policy");
    expect(policyRem?.visible).toBe(true);
    expect(diffRem?.visible).toBe(false);

    layer.setLens("decision");
    expect(policyRem?.visible).toBe(false);
    expect(diffRem?.visible).toBe(true);
  });
});
