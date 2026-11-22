import { describe, expect, it } from "vitest";
import { buildTurboLut, turboColormap, turboHex } from "./turboColormap";

describe("Turbo colormap (Mikhailov polynomial)", () => {
  it("matches the exact polynomial anchors", () => {
    expect(turboColormap(0)).toEqual([0.13572138, 0.09140261, 0.1066733]);
    const quarter = turboColormap(0.25);
    expect(quarter[0]).toBeCloseTo(0.14831316, 6);
    expect(quarter[1]).toBeCloseTo(0.74046542, 6);
    expect(quarter[2]).toBeCloseTo(0.8807233, 6);
    const half = turboColormap(0.5);
    expect(half[0]).toBeCloseTo(0.58852206, 6);
    expect(half[1]).toBeCloseTo(0.98186438, 6);
    expect(half[2]).toBeCloseTo(0.3131687, 6);
    const end = turboColormap(1);
    expect(end[0]).toBeCloseTo(0.56585921, 6);
    expect(end[1]).toBeCloseTo(0.05038886, 6);
    expect(end[2]).toBe(0);
  });

  it("clamps x outside [0,1]", () => {
    expect(turboColormap(-3)).toEqual(turboColormap(0));
    expect(turboColormap(4)).toEqual(turboColormap(1));
  });

  it("always returns finite channels in [0,1]", () => {
    for (let i = 0; i <= 100; i += 1) {
      const channels = turboColormap(i / 100);
      for (const channel of channels) {
        expect(Number.isFinite(channel)).toBe(true);
        expect(channel).toBeGreaterThanOrEqual(0);
        expect(channel).toBeLessThanOrEqual(1);
      }
    }
  });

  it("builds a 256-entry RGBA LUT", () => {
    const lut = buildTurboLut();
    expect(lut.length).toBe(256 * 4);
    expect(lut[3]).toBe(255);
    expect(lut[lut.length - 1]).toBe(255);
  });

  it("emits valid hex colors", () => {
    expect(turboHex(0)).toMatch(/^#[0-9a-f]{6}$/);
    expect(turboHex(1)).toMatch(/^#[0-9a-f]{6}$/);
  });
});
