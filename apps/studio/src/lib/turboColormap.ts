export type Rgb = [number, number, number];

const K_RED_VEC4 = [0.13572138, 4.6153926, -42.66032258, 132.13108234] as const;
const K_GREEN_VEC4 = [0.09140261, 2.19418839, 4.84296658, -14.18503333] as const;
const K_BLUE_VEC4 = [0.1066733, 12.64194608, -60.58204836, 110.36276771] as const;
const K_RED_VEC2 = [-152.94239396, 59.28637943] as const;
const K_GREEN_VEC2 = [4.27729857, 2.82956604] as const;
const K_BLUE_VEC2 = [-89.90310912, 27.34824973] as const;

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

/**
 * Exact Mikhailov Turbo polynomial, identical to the GLSL implementation in
 * VoxelMeshLayer.ts.
 */
export function turboColormap(t: number): Rgb {
  const x = clamp01(t);
  const x2 = x * x;
  const x3 = x2 * x;
  const v4 = [1, x, x2, x3];
  const v2 = [x2 * x2, x3 * x2];
  const dot4 = (coefficients: readonly number[]): number =>
    coefficients.reduce((sum, coefficient, index) => sum + coefficient * (v4[index] ?? 0), 0);
  const dot2 = (coefficients: readonly number[]): number =>
    coefficients.reduce((sum, coefficient, index) => sum + coefficient * (v2[index] ?? 0), 0);
  return [
    clamp01(dot4(K_RED_VEC4) + dot2(K_RED_VEC2)),
    clamp01(dot4(K_GREEN_VEC4) + dot2(K_GREEN_VEC2)),
    clamp01(dot4(K_BLUE_VEC4) + dot2(K_BLUE_VEC2)),
  ];
}

export function turboHex(t: number): string {
  const [r, g, b] = turboColormap(t);
  const hex = (value: number): string =>
    Math.round(value * 255)
      .toString(16)
      .padStart(2, "0");
  return `#${hex(r)}${hex(g)}${hex(b)}`;
}

export function buildTurboLut(size = 256): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(size * 4);
  for (let i = 0; i < size; i += 1) {
    const [r, g, b] = turboColormap(size === 1 ? 0 : i / (size - 1));
    const offset = i * 4;
    lut[offset] = Math.round(r * 255);
    lut[offset + 1] = Math.round(g * 255);
    lut[offset + 2] = Math.round(b * 255);
    lut[offset + 3] = 255;
  }
  return lut;
}
