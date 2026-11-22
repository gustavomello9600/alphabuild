import { describe, expect, it } from "vitest";
import {
  clamp,
  formatCompliance,
  formatCount,
  formatMaxDisplacement,
  formatMilliseconds,
  formatPercent,
  formatReward,
  formatScientific,
  formatSigned,
  formatVolume,
  lerp,
  safeRatio,
  formatNumber,
} from "./formatters";

describe("scientific formatters", () => {
  it("formats numbers with tabular-safe output", () => {
    expect(formatNumber(1.23456)).toBe("1.23");
    expect(formatNumber(null)).toBe("—");
    expect(formatNumber(Number.NaN)).toBe("—");
    expect(formatSigned(0.5821)).toBe("+0.582");
    expect(formatSigned(-0.1)).toBe("-0.100");
  });

  it("formats scientific values for compliance", () => {
    expect(formatScientific(0)).toBe("0");
    expect(formatScientific(312.4)).toBe("312.4");
    expect(formatScientific(1_000_000)).toBe("1.00e+6");
    expect(formatCompliance(412.83)).toBe("412.8");
  });

  it("converts raw displacement to meters with the v3.3 scale", () => {
    expect(formatMaxDisplacement(500_000, 500_000)).toBe("1.00 m");
    expect(formatMaxDisplacement(41.2)).toMatch(/m$/);
    expect(formatMaxDisplacement(null)).toBe("—");
  });

  it("formats latency, percentages, volume and counts", () => {
    expect(formatMilliseconds(0.0184)).toBe("18 µs");
    expect(formatMilliseconds(18.4)).toBe("18.4 ms");
    expect(formatMilliseconds(1400)).toBe("1.40 s");
    expect(formatPercent(0.112)).toBe("11.20%");
    expect(formatVolume(0.0842)).toBe("8.42%");
    expect(formatCount(16384)).toBe("16,384");
    expect(formatReward(-0.6)).toBe("-0.6000");
  });

  it("clamps and interpolates safely", () => {
    expect(clamp(4, 0, 3)).toBe(3);
    expect(clamp(-2, 0, 3)).toBe(0);
    expect(lerp(0, 10, 0.25)).toBe(2.5);
    expect(safeRatio(1, 0)).toBe(0);
    expect(safeRatio(3, 4)).toBe(0.75);
  });
});

