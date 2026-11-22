const DASH = "—";

export function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return value.toFixed(digits);
}

export function formatSigned(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}`;
}

export function formatScientific(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 0.01 && abs < 10000) {
    return Number(value.toPrecision(digits + 2)).toString();
  }
  return value.toExponential(digits);
}

export function formatCompliance(value: number | null | undefined): string {
  return formatScientific(value, 2);
}

export function formatMaxDisplacement(
  value: number | null | undefined,
  scale = 500_000,
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  const meters = value / scale;
  if (meters >= 0.01 && meters < 1000) return `${meters.toFixed(2)} m`;
  return `${formatScientific(meters, 2)} m`;
}

export function formatMilliseconds(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  if (value < 1) return `${(value * 1000).toFixed(0)} µs`;
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${value.toFixed(1)} ms`;
}

export function formatPercent(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatReward(value: number | null | undefined): string {
  return formatSigned(value, 4);
}

export function formatVolume(value: number | null | undefined): string {
  return formatPercent(value, 2);
}

export function formatCount(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH;
  return Math.round(value).toLocaleString("en-US");
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function safeRatio(numerator: number, denominator: number): number {
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator === 0) {
    return 0;
  }
  return numerator / denominator;
}
