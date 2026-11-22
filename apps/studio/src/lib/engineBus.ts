import type { ViewportEngine } from "../engine3d/ViewportEngine";

let active: ViewportEngine | null = null;
const engines = new Map<string, ViewportEngine>();

export function setActiveEngine(engine: ViewportEngine | null): void {
  active = engine;
}

export function getActiveEngine(): ViewportEngine | null {
  return active;
}

export function registerEngine(slot: string, engine: ViewportEngine): void {
  engines.set(slot, engine);
}

export function unregisterEngine(slot: string): void {
  engines.delete(slot);
}

export function getEngine(slot: string): ViewportEngine | null {
  return engines.get(slot) ?? null;
}
