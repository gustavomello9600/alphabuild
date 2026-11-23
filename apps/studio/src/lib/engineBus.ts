import type { ViewportEngine } from "../engine3d/ViewportEngine";

let active: ViewportEngine | null = null;
const engines = new Map<string, ViewportEngine>();

declare global {
  interface Window {
    __abEngine?: ViewportEngine | null;
  }
}

export function setActiveEngine(engine: ViewportEngine | null): void {
  active = engine;
  if (import.meta.env.DEV && typeof window !== "undefined") {
    window.__abEngine = engine;
  }
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
