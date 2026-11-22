import { create } from "zustand";
import type { AppMode } from "../types/contracts";

export type ThemeMode = "dark" | "tcc-light";

interface AppStore {
  mode: AppMode;
  theme: ThemeMode;
  libraryOpen: boolean;
  paletteOpen: boolean;
  exportOpen: boolean;
  computeOpen: boolean;
  figureMode: boolean;
  setMode: (mode: AppMode) => void;
  setTheme: (theme: ThemeMode) => void;
  toggleTheme: () => void;
  toggleLibrary: () => void;
  setLibraryOpen: (open: boolean) => void;
  setPaletteOpen: (open: boolean) => void;
  setExportOpen: (open: boolean) => void;
  setComputeOpen: (open: boolean) => void;
  setFigureMode: (value: boolean) => void;
}

export const useAppStore = create<AppStore>((set) => ({
  mode: "inspector",
  theme: "dark",
  libraryOpen: false,
  paletteOpen: false,
  exportOpen: false,
  computeOpen: false,
  figureMode: false,
  setMode: (mode) => set({ mode }),
  setTheme: (theme) => set({ theme }),
  toggleTheme: () => set((state) => ({ theme: state.theme === "dark" ? "tcc-light" : "dark" })),
  toggleLibrary: () => set((state) => ({ libraryOpen: !state.libraryOpen })),
  setLibraryOpen: (libraryOpen) => set({ libraryOpen }),
  setPaletteOpen: (paletteOpen) => set({ paletteOpen }),
  setExportOpen: (exportOpen) => set({ exportOpen }),
  setComputeOpen: (computeOpen) => set({ computeOpen }),
  setFigureMode: (figureMode) => set({ figureMode }),
}));
