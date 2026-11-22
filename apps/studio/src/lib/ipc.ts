import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type {
  CachedInferOverlay,
  ComparisonMetrics,
  ComputeConnectionState,
  ComputeProfile,
  ComputeStatusEvent,
  DataSourceRef,
  DatabaseDescriptor,
  EntryListPage,
  EntryListQuery,
  EntryMetadata,
  ExportResult,
  ExportSource,
  FemEvalXRayRequest,
  FrameRef,
  HarvestClusterStatus,
  InferXRayRequest,
  LoadedModelInfo,
  LocalTopologyReport,
  MeshOptions,
  ModelSummary,
  SessionCommand,
  SessionHandle,
  SessionProgressEvent,
  SessionStateEvent,
  SessionStepReadyEvent,
  SidecarStatus,
  StartSessionRequest,
  StudioVoxelState,
} from "../types/contracts";

export function isTauriRuntime(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

function requireRuntime(): void {
  if (!isTauriRuntime()) {
    throw new Error(
      "Tauri runtime indisponível. Rode `npm run tauri dev` ou abra o aplicativo desktop.",
    );
  }
}

async function call<T>(command: string, args: Record<string, unknown> = {}): Promise<T> {
  requireRuntime();
  return invoke<T>(command, args);
}

function toArrayBuffer(value: ArrayBuffer | Uint8Array | number[]): ArrayBuffer {
  if (value instanceof ArrayBuffer) return value;
  if (value instanceof Uint8Array) {
    return value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength) as ArrayBuffer;
  }
  return new Uint8Array(value).buffer;
}

// ---------------------------------------------------------------- datasets
export const listLocalDatabases = (extraDirs: string[] = []): Promise<DatabaseDescriptor[]> =>
  call("list_local_databases", { extra_dirs: extraDirs });

export const openDatabaseFile = (path: string): Promise<DatabaseDescriptor> =>
  call("open_database_file", { path });

export const listEntries = (
  source: DataSourceRef,
  query: EntryListQuery,
): Promise<EntryListPage> => call("list_entries", { source, query });

export const getEntryMetadata = (
  source: DataSourceRef,
  entryId: string,
): Promise<EntryMetadata> => call("get_entry_metadata", { source, entry_id: entryId });

export async function getFrameMesh(
  source: DataSourceRef,
  entryId: string,
  step: number,
  options: MeshOptions,
): Promise<ArrayBuffer> {
  const result = await call<ArrayBuffer | Uint8Array | number[]>("get_frame_mesh", {
    source,
    entry_id: entryId,
    step,
    options,
  });
  return toArrayBuffer(result);
}

export const prefetchFrames = (
  source: DataSourceRef,
  entryId: string,
  start: number,
  count: number,
  options: MeshOptions,
): Promise<void> =>
  call("prefetch_frames", {
    source,
    entry_id: entryId,
    start,
    count,
    options,
  });

export const computeAbComparison = (a: FrameRef, b: FrameRef): Promise<ComparisonMetrics> =>
  call("compute_ab_comparison", { a, b });

// ---------------------------------------------------------------- studio
export const studioAnalyzeLocal = (state: StudioVoxelState): Promise<LocalTopologyReport> =>
  call("studio_analyze_local", { state });

export async function studioBuildMesh(
  state: StudioVoxelState,
  overlay: CachedInferOverlay | null = null,
): Promise<ArrayBuffer> {
  const result = await call<ArrayBuffer | Uint8Array | number[]>("studio_build_mesh", {
    state,
    overlay,
  });
  return toArrayBuffer(result);
}

// ---------------------------------------------------------------- compute
export const computeListProfiles = (): Promise<ComputeProfile[]> =>
  call("compute_list_profiles");

export const computeSaveProfile = (profile: ComputeProfile): Promise<ComputeProfile> =>
  call("compute_save_profile", { profile });

export const computeDeleteProfile = (profileId: string): Promise<void> =>
  call("compute_delete_profile", { profile_id: profileId });

export const computeActivateProfile = (profileId: string): Promise<ComputeConnectionState> =>
  call("compute_activate_profile", { profile_id: profileId });

export const computeSidecarControl = (
  action: "start" | "stop" | "status",
  pythonBin: string | null,
  port: number,
): Promise<SidecarStatus> =>
  call("compute_sidecar_control", { action, python_bin: pythonBin, port });

export const computeListModels = (): Promise<ModelSummary[]> => call("compute_list_models");

export const computeLoadModel = (
  modelId: string,
  device = "auto",
  openvinoHint = "LATENCY",
): Promise<LoadedModelInfo> =>
  call("compute_load_model", {
    req: { model_id: modelId, device, openvino_hint: openvinoHint },
  });

export const computeUnloadModel = (): Promise<void> => call("compute_unload_model");

export async function computeInferXray(request: InferXRayRequest): Promise<ArrayBuffer> {
  const result = await call<ArrayBuffer | Uint8Array | number[]>("compute_infer_xray", {
    req: request,
  });
  return toArrayBuffer(result);
}

export async function computeEvaluateFem(request: FemEvalXRayRequest): Promise<ArrayBuffer> {
  const result = await call<ArrayBuffer | Uint8Array | number[]>("compute_evaluate_fem", {
    req: request,
  });
  return toArrayBuffer(result);
}

export const computeHarvestStatus = (supervisorUrl?: string): Promise<HarvestClusterStatus> =>
  call("compute_harvest_status", { supervisor_url: supervisorUrl ?? null });

// ---------------------------------------------------------------- sessions
export const sessionStart = (request: StartSessionRequest): Promise<SessionHandle> =>
  call("session_start", { req: request });

export const sessionSendCommand = (
  sessionId: string,
  cmd: SessionCommand,
): Promise<void> => call("session_send_command", { session_id: sessionId, cmd });

export async function sessionGetStepMesh(
  sessionId: string,
  step: number,
): Promise<ArrayBuffer> {
  const result = await call<ArrayBuffer | Uint8Array | number[]>("session_get_step_mesh", {
    session_id: sessionId,
    step,
  });
  return toArrayBuffer(result);
}

// ---------------------------------------------------------------- export
export const exportGeometry = (
  source: ExportSource,
  format: "stl" | "obj" | "vti",
  outputPath: string,
): Promise<ExportResult> =>
  call("export_geometry", { source, format, output_path: outputPath });

export const exportPngBytes = (pngBase64: string, outputPath: string): Promise<ExportResult> =>
  call("export_png_bytes", { png_base64: pngBase64, output_path: outputPath });

// ---------------------------------------------------------------- dialogs
export async function pickDatabaseFile(): Promise<string | null> {
  requireRuntime();
  const { open } = await import("@tauri-apps/plugin-dialog");
  const selected = await open({
    multiple: false,
    directory: false,
    filters: [{ name: "AlphaBuilder SQLite", extensions: ["db", "sqlite"] }],
  });
  return typeof selected === "string" ? selected : null;
}

export async function pickSavePath(
  defaultPath: string,
  filterName: string,
  extension: string,
): Promise<string | null> {
  requireRuntime();
  const { save } = await import("@tauri-apps/plugin-dialog");
  const selected = await save({
    defaultPath,
    filters: [{ name: filterName, extensions: [extension] }],
  });
  return typeof selected === "string" ? selected : null;
}

export async function pickDirectory(): Promise<string | null> {
  requireRuntime();
  const { open } = await import("@tauri-apps/plugin-dialog");
  const selected = await open({ directory: true, multiple: false });
  return typeof selected === "string" ? selected : null;
}

// ---------------------------------------------------------------- events
export function onComputeStatus(
  handler: (payload: ComputeStatusEvent) => void,
): Promise<UnlistenFn> {
  return listen<ComputeStatusEvent>("ab://compute-status", (event) => handler(event.payload));
}

export function onSessionState(
  handler: (payload: SessionStateEvent) => void,
): Promise<UnlistenFn> {
  return listen<SessionStateEvent>("ab://session-state", (event) => handler(event.payload));
}

export function onSessionProgress(
  handler: (payload: SessionProgressEvent) => void,
): Promise<UnlistenFn> {
  return listen<SessionProgressEvent>("ab://session-progress", (event) =>
    handler(event.payload),
  );
}

export function onSessionStepReady(
  handler: (payload: SessionStepReadyEvent) => void,
): Promise<UnlistenFn> {
  return listen<SessionStepReadyEvent>("ab://session-step-ready", (event) =>
    handler(event.payload),
  );
}

export function onSidecarLog(
  handler: (payload: { level: string; line: string }) => void,
): Promise<UnlistenFn> {
  return listen<{ level: string; line: string }>("ab://sidecar-log", (event) =>
    handler(event.payload),
  );
}

export async function listenToFileDrop(
  handler: (paths: string[]) => void,
): Promise<UnlistenFn | null> {
  if (!isTauriRuntime()) return null;
  const { getCurrentWebview } = await import("@tauri-apps/api/webview");
  return getCurrentWebview().onDragDropEvent((event) => {
    if (event.payload.type === "drop") {
      const paths = event.payload.paths.filter((path) => path.endsWith(".db"));
      if (paths.length > 0) handler(paths);
    }
  });
}
