import { useCallback, useEffect, useState } from "react";
import {
  AlertTriangle,
  ChevronRight,
  Database,
  FilePlus2,
  RefreshCw,
  Search,
  X,
} from "lucide-react";
import { useAppStore } from "../store/useAppStore";
import { useInspectorStore } from "../store/useInspectorStore";
import { useCompareStore } from "../store/useCompareStore";
import {
  getEntryMetadata,
  isTauriRuntime,
  listenToFileDrop,
  listEntries,
  listLocalDatabases,
  openDatabaseFile,
  pickDatabaseFile,
} from "../lib/ipc";
import type {
  DataSourceRef,
  DatabaseDescriptor,
  EntryListQuery,
  EntrySummary,
} from "../types/contracts";
import { formatCount, formatPercent, formatReward } from "../lib/formatters";

function sourceOf(descriptor: DatabaseDescriptor): DataSourceRef {
  return {
    origin: "local",
    db_id: descriptor.db_id,
    kind: descriptor.kind,
  };
}

export function LibraryDrawer() {
  const open = useAppStore((state) => state.libraryOpen);
  const mode = useAppStore((state) => state.mode);
  const activeSlot = useCompareStore((state) => state.activeSlot);
  const setOpen = useAppStore((state) => state.setLibraryOpen);
  const setMode = useAppStore((state) => state.setMode);
  const [databases, setDatabases] = useState<DatabaseDescriptor[]>([]);
  const [selected, setSelected] = useState<DatabaseDescriptor | null>(null);
  const [entries, setEntries] = useState<EntrySummary[]>([]);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState<NonNullable<EntryListQuery["sort"]>>("recent");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshDatabases = useCallback(async () => {
    if (!isTauriRuntime()) return;
    try {
      const list = await listLocalDatabases();
      setDatabases(list);
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, []);

  useEffect(() => {
    void refreshDatabases();
    const handle = () => {
      void refreshDatabases();
      setOpen(true);
    };
    window.addEventListener("ab://library-refresh", handle);
    return () => window.removeEventListener("ab://library-refresh", handle);
  }, [refreshDatabases, setOpen]);

  useEffect(() => {
    let disposed = false;
    let unlisten: (() => void) | null = null;
    void listenToFileDrop((paths) => {
      const first = paths[0];
      if (first) void openPath(first);
    }).then((unsubscribe) => {
      if (disposed) unsubscribe?.();
      else unlisten = unsubscribe;
    });
    return () => {
      disposed = true;
      unlisten?.();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadEntries = useCallback(
    async (descriptor: DatabaseDescriptor, query: EntryListQuery) => {
      setLoading(true);
      setError(null);
      try {
        const page = await listEntries(sourceOf(descriptor), query);
        setEntries(page.items);
        setTotal(page.total_count);
      } catch (cause) {
        setEntries([]);
        setTotal(0);
        setError(cause instanceof Error ? cause.message : String(cause));
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    if (!selected || !open) return;
    void loadEntries(selected, {
      limit: 80,
      offset: 0,
      search: search || undefined,
      sort,
    });
  }, [selected, open, search, sort, loadEntries]);

  async function openPath(path: string) {
    if (!isTauriRuntime()) {
      setError("Rode o aplicativo desktop (`npm run tauri dev`) para abrir bancos SQLite.");
      return;
    }
    try {
      const descriptor = await openDatabaseFile(path);
      await refreshDatabases();
      setSelected(descriptor);
      setOpen(true);
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  async function handlePick() {
    const path = await pickDatabaseFile();
    if (path) await openPath(path);
  }

  async function handleEntry(entry: EntrySummary) {
    if (!selected) return;
    try {
      const source = sourceOf(selected);
      const metadata = await getEntryMetadata(source, entry.entry_id);
      if (mode === "compare") {
        const compare = useCompareStore.getState();
        compare.setSlot(compare.activeSlot, { source, entryId: entry.entry_id, metadata });
        setOpen(false);
        return;
      }
      useInspectorStore.getState().openEntry(source, entry.entry_id, metadata);
      setMode("inspector");
      setOpen(false);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  if (!open) return null;

  return (
    <aside className="library-drawer glass">
      <div className="dock-header">
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <LibraryGlyph />
          <span className="dock-title">Biblioteca de Dados</span>
          {mode === "compare" ? (
            <span className="badge" data-tone="accent">
              SLOT {activeSlot.toUpperCase()}
            </span>
          ) : null}
        </div>
        <button
          type="button"
          className="icon-button"
          onClick={() => setOpen(false)}
          title="Fechar (L)"
        >
          <X size={14} />
        </button>
      </div>

      <div className="library-drawer__body scroll-thin">
        <div style={{ display: "flex", gap: 6 }}>
          <button type="button" className="pill-button pill-button--accent" onClick={handlePick}>
            <FilePlus2 size={13} />
            Abrir .db
          </button>
          <button
            type="button"
            className="icon-button"
            onClick={() => void refreshDatabases()}
            title="Atualizar biblioteca"
          >
            <RefreshCw size={13} />
          </button>
        </div>

        {!isTauriRuntime() ? (
          <div className="empty-card">
            O navegador não acessa SQLite local. Rode o aplicativo desktop para inspecionar bancos
            v3.3.
          </div>
        ) : null}

        {error ? (
          <div className="empty-card" style={{ borderColor: "var(--accent-magenta)" }}>
            <AlertTriangle size={15} style={{ margin: "0 auto 6px", color: "var(--accent-magenta)" }} />
            {error}
          </div>
        ) : null}

        <div className="panel-section">
          <div className="panel-heading">
            <span>Bancos locais</span>
            <span className="mono">{databases.length}</span>
          </div>
          {databases.length === 0 ? (
            <div className="empty-card">
              Nenhum banco encontrado. Arraste um <span className="mono">.db</span> para a janela ou
              clique em Abrir.
            </div>
          ) : (
            databases.map((descriptor) => (
              <button
                key={descriptor.db_id}
                type="button"
                className="db-row"
                data-active={selected?.db_id === descriptor.db_id}
                onClick={() => setSelected(descriptor)}
              >
                <span className="db-row__title">
                  <span style={{ display: "flex", alignItems: "center", gap: 6, minWidth: 0 }}>
                    <Database size={12} style={{ flexShrink: 0, color: "var(--accent-cyan)" }} />
                    <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>
                      {descriptor.name}
                    </span>
                  </span>
                  <ChevronRight size={12} style={{ color: "var(--text-muted)" }} />
                </span>
                <span className="db-row__meta">
                  {descriptor.kind} · {formatCount(descriptor.item_count)} episódios ·{" "}
                  {formatCount(descriptor.total_steps)} passos · {descriptor.size_mb.toFixed(1)} MB
                </span>
                {descriptor.has_legacy_pickle ? (
                  <span className="badge" data-tone="warn">
                    pickle legado
                  </span>
                ) : null}
              </button>
            ))
          )}
        </div>

        {selected ? (
          <div className="panel-section">
            <div className="panel-heading">
              <span>Episódios / Partidas</span>
              <span className="mono">{formatCount(total)}</span>
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              <label style={{ position: "relative", flex: 1 }}>
                <Search
                  size={12}
                  style={{
                    position: "absolute",
                    left: 9,
                    top: 9,
                    color: "var(--text-muted)",
                  }}
                />
                <input
                  className="field-input"
                  style={{ paddingLeft: 27 }}
                  placeholder="Buscar por ID…"
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                />
              </label>
              <select
                className="field-input"
                style={{ width: 118 }}
                value={sort}
                onChange={(event) =>
                  setSort(event.target.value as NonNullable<EntryListQuery["sort"]>)
                }
              >
                <option value="recent">Recentes</option>
                <option value="reward_desc">Reward ↓</option>
                <option value="compliance_asc">Compliance ↑</option>
                <option value="seed_asc">Seed ↑</option>
              </select>
            </div>
            {selected.has_legacy_pickle ? (
              <div className="empty-card" style={{ textAlign: "left", lineHeight: 1.55 }}>
                <span className="badge" data-tone="warn" style={{ marginBottom: 6 }}>
                  LEGACY_PICKLE_BLOB
                </span>
                <div>
                  Este banco usa pickle legado. Use o servidor v4
                  <span className="mono"> /api/v1/datasets/…/migrate-v33</span> para gerar a cópia
                  v3.3 sem sobrescrever o original.
                </div>
              </div>
            ) : null}
            {loading ? <div className="empty-card">Lendo índices SQLite em modo read-only…</div> : null}
            {!loading && entries.length === 0 ? (
              <div className="empty-card">
                Nenhum episódio com esses filtros. Ajuste a busca ou abra outro banco.
              </div>
            ) : null}
            {entries.map((entry) => (
              <button
                key={entry.entry_id}
                type="button"
                className="entry-row"
                onClick={() => void handleEntry(entry)}
                disabled={selected.kind === "legacy_v1"}
              >
                <span style={{ minWidth: 0, display: "flex", flexDirection: "column", gap: 2 }}>
                  <span className="entry-row__id">{entry.entry_id}</span>
                  <span className="entry-row__metrics">
                    <span>{entry.total_steps} passos</span>
                    <span>F1 {entry.steps_phase1}</span>
                    <span>F2 {entry.steps_phase2}</span>
                    {entry.bc_type ? <span>{entry.bc_type}</span> : null}
                  </span>
                </span>
                <span
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "flex-end",
                    gap: 2,
                  }}
                >
                  <span className="mono" style={{ fontSize: 10, color: "var(--text-secondary)" }}>
                    R {formatReward(entry.final_reward)}
                  </span>
                  <span className="entry-row__metrics">
                    <span>{formatPercent(entry.final_volume)}</span>
                    {entry.seed !== null ? <span>seed {entry.seed}</span> : null}
                  </span>
                </span>
              </button>
            ))}
          </div>
        ) : null}
      </div>
    </aside>
  );
}

function LibraryGlyph() {
  return (
    <span
      style={{
        width: 18,
        height: 18,
        borderRadius: 6,
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        background: "color-mix(in srgb, var(--accent-cyan) 10%, transparent)",
        border: "1px solid var(--border-active)",
        color: "var(--accent-cyan)",
      }}
    >
      <Database size={10} />
    </span>
  );
}
