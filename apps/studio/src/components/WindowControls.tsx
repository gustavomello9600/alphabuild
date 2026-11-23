import { useEffect, useState } from "react";
import { Maximize2, Minimize2 } from "lucide-react";
import { isTauriRuntime } from "../lib/ipc";

/**
 * Fullscreen toggle for the native window. F11 works globally; the same
 * control is available in the top pill bar so the option is discoverable.
 */
export function WindowControls() {
  const [fullscreen, setFullscreen] = useState(false);

  useEffect(() => {
    if (isTauriRuntime()) {
      let disposed = false;
      let timer: number | undefined;
      const sync = () => {
        void import("@tauri-apps/api/window").then(async ({ getCurrentWindow }) => {
          try {
            const value = await getCurrentWindow().isFullscreen();
            if (!disposed) setFullscreen(value);
          } catch {
            /* capability may be unavailable in an old build */
          }
        });
      };
      sync();
      const onResize = () => {
        window.clearTimeout(timer);
        timer = window.setTimeout(sync, 120);
      };
      window.addEventListener("resize", onResize);
      return () => {
        disposed = true;
        window.clearTimeout(timer);
        window.removeEventListener("resize", onResize);
      };
    }
    const onFullscreenChange = () => setFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", onFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", onFullscreenChange);
  }, []);

  async function toggleFullscreen() {
    if (isTauriRuntime()) {
      const { getCurrentWindow } = await import("@tauri-apps/api/window");
      const windowHandle = getCurrentWindow();
      const next = !(await windowHandle.isFullscreen());
      await windowHandle.setFullscreen(next);
      setFullscreen(next);
      return;
    }
    if (document.fullscreenElement) {
      await document.exitFullscreen();
    } else {
      await document.documentElement.requestFullscreen();
    }
  }

  useEffect(() => {
    const handle = (event: KeyboardEvent) => {
      if (event.key !== "F11") return;
      event.preventDefault();
      void toggleFullscreen();
    };
    window.addEventListener("keydown", handle);
    return () => window.removeEventListener("keydown", handle);
  }, [fullscreen]);

  return (
    <button
      type="button"
      className="pill-button"
      data-active={fullscreen}
      onClick={() => void toggleFullscreen()}
      title={fullscreen ? "Sair da tela cheia (F11)" : "Tela cheia (F11)"}
    >
      {fullscreen ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
      <span className="hide-compact">{fullscreen ? "Sair" : "Tela cheia"}</span>
    </button>
  );
}
