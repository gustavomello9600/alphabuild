#!/usr/bin/env python3
"""
Token-authenticated gateway + read-only public status dashboard for the
AlphaBuilder v3.3 Harvest Supervisor (bound on 127.0.0.1:8455 behind cloudflared).

Public read-only routes (zero credentials / tokens exposed):
  - GET / , /index.html , /dashboard   -> Interactive HTML status dashboard
  - GET /api/public/dashboard          -> Read-only JSON snapshot from SQLite (mode=ro)
  - GET /favicon.ico                   -> 204 No Content

Authenticated worker routes (all require /t/<TOKEN>/ prefix, else 403):
  - GET  /t/<TOKEN>/boot/<WORKER_ID>   -> bash bootstrap script for Colab runtime
  - ANY  /t/<TOKEN>/api/v1/*           -> proxied to the Tailscale supervisor
Stdlib only. Binds 127.0.0.1 only.
"""
import base64
import hmac
import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

HOST, PORT = "127.0.0.1", 8455
UPSTREAM = "http://100.104.121.28:8450"
SUPERVISOR_DB = Path("/home/ubuntu/alphabuild_supervisor/supervisor_state.db")
TOKEN = Path("/home/ubuntu/alphabuild_supervisor/colab_token.txt").read_text().strip()
RCLONE_CONF = Path("/home/ubuntu/.config/rclone/rclone.conf")
COMMIT = "ee04deb"
DEFAULT_LEASE_TTL = 2400.0
PROD_TARGET_SHARDS = 400
PROD_TARGET_EPISODES = 4000
BRT_TZ = timezone(timedelta(hours=-3))
WORKER_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")

BOOT_TEMPLATE = r"""#!/bin/bash
set -e
mkdir -p /root/.config/rclone
echo '@@CONF@@' | base64 -d > /root/.config/rclone/rclone.conf
chmod 600 /root/.config/rclone/rclone.conf
if ! command -v rclone >/dev/null 2>&1; then
  cd /tmp && curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o rclone.zip \
    && python3 -c "import zipfile;zipfile.ZipFile('rclone.zip').extractall('rclone_x')" \
    && cp rclone_x/*/rclone /usr/local/bin/rclone && chmod +x /usr/local/bin/rclone
fi
if [ ! -x /content/env/bin/python ]; then
  cd /tmp && curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
  mv /tmp/bin/micromamba /usr/local/bin/micromamba
  micromamba create -y -q -p /content/env -c conda-forge python=3.10 fenics-dolfinx=0.10.0 mpich mpi4py scipy numpy tqdm
fi
if [ ! -d /content/alphabuild_harvest/.git ]; then
  git clone -q https://github.com/gustavomello9600/alphabuild.git /content/alphabuild_harvest
fi
cd /content/alphabuild_harvest && git fetch -q --all && git checkout -q -f @@COMMIT@@
cat > /content/run_colab_worker.py <<'PYEOF'
import sys, runpy, urllib.request
_orig = urllib.request.Request.__init__
def _init(self, url, data=None, headers={}, *a, **k):
    h = dict(headers); h.setdefault("User-Agent", "AlphaBuilderWorker/3.3")
    _orig(self, url, data, h, *a, **k)
urllib.request.Request.__init__ = _init
sys.argv = ["harvest_worker.py"] + sys.argv[1:]
runpy.run_path("/content/alphabuild_harvest/scripts/harvest_worker.py", run_name="__main__")
PYEOF
mkdir -p /content/alphabuild_work_@@WID@@
exec python3 /content/run_colab_worker.py --supervisor-url https://@@HOST@@/t/@@TOKEN@@ --worker-id @@WID@@ \
  --slots @@SLOTS@@ --executor native --python-bin /content/env/bin/python --repo-dir /content/alphabuild_harvest \
  --work-dir /content/alphabuild_work_@@WID@@ --rclone-bin /usr/local/bin/rclone --rclone-remote gdrive_harvest: \
  --min-disk-gb 5 --min-ram-mb 800 --retain-local-shards 2 --expected-commit @@COMMIT@@
"""


def _fallback_eps_per_slot_h(worker_id: str) -> float:
    wid = worker_id.lower()
    if wid.startswith("katastopoulos"):
        return 11.5
    if wid.startswith("core-ultra5"):
        return 10.5
    if wid.startswith("google-colab"):
        return 9.0
    if wid.startswith("archgustavo"):
        return 9.0
    if wid.startswith("gustavo-oracle"):
        return 8.2
    if wid.startswith("idcons-smart"):
        return 5.5
    return 8.0


def build_dashboard_data() -> Dict[str, Any]:
    now = time.time()
    conn = sqlite3.connect(f"file:{SUPERVISOR_DB}?mode=ro", uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        shard_rows = conn.execute(
            """
            SELECT shard_id, wave, seed_start, seed_count, status,
                   assigned_worker, lease_expires_at, attempt_count,
                   manifest_json, last_error, updated_at
            FROM shards
            ORDER BY wave ASC, seed_start ASC
            """
        ).fetchall()
        worker_rows = conn.execute(
            "SELECT * FROM workers ORDER BY last_seen DESC"
        ).fetchall()
    finally:
        conn.close()

    by_wave: Dict[str, Dict[str, int]] = {}
    verified_shards: List[Dict[str, Any]] = []
    active_by_worker: Dict[str, List[Dict[str, Any]]] = {}
    worker_perf: Dict[str, Dict[str, Any]] = {}

    total_verified_episodes = 0
    total_verified_records = 0
    total_db_bytes = 0
    prod_verified_episodes = 0
    prod_verified_records = 0
    prod_db_bytes = 0
    in_flight_prod_leases = 0

    for r in shard_rows:
        wave = r["wave"]
        status = r["status"]
        wmap = by_wave.setdefault(wave, {"PENDING": 0, "LEASED": 0, "RUNNING": 0, "VERIFIED": 0, "FAILED": 0})
        wmap[status] = wmap.get(status, 0) + 1

        if status in ("LEASED", "RUNNING"):
            if wave == "prod":
                in_flight_prod_leases += 1
            wid = r["assigned_worker"] or "unknown"
            ttl_rem = round(max(0.0, float(r["lease_expires_at"] or 0.0) - now), 1)
            active_by_worker.setdefault(wid, []).append(
                {
                    "shard_id": r["shard_id"],
                    "wave": wave,
                    "seed_range": [int(r["seed_start"]), int(r["seed_start"]) + int(r["seed_count"]) - 1],
                    "status": status,
                    "attempt": int(r["attempt_count"]),
                    "ttl_rem_sec": ttl_rem,
                    "ttl_rem_min": round(ttl_rem / 60.0, 1),
                }
            )
        elif status == "VERIFIED":
            m = json.loads(r["manifest_json"]) if r["manifest_json"] else {}
            wid = r["assigned_worker"] or m.get("worker_id") or "unknown"
            eps = len(m.get("seeds_completed", [])) or int(r["seed_count"])
            rec = int(m.get("record_count", 0))
            bsize = int(m.get("db_size_bytes", 0))
            dur = float(m.get("duration_total_sec", 0.0))

            total_verified_episodes += eps
            total_verified_records += rec
            total_db_bytes += bsize
            if wave == "prod":
                prod_verified_episodes += eps
                prod_verified_records += rec
                prod_db_bytes += bsize

            wp = worker_perf.setdefault(
                wid,
                {
                    "verified_shards": 0,
                    "verified_prod_shards": 0,
                    "verified_episodes": 0,
                    "verified_records": 0,
                    "verified_bytes": 0,
                    "duration_sec": 0.0,
                },
            )
            wp["verified_shards"] += 1
            if wave == "prod":
                wp["verified_prod_shards"] += 1
            wp["verified_episodes"] += eps
            wp["verified_records"] += rec
            wp["verified_bytes"] += bsize
            wp["duration_sec"] += dur

            verified_shards.append(
                {
                    "ts": float(r["updated_at"]),
                    "shard_id": r["shard_id"],
                    "wave": wave,
                    "worker_id": wid,
                    "eps": eps,
                    "records": rec,
                    "duration_sec": round(dur, 1),
                }
            )

    verified_shards.sort(key=lambda x: x["ts"])
    cum_prod_eps = 0
    cum_total_eps = 0
    cum_records = 0
    timeline: List[Dict[str, Any]] = []
    for item in verified_shards:
        cum_total_eps += item["eps"]
        if item["wave"] == "prod":
            cum_prod_eps += item["eps"]
        cum_records += item["records"]
        timeline.append(
            {
                "ts": round(item["ts"], 1),
                "time_brt": datetime.fromtimestamp(item["ts"], tz=BRT_TZ).strftime("%d/%m %H:%M"),
                "shard_id": item["shard_id"],
                "wave": item["wave"],
                "worker_id": item["worker_id"],
                "duration_sec": item["duration_sec"],
                "cum_prod_eps": cum_prod_eps,
                "cum_total_eps": cum_total_eps,
                "cum_records": cum_records,
            }
        )

    workers_out: List[Dict[str, Any]] = []
    cluster_eps_per_hour = 0.0
    fixed_nodes_eps_per_hour = 0.0
    active_slots_total = 0
    configured_slots_total = 0
    online_workers_count = 0

    for wr in worker_rows:
        wid = wr["worker_id"]
        sec_since_seen = round(max(0.0, now - float(wr["last_seen"])), 1)
        if sec_since_seen < 180.0:
            w_status = "ONLINE"
        elif sec_since_seen < 600.0:
            w_status = "STALE"
        else:
            w_status = "OFFLINE"

        slots = int(wr["slots"] or 0)
        active_shards = active_by_worker.get(wid, [])
        active_slots = len(active_shards)
        wp = worker_perf.get(
            wid,
            {
                "verified_shards": 0,
                "verified_prod_shards": 0,
                "verified_episodes": 0,
                "verified_records": 0,
                "verified_bytes": 0,
                "duration_sec": 0.0,
            },
        )

        if wp["verified_episodes"] > 0 and wp["duration_sec"] > 0.0:
            avg_sec_per_ep = round(wp["duration_sec"] / wp["verified_episodes"], 1)
            eps_per_slot_h = round(3600.0 * wp["verified_episodes"] / wp["duration_sec"], 2)
            rate_source = "medido"
        else:
            eps_per_slot_h = _fallback_eps_per_slot_h(wid)
            avg_sec_per_ep = round(3600.0 / eps_per_slot_h, 1)
            rate_source = "estimado"

        effective_slots = active_slots if active_slots > 0 else slots
        is_up = w_status != "OFFLINE"
        node_eps_per_h = round(eps_per_slot_h * effective_slots, 1) if is_up else 0.0

        if is_up:
            online_workers_count += 1
            active_slots_total += active_slots
            configured_slots_total += slots
            cluster_eps_per_hour += node_eps_per_h
            if not wid.lower().startswith("google-colab"):
                fixed_nodes_eps_per_hour += node_eps_per_h

        workers_out.append(
            {
                "worker_id": wid,
                "status": w_status,
                "is_colab": wid.lower().startswith("google-colab"),
                "arch": wr["arch"] or "?",
                "dolfinx_version": wr["dolfinx_version"] or "?",
                "code_commit": wr["code_commit"] or "?",
                "slots": slots,
                "active_slots": active_slots,
                "active_shards": active_shards,
                "verified_shards": wp["verified_shards"],
                "verified_prod_shards": wp["verified_prod_shards"],
                "verified_episodes": wp["verified_episodes"],
                "verified_records": wp["verified_records"],
                "verified_mb": round(wp["verified_bytes"] / (1024.0 * 1024.0), 1),
                "avg_sec_per_ep": avg_sec_per_ep,
                "eps_per_slot_h": eps_per_slot_h,
                "node_eps_per_h": node_eps_per_h,
                "rate_source": rate_source,
                "cpu_busy": round(float(wr["cpu_busy"] or 0.0), 1),
                "ram_free_mb": round(float(wr["ram_free_mb"] or 0.0), 0),
                "disk_free_gb": round(float(wr["disk_free_gb"] or 0.0), 1),
                "sec_since_seen": sec_since_seen,
            }
        )

    workers_out.sort(key=lambda w: (w["status"] == "OFFLINE", -w["node_eps_per_h"], w["worker_id"]))

    cluster_eps_per_hour = round(cluster_eps_per_hour, 1)
    fixed_nodes_eps_per_hour = round(fixed_nodes_eps_per_hour, 1)

    in_flight_equiv_episodes = in_flight_prod_leases * 5
    remaining_prod_episodes = max(0, PROD_TARGET_EPISODES - prod_verified_episodes - in_flight_equiv_episodes)

    eta_hours = round(remaining_prod_episodes / cluster_eps_per_hour, 2) if cluster_eps_per_hour > 0 else None
    eta_iso = (
        datetime.fromtimestamp(now + eta_hours * 3600.0, tz=BRT_TZ).isoformat()
        if eta_hours is not None
        else None
    )
    eta_human_brt = (
        datetime.fromtimestamp(now + eta_hours * 3600.0, tz=BRT_TZ).strftime("%d/%m %H:%M BRT")
        if eta_hours is not None
        else "N/A"
    )

    eta_fixed_hours = (
        round(remaining_prod_episodes / fixed_nodes_eps_per_hour, 2)
        if fixed_nodes_eps_per_hour > 0
        else None
    )
    eta_fixed_human_brt = (
        datetime.fromtimestamp(now + eta_fixed_hours * 3600.0, tz=BRT_TZ).strftime("%d/%m %H:%M BRT")
        if eta_fixed_hours is not None
        else "N/A"
    )

    prod_counts = by_wave.get("prod", {"PENDING": 0, "LEASED": 0, "RUNNING": 0, "VERIFIED": 0, "FAILED": 0})
    pilot_counts = by_wave.get("pilot", {"PENDING": 0, "LEASED": 0, "RUNNING": 0, "VERIFIED": 0, "FAILED": 0})

    return {
        "ok": True,
        "timestamp": round(now, 1),
        "updated_brt": datetime.fromtimestamp(now, tz=BRT_TZ).strftime("%d/%m/%Y %H:%M:%S BRT"),
        "expected_commit": COMMIT,
        "active_wave": "prod",
        "prod": {
            "target_shards": PROD_TARGET_SHARDS,
            "target_episodes": PROD_TARGET_EPISODES,
            "verified_shards": prod_counts.get("VERIFIED", 0),
            "verified_episodes": prod_verified_episodes,
            "verified_records": prod_verified_records,
            "verified_mb": round(prod_db_bytes / (1024.0 * 1024.0), 1),
            "running_shards": prod_counts.get("RUNNING", 0) + prod_counts.get("LEASED", 0),
            "pending_shards": prod_counts.get("PENDING", 0),
            "failed_shards": prod_counts.get("FAILED", 0),
            "in_flight_equiv_episodes": in_flight_equiv_episodes,
            "remaining_episodes": remaining_prod_episodes,
            "progress_pct": round(100.0 * prod_verified_episodes / PROD_TARGET_EPISODES, 2),
        },
        "pilot": {
            "verified_shards": pilot_counts.get("VERIFIED", 0),
            "verified_episodes": total_verified_episodes - prod_verified_episodes,
        },
        "totals": {
            "verified_episodes": total_verified_episodes,
            "verified_records": total_verified_records,
            "verified_mb": round(total_db_bytes / (1024.0 * 1024.0), 1),
            "online_workers": online_workers_count,
            "total_workers": len(workers_out),
            "active_slots": active_slots_total,
            "configured_slots": configured_slots_total,
        },
        "eta": {
            "cluster_eps_per_hour": cluster_eps_per_hour,
            "fixed_nodes_eps_per_hour": fixed_nodes_eps_per_hour,
            "eta_hours": eta_hours,
            "eta_iso": eta_iso,
            "eta_human_brt": eta_human_brt,
            "eta_fixed_hours": eta_fixed_hours,
            "eta_fixed_human_brt": eta_fixed_human_brt,
        },
        "timeline": timeline,
        "workers": workers_out,
    }


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>AlphaBuilder v3.3 — Monitor de Colheita</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0b1220;
    --card: #131c2e;
    --card2: #192338;
    --border: #26334d;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --blue: #38bdf8;
    --green: #22c55e;
    --amber: #f59e0b;
    --purple: #a855f7;
    --red: #ef4444;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 20px; background: var(--bg); color: var(--text);
    font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    line-height: 1.45;
  }
  .wrap { max-width: 1400px; margin: 0 auto; }
  header {
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
    gap: 12px; margin-bottom: 18px; padding-bottom: 14px; border-bottom: 1px solid var(--border);
  }
  .title-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  h1 { margin: 0; font-size: 1.35rem; font-weight: 700; letter-spacing: -0.01em; }
  .pill {
    display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px;
    border-radius: 999px; font-size: 0.75rem; font-weight: 600;
    background: var(--card2); border: 1px solid var(--border); color: var(--text);
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); box-shadow: 0 0 8px var(--green); }
  .dot.warn { background: var(--amber); box-shadow: 0 0 8px var(--amber); }
  .meta-right { display: flex; align-items: center; gap: 10px; font-size: 0.82rem; color: var(--muted); }
  button {
    background: var(--card2); color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 6px 12px; font-size: 0.8rem; cursor: pointer; font-weight: 600;
  }
  button:hover { border-color: var(--blue); color: var(--blue); }
  .progress-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 14px 16px; margin-bottom: 16px;
  }
  .progress-head {
    display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;
    font-size: 0.88rem;
  }
  .bar-bg {
    width: 100%; height: 14px; border-radius: 999px; background: #0f172a;
    border: 1px solid var(--border); overflow: hidden; display: flex;
  }
  .bar-verified { background: linear-gradient(90deg, #16a34a, #22c55e); height: 100%; transition: width .4s; }
  .bar-inflight { background: repeating-linear-gradient(45deg, #0284c7, #0284c7 6px, #38bdf8 6px, #38bdf8 12px); height: 100%; opacity: 0.85; transition: width .4s; }
  .kpis {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 14px; margin-bottom: 16px;
  }
  .kpi {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px;
  }
  .kpi-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }
  .kpi-val { font-size: 1.65rem; font-weight: 700; margin: 4px 0; }
  .kpi-sub { font-size: 0.8rem; color: var(--muted); }
  .charts {
    display: grid; grid-template-columns: 2fr 1.5fr 1fr; gap: 14px; margin-bottom: 16px;
  }
  @media (max-width: 1080px) { .charts { grid-template-columns: 1fr; } }
  .card {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px;
  }
  .card h2 { margin: 0 0 10px 0; font-size: 0.95rem; font-weight: 600; color: var(--text); }
  .card-head {
    display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap;
    margin-bottom: 10px;
  }
  .card-head h2 { margin: 0; }
  .pager { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  .pager button { padding: 6px 10px; font-size: 0.78rem; min-height: 32px; }
  .pager button:disabled { opacity: 0.4; cursor: not-allowed; border-color: var(--border); color: var(--muted); }
  .pager .page-info {
    font-size: 0.78rem; color: var(--muted); font-family: ui-monospace, monospace;
    min-width: 62px; text-align: center;
  }
  .chart-box { position: relative; height: 260px; width: 100%; }
  .chart-box-workers { min-height: 280px; }
  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
  th {
    text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--border);
    color: var(--muted); font-weight: 600; font-size: 0.76rem; text-transform: uppercase; white-space: nowrap;
  }
  td { padding: 10px 8px; border-bottom: 1px solid #1d2940; vertical-align: top; }
  tr:hover td { background: rgba(56, 189, 248, 0.04); }
  .badge {
    display: inline-block; padding: 2px 7px; border-radius: 6px; font-size: 0.72rem; font-weight: 600;
  }
  .b-online { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.35); }
  .b-stale { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }
  .b-offline { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.35); }
  .b-medido { background: rgba(34, 197, 94, 0.14); color: #4ade80; }
  .b-est { background: rgba(168, 85, 247, 0.16); color: #c084fc; }
  .shard-pill {
    display: inline-block; margin: 1px 3px 2px 0; padding: 2px 6px; border-radius: 5px;
    font-size: 0.72rem; background: var(--card2); border: 1px solid var(--border); font-family: ui-monospace, monospace;
  }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .muted { color: var(--muted); font-size: 0.76rem; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="title-row">
      <h1>AlphaBuilder v3.3 — Harvest Monitor</h1>
      <span class="pill"><span class="dot" id="liveDot"></span> <span id="waveBadge">wave: prod</span></span>
      <span class="pill mono" id="commitBadge">commit: ee04deb</span>
    </div>
    <div class="meta-right">
      <span id="lastUpdate">Atualizando...</span>
      <span>• próx. em <b id="countdown">10</b>s</span>
      <button onclick="refreshNow()">Atualizar agora</button>
    </div>
  </header>

  <div class="progress-card">
    <div class="progress-head">
      <div><b>Progresso Onda Prod</b> (<span id="prodEpsText">0 / 4.000</span> episódios verificados + <span id="inflightText">0</span> em andamento equiv.)</div>
      <div class="mono"><b id="prodPctText">0.00%</b></div>
    </div>
    <div class="bar-bg">
      <div class="bar-verified" id="barVerified" style="width:0%"></div>
      <div class="bar-inflight" id="barInflight" style="width:0%"></div>
    </div>
  </div>

  <div class="kpis">
    <div class="kpi">
      <div class="kpi-label">Shards Onda Prod</div>
      <div class="kpi-val" id="kpiShards">-</div>
      <div class="kpi-sub" id="kpiShardsSub">-</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Vazão Atual do Cluster</div>
      <div class="kpi-val" style="color: var(--blue)" id="kpiRate">-</div>
      <div class="kpi-sub" id="kpiRateSub">-</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Estimativa de Término (ETA)</div>
      <div class="kpi-val" style="color: var(--green)" id="kpiEta">-</div>
      <div class="kpi-sub" id="kpiEtaSub">-</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Dados Validados (Total)</div>
      <div class="kpi-val" id="kpiData">-</div>
      <div class="kpi-sub" id="kpiDataSub">-</div>
    </div>
  </div>

  <div class="charts">
    <div class="card">
      <h2>Evolução de Episódios Verificados (Onda Prod) & Projeção</h2>
      <div class="chart-box"><canvas id="chartTimeline"></canvas></div>
    </div>
    <div class="card">
      <div class="card-head">
        <h2>Episódios Gerados &amp; Vazão por Instância</h2>
        <div class="pager">
          <button id="btnPrevWorker" onclick="prevWorkerPage()">◀ Ant</button>
          <span class="page-info" id="workerPageInfo">1/1</span>
          <button id="btnNextWorker" onclick="nextWorkerPage()">Próx ▶</button>
          <button id="btnWorkerPageSize" onclick="toggleWorkerPageSize()">Todos</button>
        </div>
      </div>
      <div class="chart-box chart-box-workers" id="chartWorkersBox"><canvas id="chartWorkers"></canvas></div>
    </div>
    <div class="card">
      <h2>Status dos 400 Shards (Prod)</h2>
      <div class="chart-box"><canvas id="chartShards"></canvas></div>
    </div>
  </div>

  <div class="card">
    <h2>Estatísticas Detalhadas por Nó</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Nó / Worker</th>
            <th>Status</th>
            <th>Arch / FEniCSx</th>
            <th>Slots</th>
            <th>Shards em Execução Agora</th>
            <th>Verificados</th>
            <th>Ritmo / Slot</th>
            <th>Vazão Nó</th>
            <th>Recursos (CPU / RAM / Disco)</th>
            <th>Heartbeat</th>
          </tr>
        </thead>
        <tbody id="workersTbody">
          <tr><td colspan="10" class="muted">Carregando dados...</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

<script>
let timelineChart = null, workersChart = null, shardsChart = null;
let countdown = 10;
let latestData = null;
let workerPage = 0;
let workerPageSize = 4;

const workerBarLabelsPlugin = {
  id: "workerBarLabels",
  afterDatasetsDraw(chart) {
    const { ctx, chartArea } = chart;
    ctx.save();
    ctx.font = "600 11px ui-monospace, monospace";
    ctx.textBaseline = "middle";
    chart.data.datasets.forEach((ds, di) => {
      const meta = chart.getDatasetMeta(di);
      if (!meta.visible) return;
      meta.data.forEach((bar, i) => {
        const val = ds.data[i];
        const txt = di === 0 ? `${val} eps` : `${val} eps/h`;
        ctx.fillStyle = di === 0 ? "#86efac" : "#bae6fd";
        const w = ctx.measureText(txt).width;
        if (bar.x + 6 + w > chartArea.right) {
          ctx.textAlign = "right";
          ctx.fillText(txt, Math.max(bar.base + 4, bar.x - 6), bar.y);
        } else {
          ctx.textAlign = "left";
          ctx.fillText(txt, bar.x + 6, bar.y);
        }
      });
    });
    ctx.restore();
  }
};

function prevWorkerPage() {
  if (workerPage > 0) {
    workerPage -= 1;
    if (latestData) renderWorkersChart(latestData);
  }
}

function nextWorkerPage() {
  if (!latestData) return;
  const totalPages = Math.max(1, Math.ceil(latestData.workers.length / workerPageSize));
  if (workerPage + 1 < totalPages) {
    workerPage += 1;
    renderWorkersChart(latestData);
  }
}

function toggleWorkerPageSize() {
  if (!latestData) return;
  workerPageSize = workerPageSize <= 4 ? Math.max(8, latestData.workers.length) : 4;
  workerPage = 0;
  renderWorkersChart(latestData);
}

function renderWorkersChart(d) {
  const sorted = [...d.workers].sort((a, b) =>
    (b.verified_episodes - a.verified_episodes) ||
    (b.node_eps_per_h - a.node_eps_per_h) ||
    a.worker_id.localeCompare(b.worker_id)
  );
  const totalPages = Math.max(1, Math.ceil(sorted.length / workerPageSize));
  if (workerPage >= totalPages) workerPage = totalPages - 1;
  const start = workerPage * workerPageSize;
  const slice = sorted.slice(start, start + workerPageSize);

  document.getElementById("workerPageInfo").textContent = `Pág. ${workerPage + 1}/${totalPages}`;
  document.getElementById("btnPrevWorker").disabled = workerPage <= 0;
  document.getElementById("btnNextWorker").disabled = workerPage + 1 >= totalPages;
  document.getElementById("btnWorkerPageSize").textContent = workerPageSize <= 4 ? "Todos" : "4/pág";

  const box = document.getElementById("chartWorkersBox");
  box.style.height = `${Math.max(280, slice.length * 60)}px`;

  const wcData = {
    labels: slice.map(w => w.worker_id),
    datasets: [
      {
        label: "Episódios entregues",
        data: slice.map(w => w.verified_episodes),
        backgroundColor: "rgba(34,197,94,0.8)",
        borderRadius: 4,
        barPercentage: 0.8,
        categoryPercentage: 0.72,
        xAxisID: "x"
      },
      {
        label: "Vazão (eps/h)",
        data: slice.map(w => w.node_eps_per_h),
        backgroundColor: slice.map(w => w.is_colab ? "#a855f7" : "#38bdf8"),
        borderRadius: 4,
        barPercentage: 0.8,
        categoryPercentage: 0.72,
        xAxisID: "x1"
      }
    ]
  };
  if (!workersChart) {
    workersChart = new Chart(document.getElementById("chartWorkers"), {
      type: "bar",
      data: wcData,
      plugins: [workerBarLabelsPlugin],
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            ticks: {
              autoSkip: false,
              padding: 6,
              color: "#e2e8f0",
              font: { size: window.innerWidth <= 640 ? 11 : 12, family: "ui-monospace, monospace", weight: "600" }
            },
            grid: { display: false }
          },
          x: {
            position: "top",
            beginAtZero: true,
            grace: "22%",
            ticks: { color: "#4ade80", maxTicksLimit: 5 },
            grid: { color: "#1e293b" },
            title: { display: true, text: "episódios (eps)", color: "#4ade80", font: { size: 11 } }
          },
          x1: {
            position: "bottom",
            beginAtZero: true,
            grace: "22%",
            ticks: { color: "#38bdf8", maxTicksLimit: 5 },
            grid: { drawOnChartArea: false },
            title: { display: true, text: "vazão (eps/h)", color: "#38bdf8", font: { size: 11 } }
          }
        },
        plugins: { legend: { labels: { color: "#cbd5e1", boxWidth: 12 } } }
      }
    });
  } else {
    workersChart.data = wcData;
    workersChart.resize();
    workersChart.update("none");
  }
}

function fmtDuration(hours) {
  if (hours == null || !isFinite(hours)) return "N/A";
  const h = Math.floor(hours);
  const m = Math.round((hours - h) * 60);
  return `${h}h ${String(m).padStart(2, '0')}m`;
}

function fmtAge(sec) {
  if (sec < 60) return `${Math.round(sec)}s atrás`;
  if (sec < 3600) return `${Math.round(sec / 60)}m atrás`;
  return `${(sec / 3600).toFixed(1)}h atrás`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function render(d) {
  latestData = d;
  const p = d.prod, e = d.eta, t = d.totals;
  document.getElementById("waveBadge").textContent = `wave: ${d.active_wave}`;
  document.getElementById("commitBadge").textContent = `commit: ${d.expected_commit}`;
  document.getElementById("lastUpdate").textContent = `Atualizado: ${d.updated_brt}`;

  document.getElementById("prodEpsText").textContent = `${p.verified_episodes.toLocaleString('pt-BR')} / ${p.target_episodes.toLocaleString('pt-BR')}`;
  document.getElementById("inflightText").textContent = `~${p.in_flight_equiv_episodes}`;
  document.getElementById("prodPctText").textContent = `${p.progress_pct.toFixed(2)}%`;

  const verPct = Math.min(100, 100 * p.verified_episodes / p.target_episodes);
  const infPct = Math.min(100 - verPct, 100 * p.in_flight_equiv_episodes / p.target_episodes);
  document.getElementById("barVerified").style.width = `${verPct.toFixed(2)}%`;
  document.getElementById("barInflight").style.width = `${infPct.toFixed(2)}%`;

  document.getElementById("kpiShards").textContent = `${p.verified_shards} / ${p.target_shards}`;
  document.getElementById("kpiShardsSub").textContent = `${p.running_shards} rodando • ${p.pending_shards} na fila • ${p.failed_shards} falhas`;

  document.getElementById("kpiRate").textContent = `${e.cluster_eps_per_hour} eps/h`;
  document.getElementById("kpiRateSub").textContent = `${t.active_slots} slots ativos (${t.online_workers}/${t.total_workers} nós) • fixos: ${e.fixed_nodes_eps_per_hour} eps/h`;

  document.getElementById("kpiEta").textContent = fmtDuration(e.eta_hours);
  document.getElementById("kpiEtaSub").textContent = `Previsto: ${e.eta_human_brt} • só fixos: ${fmtDuration(e.eta_fixed_hours)} (${e.eta_fixed_human_brt})`;

  document.getElementById("kpiData").textContent = `${t.verified_episodes.toLocaleString('pt-BR')} eps`;
  document.getElementById("kpiDataSub").textContent = `${t.verified_records.toLocaleString('pt-BR')} frames • ${t.verified_mb.toFixed(1)} MB (inclui ${d.pilot.verified_episodes} piloto)`;

  // 1. Timeline + Linear Projection Chart (using numeric epoch hours relative to first prod shard or actual timestamps)
  const prodPts = d.timeline.filter(x => x.wave === "prod").map(x => ({ x: x.ts * 1000, y: x.cum_prod_eps, label: x.shard_id + " (" + x.worker_id + ")" }));
  const projPts = [];
  if (prodPts.length > 0 && e.eta_iso) {
    const lastActual = prodPts[prodPts.length - 1];
    projPts.push({ x: d.timestamp * 1000, y: p.verified_episodes });
    projPts.push({ x: new Date(e.eta_iso).getTime(), y: p.target_episodes });
  }
  const tcData = {
    datasets: [
      {
        label: "Verificados (prod)",
        data: prodPts,
        borderColor: "#22c55e",
        backgroundColor: "rgba(34,197,94,0.18)",
        fill: true,
        tension: 0.2,
        pointRadius: 3
      },
      {
        label: "Projeção atual -> 4.000",
        data: projPts,
        borderColor: "#38bdf8",
        borderDash: [6, 5],
        tension: 0,
        pointRadius: 4
      }
    ]
  };
  if (!timelineChart) {
    timelineChart = new Chart(document.getElementById("chartTimeline"), {
      type: "line",
      data: tcData,
      options: {
        responsive: true, maintainAspectRatio: false, parsing: false,
        scales: {
          x: {
            type: "linear",
            ticks: {
              color: "#94a3b8", maxTicksLimit: 7,
              callback: v => new Date(v).toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" })
            },
            grid: { color: "#1e293b" }
          },
          y: { min: 0, suggestedMax: Math.max(300, p.verified_episodes + 100), ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" } }
        },
        plugins: { legend: { labels: { color: "#cbd5e1" } } }
      }
    });
  } else {
    timelineChart.data = tcData;
    timelineChart.update("none");
  }

  // 2. Workers Paginated Horizontal Bar Chart
  renderWorkersChart(d);

  // 3. Shards Doughnut
  const scData = {
    labels: ["Verificados", "Em Execução", "Na Fila", "Falhas"],
    datasets: [{
      data: [p.verified_shards, p.running_shards, p.pending_shards, p.failed_shards],
      backgroundColor: ["#22c55e", "#38bdf8", "#475569", "#ef4444"],
      borderColor: "#131c2e",
      borderWidth: 2
    }]
  };
  if (!shardsChart) {
    shardsChart = new Chart(document.getElementById("chartShards"), {
      type: "doughnut",
      data: scData,
      options: {
        responsive: true, maintainAspectRatio: false, cutout: "62%",
        plugins: { legend: { position: "bottom", labels: { color: "#cbd5e1", boxWidth: 12 } } }
      }
    });
  } else {
    shardsChart.data = scData;
    shardsChart.update("none");
  }

  // 4. Workers Detailed Table
  const tb = document.getElementById("workersTbody");
  tb.innerHTML = d.workers.map(w => {
    const stClass = w.status === "ONLINE" ? "b-online" : (w.status === "STALE" ? "b-stale" : "b-offline");
    const rateClass = w.rate_source === "medido" ? "b-medido" : "b-est";
    const shardHtml = w.active_shards.length
      ? w.active_shards.map(s => `<span class="shard-pill" title="TTL restante: ${s.ttl_rem_min} min">${esc(s.shard_id.replace('shard_','#'))} [${s.seed_range[0]}..${String(s.seed_range[1]%1000).padStart(3,'0')}] <span class="muted">TTL ${s.ttl_rem_min}m</span></span>`).join("")
      : `<span class="muted">ocioso</span>`;
    return `<tr>
      <td><b>${esc(w.worker_id)}</b><br><span class="muted mono">${esc(w.code_commit)}</span></td>
      <td><span class="badge ${stClass}">${w.status}</span></td>
      <td class="mono">${esc(w.arch)}<br><span class="muted">dolfinx ${esc(w.dolfinx_version)}</span></td>
      <td class="mono"><b>${w.active_slots}</b> / ${w.slots}</td>
      <td>${shardHtml}</td>
      <td class="mono"><b>${w.verified_episodes}</b> eps<br><span class="muted">${w.verified_shards} shards • ${w.verified_mb} MB</span></td>
      <td class="mono">${w.eps_per_slot_h} eps/slot/h<br><span class="badge ${rateClass}">${w.rate_source} (~${w.avg_sec_per_ep}s/ep)</span></td>
      <td class="mono"><b style="color:var(--blue)">${w.node_eps_per_h} eps/h</b></td>
      <td class="mono">CPU ${w.cpu_busy}%<br><span class="muted">RAM ${Math.round(w.ram_free_mb)} MB • Disco ${w.disk_free_gb} GB</span></td>
      <td class="mono">${fmtAge(w.sec_since_seen)}</td>
    </tr>`;
  }).join("");
}

async function fetchAndRender() {
  try {
    const r = await fetch("/api/public/dashboard", { cache: "no-store" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const d = await r.json();
    render(d);
    document.getElementById("liveDot").classList.remove("warn");
  } catch (err) {
    document.getElementById("liveDot").classList.add("warn");
    document.getElementById("lastUpdate").textContent = "Erro ao atualizar: " + err.message;
  }
}

function refreshNow() {
  countdown = 10;
  fetchAndRender();
}

fetchAndRender();
setInterval(() => {
  countdown -= 1;
  if (countdown <= 0) {
    countdown = 10;
    fetchAndRender();
  }
  document.getElementById("countdown").textContent = countdown;
}, 1000);
</script>
</body>
</html>
"""


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # never log tokens
        return

    def _send(self, code: int, body: bytes, ctype: str = "text/plain", no_store: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if no_store:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _route(self) -> None:
        path = self.path.split("?", 1)[0]

        # Read-only public status dashboard (zero credentials / tokens exposed)
        if self.command == "GET" and path in ("/", "/index.html", "/dashboard"):
            self._send(200, DASHBOARD_HTML.encode("utf-8"), "text/html; charset=utf-8", no_store=True)
            return
        if self.command == "GET" and path == "/api/public/dashboard":
            try:
                payload = json.dumps(build_dashboard_data()).encode("utf-8")
                self._send(200, payload, "application/json; charset=utf-8", no_store=True)
            except Exception as e:
                err = json.dumps({"ok": False, "error": str(e)}).encode("utf-8")
                self._send(500, err, "application/json; charset=utf-8", no_store=True)
            return
        if self.command == "GET" and path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        # Token-authenticated worker / supervisor proxy routes
        parts = path.split("/")  # ['', 't', TOKEN, ...]
        if len(parts) < 4 or parts[1] != "t" or not hmac.compare_digest(parts[2], TOKEN):
            self._send(403, b"forbidden")
            return
        rest = "/" + "/".join(parts[3:])
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length else None
        if self.command == "GET" and rest.startswith("/boot/"):
            wid = rest[len("/boot/"):]
            if not WORKER_RE.match(wid):
                self._send(400, b"bad worker id")
                return
            host = self.headers.get("Host", "")
            conf = base64.b64encode(RCLONE_CONF.read_bytes()).decode()
            script = (
                BOOT_TEMPLATE.replace("@@CONF@@", conf)
                .replace("@@COMMIT@@", COMMIT)
                .replace("@@WID@@", wid)
                .replace("@@HOST@@", host)
                .replace("@@TOKEN@@", TOKEN)
                .replace("@@SLOTS@@", "2")
            )
            self._send(200, script.encode("utf-8"))
            return
        if rest.startswith("/api/v1/"):
            req = urllib.request.Request(
                UPSTREAM + rest,
                data=body,
                method=self.command,
                headers={"Content-Type": "application/json"} if body else {},
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    raw, code = r.read(), r.status
            except urllib.error.HTTPError as e:
                raw, code = e.read(), e.code
            except Exception as e:
                self._send(502, str(e).encode("utf-8"))
                return
            if code == 204 or not raw:
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._send(code, raw, "application/json")
            return
        self._send(404, b"not found")

    do_GET = _route
    do_POST = _route


if __name__ == "__main__":
    ThreadingHTTPServer.allow_reuse_address = True
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()
