"""M1 acceptance tests for the AlphaBuilder Compute & Inference Server."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from alphabuilder.server import wire
from alphabuilder.server.app import create_app
from alphabuilder.server.config import ServerConfig
from alphabuilder.server.schemas import REWARD_SPEC

REPO_ROOT = Path(__file__).resolve().parents[2]


class DummyModel:
    """Deterministic stand-in for AlphaBuilderInference (no torch needed in CI)."""

    config = {"use_swin": False, "feature_size": 8, "in_channels": 7}
    use_swin = False
    device_str = "cpu"
    epoch = 3
    val_loss = 0.0123
    in_channels = 7
    feature_size = 8

    def predict(self, state: np.ndarray):
        d, h, w = state.shape[1:]
        p_add = np.linspace(0.0, 1.0, d * h * w, dtype=np.float32).reshape(d, h, w)
        p_rem = np.linspace(1.0, 0.0, d * h * w, dtype=np.float32).reshape(d, h, w)
        return 0.25, p_add, p_rem

    def predict_with_blueprint(self, state: np.ndarray):
        value, p_add, p_rem = self.predict(state)
        blueprint = np.full(state.shape[1:], 0.75, dtype=np.float32)
        return value, p_add, p_rem, blueprint

    def predict_batch(self, states):
        return [self.predict(state) for state in states]


def make_app(tmp_path: Path, token: str = "") -> tuple[TestClient, ServerConfig]:
    checkpoints = tmp_path / "checkpoints"
    data_dir = tmp_path / "data"
    checkpoints.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (checkpoints / "dummy.pt").write_bytes(b"dummy checkpoint")
    config = ServerConfig(
        checkpoints_dir=checkpoints,
        data_dir=data_dir,
        api_token=token,
        model_factory=lambda path, device: DummyModel(),
    )
    return TestClient(create_app(config)), config


def load_dummy(client: TestClient) -> dict:
    response = client.post(
        "/api/v1/models/load",
        json={"model_id": "dummy.pt", "device": "cpu"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def infer_body() -> dict:
    return {
        "resolution": [8, 4, 2],
        "bc_type": "FULL_CLAMP",
        "load_config": {"x": 7, "y": 2, "z_start": 0, "z_end": 2},
        "solid_voxels": [0, 1, 2],
        "custom_supports": None,
        "custom_forces": None,
        "include_blueprint": True,
        "policy_mode": "masked_probs",
        "policy_temperature": 1.0,
        "policy_top_k": 32,
        "response_format": "abwire01",
    }


def session_body() -> dict:
    return {
        "kind": "MCTS_LIVE",
        "resolution": [8, 4, 2],
        "bc_type": "FULL_CLAMP",
        "load_config": {"x": 7, "y": 2, "z_start": 0, "z_end": 2},
        "initial_solid_voxels": [0],
        "persist_to_db": False,
        "start_paused": False,
        "include_blueprint": True,
        "allow_neural_only_phase2": True,
        "episode_config": {
            "max_steps": 12,
            "max_phase1_steps": 6,
            "target_volume_fraction": 0.15,
            "use_guided_value": True,
            "generation": 0,
            "guidance_decay_generations": 5,
        },
        "phase1_mcts": {
            "num_simulations": 2,
            "batch_size": 1,
            "action_batch_size": 1,
            "c_puct": 1.2,
            "top_k_expansion": 8,
            "mixmax_beta": 0.75,
        },
        "phase2_mcts": {
            "num_simulations": 2,
            "batch_size": 1,
            "action_batch_size": 1,
            "c_puct": 0.8,
            "top_k_expansion": 8,
            "mixmax_beta": 1.0,
        },
    }


def test_health_and_capabilities_contract(tmp_path: Path):
    client, _ = make_app(tmp_path)
    with client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        payload = health.json()
        assert payload["service"] == "alphabuilder-engine"
        assert payload["protocol"] == "ABWIRE01"
        assert payload["code_version"] == "v3.3"
        assert payload["auth_required"] is False

        capabilities = client.get("/api/v1/capabilities")
        assert capabilities.status_code == 200
        caps = capabilities.json()
        assert caps["reward_spec"] == REWARD_SPEC
        assert caps["reward_spec"]["version"] == "v3.3"
        assert caps["limits"]["max_voxels"] == 262144
        assert caps["limits"]["max_concurrent_sessions"] == 1
        assert isinstance(caps["features"]["has_torch"], bool)
        assert {device["id"] for device in caps["devices"]} == {
            "cuda",
            "xpu",
            "npu",
            "cpu",
        }
        assert caps["active_model"] is None


def test_auth_rejects_missing_bearer_and_accepts_valid(tmp_path: Path):
    client, _ = make_app(tmp_path, token="s3cret")
    with client:
        assert client.get("/api/v1/health").status_code == 200
        unauthorized = client.get("/api/v1/capabilities")
        assert unauthorized.status_code == 401
        assert unauthorized.json()["error"]["code"] == "UNAUTHORIZED"
        assert unauthorized.json()["error"]["request_id"].startswith("req_")
        authorized = client.get(
            "/api/v1/capabilities",
            headers={"Authorization": "Bearer s3cret"},
        )
        assert authorized.status_code == 200
        wrong = client.get(
            "/api/v1/capabilities",
            headers={"Authorization": "Bearer wrong"},
        )
        assert wrong.status_code == 401


def test_non_loopback_bind_requires_token():
    with pytest.raises(RuntimeError):
        ServerConfig(host="0.0.0.0").validate()
    ServerConfig(host="0.0.0.0", api_token="x").validate()
    ServerConfig(host="0.0.0.0", allow_insecure_remote=True).validate()
    ServerConfig(host="127.0.0.1").validate()


def test_model_management_and_path_traversal(tmp_path: Path):
    client, _ = make_app(tmp_path)
    with client:
        listing = client.get("/api/v1/models")
        assert listing.status_code == 200
        assert listing.json()["models"][0]["model_id"] == "dummy.pt"
        assert client.get("/api/v1/models/active").json() == {"loaded": False}

        for bad in ["../../etc/passwd", "..\\secret.pt", "/etc/passwd"]:
            response = client.post(
                "/api/v1/models/load", json={"model_id": bad, "device": "cpu"}
            )
            assert response.status_code == 403, bad
            assert response.json()["error"]["code"] == "PATH_OUTSIDE_SANDBOX"

        info = load_dummy(client)
        assert info["loaded"] is True
        assert info["architecture"] == "simple"
        assert info["in_channels"] == 7
        assert info["has_blueprint_head"] is True
        active = client.get("/api/v1/models/active").json()
        assert active["loaded"] is True
        assert active["model_id"] == "dummy.pt"
        unload = client.post("/api/v1/models/unload")
        assert unload.status_code == 200
        assert client.get("/api/v1/models/active").json() == {"loaded": False}


def test_infer_returns_abwire01_and_json_without_dense_tensors(tmp_path: Path):
    client, _ = make_app(tmp_path)
    with client:
        load_dummy(client)
        response = client.post("/api/v1/infer", json=infer_body())
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "application/octet-stream"
        assert len(response.content) % 4 == 0
        packet = wire.decode_packet(response.content)
        assert packet.payload_type == wire.PAYLOAD_INFER_RESULT
        assert packet.flags & wire.FLAG_ZLIB
        tags = {tensor.tag for tensor in packet.tensors}
        assert wire.TAG_POLICY_ADD in tags
        assert wire.TAG_POLICY_REMOVE in tags
        assert wire.TAG_BLUEPRINT_PROBS in tags
        assert packet.header["legal_add_count"] > 0
        assert packet.header["legal_remove_count"] == 0
        assert packet.header["resolution"] == [8, 4, 2]
        add = next(t for t in packet.tensors if t.tag == wire.TAG_POLICY_ADD)
        dense = wire.tensor_to_dense(add)
        assert dense.shape == (8, 4, 2)
        assert np.isclose(dense.sum(), 1.0, atol=1e-4)
        blueprint = next(t for t in packet.tensors if t.tag == wire.TAG_BLUEPRINT_PROBS)
        assert wire.tensor_to_dense(blueprint).shape == (8, 4, 2)

        json_response = client.post(
            "/api/v1/infer", json={**infer_body(), "response_format": "json"}
        )
        assert json_response.status_code == 200
        body = json_response.json()
        assert body["value"] == pytest.approx(0.25)
        assert body["island_analysis"]["n_islands"] >= 1
        assert len(body["top_policy_actions"]) <= 32
        assert "tensor_block" not in body


def test_infer_requires_loaded_model(tmp_path: Path):
    client, _ = make_app(tmp_path)
    with client:
        response = client.post("/api/v1/infer", json=infer_body())
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "MODEL_NOT_LOADED"


def test_infer_rejects_oversized_resolution(tmp_path: Path):
    client, _ = make_app(tmp_path)
    with client:
        load_dummy(client)
        body = infer_body()
        body["resolution"] = [256, 256, 8]
        response = client.post("/api/v1/infer", json=body)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "RESOLUTION_LIMIT_EXCEEDED"


def test_infer_abwire01_is_parsed_by_rust(tmp_path: Path):
    client, _ = make_app(tmp_path)
    with client:
        load_dummy(client)
        response = client.post("/api/v1/infer", json=infer_body())
        assert response.status_code == 200
        packet_path = tmp_path / "infer.abwire"
        packet_path.write_bytes(response.content)
    command = [
        "cargo",
        "run",
        "--quiet",
        "-p",
        "ab-core",
        "--example",
        "abwire_verify",
        "--",
        str(packet_path),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    parsed = json.loads(completed.stdout)
    assert parsed["protocol"] == "ABWIRE01"
    assert parsed["payload_type"] == 1
    assert 5 in parsed["tensors"]
    assert 11 in parsed["tensors"]


def _next_until(ws, predicate, limit: int = 80):
    frames = []
    for _ in range(limit):
        message = ws.receive()
        if "text" in message:
            payload = json.loads(message["text"])
            frames.append(("text", payload))
            if predicate("text", payload):
                return frames
        else:
            frames.append(("binary", message["bytes"]))
            if predicate("binary", message["bytes"]):
                return frames
    raise AssertionError(f"condition not reached; last frames: {frames[-5:]}")


def test_websocket_full_session_flow(tmp_path: Path):
    client, _ = make_app(tmp_path)
    with client:
        load_dummy(client)
        created = client.post("/api/v1/sessions", json=session_body())
        assert created.status_code == 201, created.text
        handle = created.json()
        assert handle["status"] == "RUNNING"
        assert handle["fem_enabled"] is False
        session_id = handle["session_id"]
        assert handle["ws_url"] == f"/api/v1/ws/sessions/{session_id}"

        with client.websocket_connect(handle["ws_url"]) as ws:
            first = ws.receive_json()
            assert first["type"] == "session_state"
            assert first["session_id"] == session_id
            assert first["resolution"] == [8, 4, 2]

            frames = _next_until(
                ws,
                lambda kind, payload: kind == "binary" and len(payload) > 32,
            )
            assert any(kind == "text" and payload["type"] == "step_progress" for kind, payload in frames)
            binary = next(payload for kind, payload in frames if kind == "binary")
            packet = wire.decode_packet(binary)
            assert packet.payload_type == wire.PAYLOAD_STEP_FRAME
            assert packet.header["step"] >= 1
            assert packet.header["phase"] in {"GROWTH", "REFINEMENT"}
            assert "mcts_stats" in packet.header
            assert {t.tag for t in packet.tensors} >= {
                wire.TAG_DENSITY,
                wire.TAG_POLICY_ADD,
                wire.TAG_POLICY_REMOVE,
            }

            ws.send_text(json.dumps({"type": "command", "op": "pause"}))
            _next_until(
                ws,
                lambda kind, payload: kind == "text"
                and payload.get("type") == "session_state"
                and payload.get("status") == "PAUSED",
            )

            paused = client.get(f"/api/v1/sessions/{session_id}").json()
            assert paused["status"] == "PAUSED"

            ws.send_text(
                json.dumps(
                    {
                        "type": "command",
                        "op": "edit_state",
                        "add_voxels": [1, 2],
                        "remove_voxels": [],
                    }
                )
            )
            _next_until(
                ws,
                lambda kind, payload: kind == "text"
                and payload.get("type") == "session_state"
                and payload.get("status") == "PAUSED",
            )

            ws.send_text(json.dumps({"type": "command", "op": "step_once"}))
            step_frames = _next_until(ws, lambda kind, _payload: kind == "binary")
            step_packet = wire.decode_packet(
                next(payload for kind, payload in step_frames if kind == "binary")
            )
            assert step_packet.header["step"] >= 2
            _next_until(
                ws,
                lambda kind, payload: kind == "text"
                and payload.get("type") == "session_state"
                and payload.get("status") == "PAUSED",
            )

            ws.send_text(json.dumps({"type": "command", "op": "cancel"}))
            ended = _next_until(
                ws,
                lambda kind, payload: kind == "text"
                and payload.get("type") == "session_ended",
            )
            ended_payload = next(
                payload
                for kind, payload in ended
                if kind == "text" and payload.get("type") == "session_ended"
            )
            assert ended_payload["status"] == "CANCELLED"
            assert ended_payload["reason"] == "CANCELLED"


def test_rest_session_commands_and_busy_guard(tmp_path: Path):
    client, _ = make_app(tmp_path)
    with client:
        load_dummy(client)
        created = client.post("/api/v1/sessions", json=session_body())
        assert created.status_code == 201
        session_id = created.json()["session_id"]
        busy = client.post("/api/v1/infer", json=infer_body())
        assert busy.status_code == 409
        assert busy.json()["error"]["code"] == "ENGINE_BUSY"
        pause = client.post(
            f"/api/v1/sessions/{session_id}/command", json={"op": "pause"}
        )
        assert pause.status_code == 200
        listing = client.get("/api/v1/sessions").json()["sessions"]
        assert any(item["session_id"] == session_id for item in listing)
        deleted = client.delete(f"/api/v1/sessions/{session_id}")
        assert deleted.status_code == 200


def test_simp_and_fem_report_capability_unavailable_without_fenics(tmp_path: Path):
    from alphabuilder.server.worker import has_fenics

    if has_fenics():
        pytest.skip("FEniCSx is installed on this host")
    client, _ = make_app(tmp_path)
    with client:
        load_dummy(client)
        fem = client.post(
            "/api/v1/fem/evaluate", json={**infer_body(), "response_format": "json"}
        )
        assert fem.status_code == 501
        assert fem.json()["error"]["code"] == "CAPABILITY_UNAVAILABLE"
        simp = client.post(
            "/api/v1/simp/start",
            json={
                "resolution": [8, 4, 2],
                "bc_type": "FULL_CLAMP",
                "load_config": {"x": 7, "y": 2, "z_start": 0, "z_end": 2},
            },
        )
        assert simp.status_code == 501


def test_datasets_empty_and_path_traversal(tmp_path: Path):
    client, _ = make_app(tmp_path)
    with client:
        assert client.get("/api/v1/datasets").json() == {"databases": []}
        traversal = client.get("/api/v1/datasets/../../etc/passwd/episodes")
        assert traversal.status_code in {403, 404}
