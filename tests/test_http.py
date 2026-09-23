"""HTTP 层测试：在随机端口上线程内启动真实 app，走 urllib 调用。"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from app import AuditHandler


@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), AuditHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def request(base, method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_health(server):
    status, body = request(server, "GET", "/health")
    assert status == 200
    assert body == {"status": "ready", "service": "track-pair-audit"}


def test_audit_ok(server):
    payload = {
        "hits": [{"id": f"h{k}", "position": k} for k in range(4)],
        "candidates": [
            {"id": "p", "left_endpoint": "h0", "right_endpoint": "h3", "residual": 2},
            {"id": "q", "left_endpoint": "h1", "right_endpoint": "h2", "residual": 1},
        ],
    }
    status, body = request(server, "POST", "/audit", payload)
    assert status == 200
    assert body["optimal_count"] == "1"
    assert [p["id"] for p in body["canonical_pairs"]] == ["p", "q"]
    assert body["unmatched_hits"] == []


def test_audit_error_shape_has_no_audit_fields(server):
    payload = {
        "hits": [{"id": f"h{k}", "position": k} for k in range(4)],
        "candidates": [
            {"id": "x", "left_endpoint": "h0", "right_endpoint": "nope", "residual": 0}
        ],
    }
    status, body = request(server, "POST", "/audit", payload)
    assert status == 400
    assert set(body.keys()) == {"errors"}
    assert body["errors"][0]["field"] == "/candidates/0/right_endpoint"


def test_bad_json(server):
    req = urllib.request.Request(
        server + "/audit", data=b"{not json", method="POST"
    )
    req.add_header("Content-Type", "application/json")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 400


def test_unknown_route(server):
    status, _ = request(server, "GET", "/")
    assert status == 404


# ---------------------------------------------------------------- sensitivity


def test_sensitivity_ok(server):
    payload = {
        "hits": [{"id": f"h{k}", "position": k} for k in range(4)],
        "candidates": [
            {"id": "p", "left_endpoint": "h0", "right_endpoint": "h3", "residual": 2},
            {"id": "q", "left_endpoint": "h1", "right_endpoint": "h2", "residual": 1},
        ],
    }
    status, body = request(server, "POST", "/sensitivity", payload)
    assert status == 200
    assert {p["id"] for p in body["profiles"]} == {"p", "q"}
    prof = {p["id"]: p for p in body["profiles"]}
    assert prof["p"]["forced"]["paired_hits"] == 4
    assert prof["p"]["forced"]["total_residual"] == 3
    assert prof["p"]["disabled"]["paired_hits"] == 2  # 只剩 q
    for mode in ("disabled", "forced"):
        assert isinstance(prof["p"][mode]["optimal_count"], str)


def test_sensitivity_empty_candidates(server):
    payload = {
        "hits": [{"id": f"h{k}", "position": k} for k in range(4)],
        "candidates": [],
    }
    status, body = request(server, "POST", "/sensitivity", payload)
    assert status == 200
    assert body == {"profiles": []}


def test_sensitivity_error_shape_matches_audit(server):
    payload = {
        "hits": [{"id": f"h{k}", "position": k} for k in range(4)],
        "candidates": [
            {"id": "x", "left_endpoint": "h0", "right_endpoint": "nope", "residual": 0}
        ],
    }
    status, body = request(server, "POST", "/sensitivity", payload)
    assert status == 400
    assert set(body.keys()) == {"errors"}
    assert body["errors"][0]["field"] == "/candidates/0/right_endpoint"


def test_sensitivity_witness_forced(server):
    payload = {
        "hits": [{"id": f"h{k}", "position": k} for k in range(6)],
        "candidates": [
            {"id": "bait", "left_endpoint": "h0", "right_endpoint": "h3", "residual": 0},
            {"id": "inner", "left_endpoint": "h1", "right_endpoint": "h2", "residual": 10},
            {"id": "tail", "left_endpoint": "h4", "right_endpoint": "h5", "residual": 1},
            {"id": "seq01", "left_endpoint": "h0", "right_endpoint": "h1", "residual": 1},
            {"id": "seq23", "left_endpoint": "h2", "right_endpoint": "h3", "residual": 1},
        ],
        "candidate": "bait",
        "mode": "forced",
    }
    status, body = request(server, "POST", "/sensitivity/witness", payload)
    assert status == 200
    assert body["id"] == "bait"
    assert body["mode"] == "forced"
    assert [p["id"] for p in body["canonical_pairs"]] == ["bait", "inner", "tail"]
    assert body["unmatched_hits"] == []
    assert body["total_residual"] == 11


def test_sensitivity_witness_unknown_candidate(server):
    payload = {
        "hits": [{"id": f"h{k}", "position": k} for k in range(4)],
        "candidates": [],
        "candidate": "ghost",
        "mode": "disabled",
    }
    status, body = request(server, "POST", "/sensitivity/witness", payload)
    assert status == 400
    assert set(body.keys()) == {"errors"}
    assert any(e["field"] == "/candidate" for e in body["errors"])


def test_sensitivity_witness_bad_mode(server):
    payload = {
        "hits": [{"id": f"h{k}", "position": k} for k in range(4)],
        "candidates": [
            {"id": "p", "left_endpoint": "h0", "right_endpoint": "h1", "residual": 0}
        ],
        "candidate": "p",
        "mode": "maybe",
    }
    status, body = request(server, "POST", "/sensitivity/witness", payload)
    assert status == 400
    assert any(e["field"] == "/mode" for e in body["errors"])


def test_unknown_post_route(server):
    status, body = request(server, "POST", "/nope", {"x": 1})
    assert status == 404
    assert "errors" in body
