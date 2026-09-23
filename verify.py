"""Compose verify 服务的单次复核入口。

依次执行：
1. 代码测试（pytest 单元测试，含随机暴力交叉验证）；
2. 构建检查（语法编译、模块导入、镜像内关键文件齐备）；
3. API/HTTP 冒烟（健康路径 + 嵌套同优、交叉低价诱饵、空候选、非法引用，
   以及敏感度的必选弧禁用退化、诱饵强制见证、空剖面、非法 target、
   旧审计回归场景）。

任一步失败即以非零退出码结束，全部通过退出码 0。
"""

from __future__ import annotations

import json
import os
import py_compile
import sys
import urllib.error
import urllib.request

API_BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
REQUIRE_FILES = os.environ.get("REQUIRE_BUILD_FILES", "").split(",")

failures: list[str] = []


def check(condition: bool, name: str, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {name}{(' - ' + detail) if detail and not condition else ''}")
    if not condition:
        failures.append(name)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------- 1. 代码测试


def run_unit_tests() -> None:
    section("代码测试 (pytest)")
    import pytest

    rc = pytest.main(["-q", "tests"])
    check(rc == 0, "pytest 单元测试", f"退出码 {rc}")


# ---------------------------------------------------------------- 2. 构建检查


def run_build_checks() -> None:
    section("构建检查")
    for path in ["solver.py", "app.py", "verify.py", os.path.join("tests", "test_solver.py")]:
        try:
            py_compile.compile(path, doraise=True)
            ok = True
        except py_compile.PyCompileError as exc:
            ok = False
            print(exc)
        check(ok, f"语法编译: {path}")

    for fname in REQUIRE_FILES:
        fname = fname.strip()
        if not fname:
            continue
        check(os.path.isfile(fname), f"镜像内文件齐备: {fname}")

    try:
        import solver  # noqa: F401
        import app  # noqa: F401

        ok = True
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(exc)
    check(ok, "模块导入 solver/app")


# ---------------------------------------------------------------- 3. API/HTTP 冒烟


def http_request(method: str, path: str, payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        API_BASE + path, data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def hits(n):
    return [{"id": f"h{k}", "position": k * 10} for k in range(n)]


def cand(cid, a, b, r):
    return {
        "id": cid,
        "left_endpoint": f"h{a}",
        "right_endpoint": f"h{b}",
        "residual": r,
    }


def run_smoke() -> None:
    section(f"API/HTTP 冒烟 ({API_BASE})")

    status, body = http_request("GET", "/health")
    check(status == 200 and body.get("status") == "ready", "健康路径 /health 报告就绪",
          f"HTTP {status} {body}")

    # 场景 1：嵌套同优 —— 顺序配对与嵌套配对同为 2 对同残差。
    payload = {
        "hits": hits(4),
        "candidates": [
            cand("a_out", 0, 3, 1),
            cand("a_in", 1, 2, 5),
            cand("b_left", 0, 1, 3),
            cand("b_right", 2, 3, 3),
        ],
    }
    status, body = http_request("POST", "/audit", payload)
    ok = (
        status == 200
        and body["optimal_count"] == "2"
        and [p["id"] for p in body["canonical_pairs"]] == ["a_out", "a_in"]
        and set(body["classification"]["optional"])
        == {"a_out", "a_in", "b_left", "b_right"}
    )
    check(ok, "嵌套同优方案并列计数与规范解", f"HTTP {status} {body}")

    # 场景 2：交叉低价诱饵 —— 0 残差的交叉弧不得胜出。
    payload = {
        "hits": hits(6),
        "candidates": [
            cand("bait", 0, 3, 0),
            cand("inner", 1, 2, 10),
            cand("tail", 4, 5, 1),
            cand("seq01", 0, 1, 1),
            cand("seq23", 2, 3, 1),
        ],
    }
    status, body = http_request("POST", "/audit", payload)
    ok = (
        status == 200
        and [p["id"] for p in body["canonical_pairs"]] == ["seq01", "seq23", "tail"]
        and "bait" in body["classification"]["never"]
        and body["total_residual"] == 3
    )
    check(ok, "交叉低价诱饵不被接受", f"HTTP {status} {body}")

    # 场景 3：合法空候选 —— 唯一空方案。
    payload = {"hits": hits(4), "candidates": []}
    status, body = http_request("POST", "/audit", payload)
    ok = (
        status == 200
        and body["optimal_count"] == "1"
        and body["canonical_pairs"] == []
        and len(body["unmatched_hits"]) == 4
        and body["classification"] == {"required": [], "optional": [], "never": []}
    )
    check(ok, "合法空候选返回唯一空方案", f"HTTP {status} {body}")

    # 场景 4：非法引用 —— 错误带字段路径，且不夹带任何审计字段。
    payload = {"hits": hits(4), "candidates": [cand("bad", 0, 99, 0)]}
    # 99 不在 id 中，手工构造以模拟未知端点字符串。
    payload["candidates"][0]["right_endpoint"] = "h99"
    status, body = http_request("POST", "/audit", payload)
    ok = (
        status == 400
        and isinstance(body.get("errors"), list)
        and any(e.get("field") == "/candidates/0/right_endpoint" for e in body["errors"])
        and "optimal_count" not in body
    )
    check(ok, "非法引用返回字段路径错误且无审计结果", f"HTTP {status} {body}")

    # 附加：重复端点对、位置冲突、规模越界。
    status, body = http_request(
        "POST",
        "/audit",
        {"hits": hits(4), "candidates": [cand("a", 0, 1, 0), cand("b", 0, 1, 1)]},
    )
    check(
        status == 400
        and any(e["field"] == "/candidates/1" for e in body["errors"])
        and "optimal_count" not in body,
        "重复端点对被拒绝", f"HTTP {status} {body}",
    )

    conflict = hits(4)
    conflict[2]["position"] = conflict[1]["position"]
    status, body = http_request("POST", "/audit", {"hits": conflict, "candidates": []})
    check(
        status == 400
        and any(e["field"] == "/hits/2/position" for e in body["errors"]),
        "位置冲突被拒绝", f"HTTP {status} {body}",
    )

    status, body = http_request("POST", "/audit", {"hits": hits(3), "candidates": []})
    check(
        status == 400 and any(e["field"] == "/hits" for e in body["errors"]),
        "击中规模越界被拒绝", f"HTTP {status} {body}",
    )

    # 未知路径返回 404。
    status, _ = http_request("GET", "/nope")
    check(status == 404, "未知路径返回 404", f"HTTP {status}")


def run_sensitivity_smoke() -> None:
    section("敏感度/见证 API 冒烟")

    # bait(0,2) 与 blocker(1,4) 真正交叉；唯一 3 对最优是三条顺序相邻弧。
    payload = {
        "hits": hits(6),
        "candidates": [
            cand("bait", 0, 2, 0),
            cand("blocker", 1, 4, 0),
            cand("seq01", 0, 1, 1),
            cand("seq23", 2, 3, 1),
            cand("seq45", 4, 5, 1),
        ],
    }

    # 旧审计回归：/audit 响应结构与数值不变。
    status, body = http_request("POST", "/audit", payload)
    audit_ok = (
        status == 200
        and body["optimal_count"] == "1"
        and body["paired_hits"] == 6
        and body["total_residual"] == 3
        and [p["id"] for p in body["canonical_pairs"]] == ["seq01", "seq23", "seq45"]
        and body["unmatched_hits"] == []
        and body["classification"]["required"] == ["seq01", "seq23", "seq45"]
        and set(body["classification"]["never"]) == {"bait", "blocker"}
    )
    check(audit_ok, "旧审计 /audit 回归无变化", f"HTTP {status} {body}")

    # 全量剖面：必选弧禁用后的退化 + 交叉诱饵强制后的反事实值。
    status, body = http_request("POST", "/sensitivity", payload)
    profiles = {p["id"]: p for p in body.get("profiles", [])} if status == 200 else {}
    ok = (
        status == 200
        and set(profiles) == {"bait", "blocker", "seq01", "seq23", "seq45"}
        # 禁用必选 seq01：6 -> 4 配对点；两个并列最优方案（残差均为 1）。
        and profiles["seq01"]["disabled"]
        == {"paired_hits": 4, "total_residual": 1, "optimal_count": "2"}
        and profiles["seq45"]["disabled"]
        == {"paired_hits": 4, "total_residual": 1, "optimal_count": "1"}
        # 从不出现的 bait/blocker 被禁用时目标不退化，仍是唯一最优。
        and profiles["bait"]["disabled"]
        == {"paired_hits": 6, "total_residual": 3, "optimal_count": "1"}
        # 强制交叉诱饵 bait：只能 2 对（bait+seq45），残差 1。
        and profiles["bait"]["forced"]
        == {"paired_hits": 4, "total_residual": 1, "optimal_count": "1"}
        # 强制必选弧：目标不变。
        and profiles["seq23"]["forced"]
        == {"paired_hits": 6, "total_residual": 3, "optimal_count": "1"}
    )
    check(ok, "必选弧禁用退化与交叉诱饵强制剖面正确", f"HTTP {status} {body}")

    # 空剖面：合法空候选返回空列表。
    status, body = http_request(
        "POST", "/sensitivity", {"hits": hits(4), "candidates": []}
    )
    check(
        status == 200 and body == {"profiles": []},
        "空候选敏感度剖面为空列表", f"HTTP {status} {body}",
    )

    # 敏感度非法引用：只返回字段路径错误，无任何剖面字段。
    bad_ref = {"hits": hits(4), "candidates": [cand("x", 0, 99, 0)]}
    bad_ref["candidates"][0]["right_endpoint"] = "h99"
    status, body = http_request("POST", "/sensitivity", bad_ref)
    check(
        status == 400
        and set(body.keys()) == {"errors"}
        and any(e["field"] == "/candidates/0/right_endpoint" for e in body["errors"]),
        "敏感度非法引用只报字段路径", f"HTTP {status} {body}",
    )

    # 交叉诱饵强制后的最优见证。
    status, body = http_request(
        "POST",
        "/sensitivity/witness",
        {**payload, "target": "bait", "mode": "forced"},
    )
    ok = (
        status == 200
        and [p["id"] for p in body["canonical_pairs"]] == ["bait", "seq45"]
        and body["unmatched_hits"] == ["h1", "h3"]
        and body["paired_hits"] == 4
        and body["total_residual"] == 1
        and body["optimal_count"] == "1"
    )
    check(ok, "交叉诱饵强制后的最优见证", f"HTTP {status} {body}")

    # 禁用见证：禁用从不出现的 blocker 等价原审计规范解。
    status, body = http_request(
        "POST",
        "/sensitivity/witness",
        {**payload, "target": "blocker", "mode": "disabled"},
    )
    check(
        status == 200
        and [p["id"] for p in body["canonical_pairs"]] == ["seq01", "seq23", "seq45"]
        and body["optimal_count"] == "1",
        "禁用从不出现候选的见证等同原规范解", f"HTTP {status} {body}",
    )

    # 见证非法 target / mode：字段路径错误，不夹带见证字段。
    status, body = http_request(
        "POST",
        "/sensitivity/witness",
        {**payload, "target": "ghost", "mode": "forced"},
    )
    check(
        status == 400
        and set(body.keys()) == {"errors"}
        and any(e["field"] == "/target" for e in body["errors"]),
        "见证未知 target 报 /target 字段错误", f"HTTP {status} {body}",
    )
    status, body = http_request(
        "POST",
        "/sensitivity/witness",
        {**payload, "target": "bait", "mode": "sideways"},
    )
    check(
        status == 400 and any(e["field"] == "/mode" for e in body["errors"]),
        "见证非法 mode 报 /mode 字段错误", f"HTTP {status} {body}",
    )


def main() -> int:
    run_unit_tests()
    run_build_checks()
    run_smoke()
    run_sensitivity_smoke()

    print("\n=== 汇总 ===")
    if failures:
        print(f"失败 {len(failures)} 项: {failures}")
        return 1
    print("全部复核通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
