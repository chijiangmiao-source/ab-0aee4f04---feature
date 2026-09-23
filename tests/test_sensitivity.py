"""sensitivity / sensitivity_witness 的单元测试。

固定场景覆盖必选弧禁用退化、交叉诱饵强制、空剖面与校验错误；
随机场景与暴力枚举（force/forbid 过滤）逐条对照剖面与见证。
"""

from __future__ import annotations

import itertools
import random
import time

import pytest

from solver import (
    MAX_CANDIDATES,
    MAX_HITS,
    ValidationError,
    audit,
    sensitivity,
    sensitivity_witness,
)

from tests.brute import brute_solve


def make_hits(n, prefix="h"):
    return [{"id": f"{prefix}{k}", "position": k * 10} for k in range(n)]


def cand(cid, left, right, residual):
    return {
        "id": cid,
        "left_endpoint": left,
        "right_endpoint": right,
        "residual": residual,
    }


def records_of(candidates):
    pos = {f"h{k}": k for k in range(MAX_HITS + 1)}
    return [
        (c["id"], pos[c["left_endpoint"]], pos[c["right_endpoint"]], c["residual"])
        for c in candidates
    ]


# ---------------------------------------------------------------- 固定场景


def test_empty_candidates_profile_empty():
    payload = {"hits": make_hits(4), "candidates": []}
    res = sensitivity(payload)
    assert res == {"profiles": []}


def test_profile_blocks_match_audit_when_arc_in_optimal():
    # 嵌套同优：所有候选都出现在某一个最优方案中。
    payload = {
        "hits": make_hits(4),
        "candidates": [
            cand("a_out", "h0", "h3", 1),
            cand("a_in", "h1", "h2", 5),
            cand("b_left", "h0", "h1", 3),
            cand("b_right", "h2", "h3", 3),
        ],
    }
    res = sensitivity(payload)
    by_id = {p["id"]: p for p in res["profiles"]}
    for p in res["profiles"]:
        # 每条都出现在恰好一个最优方案（共 2 个）：
        # 禁用或强制都不改变两级目标，方案数均为 1。
        assert p["disabled"] == {
            "paired_hits": 4,
            "total_residual": 6,
            "optimal_count": "1",
        }
        assert p["forced"] == {
            "paired_hits": 4,
            "total_residual": 6,
            "optimal_count": "1",
        }
    assert set(by_id) == {"a_out", "a_in", "b_left", "b_right"}


def test_required_arc_disabled_degrades():
    # 交叉低价诱饵场景：seq01/seq23/tail 必选；禁用必选弧后退化。
    payload = {
        "hits": make_hits(6),
        "candidates": [
            cand("bait", "h0", "h3", 0),
            cand("inner", "h1", "h2", 10),
            cand("tail", "h4", "h5", 1),
            cand("seq01", "h0", "h1", 1),
            cand("seq23", "h2", "h3", 1),
        ],
    }
    audit_res = audit(payload)
    assert audit_res["classification"]["required"] == ["seq01", "seq23", "tail"]

    res = sensitivity(payload)
    by_id = {p["id"]: p for p in res["profiles"]}

    # 非必选弧禁用：目标不变；方案唯一，故方案数归零。
    for cid in ("bait", "inner"):
        assert by_id[cid]["disabled"] == {
            "paired_hits": 6,
            "total_residual": 3,
            "optimal_count": "1",
        }
    # 必选弧禁用：tail 禁用后只剩 2 对（残差 2）；seq01/seq23 禁用后嵌套
    # 方案 bait+inner+tail 顶上，仍 3 对但残差 3 -> 11。
    assert by_id["tail"]["disabled"] == {
        "paired_hits": 4,
        "total_residual": 2,
        "optimal_count": "1",
    }
    assert by_id["seq01"]["disabled"] == {
        "paired_hits": 6,
        "total_residual": 11,
        "optimal_count": "1",
    }

    # 强制从不出现的诱饵：bait 与 inner 嵌套、与 tail 并列，仍可 3 对，
    # 残差 0+10+1=11。
    assert by_id["bait"]["forced"] == {
        "paired_hits": 6,
        "total_residual": 11,
        "optimal_count": "1",
    }
    # 强制必选弧：目标不变，唯一方案。
    for cid in ("seq01", "seq23", "tail"):
        assert by_id[cid]["forced"] == {
            "paired_hits": 6,
            "total_residual": 3,
            "optimal_count": "1",
        }


def test_disabled_count_sums_over_optional_arcs():
    payload = {
        "hits": make_hits(4),
        "candidates": [
            cand("p01", "h0", "h1", 0),
            cand("p23", "h2", "h3", 0),
            cand("out", "h0", "h3", 0),
            cand("in", "h1", "h2", 0),
        ],
    }
    res = sensitivity(payload)
    for p in res["profiles"]:
        # 两个最优方案，每弧恰在其一：禁用后剩 1 个同优方案。
        assert p["disabled"]["optimal_count"] == "1"
        assert p["forced"]["optimal_count"] == "1"
        assert p["disabled"]["paired_hits"] == 4
        assert p["forced"]["paired_hits"] == 4


def test_witness_forced_crossing_bait():
    payload = {
        "hits": make_hits(6),
        "candidates": [
            cand("bait", "h0", "h3", 0),
            cand("inner", "h1", "h2", 10),
            cand("tail", "h4", "h5", 1),
            cand("seq01", "h0", "h1", 1),
            cand("seq23", "h2", "h3", 1),
        ],
    }
    res = sensitivity_witness({**payload, "target": "bait", "mode": "forced"})
    assert res["target"] == "bait"
    assert res["mode"] == "forced"
    # bait 强制后，内部最优选 inner，右侧 tail 仍可并列：bait+inner+tail。
    assert [p["id"] for p in res["canonical_pairs"]] == ["bait", "inner", "tail"]
    assert res["unmatched_hits"] == []
    assert res["paired_hits"] == 6
    assert res["total_residual"] == 11
    assert res["optimal_count"] == "1"
    # 强制结果含完整配对对象。
    first = res["canonical_pairs"][0]
    assert first == {
        "id": "bait",
        "left_endpoint": "h0",
        "right_endpoint": "h3",
        "residual": 0,
    }


def test_witness_disabled_required_arc():
    payload = {
        "hits": make_hits(6),
        "candidates": [
            cand("bait", "h0", "h3", 0),
            cand("inner", "h1", "h2", 10),
            cand("tail", "h4", "h5", 1),
            cand("seq01", "h0", "h1", 1),
            cand("seq23", "h2", "h3", 1),
        ],
    }
    res = sensitivity_witness({**payload, "target": "tail", "mode": "disabled"})
    ids = [p["id"] for p in res["canonical_pairs"]]
    assert "tail" not in ids
    assert res["paired_hits"] == 4
    assert res["total_residual"] == 2
    assert "h4" in res["unmatched_hits"] and "h5" in res["unmatched_hits"]


def test_witness_disabled_never_arc_equals_audit_canonical():
    payload = {
        "hits": make_hits(6),
        "candidates": [
            cand("bait", "h0", "h3", 0),
            cand("inner", "h1", "h2", 10),
            cand("tail", "h4", "h5", 1),
            cand("seq01", "h0", "h1", 1),
            cand("seq23", "h2", "h3", 1),
        ],
    }
    res = sensitivity_witness({**payload, "target": "bait", "mode": "disabled"})
    base = audit(payload)
    assert [p["id"] for p in res["canonical_pairs"]] == [
        p["id"] for p in base["canonical_pairs"]
    ]
    assert res["unmatched_hits"] == base["unmatched_hits"]
    assert res["optimal_count"] == base["optimal_count"]


def test_witness_unmatched_order_by_position():
    payload = {
        "hits": make_hits(5),
        "candidates": [cand("m", "h1", "h3", 0)],
    }
    forced = sensitivity_witness({**payload, "target": "m", "mode": "forced"})
    assert [p["id"] for p in forced["canonical_pairs"]] == ["m"]
    assert forced["unmatched_hits"] == ["h0", "h2", "h4"]


# ---------------------------------------------------------------- 校验错误


def _errors(payload, fn=sensitivity):
    with pytest.raises(ValidationError) as exc:
        fn(payload)
    return exc.value.errors


def test_sensitivity_validation_same_as_audit():
    # 未知端点引用：字段路径错误，且不返回任何剖面。
    bad = {"hits": make_hits(4), "candidates": [cand("c", "h0", "ghost", 0)]}
    errors = _errors(bad)
    assert any(e["field"] == "/candidates/0/right_endpoint" for e in errors)

    errors = _errors({"hits": make_hits(3), "candidates": []})
    assert any(e["field"] == "/hits" for e in errors)

    errors = _errors("not-an-object")
    assert errors[0]["field"] == ""


def test_witness_illegal_target_returns_field_path_only():
    base = {"hits": make_hits(4), "candidates": [cand("c", "h0", "h1", 0)]}

    errors = _errors({**base, "target": "nope", "mode": "forced"}, sensitivity_witness)
    assert any(e["field"] == "/target" for e in errors)

    errors = _errors({**base, "target": "c", "mode": "sideways"}, sensitivity_witness)
    assert any(e["field"] == "/mode" for e in errors)

    errors = _errors({**base, "mode": "forced"}, sensitivity_witness)
    assert any(e["field"] == "/target" for e in errors)

    # 数据本身非法时仍只报数据字段，不触及 target 检查。
    bad = {"hits": make_hits(4), "candidates": [cand("c", "h0", "ghost", 0)]}
    errors = _errors({**bad, "target": "c", "mode": "forced"}, sensitivity_witness)
    assert any(e["field"] == "/candidates/0/right_endpoint" for e in errors)
    assert all(e["field"] != "/target" for e in errors)


# ---------------------------------------------------------------- 随机暴力对照


@pytest.mark.parametrize("seed", range(50))
def test_sensitivity_matches_bruteforce(seed):
    rng = random.Random(1000 + seed)
    n = rng.randint(4, 9)
    possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
    rng.shuffle(possible)
    candidates = []
    counter = itertools.count()
    for a, b in possible:
        if rng.random() < 0.4:
            candidates.append(
                cand(
                    f"cid{next(counter):03d}",
                    f"h{a}",
                    f"h{b}",
                    rng.choice([0, 0, 1, 2, 5]),
                )
            )

    payload = {"hits": make_hits(n), "candidates": candidates}
    recs = records_of(candidates)
    base = brute_solve(n, recs)
    res = sensitivity(payload)
    by_id = {p["id"]: p for p in res["profiles"]}

    assert len(by_id) == len(candidates)
    for cid, _a, _b, _r in recs:
        dis = brute_solve(n, recs, forbid=(cid,))
        frc = brute_solve(n, recs, force=(cid,))
        block = by_id[cid]
        assert block["disabled"] == {
            "paired_hits": 2 * dis["max_pairs"],
            "total_residual": dis["min_cost"],
            "optimal_count": str(dis["optimal_count"]),
        }
        assert block["forced"] == {
            "paired_hits": 2 * frc["max_pairs"],
            "total_residual": frc["min_cost"],
            "optimal_count": str(frc["optimal_count"]),
        }
        # 退化只可能单向：强制不增配对数、禁用不减配对数。
        assert block["forced"]["paired_hits"] <= 2 * base["max_pairs"]
        assert block["disabled"]["paired_hits"] <= 2 * base["max_pairs"]


@pytest.mark.parametrize("seed", range(40))
def test_witness_matches_bruteforce(seed):
    rng = random.Random(2000 + seed)
    n = rng.randint(4, 9)
    possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
    rng.shuffle(possible)
    candidates = []
    counter = itertools.count()
    for a, b in possible:
        if rng.random() < 0.4:
            candidates.append(
                cand(
                    f"cid{next(counter):03d}",
                    f"h{a}",
                    f"h{b}",
                    rng.choice([0, 0, 1, 2, 5]),
                )
            )
    if not candidates:
        return

    payload = {"hits": make_hits(n), "candidates": candidates}
    recs = records_of(candidates)
    hit_ids = [f"h{k}" for k in range(n)]

    for cid, _a, _b, _r in recs:
        for mode, kw in (("disabled", {"forbid": (cid,)}), ("forced", {"force": (cid,)})):
            ref = brute_solve(n, recs, **kw)
            got = sensitivity_witness({**payload, "target": cid, "mode": mode})
            assert [p["id"] for p in got["canonical_pairs"]] == ref["canonical"]
            assert got["unmatched_hits"] == [hit_ids[k] for k in ref["canonical_unmatched"]]
            assert got["optimal_count"] == str(ref["optimal_count"])
            assert got["paired_hits"] == 2 * ref["max_pairs"]
            assert got["total_residual"] == ref["min_cost"]
            # 约束确实生效。
            ids = {p["id"] for p in got["canonical_pairs"]}
            if mode == "forced":
                assert cid in ids
            else:
                assert cid not in ids


# ---------------------------------------------------------------- 性能


def test_max_scale_sensitivity_performance():
    n = MAX_HITS
    rng = random.Random(42)
    possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
    rng.shuffle(possible)
    candidates = [
        cand(f"c{k:04d}", f"h{a}", f"h{b}", rng.randrange(1000))
        for k, (a, b) in enumerate(possible[:MAX_CANDIDATES])
    ]
    payload = {"hits": make_hits(n), "candidates": candidates}

    start = time.monotonic()
    res = sensitivity(payload)
    elapsed = time.monotonic() - start
    assert len(res["profiles"]) == MAX_CANDIDATES
    # 全量四千候选剖面必须在与审计同量级内完成（不逐条重跑审计）。
    assert elapsed < 5.0

    start = time.monotonic()
    w = sensitivity_witness({**payload, "target": candidates[-1]["id"], "mode": "forced"})
    assert time.monotonic() - start < 2.0
    assert candidates[-1]["id"] in {p["id"] for p in w["canonical_pairs"]}
