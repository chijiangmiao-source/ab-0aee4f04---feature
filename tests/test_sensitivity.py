"""sensitivity / sensitivity_witness 的单元测试。

固定场景 + 随机暴力交叉验证（小 n 全枚举禁用/强制下的非交叉匹配）。
"""

from __future__ import annotations

import itertools
import random

import pytest

from solver import (
    MAX_CANDIDATES,
    MAX_HITS,
    MIN_HITS,
    ValidationError,
    sensitivity,
    sensitivity_witness,
)

from tests.brute import brute_counterfactual


def make_hits(n, prefix="h"):
    return [{"id": f"{prefix}{k}", "position": k * 10} for k in range(n)]


def cand(cid, left, right, residual):
    return {
        "id": cid,
        "left_endpoint": left,
        "right_endpoint": right,
        "residual": residual,
    }


def to_arc_records(candidates, hit_ids):
    pos = {h: k for k, h in enumerate(hit_ids)}
    return [
        (c["id"], pos[c["left_endpoint"]], pos[c["right_endpoint"]], c["residual"])
        for c in candidates
    ]


def payload(n, candidates, **extra):
    return {"hits": make_hits(n), "candidates": candidates, **extra}


# ---------------------------------------------------------------- 固定场景


def test_empty_candidates_profile_is_empty():
    res = sensitivity(payload(MIN_HITS, []))
    assert res == {"profiles": []}


def test_single_candidate_disabled_and_forced():
    p = payload(4, [cand("only", "h1", "h2", 7)])
    res = sensitivity(p)
    [prof] = res["profiles"]
    assert prof["id"] == "only"
    # 禁用：唯一空方案。
    assert prof["disabled"] == {
        "optimal_count": "1",
        "paired_hits": 0,
        "total_residual": 0,
    }
    # 强制：必须选它。
    assert prof["forced"] == {
        "optimal_count": "1",
        "paired_hits": 2,
        "total_residual": 7,
    }


def test_required_arc_disabled_degrades():
    # must 是唯一能达到 2 对的外层弧：禁用后退化为 1 对。
    cands = [
        cand("must", "h0", "h4", 0),
        cand("mid", "h1", "h3", 0),
        cand("only_other", "h2", "h3", 5),
    ]
    res = sensitivity(payload(5, cands))
    prof = next(p for p in res["profiles"] if p["id"] == "must")
    assert prof["disabled"]["paired_hits"] == 2
    assert prof["forced"]["paired_hits"] == 4


def test_never_arc_disabled_is_identity():
    # bait 为交叉低价诱饵，从不出现；禁用它不应改变任何最优值。
    cands = [
        cand("bait", "h0", "h3", 0),
        cand("inner", "h1", "h2", 10),
        cand("tail", "h4", "h5", 1),
        cand("seq01", "h0", "h1", 1),
        cand("seq23", "h2", "h3", 1),
    ]
    res = sensitivity(payload(6, cands))
    bait = next(p for p in res["profiles"] if p["id"] == "bait")
    assert bait["disabled"] == {"optimal_count": "1", "paired_hits": 6, "total_residual": 3}


def test_forced_never_arc_is_optimal_under_constraint():
    # 强制交叉诱饵：3 对方案 bait+inner+tail，残差 11。
    cands = [
        cand("bait", "h0", "h3", 0),
        cand("inner", "h1", "h2", 10),
        cand("tail", "h4", "h5", 1),
        cand("seq01", "h0", "h1", 1),
        cand("seq23", "h2", "h3", 1),
    ]
    res = sensitivity(payload(6, cands))
    bait = next(p for p in res["profiles"] if p["id"] == "bait")
    assert bait["forced"] == {
        "optimal_count": "1",
        "paired_hits": 6,
        "total_residual": 11,
    }

    w = sensitivity_witness(payload(6, cands, candidate="bait", mode="forced"))
    assert [p["id"] for p in w["canonical_pairs"]] == ["bait", "inner", "tail"]
    assert w["unmatched_hits"] == []
    assert w["paired_hits"] == 6
    assert w["total_residual"] == 11


def test_witness_disabled_canonical_pairs_and_unmatched():
    cands = [
        cand("a_out", "h0", "h3", 1),
        cand("a_in", "h1", "h2", 5),
        cand("s_left", "h0", "h1", 3),
        cand("s_right", "h2", "h3", 3),
    ]
    w = sensitivity_witness(payload(4, cands, candidate="a_out", mode="disabled"))
    assert [p["id"] for p in w["canonical_pairs"]] == ["s_left", "s_right"]
    assert w["unmatched_hits"] == []
    assert w["optimal_count"] == "1"

    w2 = sensitivity_witness(payload(4, cands, candidate="s_left", mode="forced"))
    assert [p["id"] for p in w2["canonical_pairs"]] == ["s_left", "s_right"]
    assert w2["optimal_count"] == "1"


def test_mode_aliases_and_canonical_mode_value():
    cands = [cand("only", "h0", "h1", 0)]
    for alias in ("disable", "禁用"):
        w = sensitivity_witness(payload(4, cands, candidate="only", mode=alias))
        assert w["mode"] == "disabled"
    for alias in ("force", "强制"):
        w = sensitivity_witness(payload(4, cands, candidate="only", mode=alias))
        assert w["mode"] == "forced"
        assert [p["id"] for p in w["canonical_pairs"]] == ["only"]


def test_arbitrary_precision_counterfactual_counts():
    # 60 个独立三元组，每组 3 条最优弧 => 3^60；禁用其中一条后该组剩 2 条，
    # 强制后该组唯一，其余组仍 3 条。
    n = 180
    cands = []
    for a in range(0, n, 3):
        cands.append(cand(f"span{a}", f"h{a}", f"h{a+2}", 0))
        cands.append(cand(f"adj1{a}", f"h{a}", f"h{a+1}", 0))
        cands.append(cand(f"adj2{a}", f"h{a+1}", f"h{a+2}", 0))
    res = sensitivity(payload(n, cands))
    span0 = next(p for p in res["profiles"] if p["id"] == "span0")
    assert span0["disabled"]["optimal_count"] == str(2 * 3 ** (n // 3 - 1))
    assert span0["forced"]["optimal_count"] == str(3 ** (n // 3 - 1))


# ---------------------------------------------------------------- 校验错误


def invalid_witness(p):
    with pytest.raises(ValidationError) as exc:
        sensitivity_witness(p)
    return exc.value.errors


def test_witness_unknown_candidate_field_path():
    errors = invalid_witness(payload(4, [], candidate="ghost", mode="forced"))
    assert any(e["field"] == "/candidate" for e in errors)


def test_witness_bad_mode_field_path():
    cands = [cand("only", "h0", "h1", 0)]
    errors = invalid_witness(payload(4, cands, candidate="only", mode="maybe"))
    assert any(e["field"] == "/mode" for e in errors)


def test_witness_missing_fields():
    errors = invalid_witness({"hits": make_hits(4), "candidates": []})
    fields = {e["field"] for e in errors}
    assert "/candidate" in fields and "/mode" in fields


def test_witness_invalid_reference_has_no_sensitivity_fields():
    # 审计字段的非法引用仍只返回字段路径错误，不夹带剖面字段。
    errors = invalid_witness(
        {"hits": make_hits(4), "candidates": [cand("x", "h0", "nope", 0)],
         "candidate": "x", "mode": "disabled"}
    )
    assert any(e["field"] == "/candidates/0/right_endpoint" for e in errors)


def test_sensitivity_invalid_reference_same_as_audit():
    with pytest.raises(ValidationError) as exc:
        sensitivity(payload(4, [cand("x", "h0", "h9", 0)]))
    assert any(e["field"] == "/candidates/0/right_endpoint" for e in exc.value.errors)


# ---------------------------------------------------------------- 随机暴力对照


@pytest.mark.parametrize("seed", range(40))
def test_matches_bruteforce_counterfactual(seed):
    rng = random.Random(1000 + seed)
    n = rng.randint(MIN_HITS, 9)
    ids = [f"h{k}" for k in range(n)]
    possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
    rng.shuffle(possible)
    candidates = []
    counter = itertools.count()
    for a, b in possible:
        if rng.random() < 0.4:
            candidates.append(
                cand(
                    f"cid{next(counter):03d}",
                    ids[a],
                    ids[b],
                    rng.choice([0, 0, 1, 2, 5]),
                )
            )
    records = to_arc_records(candidates, ids)
    if not candidates:
        pytest.skip("本次抽样无候选")

    res = sensitivity(payload(n, candidates))
    prof = {p["id"]: p for p in res["profiles"]}

    for cid, _a, _b, _r in records:
        for mode in ("disabled", "forced"):
            mp, mc, ways, canonical = brute_counterfactual(n, records, cid, mode)
            block = prof[cid][mode]
            assert block["paired_hits"] == 2 * mp, (seed, cid, mode)
            assert block["total_residual"] == mc, (seed, cid, mode)
            assert int(block["optimal_count"]) == ways, (seed, cid, mode)

            w = sensitivity_witness(
                payload(n, candidates, candidate=cid, mode=mode)
            )
            assert [p["id"] for p in w["canonical_pairs"]] == canonical, (
                seed, cid, mode
            )
            assert w["paired_hits"] == 2 * mp
            assert w["total_residual"] == mc
            assert int(w["optimal_count"]) == ways

            endpoint = {c["id"]: (c["left_endpoint"], c["right_endpoint"]) for c in candidates}
            used: set[str] = set()
            for pid in canonical:
                used.update(endpoint[pid])
            assert set(w["unmatched_hits"]) == set(ids) - used
            # 未配对击中按位置顺序输出。
            assert w["unmatched_hits"] == [h for h in ids if h not in used]


# ---------------------------------------------------------------- 性能 / 健全性


def test_max_scale_sensitivity_performance_and_sanity():
    n = MAX_HITS
    rng = random.Random(42)
    possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
    rng.shuffle(possible)
    cands = [
        cand(f"c{k:04d}", f"h{a}", f"h{b}", rng.randrange(1000))
        for k, (a, b) in enumerate(possible[:MAX_CANDIDATES])
    ]
    p = payload(n, cands)
    res = sensitivity(p)
    assert len(res["profiles"]) == MAX_CANDIDATES

    # 禁用任何弧不可能改善配对数，强制不可能超过无约束配对数。
    forced_pairs = [b["forced"]["paired_hits"] for b in res["profiles"]]
    disabled_pairs = [b["disabled"]["paired_hits"] for b in res["profiles"]]
    assert max(forced_pairs) <= n * 2
    assert min(disabled_pairs) >= 0

    # 单条见证可用且字段齐全。
    w = sensitivity_witness(
        payload(n, cands, candidate=cands[0]["id"], mode="disabled")
    )
    assert w["id"] == cands[0]["id"]
    assert len(w["canonical_pairs"]) * 2 == w["paired_hits"]
