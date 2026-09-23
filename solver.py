"""硅微条击中配对审计求解器。

输入校验通过后，求解点列上的带权非交叉匹配问题（允许弧段相互嵌套）：

* 击中沿位置严格递增排列，候选对连接其中两个击中（左端点位置 < 右端点位置）；
* 选中的对端点互异，按位置绘制后两两不交叉（允许嵌套与并列）；
* 目标依次为：最大化已配对击中数（等价于最大化对数 * 2）、最小化残差总和；
* 在所有达到前两级目标的方案上统计任意精度方案数，输出按左端位置顺序下
  配对标识序列字典序最小的规范方案，并把每条候选对判为 required / optional / never。

算法采用区间 inside DP（求最优目标、方案数、规范序列）与 outside DP
（inside-outside，统计每条候选对出现在多少个最优方案中）。
所有计数使用 Python 任意精度整数；区间 DP 复杂度 O(n*|C| + n^3)，
n <= 180、|C| <= 4000。

敏感度（/sensitivity）复用同一份 inside 状态推导全部反事实最优值，
不逐条重跑完整审计：

* outside 计数（只统计最优规则）给出每条候选在最优方案中的出现次数
  used(e)：禁用非必选候选时目标不变、方案数 total-used(e)；强制本就出现
  的候选时方案数即 used(e)。
* 额外一张全规则 outside 目标表（对父区间所有规则求外部上下文最优，而非
  只下发最优规则），与 inside 表联合聚合：必选弧禁用时按顶点 a 在解析树
  上的两种互斥情形（作为左边界节点 / 作为入弧右端点）求最优；从不出现的
  候选强制选用时以其为节点首步聚合上下文、内部与右兄弟。整体代价
  O(n*|C| + n^3)。

见证（/sensitivity/witness）用带固定/禁用约束的记忆化区间递归，在同样的
无交叉约束、目标顺序与字典序裁决规则下取回约束后的规范配对与未配对击中。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

MIN_HITS = 4
MAX_HITS = 180
MAX_CANDIDATES = 4000

MODE_DISABLED = "disabled"
MODE_FORCED = "forced"
WITNESS_MODES = (MODE_DISABLED, MODE_FORCED)


class ValidationError(Exception):
    """携带字段路径的请求校验错误。"""

    def __init__(self, errors: list[dict[str, str]]):
        super().__init__("; ".join(e["message"] for e in errors))
        self.errors = errors


def _err(errors: list[dict[str, str]], field: str, message: str) -> None:
    errors.append({"field": field, "message": message})


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_payload(
    payload: Any,
) -> tuple[list[dict[str, Any]], list[tuple[str, int, int, int]]]:
    """校验 /audit 与 /sensitivity 共用的请求体，返回 (击中, 候选记录)。"""

    errors: list[dict[str, str]] = []

    if not isinstance(payload, dict):
        raise ValidationError([{"field": "", "message": "请求体必须是 JSON 对象"}])

    if "hits" not in payload:
        _err(errors, "/hits", "缺少 hits 字段")
    if "candidates" not in payload:
        _err(errors, "/candidates", "缺少 candidates 字段")
    if errors:
        raise ValidationError(errors)

    raw_hits = payload["hits"]
    raw_candidates = payload["candidates"]

    if not isinstance(raw_hits, list):
        _err(errors, "/hits", "hits 必须是数组")
        raw_hits = []
    if not isinstance(raw_candidates, list):
        _err(errors, "/candidates", "candidates 必须是数组")
        raw_candidates = []

    if isinstance(raw_hits, list) and not (MIN_HITS <= len(raw_hits) <= MAX_HITS):
        _err(
            errors,
            "/hits",
            f"击中数量必须在 {MIN_HITS} 到 {MAX_HITS} 之间，收到 {len(raw_hits)}",
        )
    if isinstance(raw_candidates, list) and len(raw_candidates) > MAX_CANDIDATES:
        _err(
            errors,
            "/candidates",
            f"候选配对数量不能超过 {MAX_CANDIDATES}，收到 {len(raw_candidates)}",
        )

    hits: list[dict[str, Any]] = []
    seen_hit_ids: set[str] = set()

    for k, hit in enumerate(raw_hits if isinstance(raw_hits, list) else []):
        base = f"/hits/{k}"
        if not isinstance(hit, dict):
            _err(errors, base, "击中必须是对象")
            continue
        hid = hit.get("id")
        pos = hit.get("position")
        if not isinstance(hid, str) or not hid:
            _err(errors, f"{base}/id", "击中 id 必须是非空字符串")
        elif hid in seen_hit_ids:
            _err(errors, f"{base}/id", f"击中 id 重复: {hid}")
        else:
            seen_hit_ids.add(hid)
        if not _is_int(pos):
            _err(errors, f"{base}/position", "position 必须是整数")
            pos = None
        hits.append({"id": hid, "position": pos})

    if not errors:
        for k in range(1, len(hits)):
            if hits[k]["position"] <= hits[k - 1]["position"]:
                _err(
                    errors,
                    f"/hits/{k}/position",
                    f"位置必须严格递增: {hits[k - 1]['position']} 之后出现 "
                    f"{hits[k]['position']}",
                )
                break

    candidate_records: list[tuple[str, int, int, int]] = []
    if not any(e["field"].startswith("/hits") for e in errors):
        index_by_id = {h["id"]: k for k, h in enumerate(hits)}
        seen_pair_ids: set[str] = set()
        seen_endpoint_pairs: set[tuple[int, int]] = set()

        for k, cand in enumerate(raw_candidates if isinstance(raw_candidates, list) else []):
            base = f"/candidates/{k}"
            if not isinstance(cand, dict):
                _err(errors, base, "候选配对必须是对象")
                continue
            cid = cand.get("id")
            left = cand.get("left_endpoint")
            right = cand.get("right_endpoint")
            residual = cand.get("residual")

            if not isinstance(cid, str) or not cid:
                _err(errors, f"{base}/id", "候选 id 必须是非空字符串")
            elif cid in seen_pair_ids:
                _err(errors, f"{base}/id", f"候选 id 重复: {cid}")
            else:
                seen_pair_ids.add(cid)

            if not _is_int(residual):
                _err(errors, f"{base}/residual", "residual 必须是非负整数")
            elif residual < 0:
                _err(errors, f"{base}/residual", f"residual 不能为负，收到 {residual}")

            if not isinstance(left, str):
                _err(errors, f"{base}/left_endpoint", "left_endpoint 必须是字符串标识")
            elif left not in index_by_id:
                _err(errors, f"{base}/left_endpoint", f"未知端点标识: {left}")

            if not isinstance(right, str):
                _err(errors, f"{base}/right_endpoint", "right_endpoint 必须是字符串标识")
            elif right not in index_by_id:
                _err(errors, f"{base}/right_endpoint", f"未知端点标识: {right}")

            if (
                isinstance(left, str)
                and isinstance(right, str)
                and left in index_by_id
                and right in index_by_id
            ):
                a = index_by_id[left]
                b = index_by_id[right]
                if a >= b:
                    _err(
                        errors,
                        f"{base}/right_endpoint",
                        "右端点位置必须严格大于左端点位置，且两端点必须不同",
                    )
                elif (a, b) in seen_endpoint_pairs:
                    _err(errors, base, f"重复端点对: ({left}, {right})")
                else:
                    seen_endpoint_pairs.add((a, b))
                    if isinstance(cid, str) and cid and _is_int(residual) and residual >= 0:
                        candidate_records.append((cid, a, b, residual))

    if errors:
        raise ValidationError(errors)

    return hits, candidate_records


def _build_arcs(
    n: int, candidates: list[tuple[str, int, int, int]]
) -> list[list[tuple[int, int, str]]]:
    # arcs[i]: 以位置 i 为左端点的候选 (右端点, 残差, id)，按右端点升序。
    arcs: list[list[tuple[int, int, str]]] = [[] for _ in range(n)]
    for cid, a, b, r in candidates:
        arcs[a].append((b, r, cid))
    for row in arcs:
        row.sort(key=lambda t: (t[0], t[2]))
    return arcs


def _inside_dp(
    n: int, arcs: list[list[tuple[int, int, str]]]
) -> tuple[
    list[list[int]],
    list[list[int]],
    list[list[int]],
    list[list[Optional[list[str]]]],
    list[list[Optional[tuple[Any, ...]]]],
]:
    """区间 inside DP。

    P[i][j]/C[i][j]/W[i][j]：区间 [i,j) 上的最大对数、最小残差、最优方案数；
    seq[i][j]：按左端位置顺序的最小 id 序列；choice 记录规范方案首步。
    """

    P = [[0] * (n + 1) for _ in range(n + 1)]
    C = [[0] * (n + 1) for _ in range(n + 1)]
    W = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        W[i][i] = 1

    def take(pairs: int, cost: int, ways: int, sequence: Optional[list[str]]) -> None:
        """把一条规则的结果并入当前区间的最优值。"""
        if ways == 0:
            return
        if best[0] is None or pairs > best[0] or (pairs == best[0] and cost < best[1]):
            best[0] = pairs
            best[1] = cost
            best[2] = ways
            best[3] = sequence
        elif pairs == best[0] and cost == best[1]:
            best[2] += ways
            if sequence is not None and (best[3] is None or sequence < best[3]):
                best[3] = sequence

    seq: list[list[Optional[list[str]]]] = [[None] * (n + 1) for _ in range(n + 1)]
    choice: list[list[Optional[tuple[Any, ...]]]] = [
        [None] * (n + 1) for _ in range(n + 1)
    ]
    for i in range(n + 1):
        seq[i][i] = []

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            best: list[Any] = [None, None, 0, None]  # pairs, cost, ways, 最小序列
            # 规则 1：i 未配对。
            take(P[i + 1][j], C[i + 1][j], W[i + 1][j], seq[i + 1][j])
            skip_ties = (P[i + 1][j], C[i + 1][j])
            # 规则 2：i 与 k 配对，内部 [i+1,k) 与外部 [k+1,j) 独立。
            for k, r, cid in arcs[i]:
                if k >= j:
                    continue
                take(
                    1 + P[i + 1][k] + P[k + 1][j],
                    r + C[i + 1][k] + C[k + 1][j],
                    W[i + 1][k] * W[k + 1][j],
                    [cid] + seq[i + 1][k] + seq[k + 1][j],
                )

            P[i][j] = best[0]
            C[i][j] = best[1]
            W[i][j] = best[2]
            seq[i][j] = best[3]

            # 确定取得最小 id 序列的首步规则。
            chosen: Optional[tuple[Any, ...]] = None
            if skip_ties == (P[i][j], C[i][j]) and seq[i + 1][j] == best[3]:
                chosen = ("s",)
            else:
                for k, r, cid in arcs[i]:
                    if k >= j:
                        continue
                    if (
                        1 + P[i + 1][k] + P[k + 1][j] == P[i][j]
                        and r + C[i + 1][k] + C[k + 1][j] == C[i][j]
                    ):
                        candidate = [cid] + seq[i + 1][k] + seq[k + 1][j]  # type: ignore[operator]
                        if candidate == best[3]:
                            chosen = ("p", k, cid)
                            break
            choice[i][j] = chosen

    return P, C, W, seq, choice


def _outside_counts(
    n: int,
    candidates: list[tuple[str, int, int, int]],
    arcs: list[list[tuple[int, int, str]]],
    P: list[list[int]],
    C: list[list[int]],
    W: list[list[int]],
) -> dict[str, int]:
    """outside DP：每条候选弧在全局最优方案中的出现次数。"""

    Out = [[0] * (n + 1) for _ in range(n + 1)]
    Out[0][n] = 1
    for length in range(n, 0, -1):
        for h in range(0, n - length + 1):
            m = h + length
            outside = Out[h][m]
            if outside == 0:
                continue
            # 跳过规则：父 [h,m) -> 子 [h+1,m)。
            if P[h + 1][m] == P[h][m] and C[h + 1][m] == C[h][m]:
                Out[h + 1][m] += outside
            # 配对规则：父 [h,m) 经弧 (h,k) -> 左子 [h+1,k)、右子 [k+1,m)。
            for k, r, _cid in arcs[h]:
                if k >= m:
                    continue
                if (
                    1 + P[h + 1][k] + P[k + 1][m] == P[h][m]
                    and r + C[h + 1][k] + C[k + 1][m] == C[h][m]
                ):
                    Out[h + 1][k] += outside * W[k + 1][m]
                    Out[k + 1][m] += outside * W[h + 1][k]

    # 弧 (a,b) 作为某父区间 [a,m) 的首步规则出现：
    # 出现方案数 = W[a+1][b] * Σ_m O[a][m] * W[b+1][m]（仅计最优规则）。
    used_count: dict[str, int] = {}
    for cid, a, b, r in candidates:
        count = 0
        interior_ways = W[a + 1][b]
        for m in range(b + 1, n + 1):
            if (
                1 + P[a + 1][b] + P[b + 1][m] == P[a][m]
                and r + C[a + 1][b] + C[b + 1][m] == C[a][m]
            ):
                count += Out[a][m] * interior_ways * W[b + 1][m]
        used_count[cid] = count
    return used_count


def _pair_entry(cid: str, hits: list[dict[str, Any]], info: dict[str, tuple[int, int, int]]) -> dict[str, Any]:
    a, b, r = info[cid]
    return {
        "id": cid,
        "left_endpoint": hits[a]["id"],
        "right_endpoint": hits[b]["id"],
        "residual": r,
    }


def audit(payload: Any) -> dict[str, Any]:
    """执行完整审计，返回可直接 JSON 序列化的结果。"""

    hits, candidates = validate_payload(payload)
    n = len(hits)
    arcs = _build_arcs(n, candidates)
    P, C, W, _seq, choice = _inside_dp(n, arcs)
    total = W[0][n]
    used_count = _outside_counts(n, candidates, arcs, P, C, W)

    required: list[str] = []
    optional: list[str] = []
    never: list[str] = []
    for cid, _a, _b, _r in candidates:
        c = used_count[cid]
        if c == 0:
            never.append(cid)
        elif c == total:
            required.append(cid)
        else:
            optional.append(cid)

    # ---------- 规范方案回溯 ----------
    canonical_ids: list[str] = []
    unmatched_idx: list[int] = []

    def build(i: int, j: int) -> None:
        while i < j:
            step = choice[i][j]
            assert step is not None
            if step[0] == "s":
                unmatched_idx.append(i)
                i += 1
            else:
                k, cid = step[1], step[2]
                canonical_ids.append(cid)
                build(i + 1, k)
                i = k + 1

    build(0, n)

    info = {cid: (a, b, r) for cid, a, b, r in candidates}
    canonical_pairs = [_pair_entry(cid, hits, info) for cid in canonical_ids]

    return {
        # 以字符串承载任意精度十进制整数，避免客户端 JSON 大整数精度损失。
        "optimal_count": str(total),
        "paired_hits": 2 * P[0][n],
        "total_residual": C[0][n],
        "canonical_pairs": canonical_pairs,
        "unmatched_hits": [hits[k]["id"] for k in unmatched_idx],
        "classification": {
            "required": sorted(required),
            "optional": sorted(optional),
            "never": sorted(never),
        },
    }


# ---------------------------------------------------------------- 敏感度推导


def _outside_objective(
    n: int,
    arcs: list[list[tuple[int, int, str]]],
    P: list[list[int]],
    C: list[list[int]],
    W: list[list[int]],
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """全规则 outside 目标 DP。

    O[i][j]：把 [i,j) 当作一个子区间时，其外部上下文（[0,i) 与 [j,n) 上
    与之相容的非交叉匹配，含可能的包围弧链）能取得的（最大对数、最小残差、
    方案数）。与审计用的 outside 计数不同，这里对父区间的 *所有* 规则取最优，
    因此可推导任意候选被强制选用时的反事实最优值。
    """

    NEG = -1
    Op = [[NEG] * (n + 1) for _ in range(n + 1)]
    Oc = [[0] * (n + 1) for _ in range(n + 1)]
    Ow = [[0] * (n + 1) for _ in range(n + 1)]
    Op[0][n], Oc[0][n], Ow[0][n] = 0, 0, 1

    def merge(
        i: int, j: int, pairs: int, cost: int, ways: int
    ) -> None:
        if ways == 0:
            return
        if Op[i][j] == NEG or pairs > Op[i][j] or (pairs == Op[i][j] and cost < Oc[i][j]):
            Op[i][j] = pairs
            Oc[i][j] = cost
            Ow[i][j] = ways
        elif pairs == Op[i][j] and cost == Oc[i][j]:
            Ow[i][j] += ways

    for length in range(n, 0, -1):
        for h in range(0, n - length + 1):
            m = h + length
            ow = Ow[h][m]
            if ow == 0:
                continue
            op, oc = Op[h][m], Oc[h][m]
            # 跳过规则：父 [h,m) -> 子 [h+1,m)。
            merge(h + 1, m, op, oc, ow)
            # 配对规则：父 [h,m) 经弧 (h,k) -> 左子 [h+1,k)、右子 [k+1,m)。
            for k, r, _cid in arcs[h]:
                if k >= m:
                    continue
                merge(
                    h + 1,
                    k,
                    op + 1 + P[k + 1][m],
                    oc + r + C[k + 1][m],
                    ow * W[k + 1][m],
                )
                merge(
                    k + 1,
                    m,
                    op + 1 + P[h + 1][k],
                    oc + r + C[h + 1][k],
                    ow * W[h + 1][k],
                )

    return Op, Oc, Ow


def _counts_at_left_node(
    Op: list[list[int]],
    Oc: list[list[int]],
    Ow: list[list[int]],
    P: list[list[int]],
    C: list[list[int]],
    W: list[list[int]],
    a: int,
    m: int,
    arcs_a: list[tuple[int, int, str]],
    skip_k: int,
) -> tuple[int, int, int]:
    """在解析节点 [a,m)（以 a 为左边界）上求首步不使用弧 (a, skip_k) 时的
    （对数, 残差, 方案数），含外部上下文 O[a][m]。"""

    op, oc, ow = Op[a][m], Oc[a][m], Ow[a][m]
    best_p, best_c, best_w = -1, 0, 0

    def merge(pairs: int, cost: int, ways: int) -> None:
        nonlocal best_p, best_c, best_w
        if ways == 0:
            return
        if pairs > best_p or (pairs == best_p and cost < best_c):
            best_p, best_c, best_w = pairs, cost, ways
        elif pairs == best_p and cost == best_c:
            best_w += ways

    # 跳过 a。
    merge(op + P[a + 1][m], oc + C[a + 1][m], ow * W[a + 1][m])
    # a 与 k 配对，k < m 且 k != skip_k。
    for k, r, _cid in arcs_a:
        if k >= m:
            break  # arcs_a 按右端点升序
        if k == skip_k:
            continue
        merge(
            op + 1 + P[a + 1][k] + P[k + 1][m],
            oc + r + C[a + 1][k] + C[k + 1][m],
            ow * W[a + 1][k] * W[k + 1][m],
        )
    return best_p, best_c, best_w


def _disabled_from_outside(
    Op: list[list[int]],
    Oc: list[list[int]],
    Ow: list[list[int]],
    P: list[list[int]],
    C: list[list[int]],
    W: list[list[int]],
    n: int,
    arcs_a: list[tuple[int, int, str]],
    incoming_a: list[tuple[int, int, str]],
    a: int,
    b: int,
) -> tuple[int, int, int]:
    """从全规则 outside 上下文推导禁用弧 e=(a,b) 后的反事实最优值。

    每个方案在解析树上对顶点 a 只有两种互斥且完备的情形：

    * A：a 未配对或是某条弧的左端点——路径上存在唯一节点 [a,m)。m<=b 时 e
      本就不在区间内，直接取 inside；m>b 时首步规则排除 e。
    * B：a 是入弧 g=(i,a) 的右端点——唯一节点 [i,m) 以 g 为首步规则，
      内部 [i+1,a) 与右兄弟 [a+1,m) 自由（均不可能含 e）。

    各 (情形, 节点) 互斥，方案数可直接相加；上下文与各兄弟内解天然不含 e，
    故自由部分一律取共享 inside 表。
    """

    best_p, best_c, best_w = -1, 0, 0

    def merge(pairs: int, cost: int, ways: int) -> None:
        nonlocal best_p, best_c, best_w
        if ways == 0:
            return
        if pairs > best_p or (pairs == best_p and cost < best_c):
            best_p, best_c, best_w = pairs, cost, ways
        elif pairs == best_p and cost == best_c:
            best_w += ways

    # 情形 A：存在以 a 为左边界的节点 [a,m)。
    for m in range(a + 1, n + 1):
        if Ow[a][m] == 0:
            continue
        if m <= b:
            merge(
                Op[a][m] + P[a][m],
                Oc[a][m] + C[a][m],
                Ow[a][m] * W[a][m],
            )
        else:
            merge(*_counts_at_left_node(Op, Oc, Ow, P, C, W, a, m, arcs_a, b))

    # 情形 B：a 作为入弧 (i,a) 的右端点，节点 [i,m) 首步为该入弧。
    for i, r, _cid in incoming_a:
        for m in range(a + 1, n + 1):
            if Ow[i][m] == 0:
                continue
            ways = Ow[i][m] * W[i + 1][a] * W[a + 1][m]
            if ways == 0:
                continue
            merge(
                Op[i][m] + 1 + P[i + 1][a] + P[a + 1][m],
                Oc[i][m] + r + C[i + 1][a] + C[a + 1][m],
                ways,
            )

    return best_p, best_c, best_w


def _forced_from_outside(
    Op: list[list[int]],
    Oc: list[list[int]],
    Ow: list[list[int]],
    P: list[list[int]],
    C: list[list[int]],
    W: list[list[int]],
    n: int,
    a: int,
    b: int,
    r: int,
) -> tuple[int, int, int]:
    """从全规则 outside 上下文聚合强制选用弧 (a,b) 的反事实最优值。

    含 e 的方案对应唯一分解路径：e 作为区间 [a,m) 的首步规则，m 唯一；
    对 m>b 聚合 上下文 O[a][m] + e + 内部 [a+1,b) + 右兄弟 [b+1,m)。
    """

    best_p, best_c, best_w = -1, 0, 0
    pin, cin, win = P[a + 1][b], C[a + 1][b], W[a + 1][b]
    for m in range(b + 1, n + 1):
        ways = Ow[a][m] * win * W[b + 1][m]
        if ways == 0:
            continue
        pairs = Op[a][m] + 1 + pin + P[b + 1][m]
        cost = Oc[a][m] + r + cin + C[b + 1][m]
        if pairs > best_p or (pairs == best_p and cost < best_c):
            best_p, best_c, best_w = pairs, cost, ways
        elif pairs == best_p and cost == best_c:
            best_w += ways
    return best_p, best_c, best_w


def _profile_block(pairs: int, cost: int, ways: int) -> dict[str, Any]:
    return {
        "paired_hits": 2 * pairs,
        "total_residual": cost,
        "optimal_count": str(ways),
    }


def sensitivity(payload: Any) -> dict[str, Any]:
    """对每条候选返回禁用/强制两种反事实下的最优剖面。

    共享一份 inside 表与一份全规则 outside 目标表；全部反事实值由区间状态与
    外部上下文聚合得到，不逐条重跑完整审计，总代价 O(n*|C| + n^3)。
    """

    hits, candidates = validate_payload(payload)
    n = len(hits)
    arcs = _build_arcs(n, candidates)
    P, C, W, _seq, _choice = _inside_dp(n, arcs)
    used_count = _outside_counts(n, candidates, arcs, P, C, W)
    Op, Oc, Ow = _outside_objective(n, arcs, P, C, W)
    total = W[0][n]
    p0, c0 = P[0][n], C[0][n]

    # incoming[a]：以 a 为右端点的候选 (左端点, 残差, id)。
    incoming: list[list[tuple[int, int, str]]] = [[] for _ in range(n)]
    for cid, ia, ib, ir in candidates:
        incoming[ib].append((ia, ir, cid))

    profiles: list[dict[str, Any]] = []
    for cid, a, b, r in candidates:
        used = used_count[cid]

        # 禁用：e 并非出现在全部最优方案中时目标不退化，方案数 total-used；
        # 必选弧禁用后从共享区间状态与外部上下文聚合反事实最优值。
        if used == total:
            dp_pairs, dp_cost, dp_ways = _disabled_from_outside(
                Op, Oc, Ow, P, C, W, n, arcs[a], incoming[a], a, b
            )
        else:
            dp_pairs, dp_cost, dp_ways = p0, c0, total - used
        disabled_block = _profile_block(dp_pairs, dp_cost, dp_ways)

        # 强制：e 本就可出现在最优方案中则目标不变，方案数即其出现次数；
        # 从不出现的候选由外部上下文聚合强制选用 e 后的最优值。
        if used > 0:
            fp_pairs, fp_cost, fp_ways = p0, c0, used
        else:
            fp_pairs, fp_cost, fp_ways = _forced_from_outside(
                Op, Oc, Ow, P, C, W, n, a, b, r
            )
        forced_block = _profile_block(fp_pairs, fp_cost, fp_ways)

        profiles.append({"id": cid, "disabled": disabled_block, "forced": forced_block})

    return {"profiles": profiles}


# ---------------------------------------------------------------- 反事实见证


def sensitivity_witness(payload: Any) -> dict[str, Any]:
    """按候选标识与约束模式返回约束后的规范配对与未配对击中。"""

    hits, candidates = validate_payload(payload)

    errors: list[dict[str, str]] = []
    target = payload.get("target") if isinstance(payload, dict) else None
    mode = payload.get("mode") if isinstance(payload, dict) else None
    if not isinstance(target, str) or not target:
        _err(errors, "/target", "target 必须是非空候选标识字符串")
    if not isinstance(mode, str) or mode not in WITNESS_MODES:
        _err(
            errors,
            "/mode",
            f"mode 必须是 {WITNESS_MODES[0]} 或 {WITNESS_MODES[1]}",
        )
    if not errors and target not in {cid for cid, _a, _b, _r in candidates}:
        _err(errors, "/target", f"未知候选标识: {target}")
    if errors:
        raise ValidationError(errors)

    n = len(hits)
    arcs = _build_arcs(n, candidates)
    info = {cid: (a, b, r) for cid, a, b, r in candidates}

    fixed_ids = {target} if mode == MODE_FORCED else set()
    forbidden_ids = {target} if mode == MODE_DISABLED else set()
    endpoint_of = {cid: (ea, eb) for cid, ea, eb, _r in candidates}

    def contained(cid: str, i: int, j: int) -> bool:
        x, y = endpoint_of[cid]
        return i <= x and y <= j

    @lru_cache(maxsize=None)
    def solve(
        i: int, j: int, fixed: tuple[str, ...]
    ) -> tuple[bool, int, int, int, Optional[tuple[str, ...]]]:
        """返回 (可行, 对数, 残差, 方案数, 字典序最小 id 序列)。

        fixed 中的候选必须全部包含在 [i,j) 的解里（调用时保证均落在区间内）。
        """

        if i == j:
            return (len(fixed) == 0, 0, 0, 1, ())

        best: list[Any] = [False, None, None, 0, None]

        def consider(
            feasible: bool,
            pairs: int,
            cost: int,
            ways: int,
            sequence: Optional[tuple[str, ...]],
        ) -> None:
            if not feasible:
                return
            if not best[0] or pairs > best[1] or (pairs == best[1] and cost < best[2]):
                best[0] = True
                best[1] = pairs
                best[2] = cost
                best[3] = ways
                best[4] = sequence
            elif pairs == best[1] and cost == best[2]:
                # 方案数对所有达到两级目标的规则累加；序列单独取字典序最小。
                best[3] += ways
                if sequence < best[4]:
                    best[4] = sequence

        # 规则 1：i 未配对（仅当不含以 i 为左端点的固定弧）。
        if all(endpoint_of[cid][0] != i for cid in fixed):
            child = solve(i + 1, j, fixed)
            consider(child[0], child[1], child[2], child[3], child[4])

        # 规则 2：i 与 k 配对；固定弧按几何关系分派给两个子区间。
        for k, r, cid in arcs[i]:
            if k >= j or cid in forbidden_ids:
                continue
            remaining = tuple(x for x in fixed if x != cid)
            inner: list[str] = []
            outer: list[str] = []
            feasible = True
            for x in remaining:
                if contained(x, i + 1, k):
                    inner.append(x)
                elif contained(x, k + 1, j):
                    outer.append(x)
                else:
                    feasible = False  # 与所选弧交叉/共端点，无法共存
                    break
            if not feasible:
                continue
            left_res = solve(i + 1, k, tuple(inner))
            right_res = solve(k + 1, j, tuple(outer))
            if not left_res[0] or not right_res[0]:
                continue
            sequence = (cid,) + left_res[4] + right_res[4]
            consider(
                True,
                1 + left_res[1] + right_res[1],
                r + left_res[2] + right_res[2],
                left_res[3] * right_res[3],
                sequence,
            )

        return best[0], best[1], best[2], best[3], best[4]

    feasible, pairs, cost, ways, sequence = solve(
        0, n, tuple(sorted(fixed_ids))
    )
    # 强制模式总可行（e 单独成配即一个合法方案）；防御性保留断言。
    assert feasible and sequence is not None

    matched: set[int] = set()
    canonical_pairs: list[dict[str, Any]] = []
    for cid in sequence:
        a, b, _r = info[cid]
        matched.add(a)
        matched.add(b)
        canonical_pairs.append(_pair_entry(cid, hits, info))

    return {
        "target": target,
        "mode": mode,
        "optimal_count": str(ways),
        "paired_hits": 2 * pairs,
        "total_residual": cost,
        "canonical_pairs": canonical_pairs,
        "unmatched_hits": [hits[k]["id"] for k in range(n) if k not in matched],
    }
