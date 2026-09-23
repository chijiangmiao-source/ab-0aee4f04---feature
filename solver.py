"""硅微条击中配对审计求解器。

输入校验通过后，求解点列上的带权非交叉匹配问题（允许弧段相互嵌套）：

* 击中沿位置严格递增排列，候选对连接其中两个击中（左端点位置 < 右端点位置）；
* 选中的对端点互异，按位置绘制后两两不交叉（允许嵌套与并列）；
* 目标依次为：最大化已配对击中数（等价于最大化对数 * 2）、最小化残差总和；
* 在所有达到前两级目标的方案上统计任意精度方案数，输出按左端位置顺序下
  配对标识序列字典序最小的规范方案，并把每条候选对判为 required / optional / never。

算法采用区间 inside DP（求最优目标、方案数、规范序列）与 outside DP
（inside-outside，统计每条候选对出现在多少个最优方案中）。
敏感度（/sensitivity）复用同一份 inside 表，并新增一张不加最优过滤的外部
上下文 DP：把区间作为分解树节点时的上下文最优偏移沿树下发，从而对每条
候选在 O(n) 内推导「禁用 / 强制」反事实最优值，无需逐条重跑完整审计。
所有计数使用 Python 任意精度整数；区间 DP 复杂度 O(n*|C| + n^3)，
n <= 180、|C| <= 4000。
"""

from __future__ import annotations

from typing import Any, Optional

MIN_HITS = 4
MAX_HITS = 180
MAX_CANDIDATES = 4000

DISABLED_MODES = {"disabled", "disable", "禁用"}
FORCED_MODES = {"forced", "force", "强制"}


class ValidationError(Exception):
    """携带字段路径的请求校验错误。"""

    def __init__(self, errors: list[dict[str, str]]):
        super().__init__("; ".join(e["message"] for e in errors))
        self.errors = errors


def _err(errors: list[dict[str, str]], field: str, message: str) -> None:
    errors.append({"field": field, "message": message})


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate(
    payload: Any,
) -> tuple[list[dict[str, Any]], list[tuple[str, int, int, int]]]:
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


def _validate_witness(
    payload: Any,
) -> tuple[list[dict[str, Any]], list[tuple[str, int, int, int]], str, str]:
    """校验 /sensitivity/witness：审计字段同 /audit，另需 candidate 与 mode。"""

    hits, candidates = _validate(payload)

    errors: list[dict[str, str]] = []
    raw_cid = payload.get("candidate") if isinstance(payload, dict) else None
    raw_mode = payload.get("mode") if isinstance(payload, dict) else None

    known = {cid for cid, _a, _b, _r in candidates}
    if not isinstance(raw_cid, str) or not raw_cid:
        _err(errors, "/candidate", "candidate 必须是非空候选标识字符串")
    elif raw_cid not in known:
        _err(errors, "/candidate", f"未知候选标识: {raw_cid}")

    if not isinstance(raw_mode, str) or not raw_mode:
        _err(errors, "/mode", "mode 必须是 'disabled' 或 'forced'")
        mode = ""
    elif raw_mode in DISABLED_MODES:
        mode = "disabled"
    elif raw_mode in FORCED_MODES:
        mode = "forced"
    else:
        _err(errors, "/mode", f"未知约束模式: {raw_mode}")
        mode = ""

    if errors:
        raise ValidationError(errors)
    return hits, candidates, raw_cid, mode


# ---------------------------------------------------------------- 区间 DP


def _inside_dp(
    n: int, arcs: list[list[tuple[int, int, str]]]
) -> tuple[
    list[list[int]],
    list[list[int]],
    list[list[int]],
    list[list[Optional[tuple[str, ...]]]],
    list[list[Optional[tuple[Any, ...]]]],
]:
    """inside 区间 DP：返回 P/C/W（对数、残差、方案数）、seq（字典序最小 id
    序列）、choice（规范回溯首步）。"""

    # P[i][j]/C[i][j]/W[i][j]：区间 [i,j) 上的最大对数、最小残差、最优方案数。
    P = [[0] * (n + 1) for _ in range(n + 1)]
    C = [[0] * (n + 1) for _ in range(n + 1)]
    W = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        W[i][i] = 1

    def take(pairs: int, cost: int, ways: int, sequence: Optional[tuple[str, ...]]) -> None:
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

    # seq[i][j]：区间 [i,j) 最优方案中按左端位置顺序的最小 id 序列；
    # choice 记录对应首步：('s',) 跳过 i，或 ('p', k, cid) 以弧 (i,k) 配对。
    seq: list[list[Optional[tuple[str, ...]]]] = [
        [None] * (n + 1) for _ in range(n + 1)
    ]
    choice: list[list[Optional[tuple[Any, ...]]]] = [
        [None] * (n + 1) for _ in range(n + 1)
    ]
    for i in range(n + 1):
        seq[i][i] = ()

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
                    (cid,) + seq[i + 1][k] + seq[k + 1][j],  # type: ignore[operator]
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
                        candidate = (cid,) + seq[i + 1][k] + seq[k + 1][j]  # type: ignore[operator]
                        if candidate == best[3]:
                            chosen = ("p", k, cid)
                            break
            choice[i][j] = chosen

    return P, C, W, seq, choice


def _context_dp(
    n: int,
    arcs: list[list[tuple[int, int, str]]],
    P: list[list[int]],
    C: list[list[int]],
    W: list[list[int]],
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """外部上下文 DP（不做最优过滤）。

    对每个可作为分解树节点的区间 [i,j)，统计当该区间被整体留出（其内部
    后续另行决定）时，外部上下文的最优偏移：XP/XC 为外部弧的最大对数与
    最小残差，XW 为达到该偏移的上下文方案数（兄弟子树按自身最优计入）。

    与审计用的 filtered outside 不同：父区间的每条结构规则（跳过 / 任意
    弧配对）都下发，即便该规则在父区间并非最优——强制一条原本从不出现
    的弧时，其上下文链恰恰可能经过这类规则。
    """

    XP = [[0] * (n + 1) for _ in range(n + 1)]
    XC = [[0] * (n + 1) for _ in range(n + 1)]
    XW = [[0] * (n + 1) for _ in range(n + 1)]
    XW[0][n] = 1

    def offer(i: int, j: int, pairs: int, cost: int, ways: int) -> None:
        if ways == 0:
            return
        old = XW[i][j]
        if old == 0 or pairs > XP[i][j] or (pairs == XP[i][j] and cost < XC[i][j]):
            XP[i][j] = pairs
            XC[i][j] = cost
            XW[i][j] = ways
        elif pairs == XP[i][j] and cost == XC[i][j]:
            XW[i][j] = old + ways

    for length in range(n, 0, -1):
        for h in range(0, n - length + 1):
            m = h + length
            outside = XW[h][m]
            if outside == 0:
                continue
            po, co = XP[h][m], XC[h][m]
            # 跳过规则：父 [h,m) -> 子 [h+1,m)，位置 h 未配对。
            offer(h + 1, m, po, co, outside)
            # 配对规则：父 [h,m) 经弧 (h,k) -> 左子 [h+1,k)、右子 [k+1,m)。
            for k, r, _cid in arcs[h]:
                if k >= m:
                    continue
                # 左子：兄弟 [k+1,m) 在节点右侧。
                offer(
                    h + 1,
                    k,
                    po + 1 + P[k + 1][m],
                    co + r + C[k + 1][m],
                    outside * W[k + 1][m],
                )
                # 右子：兄弟 [h+1,k) 在节点左侧（弧 (h,k) 内侧）。
                offer(
                    k + 1,
                    m,
                    po + 1 + P[h + 1][k],
                    co + r + C[h + 1][k],
                    outside * W[h + 1][k],
                )

    return XP, XC, XW


# ---------------------------------------------------------------- 反事实求解


def _constrained_solve(
    n: int,
    arcs: list[list[tuple[int, int, str]]],
    forbid: Optional[tuple[int, int]] = None,
    force: Optional[tuple[int, int]] = None,
):
    """带单条约束（禁用或强制某弧）的区间 DP。

    返回 (pairs, cost, ways, canonical_ids, unmatched_idx)。强制时每个区间
    维护两个状态：t=0 / t=1 分别表示方案是否已含强制弧；配对规则按两个
    子区间状态的或合并，根区间只取 t=1。端点对唯一，故跳过强制弧端点
    的规则直接不可行。无强制约束时只算 t=0，与普通审计 DP 等价。
    """

    fa, fb = force if force is not None else (-1, -1)
    states = 2 if force is not None else 1

    # 每格每状态：(对数, 残差, 方案数, 最小序列, 回溯首步)。
    P: list[list[list[int]]] = [[[0] * states for _ in range(n + 1)] for _ in range(n + 1)]
    C: list[list[list[int]]] = [[[0] * states for _ in range(n + 1)] for _ in range(n + 1)]
    W: list[list[list[int]]] = [[[0] * states for _ in range(n + 1)] for _ in range(n + 1)]
    seq: list[list[list[Optional[tuple[str, ...]]]]] = [
        [[None] * states for _ in range(n + 1)] for _ in range(n + 1)
    ]
    choice: list[list[list[Optional[tuple[Any, ...]]]]] = [
        [[None] * states for _ in range(n + 1)] for _ in range(n + 1)
    ]
    for i in range(n + 1):
        # 空区间不可能含强制弧：仅 t=0 可行。
        W[i][i][0] = 1
        seq[i][i][0] = ()

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            for t in range(states):
                best_p: Optional[int] = None
                best_c = 0
                best_w = 0
                best_seq: Optional[tuple[str, ...]] = None
                best_step: Optional[tuple[Any, ...]] = None

                def take(
                    pairs: int,
                    cost: int,
                    ways: int,
                    sequence: Optional[tuple[str, ...]],
                    step: tuple[Any, ...],
                ) -> None:
                    nonlocal best_p, best_c, best_w, best_seq, best_step
                    if ways == 0:
                        return
                    if (
                        best_p is None
                        or pairs > best_p
                        or (pairs == best_p and cost < best_c)
                    ):
                        best_p, best_c, best_w = pairs, cost, ways
                        best_seq, best_step = sequence, step
                    elif pairs == best_p and cost == best_c:
                        best_w += ways
                        if sequence is not None and (
                            best_seq is None or sequence < best_seq
                        ):
                            best_seq, best_step = sequence, step

                # 规则 1：跳过 i。t=1 方案必须真正用上强制弧两端点，
                # 故不得跳过 fa / fb；t=0 方案可令其失配，照常跳过。
                if force is None or t == 0 or (i != fa and i != fb):
                    take(
                        P[i + 1][j][t],
                        C[i + 1][j][t],
                        W[i + 1][j][t],
                        seq[i + 1][j][t],
                        ("s", t),
                    )

                # 规则 2：i 与 k 配对。
                for k, r, cid in arcs[i]:
                    if k >= j:
                        continue
                    if forbid is not None and (i, k) == forbid:
                        continue
                    if force is not None and (i, k) == force:
                        if t != 1:
                            continue
                        # 强制弧本身；两个子区间都不可能再含它。
                        ways = W[i + 1][k][0] * W[k + 1][j][0]
                        take(
                            1 + P[i + 1][k][0] + P[k + 1][j][0],
                            r + C[i + 1][k][0] + C[k + 1][j][0],
                            ways,
                            (cid,) + seq[i + 1][k][0] + seq[k + 1][j][0],  # type: ignore[operator]
                            ("p", k, cid, 0, 0),
                        )
                    else:
                        child_states = range(states)
                        for s1 in child_states:
                            for s2 in child_states:
                                if force is not None and (s1 or s2) != t:
                                    continue
                                w1 = W[i + 1][k][s1]
                                w2 = W[k + 1][j][s2]
                                if w1 == 0 or w2 == 0:
                                    continue
                                take(
                                    1 + P[i + 1][k][s1] + P[k + 1][j][s2],
                                    r + C[i + 1][k][s1] + C[k + 1][j][s2],
                                    w1 * w2,
                                    (cid,)
                                    + seq[i + 1][k][s1]
                                    + seq[k + 1][j][s2],  # type: ignore[operator]
                                    ("p", k, cid, s1, s2),
                                )

                P[i][j][t] = best_p if best_p is not None else -10**9
                C[i][j][t] = best_c
                W[i][j][t] = best_w
                seq[i][j][t] = best_seq
                choice[i][j][t] = best_step

    root_t = 1 if force is not None else 0
    canonical_ids: list[str] = []
    unmatched_idx: list[int] = []

    def build(i: int, j: int, t: int) -> None:
        while i < j:
            step = choice[i][j][t]
            assert step is not None
            if step[0] == "s":
                unmatched_idx.append(i)
                i += 1
                # t 不变（step[1] == t）。
            else:
                _kind, k, cid, s1, s2 = step
                canonical_ids.append(cid)
                build(i + 1, k, s1)
                i, t = k + 1, s2

    build(0, n, root_t)
    return P[0][n][root_t], C[0][n][root_t], W[0][n][root_t], canonical_ids, unmatched_idx


class _Counterfactual:
    """一次敏感度请求的共享状态：inside 表 + 外部上下文 + 规则聚合。

    反事实最优三元组（配对数、残差、方案数）全部由共享表 O(|C|·n)
    推导，不逐条重跑审计；规范见证（单条）另走一次约束 DP 精确回溯。
    """

    def __init__(
        self,
        n: int,
        records: list[tuple[str, int, int, int]],
    ) -> None:
        self.n = n
        self.records = records
        # arcs[a] 按右端点排序：(b, r, cid)；incoming[a] 为以 a 为右端点的弧。
        self.arcs: list[list[tuple[int, int, str]]] = [[] for _ in range(n)]
        self.incoming: list[list[tuple[int, str, int]]] = [[] for _ in range(n)]
        self.arc_id: list[dict[int, tuple[str, int]]] = [dict() for _ in range(n)]
        for cid, a, b, r in records:
            self.arcs[a].append((b, r, cid))
            self.incoming[b].append((a, cid, r))
            self.arc_id[a][b] = (cid, r)
        for a in range(n):
            self.arcs[a].sort()

        self.P, self.C, self.W, _seq, _choice = _inside_dp(n, self.arcs)
        self.XP, self.XC, self.XW = _context_dp(
            n, self.arcs, self.P, self.C, self.W
        )

    def _aggregate_rules(
        self, a: int
    ) -> dict[int, tuple[int, int, int, int, int, int]]:
        """对区间起点 a 聚合所有 m 的首步规则（跳过 + 各弧）。

        返回 m -> (v1_pairs, v1_cost, v1_ways, v2_pairs, v2_cost, v2_ways)：
        v1 为最优 (对数,残差) 与方案数，v2 为严格次优层的对应值。
        禁用唯一最优弧时其余规则退到 v2；v2 用 -10**18 哨兵表示不存在。
        """

        n = self.n
        P, C, W = self.P, self.C, self.W
        out: dict[int, tuple[int, int, int, int, int, int]] = {}
        NONE = -10**18

        for m in range(a + 1, n + 1):
            rules: list[tuple[int, int]] = [(P[a + 1][m], C[a + 1][m])]
            ways: dict[tuple[int, int], int] = {
                (P[a + 1][m], C[a + 1][m]): W[a + 1][m]
            }
            for k, r, _cid in self.arcs[a]:
                if k >= m:
                    break
                pc = (
                    1 + P[a + 1][k] + P[k + 1][m],
                    r + C[a + 1][k] + C[k + 1][m],
                )
                w = W[a + 1][k] * W[k + 1][m]
                rules.append(pc)
                ways[pc] = ways.get(pc, 0) + w

            v1p, v1c = max((p, -c) for p, c in rules)
            v1c = -v1c
            v1w = ways[(v1p, v1c)]
            rest = [(p, c) for p, c in rules if (p, c) != (v1p, v1c)]
            if rest:
                v2p, v2c = max((p, -c) for p, c in rest)
                v2c = -v2c
                v2w = ways[(v2p, v2c)]
            else:
                v2p = v2c = NONE
                v2w = 0
            out[m] = (v1p, v1c, v1w, v2p, v2c, v2w)
        return out

    def forced(self, a: int, b: int, r: int) -> tuple[int, int, int]:
        """强制选用弧 (a,b) 时的反事实最优（对数、残差、方案数）。

        固定该弧为节点 [a,m) 的首步规则：内部 [a+1,b)、后缀 [b+1,m)
        取自身最优，外部上下文取上下文 DP 的最优偏移。
        """

        P, C, W = self.P, self.C, self.W
        best: Optional[tuple[int, int, int]] = None
        for m in range(b + 1, self.n + 1):
            if self.XW[a][m] == 0:
                continue
            pairs = 1 + P[a + 1][b] + P[b + 1][m] + self.XP[a][m]
            cost = r + C[a + 1][b] + C[b + 1][m] + self.XC[a][m]
            ways = W[a + 1][b] * W[b + 1][m] * self.XW[a][m]
            if best is None or pairs > best[0] or (pairs == best[0] and cost < best[1]):
                best = (pairs, cost, ways)
            elif pairs == best[0] and cost == best[1]:
                best = (pairs, cost, best[2] + ways)
        assert best is not None  # 经跳过链 [a,n) 恒可达。
        return best

    def disabled(
        self,
        a: int,
        b: int,
        agg: dict[int, tuple[int, int, int, int, int, int]],
        incoming_best: Optional[tuple[int, int, int]],
    ) -> tuple[int, int, int]:
        """禁用弧 (a,b) 时的反事实最优（对数、残差、方案数）。

        全局方案按位置 a 的命运互斥划分：
        * a 未配对，或以左端点与 k≠b 配对——按节点 [a,m) 分解：
          m <= b 时弧 (a,b) 越界，节点取原最优；m > b 时由规则聚合
          O(1) 排除该弧，再与外部上下文拼接；
        * a 作为某外层弧 (h,a) 的右端点——该弧必不出现，直接并入
          「强制选用 (h,a)」的最优值（incoming_best 已聚合）。
        """

        P, C, W = self.P, self.C, self.W
        best: Optional[tuple[int, int, int]] = incoming_best

        def consider(pairs: int, cost: int, ways: int) -> None:
            nonlocal best
            if ways == 0:
                return
            if best is None or pairs > best[0] or (pairs == best[0] and cost < best[1]):
                best = (pairs, cost, ways)
            elif pairs == best[0] and cost == best[1]:
                best = (pairs, cost, best[2] + ways)

        for m in range(a + 1, b + 1):
            if self.XW[a][m]:
                consider(
                    P[a][m] + self.XP[a][m],
                    C[a][m] + self.XC[a][m],
                    W[a][m] * self.XW[a][m],
                )

        for m in range(b + 1, self.n + 1):
            if self.XW[a][m] == 0:
                continue
            v1p, v1c, v1w, v2p, v2c, v2w = agg[m]
            _cid, r = self.arc_id[a][b]
            bp = 1 + P[a + 1][b] + P[b + 1][m]
            bc = r + C[a + 1][b] + C[b + 1][m]
            bw = W[a + 1][b] * W[b + 1][m]
            if (bp, bc) != (v1p, v1c):
                np_, nc_, nw_ = v1p, v1c, v1w
            elif v1w > bw:
                np_, nc_, nw_ = v1p, v1c, v1w - bw
            else:
                # 该弧是唯一最优规则，其余方案退到严格次优层。
                np_, nc_, nw_ = v2p, v2c, v2w
            consider(np_ + self.XP[a][m], nc_ + self.XC[a][m], nw_ * self.XW[a][m])

        assert best is not None
        return best

    def profile(self) -> dict[str, dict[str, tuple[int, int, int]]]:
        """返回每条候选在 disabled / forced 下的最优三元组。"""

        # 先算每条弧的 forced 值；同时按右端点聚合，供禁用推导处理
        # 「禁用弧左端点被外层弧占用」这一互斥情形。
        forced_by_id: dict[str, tuple[int, int, int]] = {}
        incoming_best: list[Optional[tuple[int, int, int]]] = [
            None for _ in range(self.n)
        ]

        for a in range(self.n):
            for b, r, cid in self.arcs[a]:
                value = self.forced(a, b, r)
                forced_by_id[cid] = value
                incoming_best[b] = self._merge(incoming_best[b], value)

        result: dict[str, dict[str, tuple[int, int, int]]] = {}
        for a in range(self.n):
            if not self.arcs[a]:
                continue
            agg = self._aggregate_rules(a)
            for b, _r, cid in self.arcs[a]:
                result[cid] = {
                    "disabled": self.disabled(a, b, agg, incoming_best[a]),
                    "forced": forced_by_id[cid],
                }
        return result

    @staticmethod
    def _merge(
        cur: Optional[tuple[int, int, int]], other: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        if cur is None or other[0] > cur[0] or (
            other[0] == cur[0] and other[1] < cur[1]
        ):
            return other
        if other[0] == cur[0] and other[1] == cur[1]:
            return (cur[0], cur[1], cur[2] + other[2])
        return cur

    def single(self, a: int, b: int) -> dict[str, tuple[int, int, int]]:
        """只推导一条弧 (a,b) 的禁用 / 强制三元组（供 witness 单条请求）。"""

        incoming_best: Optional[tuple[int, int, int]] = None
        for h, _cid, r in self.incoming[a]:
            incoming_best = self._merge(incoming_best, self.forced(h, a, r))
        agg = self._aggregate_rules(a)
        _cid, r = self.arc_id[a][b]
        return {
            "disabled": self.disabled(a, b, agg, incoming_best),
            "forced": self.forced(a, b, r),
        }


# ---------------------------------------------------------------- 对外入口


def audit(payload: Any) -> dict[str, Any]:
    """执行完整审计，返回可直接 JSON 序列化的结果。"""

    hits, candidates = _validate(payload)
    n = len(hits)

    # arcs[i]: 以位置 i 为左端点的候选 (右端点, 残差, id)。
    arcs: list[list[tuple[int, int, str]]] = [[] for _ in range(n)]
    for cid, a, b, r in candidates:
        arcs[a].append((b, r, cid))

    P, C, W, _seq, choice = _inside_dp(n, arcs)
    total = W[0][n]

    # ---------- outside DP ----------
    # O[i][j]：根区间 [0,n) 的最优方案中，[i,j) 作为一个内部最优子区间出现的
    # 方案数（外部上下文数）。按父区间向其两个子区间下发贡献。
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

    # ---------- 候选对出现次数与分类 ----------
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

    required: list[str] = []
    optional: list[str] = []
    never: list[str] = []
    for cid, a, b, _r in candidates:
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
    canonical_pairs = [
        {
            "id": cid,
            "left_endpoint": hits[info[cid][0]]["id"],
            "right_endpoint": hits[info[cid][1]]["id"],
            "residual": info[cid][2],
        }
        for cid in canonical_ids
    ]

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


def sensitivity(payload: Any) -> dict[str, Any]:
    """对每条候选返回禁用 / 强制下的反事实最优三元组。"""

    hits, candidates = _validate(payload)
    cf = _Counterfactual(len(hits), candidates)
    profile = cf.profile()

    out: list[dict[str, Any]] = []
    for cid, _a, _b, _r in candidates:
        dp, dc, dw = profile[cid]["disabled"]
        fp, fc, fw = profile[cid]["forced"]
        out.append(
            {
                "id": cid,
                "disabled": {
                    "optimal_count": str(dw),
                    "paired_hits": 2 * dp,
                    "total_residual": dc,
                },
                "forced": {
                    "optimal_count": str(fw),
                    "paired_hits": 2 * fp,
                    "total_residual": fc,
                },
            }
        )
    return {"profiles": out}


def sensitivity_witness(payload: Any) -> dict[str, Any]:
    """返回单条候选在指定约束模式下的规范配对与未配对击中。

    三元组与 /sensitivity 同源（共享区间/上下文推导），规范见证由单条
    约束 DP 精确回溯，沿用无交叉、目标顺序与字典序裁决规则。
    """

    hits, candidates, cid, mode = _validate_witness(payload)
    n = len(hits)
    cf = _Counterfactual(n, candidates)
    target = next((a, b, r) for x, a, b, r in candidates if x == cid)
    a, b, _r = target
    triple = cf.single(a, b)[mode]

    if mode == "disabled":
        _p, _c, _w, pair_ids, unmatched_idx = _constrained_solve(
            n, cf.arcs, forbid=(a, b)
        )
    else:
        _p, _c, _w, pair_ids, unmatched_idx = _constrained_solve(
            n, cf.arcs, force=(a, b)
        )

    info = {cid: (a, b, r) for cid, a, b, r in candidates}
    canonical_pairs = [
        {
            "id": pair_id,
            "left_endpoint": hits[info[pair_id][0]]["id"],
            "right_endpoint": hits[info[pair_id][1]]["id"],
            "residual": info[pair_id][2],
        }
        for pair_id in pair_ids
    ]

    pairs, cost, ways = triple
    return {
        "id": cid,
        "mode": mode,
        "optimal_count": str(ways),
        "paired_hits": 2 * pairs,
        "total_residual": cost,
        "canonical_pairs": canonical_pairs,
        "unmatched_hits": [hits[k]["id"] for k in unmatched_idx],
    }
