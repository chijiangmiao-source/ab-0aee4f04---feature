# 硅微条击中配对审计服务

把一列按位置严格递增的硅微条击中还原为互不交叉的粒子径迹配对，并在全部
最优方案上做审计：方案计数、规范解、未配对击中与每条候选对的必选/可选/从不分类。
此外提供反事实敏感度接口，评估单条候选被屏蔽或强制采纳时最优解释的退化程度。

## 问题与目标

- 输入：4–180 个击中（`id` 唯一、`position` 严格递增），至多 4000 条候选配对
  （`id` 唯一、端点对不重复、`residual` 为非负整数）。
- 选中的配对必须端点互异，按位置绘制后两两不交叉（允许嵌套）。
- 优化顺序（字典序）：
  1. 最大化已配对击中数（即最大化配对对数）；
  2. 最小化残差总和；
  3. 以“左端位置顺序下的配对 id 序列”字典序最小者为规范解。
- 审计输出：
  - `optimal_count`：任意精度十进制（字符串承载），达到前两级目标的方案数；
  - `canonical_pairs`：规范配对（含端点与残差，按左端位置顺序）；
  - `unmatched_hits`：规范解中的未配对击中；
  - `classification`：依据全部最优方案给出 `required` / `optional` / `never`。
- 空候选列表合法，返回计数 1 的唯一空方案。
- 重复端点对、未知端点、位置不严格递增/冲突、标识重复、规模越界等均返回
  `400 {"errors": [{"field": "/路径", "message": "..."}]}`，错误响应不夹带任何审计字段。

## 敏感度接口

### `POST /sensitivity`

请求体与 `/audit` 完全相同（`{hits, candidates}`），一次返回每条候选在两种
反事实约束下的最优剖面，不逐条重跑审计：

```json
{
  "profiles": [
    {
      "id": "a_out",
      "disabled": {"paired_hits": 4, "total_residual": 6, "optimal_count": "1"},
      "forced":   {"paired_hits": 4, "total_residual": 6, "optimal_count": "1"}
    }
  ]
}
```

- `disabled`：屏蔽该候选后的最佳配对数（`paired_hits`）、残差总和与
  任意精度最优方案数（字符串）；
- `forced`：强制选用该候选（其余候选仍按无交叉约束择优）后的同组指标；
- 目标退化只可能表现为配对数下降或残差上升。空候选列表返回
  `{"profiles": []}`；非法请求沿用字段路径错误，不夹带剖面字段。

### `POST /sensitivity/witness`

在 `/audit` 数据上增加 `target`（候选标识）与 `mode`（`disabled` / `forced`），
返回该约束下的规范见证，裁决规则（无交叉、目标顺序、字典序、计数精度）与
`/audit` 完全一致：

```json
{
  "target": "bait",
  "mode": "forced",
  "optimal_count": "1",
  "paired_hits": 4,
  "total_residual": 1,
  "canonical_pairs": [ ... ],
  "unmatched_hits": ["h1", "h3"]
}
```

`target` 未引用任何候选时返回 `400` 且字段路径为 `/target`；`mode` 非法时
字段路径为 `/mode`。

## 算法

区间非交叉匹配（允许嵌套），状态为区间 `[i,j)`：

- inside DP：`跳过 i` 或 `i 与 k 配对`（内部 `[i+1,k)` 与外部 `[k+1,j)` 独立），
  维护最大对数、最小残差、任意精度方案数，以及字典序最小 id 序列；
- outside DP（inside-outside）：统计每条候选弧出现在多少个最优方案中，
  据此分类 required / optional / never。
- 敏感度复用同一份 inside 状态：非退化情形直接由出现计数 total−used(e) /
  used(e) 得到；另做一次全规则 outside 目标 DP（外部上下文对父区间所有规则
  择优，而非只下发最优规则），与 inside 联合聚合必选弧禁用与从不出现候选
  强制的反事实最优值，复杂度同为 O(n·|C| + n³)；全量四千候选实测约 0.3 秒。
- 见证用带固定/禁用集合的记忆化区间递归取回约束后的规范方案。

复杂度 O(n·|C| + n³)（n ≤ 180，|C| ≤ 4000），最大规模审计实测约 0.2 秒。

## 文件

| 文件 | 说明 |
| --- | --- |
| `solver.py` | 校验 + 区间 DP 求解 + 敏感度/见证推导（仅标准库） |
| `app.py` | HTTP 服务：`GET /health`、`POST /audit`、`POST /sensitivity`、`POST /sensitivity/witness`（仅标准库） |
| `verify.py` | 单次复核：pytest、构建检查、API/HTTP 冒烟 |
| `tests/` | 单元测试、进程内 HTTP 测试、随机暴力枚举交叉验证 |
| `Dockerfile` | API 镜像定义 |
| `docker-compose.yml` | `api` 服务 + `verify` 复核服务 |

## 运行

```bash
# 默认宿主机端口 8080，可用 HOST_PORT 覆盖
HOST_PORT=9090 docker compose up --build -d api

curl -s http://localhost:9090/health
# {"status":"ready","service":"track-pair-audit"}
```

审计请求示例：

```bash
curl -s -X POST http://localhost:9090/audit \
  -H 'Content-Type: application/json' \
  -d '{
    "hits": [
      {"id":"h0","position":0},{"id":"h1","position":10},
      {"id":"h2","position":20},{"id":"h3","position":30}
    ],
    "candidates": [
      {"id":"a_out","left_endpoint":"h0","right_endpoint":"h3","residual":1},
      {"id":"a_in","left_endpoint":"h1","right_endpoint":"h2","residual":5},
      {"id":"b_left","left_endpoint":"h0","right_endpoint":"h1","residual":3},
      {"id":"b_right","left_endpoint":"h2","right_endpoint":"h3","residual":3}
    ]
  }'
```

敏感度（同一份数据逐条给出禁用/强制剖面）：

```bash
curl -s -X POST http://localhost:9090/sensitivity \
  -H 'Content-Type: application/json' -d @body.json
```

反事实见证（强制选用交叉诱饵后的规范配对与未配对击中）：

```bash
curl -s -X POST http://localhost:9090/sensitivity/witness \
  -H 'Content-Type: application/json' \
  -d "$(cat body.json | python3 -c 'import json,sys; d=json.load(sys.stdin); d.update(target="a_out", mode="forced"); print(json.dumps(d))')"
```

## 复核（verify 单次服务）

```bash
docker compose build && docker compose run --rm verify
```

`verify` 服务等待 `api` 健康后依次执行：

1. 代码测试（185 项，含随机输入与暴力枚举的审计/敏感度/见证对照）；
2. 构建检查（语法编译、模块导入、镜像内关键文件齐备）；
3. API/HTTP 冒烟（健康路径、嵌套同优、交叉低价诱饵、空候选、非法引用、
   重复端点对、位置冲突、规模越界、未知路径，以及敏感度的必选弧禁用退化、
   交叉诱饵强制见证、空剖面、非法 target/mode、旧审计回归）。

全部通过退出码 0，任一失败非零。

## 本地开发

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
PORT=8080 python app.py
```
