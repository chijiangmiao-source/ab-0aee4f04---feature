# 硅微条击中配对审计服务

把一列按位置严格递增的硅微条击中还原为互不交叉的粒子径迹配对，并在全部
最优方案上做审计：方案计数、规范解、未配对击中与每条候选对的必选/可选/从不分类。
另提供反事实敏感度剖面：评估每条候选被屏蔽或被标定流程强制采纳时，
最佳解释（配对数 / 残差 / 最优方案数）的变化。

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

请求体与 `/audit` 完全相同（`{hits, candidates}`）。一次返回每条候选在两种
反事实约束下的最优三元组，空候选列表合法并返回 `{"profiles": []}`：

```json
{
  "profiles": [
    {
      "id": "a_out",
      "disabled": {"optimal_count": "2", "paired_hits": 4, "total_residual": 6},
      "forced":   {"optimal_count": "1", "paired_hits": 4, "total_residual": 6}
    }
  ]
}
```

- `disabled`：该候选被屏蔽（不得选用）时的最佳配对数、残差总和与任意精度
  最优方案数；必选弧被禁用会体现为配对数退化或残差上升。
- `forced`：该候选被强制采纳时的对应值；强制从不出现的候选（如交叉诱饵）
  同样给出含该弧的最优解释。

全量 4000 候选只做一次共享区间/上下文 DP（实测约 0.4 秒），不逐条重跑审计。

### `POST /sensitivity/witness`

请求体在审计数据上追加 `candidate`（候选 id）与 `mode`：

- `mode` 取 `disabled`（别名 `disable`、`禁用`）或 `forced`（别名 `force`、
  `强制`），响应中统一规范化为英文；
- 返回该约束下的规范见证：`id`、`mode`、`optimal_count`、`paired_hits`、
  `total_residual`、`canonical_pairs` 与 `unmatched_hits`，裁决规则与
  `/audit` 完全一致（无交叉、目标顺序、id 字典序）。

未知候选标识返回 `400` 且字段路径为 `/candidate`；非法 mode 路径为 `/mode`；
审计数据本身的非法引用仍只返回 `/hits/...`、`/candidates/...` 字段路径错误，
不夹带任何剖面或见证字段。

## 算法

区间非交叉匹配（允许嵌套），状态为区间 `[i,j)`：

- inside DP：`跳过 i` 或 `i 与 k 配对`（内部 `[i+1,k)` 与外部 `[k+1,j)` 独立），
  维护最大对数、最小残差、任意精度方案数，以及字典序最小 id 序列；
- outside DP（inside-outside）：统计每条候选弧出现在多少个最优方案中，
  据此分类 required / optional / never；
- 敏感度复用同一份 inside 表，另做一张**不过滤最优规则**的外部上下文 DP
  （含嵌套包围与并列两种传播），再对每个左端点做首步规则的最优/严格次优
  聚合：禁用一条弧时 O(1) 扣除其贡献（唯一最优时退到次优层），并并入
  “该左端点被外层弧占用”的互斥情形；强制一条弧时固定其内部、后缀与上下文。
- 见证的规范方案由单条两状态（是否已含强制弧）约束 DP 精确回溯；禁用则为
  跳过该弧的普通区间 DP。

复杂度 O(n·|C| + n³)（n ≤ 180，|C| ≤ 4000），最大规模审计约 0.15 秒、
全量敏感度剖面约 0.4 秒。

## 文件

| 文件 | 说明 |
| --- | --- |
| `solver.py` | 校验 + 区间 DP 审计求解 + 共享上下文敏感度推导（仅标准库） |
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

敏感度请求示例：

```bash
# 全量剖面
curl -s -X POST http://localhost:9090/sensitivity \
  -H 'Content-Type: application/json' \
  -d @audit-body.json

# 单候选规范见证
curl -s -X POST http://localhost:9090/sensitivity/witness \
  -H 'Content-Type: application/json' \
  -d '{ ...同 /audit 的 hits/candidates..., "candidate": "a_out", "mode": "forced" }'
```

## 复核（verify 单次服务）

```bash
docker compose build && docker compose run --rm verify
```

`verify` 服务等待 `api` 健康后依次执行：

1. 代码测试（134 项通过，含 60 组审计随机输入与 40 组反事实随机输入的
   暴力枚举对照：方案数、规范解、分类、禁用/强制三元组与见证）；
2. 构建检查（语法编译、模块导入、镜像内关键文件齐备）；
3. API/HTTP 冒烟：
   - 旧审计：健康路径、嵌套同优、交叉低价诱饵、空候选、非法引用、
     重复端点对、位置冲突、规模越界、未知路径；
   - 敏感度：必选弧禁用后退化与旧审计回归、交叉诱饵强制后的最优见证、
     禁用见证回落、批量剖面与见证一致、空剖面、非法引用与未知候选错误。

全部通过退出码 0，任一失败非零。

## 本地开发

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
PORT=8080 python app.py
```
