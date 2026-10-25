# 药化检查、修复与替代候选（维护说明）

工作台的“药化检查与修复”页支持 SDF/MOL、多行 SMILES（每行可附名称）和已完成的历史任务。
独立任务最多 100 条输入，超量明确失败；错误记录保留原顺序和原因。
从头生成的测试集、自定义口袋、批量来源共用同一组选项，默认关闭。
页面沿用 Alpine/classic scripts，支持中英文代理前缀，无需前端构建。

## 模式与默认值

| 参数 | 默认值 / 范围 | 含义 |
| --- | --- | --- |
| `operation` | `propose`；`check` / `repair` / `propose` | 仅检查；可靠修复；修复后追加替代候选。生成页面使用 `propose` |
| `profile` | `local`；`local` / `aggressive` | 局部模式保持环系；激进模式允许环系、骨架改造，仍作为独立候选 |
| `top_k` | 3；1–10 | 每条输入最多返回的候选数 |
| `min_similarity` | 0.55；0–1 | 候选相对同一主结构的指纹相似度下限 |
| `min_heavy_delta` / `max_heavy_delta` | −3 / 4；分别为 −10–0 / 0–10 | 重原子数变化范围 |
| `time_budget_seconds` | 30；1–120 | 每条输入的候选搜索预算；整个任务还包含解析、几何与评估时间 |
| `evaluation.mode` | `chemistry_3d` | 化学与局部三维检查；`vina_dock` 另外执行真实 Vina 对接 |
| `evaluation.exhaustiveness` | 8；1–32 | Vina 搜索强度 |
| `evaluation.timeout_seconds` | 120；5–600 | 每个结构的对接硬超时，单位秒 |
| `evaluation.seed` | 42；0–2147483647 | 构象/对接随机种子 |
| `evaluation.center_xyz` / `size_xyz` | 可选，必须成对提供 | 三个有限坐标；box 每轴长度 10–40 Å |

独立对接任务需要蛋白及参考配体或显式 box。参考配体必须有有限三维坐标；
自动 box 以其重原子范围中心为中心，每轴取 `max(extent + 10, 20)` Å，超过 40 Å 时要求显式设置。
历史任务和生成后处理可以继承受体及口袋上下文。

## 三个药化接口

接口沿用现有身份校验、用户路径 ACL 和提交限流；报告与文件按 run 所有者授权。
前端必须使用现有 API 前缀处理，不能拼接服务器文件路径下载。

| 接口 | 输入 / 返回 |
| --- | --- |
| `POST /api/medchem` | 下述请求；返回 `job_id`、`run_id`、`status` |
| `GET /api/medchem/runs/{run_id}/report` | 返回结构化报告；也接受生成父任务 ID，单个子任务时返回其报告，多个时通过 `reports` 返回选择列表 |
| `GET /api/medchem/runs/{run_id}/artifacts/{artifact_id}` | 下载报告登记的 SVG、SDF、PDB、PDBQT、CSV 或审计 JSON；不接受任意路径 |

请求中 `input_path`、`smiles`、`source_run_id` 三选一。文件应先走现有上传接口。
`input_aligned` 默认 `false`；`options` 省略时使用上表默认值，未知字段会被拒绝。

```json
{
  "smiles": "CCO example",
  "input_aligned": false,
  "options": {
    "operation": "propose",
    "profile": "local",
    "top_k": 3,
    "min_similarity": 0.55,
    "min_heavy_delta": -3,
    "max_heavy_delta": 4,
    "time_budget_seconds": 30,
    "evaluation": {
      "mode": "chemistry_3d",
      "exhaustiveness": 8,
      "timeout_seconds": 120,
      "seed": 42
    }
  }
}
```

其他顶层字段：`protein_path?`、`reference_ligand_path?`，以及替代 `smiles` 的
`input_path?: string` / `source_run_id?: positive integer`。SMILES 文本最多 200000 字符。
三类生成接口 `/api/generate`、`/api/generate/custom`、`/api/generate/batch`
新增 `medchem: MedchemOptions | null`，其对象内容就是上例的 `options`。

报告 `schema_version: 1` 包含 `run_id`、`source_run_id?`、`status`、`options`、
`rule_version`、`summary`、`records`、`artifacts`，批量报告另有 `pockets`。
`summary` 记录 total/processed/valid/repaired/candidates/failed；运行中字段可尚未齐全。
每条记录有 `record_id`、`name`、`status`、`findings`、`changes`、`rejected_counts`、
`original`、`primary`、`candidates`，失败或歧义时附原因。

每个结构有独立 `variant_id`、`stage`、`smiles`、`properties`、`atom_mapping`、
`geometry`、`evaluation`、图像/结构 artifact ID；候选另有变换、理由、相似度和 reward。
`geometry.status` 为 `passed` / `failed`，另用 `coordinate_source`、`aligned`、
`pocket_status` 说明坐标来源及口袋检查。`evaluation` 单独保存 completed/timeout/failed/skipped、
affinity、reason、pose artifact ID 和被评估的 variant ID。可选字段缺失时前端必须容错。
多口袋三维查看优先使用结构自身的 `protein_artifact_id`，再考虑报告级受体。

## 原始结构、主结果和评分

`original` 保留解析后的原始化学图，原始输入另保存在记录中；缺少三维坐标时可生成构象，
其来源必须标明。只有通过身份和几何约束的可靠修复才成为 `primary`。
歧义修复保留原结构并给出原因；替代候选由同一个主结构独立产生，不能因 reward 高而覆盖主结果。
候选是设计假设，reward、结构警示、QED/SA 和 Vina 分数均不保证药效、安全性或真实结合。

`chemistry_3d` 的独立任务和候选检查不运行 Vina。外部 SDF/MOL 默认未对齐；
任意嵌入的 SMILES 构象不用于蛋白碰撞或原位 `score_only`。
已确认处于蛋白坐标系且通过几何检查的结构才进行口袋检查。
未对齐结构可以在明确受体及 box 后做真实对接。

`vina_dock` 对有效的 original、primary、候选使用同一 box 分别计算，每个结构独立记录结果、超时或失败。
PDBQT 始终保存 Vina 返回的真实姿势；没有可靠原子映射时，不把它伪造成“已对接 SDF”。
SDF 是对应阶段的结构快照，不能将其与对接姿势混淆；旧 SDF/PT 分数不继承到新结构。

生成页原有的“主结果自动评估”仍由 `auto_evaluate` 控制，与候选评估选项独立。
完整结果放在记录的 `primary_evaluation`，界面用 `primary.main_evaluation`
区分 chemistry / score_only / dock，并复用同一主结构快照。
旧评估器不支持多片段结构时跳过主结果自动评估并注明原因，保留完整化学图、结构文件和药化属性；
不会为获得分数而静默删掉片段。

## 生成后处理、历史和取消

启用后，GPU 生成结束再提交独立 CPU 药化阶段；批量任务由一个 CPU 子任务处理各口袋，
每个口袋保留自己的受体和 box。自动评估/提取消费同一 primary 快照，避免各自重建覆盖修复。
`sample_only` 仍可执行药化阶段，但不启用既有主结果自动评估。原始生成 PT 不被修改。
生成任务与药化子任务有独立状态，应通过关联报告判断后处理是否结束。

历史页可打开报告、以已完成任务为输入，或通过既有 `/api/runs/{id}/rerun` 重跑。
读取旧药化报告时使用其 primary 快照，保留来源、失败记录和坐标上下文，重新计算新结构的评分。
历史批量生成必须有完整口袋清单；缺少清单时明确失败，不能随意选择一个文件代替全部口袋。
重跑仍执行公开参数和路径校验，客户端不能覆盖内部 PT 路径、输入限额或配额标记。

状态/取消复用 `GET /api/jobs/{job_id}` 和 `DELETE /api/jobs/{job_id}`。
报告以原子替换方式写入；`partial` 表示部分记录或评估失败，其余结果仍可查看。
正常取消终止工作进程并保留已落盘记录，未完成项不冒充成功；批量取消会合并当前口袋已落盘部分。
异常强杀只能恢复最后一次落盘快照。报告接口将数据库 cancelled/failed 终态优先展示。

## 代码入口与验证

维护入口：`server/medchem_schemas.py`、`server/routes/medchem.py`、
`server/submissions/medchem.py`、`server/runners/medchem.py`、
`src/diffdynamic/chemistry/medchem/`、`rdkit-structure-repair/src/structure_repair/optimize/`、
`ui/static/js/workbench/medchem.js`。修改 schema 时同步 UI 数值范围和契约测试。

在现有科学环境和仓库根目录运行：

```bash
PYTHONPATH=src:rdkit-structure-repair/src python -m pytest \
  tests/test_medchem_chemistry_pipeline.py tests/test_medchem_docking.py \
  tests/test_medchem_web.py -q
PYTHONPATH=src:rdkit-structure-repair/src python -m pytest \
  tests/test_refactor_installation.py -k 'archive or repair or medchem or web' -q
node --experimental-vm-modules ui/tests/check-contracts.cjs
node ui/tests/browser-smoke.cjs
```

安装测试离线构建 wheel/sdist，并在无关目录验证 wheel/editable 安装：包含页面 JS/CSS、
六份 catalog YAML，实际加载 34 条规则，并生成至少一个通过几何检查的候选。
浏览器测试需要 Playwright 与兼容 Chromium，可用 `DD_CHROMIUM_EXECUTABLE` 指定浏览器；
其 API 为模拟接口，覆盖三种路径及四种屏幕宽度，不能替代真实 Vina 环境验证。
