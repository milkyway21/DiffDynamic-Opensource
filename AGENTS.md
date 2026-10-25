# AGENTS.md — AI 编码代理工作规范

面向在本仓库工作的 AI 编码代理（以及新加入的人类协作者）。先读完本文件再动手。

## 项目一句话

DiffDynamic：基于预训练扩散模型的蛋白口袋 3D 分子生成与评估框架（**仅推理**，
无训练链路），附 FastAPI 后端 + 纯静态前端的 Web 工作台。

## 环境规则（重要）

- 唯一验证环境：conda env `diffdynamic`，Python 3.8，PyTorch 2.4.1 + CUDA 11.8。
  精确版本记录在 `requirements-validated.txt`。
- **不要升级或替换科学计算依赖**（torch、rdkit、torch-scatter/torch-sparse、
  PyG 等）；图形扩展 wheel 必须匹配 PyTorch/CUDA 构建。
- 安装本包固定用：`python -m pip install --no-deps --no-build-isolation -e .`
  （`setup.py` 的 install_requires 为空，依赖由环境提供）。
- 默认不引入新依赖；确有必要时先确认，并同步 `requirements.txt` 与验证记录。
- OpenBabel 走 conda（`conda install -c conda-forge openbabel`），不要 pip 装。

## 代码地图

- `src/diffdynamic/` 是唯一算法维护点，九个子包：
  models（网络/扩散）、data（解析/图）、sampling（baseline/dynamic/Prudent/
  骨架/优化）、chemistry（重建/片段/药化）、evaluation（指标/对接/隔离子进程）、
  analysis（口袋/绘图）、batch（批量编排）、io（路径/配置/PT 交换）、cli（命令）。
- 仓库顶层 `*.py` 与 `scripts/*.py` 大多是**兼容适配器**（约十行转发到 src 包）：
  改行为去 `src/diffdynamic/` 对应模块改，不要在适配器里写逻辑。
- `server/` 为 FastAPI 后端：`api.py`（中间件/限流/路径 ACL）、`auth.py`
  （平台会话 + 用户 slug 隔离）、`config.py`、`scheduler.py`（单进程调度）、
  `quota.py`（日配额，北京时间）、`routes/`、`runners/`（子进程任务）。
- `ui/` 为纯静态前端（Alpine.js + 3Dmol + Three.js，全部 vendor 化），
  **没有 npm/构建步骤**；改动直接编辑 `ui/static/` 与 `ui/templates/`。
- `tools/`：`project.py`（weights / audit / export 子命令）、`checks.py`
  （全量测试入口，使用临时运行时存储）。
- `configs/sampling.yml` 默认采样配置；`configs/sampling_strategies/` 预设；
  `configs/ablation/` 消融配置。
- `local/`（archive / research / inventory）与 `data/`、`outputs/`、`uploads/`、
  `diffdynamic.db` 等运行时产物均为 gitignored：**不要把生成物或数据提交入库**。
- `utils/` 中的 `gspt1_*` / `ikzf2_*` 实验模块被 `publish-manifest.json` 排除，
  不随发行；`rdkit-structure-repair/` 是随包分发的独立子库。

## 改动后必须跑的验证

```bash
python tools/checks.py                  # pytest 全量（临时数据库/输出目录）
node ui/tests/check-contracts.cjs       # 前端语法与契约（需 Node 18+）
python tools/project.py audit           # 发布清单与敏感信息审计
```

涉及 Web 路由/配额/认证时，另跑相关子集（如 `tests/test_web_limits.py`、
`tests/test_quota_daily.py`、`tests/test_go_live_security.py`）；
前端行为可用 `ui/tests/browser-smoke.cjs`（可选 Playwright）。

## 硬性约束

- **Web 服务器必须单进程**：内存态 JobScheduler、队列上限、限流器不跨 uvicorn
  worker 共享；扩容只改 `server_config.json` 的 `gpu_ids` / `max_concurrent_jobs`。
- 发布只走 `python tools/project.py export`（`publish-manifest.json` 白名单），
  工具绝不自动 stage/commit/push、不改 remote。
- 文档与代码中**不得写入绝对路径、服务器 IP、账号凭据**；生产运维细节只留在
  `local/` 下。
- 兼容契约由回归测试锁定（CLI 参数/默认值、PT 字段、SDF 属性、评估表格式）：
  `tests/test_refactor_*.py`、`tests/test_targetdiff_baseline_lock.py` 等，
  破坏即测试失败。
- 安全边界不要放松：路径 ACL、内部 key 仅限 loopback 且需显式开关、
  用户目录隔离、日配额——相关改动必须附带测试。
- `src/diffdynamic/chemistry/reconstruct_targetdiff_legacy.py` 是字节冻结的
  上游实现（SHA256 被回归测试校验），**不要重排格式**。

## 文档索引

- `README.md` / `README.zh-CN.md`：双语入口（安装、数据、命令、Web、验证）。
- `PROJECT.md`：项目说明（定位、功能清单、布局、入口）。
- `docs/ARCHITECTURE.md`：模块边界与调用关系。
- `docs/MEDCHEM.md`：药化模块参数、API 与报告格式。
- `docs/POCKET_QUALITY.md`：口袋八维评分。
- `docs/PUBLICATION.md`：GitHub 上传分类与发布流程。
