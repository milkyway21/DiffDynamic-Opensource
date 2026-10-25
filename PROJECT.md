# PROJECT.md — DiffDynamic 项目说明

## 定位

面向蛋白口袋的三维分子生成与评估框架。基于 TargetDiff 一系的 EGNN 扩散模型
预训练权重，叠加自研的动态采样与多轮筛选策略。**仅推理**：仓库不含训练链路，
预训练权重通过 `python tools/project.py weights --download` 获取
（URL 与 SHA256 固定于 `weights-manifest.json`）。

## 功能清单

- **采样**：baseline（标准反向扩散）、dynamic（动态跳步与梯度融合）、
  Prudent（多轮生成 + Vina 打分择优推进）
- **骨架约束生成**：scaffold placement 与 cascade 流水线
- **分子优化**与 **medchem 药化检查/修复**（三模式，见 `docs/MEDCHEM.md`）
- **评估**：Vina 对接（隔离子进程）、SA/QED 等分子指标、PT 结果提取为
  SDF/Excel
- **口袋分析**：八维口袋质量评分（见 `docs/POCKET_QUALITY.md`）与可视化
- **批量编排**：多 GPU 派发、断点续跑、结果汇总
- **Web 工作台**：自定义/测试集口袋生成、批量生成、骨架投放、优化、
  评估/提取、分子下载、进度/取消、历史重跑；用户级目录隔离、限流与日配额、
  管理员运行时调参

## 仓库布局

| 目录 | 内容 |
|---|---|
| `src/diffdynamic/` | 算法库：models / data / sampling / chemistry / evaluation / analysis / batch / io / cli |
| `server/` | FastAPI 后端：API、认证、调度、配额、路由与子进程 runner |
| `ui/` | 纯静态前端（Alpine.js + 3Dmol + Three.js，无构建步骤）与契约测试 |
| `scripts/` | 兼容入口（`sample_diffusion.py` 等，转发到 src 包） |
| `tools/` | `project.py`（weights/audit/export）、`checks.py`（测试入口） |
| `tests/` | pytest 回归：安全、配额、兼容契约、算法 |
| `configs/` | `sampling.yml`、策略预设、ablation 配置 |
| `docs/` | 架构、药化、口袋质量、发布说明 |
| `rdkit-structure-repair/` | 随包分发的 RDKit 结构修复子库 |
| `local/` | gitignored：历史归档、研究数据、迁移台账 |

## 入口

- 统一 CLI：`diffdynamic sample | batch | evaluate | pocket | extract | cascade | plot`
- 兼容命令：仓库顶层与 `scripts/` 下的旧脚本保持原参数与输出格式
- Web：`bash start_server.sh`（本机开发加 `DD_AUTH_MODE=off DD_HOST=127.0.0.1`）

## 环境与数据

Python 3.8 / PyTorch 2.4.1 + CUDA 11.8 / PyG 栈；OpenBabel（conda 安装）与
`vina` Python 包为重建和评估链路必需。完整搭建步骤见 `README.md` 的
"Set up from scratch"。

测试集采样需要 CrossDocked 测试划分（目录约定与来源见 README 的 "Data"）；
自定义口袋模式（`--protein_path` / `--ligand_path`）不需要任何数据集。

## 架构速览

调用链：CLI 与兼容适配层 → `sampling` 流水线 → `chemistry` 重建 →
`evaluation`（隔离子进程打分）→ `io`（PT 结果交换）；`batch` 负责多卡编排。
Web 层（`server/`）把同一批入口脚本作为子进程 runner 调度，前端为纯静态页面。
模块边界与依赖关系详见 `docs/ARCHITECTURE.md`。

## 许可与致谢

网络骨架与扩散公式源自 TargetDiff，另有 liGAN（GPLv2，字节冻结文件）、
RDKit Contrib（SA 评分）、Lilly medchem 规则等第三方来源。
声明与许可保留见 `LICENSE`、`THIRD_PARTY_NOTICES.md` 与源文件头。
