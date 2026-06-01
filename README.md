# DiffDynamic — 基于扩散模型的结构化药物设计平台

DiffDynamic 是一个基于扩散模型的 3D 结构化药物设计（SBDD）框架。在 [TargetDiff](https://github.com/DeepGraphLearning/TargetDiff) 的基础上，引入了多项推理阶段优化技术，显著提升生成分子的质量与多样性，并提供完整的 Web 交互界面用于论文演示。

## 核心创新

### 1. 两阶段动态跳步采样（Dynamic Skip-Step Sampling）

**AIDD 领域首创。** 扩散去噪轨迹中，不同时间步携带的结构信息量差异巨大：

- **高噪声区（t=999→650）**：模型确定拓扑结构（原子连接、环系统、骨架形状），每步携带**低边际信息**，可大幅跳步
- **低噪声区（t=650→0）**：模型精修几何结构（键角、扭转角、精确原子坐标），每步携带**高边际信息**，需密集步长

基于此洞察，采样轨迹在可配置的时间边界 `t_boundary` 处分段：

| 阶段 | 范围 | 目的 | 步长调度 |
|------|------|------|----------|
| 大步跳跃 | t=999 → t_boundary | 拓扑探索：自适应大步长跳跃 | Lambda/线性调度（步长 15-58） |
| 精细修复 | t_boundary → t=0 | 几何精修：密集小步长 | Lambda/线性调度（步长 3-60） |

配合**结构修复机制**（targetdiff_baseline_refine），在大步跳跃后执行浅层正向扩散至 t=9 再完整反向扩散至 t=0，修复局部几何缺陷。

**实际效果**：相比均匀 1000 步采样，减少约 60-70% 的神经网络评估次数，同时保持或提升输出质量。

### 2. 梯度融合（Gradient Fusion）

在每个去噪步骤中混合两种梯度信号：

```
局部梯度 = posterior_mean(x_t) - x_t          # 标准扩散方向
全局梯度 = predicted_x0(x_t)   - x_t          # 直达目标方向
融合梯度 = λ(t) × 全局梯度 + (1-λ(t)) × 局部梯度
```

其中 λ(t) 从 1.0 衰减至 0.0：
- **早期步骤**（高 t）：依赖全局信号建立整体分子构象
- **后期步骤**（低 t）：依赖局部后验精修键级几何

支持 5 种衰减调度：`quadratic`（默认）、`linear`、`exponential`、`adaptive`、`time`。

### 3. Prudent 多轮过滤

将分子属性过滤嵌入扩散过程内部（而非后处理），实现生成-过滤-精修循环：

```
每轮：
  1. 大步跳跃 → 生成候选分子
  2. 每个候选开启 N 条并行去噪链（如 N=20）
  3. 各链精修至 t=0
  4. 评分：先快速 QED/SA 门控，通过后再运行 Vina 对接
  5. 综合评分排序，保留 Top-K
```

**关键设计**：QED/SA 门控避免对低药物相似性分子运行昂贵的 Vina 对接，大幅节省计算。

### 4. 骨架约束生成

两种模式生成保留化学骨架的分子：

- **骨架进化（Evolve）**：基于 SDEdit 噪声-去噪，种群进化保持多样性，噪声退火从高到低
- **骨架生长（Grow）**：固定骨架原子，在口袋内从头生成额外原子

### 5. 多维分子评分

```
最终分数 = 100 × 基础分 × PAINS乘数 × 稳定性乘数
基础分 = 0.4×亲和力归一化 + 0.3×QED + 0.2×SA归一化 + 0.1×Lipinski通过率
```

集成 Lilly Medchem Rules（PAINS 过滤、反应性基团检测）作为惩罚乘数。

## 创新总结

| 组件 | 新颖性 | 创新内容 | 继承内容 |
|------|--------|----------|----------|
| **动态跳步采样** | **AIDD 首创** | 两阶段拓扑/几何解耦；信息密度比例计算分配；跳步+结构修复 | 非均匀步长通用概念（DDIM、DPM-Solver） |
| **梯度融合** | 新颖公式 | 后验均值与 x₀ 预测方向混合，5 种调度策略 | Classifier-free guidance 概念 |
| **Prudent 过滤** | **SBDD 扩散首创** | 扩散内属性门控过滤；对接分数驱动选择；检查点重噪声 | 拒绝采样、属性过滤等独立组件 |
| **骨架生成** | 工程贡献 | Murcko 骨架检测 + 进化/生长流水线 | RePaint 掩码扩散 |

**最核心创新**是动态跳步采样——首个认识到分子扩散去噪步骤信息量差异巨大、并据此分配计算资源的 AIDD 系统。

## 项目结构

```
├── server/                  # FastAPI 后端
│   ├── main.py              # 入口 (python -m server.main)
│   ├── api.py               # REST API 端点
│   ├── database.py          # SQLAlchemy ORM (SQLite)
│   ├── jobs.py              # 异步任务调度 (GPU 分配 + 子进程管理)
│   ├── config.py            # 运行时配置
│   ├── data_manager.py      # 文件系统扫描、GPU 信息
│   └── molecule_ingest.py   # 评估结果 → 分子数据库
│
├── ui/                      # 前端 (纯 HTML/JS/CSS)
│   ├── templates/index.html # 单页应用
│   └── static/
│       ├── js/app.js        # UI 逻辑
│       └── css/style.css    # 样式
│
├── DiffDynamic/             # 核心 ML 管线
│   ├── scripts/sample_diffusion.py              # 分子生成（~7000 行）
│   ├── evaluate_pt_with_correct_reconstruct.py  # 分子评估
│   ├── evaluate_pocket_quality.py               # 口袋质量评估
│   ├── extract_pt_to_sdf_excel.py               # .pt → SDF/Excel
│   ├── configs/sampling.yml                     # 采样配置
│   ├── models/            # EGNN, UniTransformer, DiffDynamic
│   ├── utils/             # 数据加载、Vina 对接、评分
│   └── datasets/          # 蛋白/配体数据处理
│
├── requirements.txt         # Python 依赖
└── start_server.sh          # 启动脚本
```

## 快速开始

### 环境要求

- Python 3.8+
- Conda 环境 `diffdynamic`（含 PyTorch 1.12+、RDKit、AutoDock Vina）
- NVIDIA GPU（用于分子生成和评估）

### 安装

```bash
conda activate diffdynamic
pip install -r requirements.txt
```

### 启动 Web 服务

```bash
bash start_server.sh
```

服务启动后访问 http://localhost:7860/ 即可使用 Web 界面。API 文档位于 http://localhost:7860/docs。

### Web 界面功能

| 页面 | 功能 |
|------|------|
| **项目情况** | 项目介绍、核心创新、使用方法 |
| **生成** | 测试集生成（Dynamic/Prudent 模式）、自定义口袋生成 |
| **评估** | PT 文件评估、分子提取 |
| **分子** | 分子数据库搜索（口袋、SMILES、Vina 评分等） |
| **历史** | 操作记录、运行日志 |
| **配置** | 采样配置编辑、GPU 状态 |

### 命令行使用

```bash
# 测试集采样（data_id 0-99）
python DiffDynamic/scripts/sample_diffusion.py DiffDynamic/configs/sampling.yml \
    --data_id 0 --device cuda:0

# 自定义蛋白口袋生成
python DiffDynamic/scripts/sample_diffusion.py DiffDynamic/configs/sampling.yml \
    --protein_path /path/to/protein.pdb \
    --ligand_path /path/to/reference.sdf \
    --device cuda:0

# 评估 .pt 结果
python DiffDynamic/evaluate_pt_with_correct_reconstruct.py \
    DiffDynamic/outputs/result_0_*.pt \
    --protein_root DiffDynamic/data/crossdocked_pocket10_test_only

# 提取为 SDF + Excel
python DiffDynamic/extract_pt_to_sdf_excel.py \
    DiffDynamic/outputs/result_0_*.pt
```

## 配置说明

所有采样参数在 `DiffDynamic/configs/sampling.yml` 中：

```yaml
model:
  checkpoint: ./pretrained_models/pretrained_diffusion.pt
  use_grad_fusion: true
  grad_fusion_lambda:
    mode: quadratic    # 融合调度模式
    start: 1.0         # 初始全局梯度权重
    end: 0.0           # 最终局部梯度权重

sample:
  mode: dynamic        # baseline | dynamic | optimization | prudent
  num_steps: 1000
  dynamic:
    time_boundary: 650 # 拓扑/几何分界点
    large_step:
      schedule: lambda
      lambda_coeff_a: 58  # 大步跳跃步长系数
    refine:
      schedule: lambda
      lambda_coeff_a: 60  # 精修步长系数
    prudent:
      enable: false
      n_sampling: 20      # 每候选并行链数
      advance_top_k: 5    # 每轮保留数
```

## 评估指标

| 工具 | 指标 | 说明 |
|------|------|------|
| AutoDock Vina | 结合亲和力 (kcal/mol) | 越负越好 |
| RDKit | QED（药物相似性）| 0-1，越高越好 |
| RDKit | SA（合成可及性）| 0-1，越高越好 |
| Lilly Medchem Rules | PAINS、反应性基团 | 通过/不通过 |
| Lipinski | 五规则 | 0-5 通过数 |

## 与 TargetDiff 的关系

| 方面 | TargetDiff | DiffDynamic |
|------|-----------|-------------|
| 模型架构 | ScorePosNet3D + UniTransformer | 相同（未修改） |
| 训练 | 自定义训练循环 | 使用 TargetDiff 预训练检查点 |
| 采样 | 固定步长后验去噪 | **两阶段动态跳步 + 结构修复** + 梯度融合 |
| 分子过滤 | 无（后处理评估） | Prudent 多轮过滤 |
| 骨架支持 | 无 | 进化 + 生长模式 |
| 评分 | 基础 Vina + QED/SA | 多维综合评分 + Lilly 规则 |

## 技术栈

- **后端**：FastAPI + SQLAlchemy + SQLite (WAL 模式)
- **前端**：原生 HTML/JS/CSS（零框架依赖）
- **ML**：PyTorch + EGNN/UniTransformer + RDKit + AutoDock Vina
- **部署**：Uvicorn (ASGI)，绑定 0.0.0.0 支持网络访问

## 数据说明

以下数据不包含在 Git 仓库中：

| 目录 | 大小 | 说明 |
|------|------|------|
| `DiffDynamic/data/` | ~68MB | 测试集蛋白口袋数据（CrossDocked 子集） |
| `DiffDynamic/outputs/` | 运行时生成 | 分子生成结果和评估输出 |
| `DiffDynamic/pretrained_models/` | ~64MB | 预训练模型权重 |

## 致谢

本项目基于 [TargetDiff](https://github.com/DeepGraphLearning/TargetDiff)（Guan et al., ICLR 2023）构建。感谢其在扩散模型结构化药物设计方面的奠基性工作。
