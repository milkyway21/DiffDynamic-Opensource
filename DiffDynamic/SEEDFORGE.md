# SeedForge: 基于多参数迭代优化的骨架约束分子生成方法

## 1. 概述

SeedForge 是一个基于扩散模型（DiffDynamic）的**骨架约束分子生成参数优化框架**。它通过系统性地调整 scaffold grow 模式的生成参数，在目标蛋白口袋上迭代生成-评估-优化，找到最优的参数组合，从而显著提升生成分子的结合亲和力、药物相似性和完整性。

### 核心思想

骨架约束生成（Scaffold-Constrained Generation）是一种在固定分子骨架的基础上，通过扩散模型生成新分子侧链的方法。不同的生成参数（起始噪声水平、去噪步长、原子生成策略等）会显著影响生成分子的质量。SeedForge 通过**10轮迭代实验**，系统性地探索参数空间，找到最优配置。

### 命名由来

**Seed** = 随机种子（控制生成多样性）+ **Forge** = 锻造（迭代精炼参数）

---

## 2. 方法设计

### 2.1 优化流程

```
┌─────────────────────────────────────────────────────────────┐
│                    SeedForge 优化循环                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Round N:                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│   │ 生成分子  │───▶│ Vina评估  │───│ 指标分析  │              │
│   │(50个/轮) │    │(ScoreOnly)│   │(Vina/QED/│              │
│   └──────────┘    └──────────┘    │SA/完整性) │              │
│                                    └─────┬────┘              │
│                                          │                   │
│                                          ▼                   │
│                                   ┌──────────┐              │
│                                   │ 参数调整  │              │
│                                   │(基于结果) │              │
│                                   └──────────┘              │
│                                                             │
│   Round N+1: 使用新参数重复...                                │
│                                                             │
│   最终: 选择综合评分最高的参数组合                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 参数探索策略

每轮使用不同的参数组合，覆盖以下维度：

| 维度 | 变化范围 | 说明 |
|------|----------|------|
| `start_t` | 350 - 550 | 起始噪声时间步，控制分子多样性 |
| `stride` | 8 - 20 | 去噪步长，控制生成精细度 |
| `n_extra_mode` | prior_minus_scaffold / pocket_prior | 新原子生成策略 |
| `lambda_coeff_a/b` | 25-60 / 3-10 | 引导强度系数 |
| `n_extra_fixed/min/max` | 8-12 / 3-5 / 20-25 | 生成原子数量范围 |

### 2.3 评分体系

综合评分（Composite Score）由四个维度加权：

```
Composite = 0.60 × VinaScore_norm + 0.20 × QED + 0.10 × SA + 0.10 × Completeness
```

- **VinaScore (60%)**: AutoDock Vina score_only 模式的结合亲和力（越负越好），归一化到 [0,1]
- **QED (20%)**: 药物相似性评分 [0,1]
- **SA (10%)**: 合成可及性评分 [0,1]
- **Completeness (10%)**: 完整分子比例（无碎片）[0,1]

---

## 3. 实验结果

### 3.1 实验设置

- **目标口袋**: data_id=10 (PARP1, PDB: 5wtc)
- **GPU**: cuda:1
- **每轮分子数**: 50
- **评估模式**: VinaScore (score_only, 20秒超时)
- **总轮数**: 10

### 3.2 排名结果

| 排名 | 轮次 | 参数名 | VinaScore | QED | SA | 完整率 | 综合分 |
|------|------|--------|-----------|-----|-----|--------|--------|
| 1 | 4 | **pocket_prior** | **-13.119** | **0.601** | **0.840** | 100% | **0.8290** |
| 2 | 9 | best_rerun | -13.119 | 0.601 | 0.840 | 100% | 0.8290 |
| 3 | 0 | baseline | -12.848 | 0.446 | 0.691 | 100% | 0.7721 |
| 4 | 1 | low_start_t | -12.333 | 0.431 | 0.729 | 100% | 0.7524 |
| 5 | 2 | high_start_t | -12.107 | 0.431 | 0.719 | 100% | 0.7424 |
| 6 | 7 | high_fine | -12.107 | 0.431 | 0.719 | 100% | 0.7424 |
| 7 | 5 | strong_guidance | -12.204 | 0.416 | 0.709 | 100% | 0.7422 |
| 8 | 6 | gentle_guidance | -12.236 | 0.411 | 0.693 | 100% | 0.7410 |
| 9 | 3 | fine_stride | -12.148 | 0.413 | 0.720 | 100% | 0.7405 |
| 10 | 8 | more_atoms | -12.148 | 0.413 | 0.720 | 100% | 0.7405 |

### 3.3 关键发现

**1. `pocket_prior` 是最关键的参数改进**

- 相比默认的 `prior_minus_scaffold`，`pocket_prior` 模式使综合评分从 0.7721 提升至 **0.8290**（+7.4%）
- VinaScore 从 -12.848 提升至 **-13.119**（+2.1%）
- QED 从 0.446 提升至 **0.601**（+34.8%）
- SA 从 0.691 提升至 **0.840**（+21.6%）

**机制解释**: `pocket_prior` 模式根据口袋体积分布生成新原子，而非简单地从先验分布中减去骨架原子。这使得生成的侧链更好地适配口袋空间，产生更紧凑、更具药物相似性的分子。

**2. `start_t` 和 `stride` 对结果影响有限**

- `start_t=350`（低噪声）和 `start_t=550`（高噪声）的差异仅 0.01 综合分
- `stride=8`（精细步长）和 `stride=20`（粗步长）的差异也极小
- 这表明 scaffold grow 模式对去噪路径的敏感度较低

**3. 引导强度调整效果不显著**

- `strong_guidance`（lambda 60/10）和 `gentle_guidance`（lambda 25/3）均低于 baseline
- 默认的 lambda 40/5 已经是较优的选择

**4. 所有轮次的分子完整性均为 100%**

- scaffold grow 模式天然生成完整分子（无碎片问题）
- 这验证了骨架约束策略的有效性

### 3.4 最优参数配置

```yaml
sample:
  scaffold:
    enable: true
    mode: grow
    scaffold_source: auto_murcko
    fix_scaffold_pos: true
    fix_scaffold_type: true
    schedule: lambda
    qed_weight: 1.0
    sa_weight: 1.0
    diversity_weight: 0.5
    min_qed: 0.15
    min_sa: 0.15
    filter_incomplete: true
    diversity_filter:
      enable: true
      max_tanimoto: 0.9
    grow:
      num_samples: 50
      start_t: 450
      stride: 15
      step_size: 0.33
      lambda_coeff_a: 40
      lambda_coeff_b: 5
      n_extra_mode: pocket_prior    # 关键改进
      n_extra_fixed: 8
      n_extra_min: 3
      n_extra_max: 20
      min_qed: 0.2
      min_sa: 0.2
      filter_incomplete: false
```

---

## 4. 碎片去除功能

### 4.1 问题背景

在分子生成过程中，某些方法（如 dynamic 模式）可能产生包含多个不相连片段的分子（SMILES 中含 `.`）。这些碎片分子在对接时会产生误导性结果。

### 4.2 解决方案

在 SDF 重建和对接**之前**，自动检测并去除小碎片，仅保留最大连通片段：

```python
def remove_small_fragments(mol):
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) <= 1:
        return mol  # 无需处理
    largest = max(frags, key=lambda m: m.GetNumAtoms())
    return largest
```

### 4.3 使用方式

**CLI 使用**:
```bash
# 评估时去除碎片
python evaluate_pt_with_correct_reconstruct.py result.pt --remove-fragments

# 提取时去除碎片
python extract_pt_to_sdf_excel.py result.pt --remove-fragments
```

**Web UI 使用**: 在任意生成/评估表单中勾选「对接前去除小碎片」复选框。

**API 使用**:
```json
{
  "mode": "dynamic",
  "data_id": 0,
  "batch_size": 50,
  "auto_extract": true,
  "remove_fragments": true
}
```

### 4.4 设计原则

- **默认关闭**: 保持与原有流程的兼容性，不影响公平的能力测试
- **全局开关**: 在生成、优化、骨架、级联、提取等所有流程中均可启用
- **可追溯**: 启用碎片去除的操作会被记录在 job history 中

---

## 5. 文件结构

```
DiffDynamic/
├── seedforge_optimize.py              # SeedForge 优化主脚本
├── scaffold_cascade_pipeline.py       # 骨架级联生成脚本
├── evaluate_pt_with_correct_reconstruct.py  # 评估脚本（含碎片去除）
├── extract_pt_to_sdf_excel.py         # 提取脚本（含碎片去除）
├── SEEDFORGE.md                       # 本文档
├── outputs/
│   └── seedforge/
│       ├── r00_baseline_*/            # 各轮结果
│       ├── r01_low_start_t_*/
│       ├── ...
│       ├── r09_best_rerun_*/
│       └── optimization_report_*.json # 完整优化报告
└── server/
    ├── api.py                         # FastAPI 后端（含 remove_fragments）
    └── jobs.py                        # 任务调度器（含 ADT_PYTHON 修复）
```

---

## 6. 未来方向

1. **跨口袋泛化**: 在更多蛋白口袋上验证 `pocket_prior` 的普适性
2. **自适应参数**: 根据口袋特征（体积、极性、疏水性）自动选择参数
3. **多目标优化**: 引入 Pareto 优化，在 VinaScore、QED、合成可及性之间寻找帕累托前沿
4. **种子级联**: 结合 SeedForge 最优参数与 ScaffoldCascade 多种子策略
5. **在线学习**: 根据前几轮结果动态调整后续轮次的参数探索方向

---

## 7. 引用

如果使用 SeedForge 优化流程，请引用：

```
SeedForge: 基于多参数迭代优化的骨架约束分子生成方法
DiffDynamic 项目, 2026
```
