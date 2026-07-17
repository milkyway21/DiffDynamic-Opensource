# DiffDynamic 口袋质量评估（Pocket Quality）

面向协作者与审稿人的说明：本评分体系测什么、生物化学上是否合理、以及与当前代码（[`evaluate_pocket_quality.py`](../evaluate_pocket_quality.py)）的一一对应关系。

**实现入口**：`evaluate_pocket_quality.py`  
**Web API**：`POST /api/pocket-eval`（与分子级 `POST /api/evaluate` 不是同一套分数）  
**校准数据**：CrossDocked 测试集 `jsdpt3010`（data_id 0–99），见 `pocket_quality_vis/jsdpt3010_run5_f95_vis/`

---

## 1. 定义与边界

### 1.1 本评分是什么

八维加权综合分（各维 ∈ [0, 1]，overall ∈ [0, 1]），用**扩散模型在该口袋上已生成并（通常）已对接的分子集合**，间接评估「该口袋是否适合产出可结合、偏药物样的分子」——即**生成结果驱动的口袋可药性综合分**。

### 1.2 本评分不是什么

| 不是 | 说明 |
|------|------|
| 纯物理口袋评分 | 不单独依赖几何空洞、静电势等；默认体积维（H）常由生成分子 MC 占据估计 |
| 实验亲和力 | Vina 分数是 docking 打分函数，不是实测 ΔG |
| 分子级综合分 | `evaluate_pt_with_correct_reconstruct.py` 的 0–100 分是另一套 |
| Prudent 筛选分 | 采样循环内 YAML 权重综合分是第三套 |

**论文/演示表述建议**：写清「基于生成+对接结果的口袋可药性代理指标」，避免写成「与生成无关的口袋物理质量」。

### 1.3 输入依赖

```
.pt（pred_ligand_pos / pred_ligand_v）──重建──► 分子 3D ──► B, D, E, F, G, H(MC)
                                                    │
eval_*/complete_molecules_*.xlsx（或 eval_results_*.pt）──► A Vina ──► C LE
可选 --fpocket_protein_pdb ──► H（FPocket 配体/蛋白体积）
```

- **想法 A / C** 需要已有 Vina 评估输出；同目录下 `eval_*_{id}_*` 与 `.pt` 的 `data_id` 对齐。  
- **想法 E** 可用 `--idea_e_expected_n_molecules N` 指定应生成数作分母（推荐与实际采样规模一致，如 jsdpt3010 用 100）。

---

## 2. 八维：生物化学含义与实现

每维最终映射到 **[0, 1]**。失败维（`success=False`）不参与加权，其余权重重归一化。

### A — Vina 对接（`score_a` / `vina_docking`）

| | |
|--|--|
| **测什么** | 生成分子与口袋的对接亲和力分布（kcal/mol，越负越好） |
| **为何合理** | SBDD 中 docking 分数是常用可药性/结合代理；均值看整体，最佳看「能否出强 binder」，命中率看「可药性命中密度」 |
| **局限** | Vina ≠ 实验 ΔG；依赖对接协议与蛋白准备；正分常表示失败 |

**实现**（过滤 `vina < 0` 后）：

\[
\begin{align*}
\text{mean} &= \mathrm{clip}\big((- \bar v - 6)/6,\,0,\,1\big) \\
\text{best} &= \mathrm{clip}\big((- v_{\min} - 6)/6,\,0,\,1\big) \\
\text{hit} &= \mathrm{mean}(v \le -7) \\
\text{score}_A &= 0.5\cdot\text{mean} + 0.3\cdot\text{best} + 0.2\cdot\text{hit}
\end{align*}
\]

含义：均值约 −6 → 0 分、−12 → 满分；命中阈值 −7 kcal/mol。  
标签：≥0.5 high，≥0.2 medium。  
返回的 `vina_scores` 保留与分子下标对齐的完整列表（含失败正值），供 C 自行跳过。

### B — 原子分布聚类 / 结合模式收敛（`score_b` / `clustering`）

| | |
|--|--|
| **测什么** | 多分子 3D 原子坐标的 DBSCAN 簇数与紧凑度 →「结合模式是否收敛」 |
| **为何合理** | 可药口袋常约束配体姿态；少而紧的簇暗示模式一致（文献直觉上的「约束性」代理） |
| **局限** | 混有**模型采样行为**；簇少也可能是 mode collapse。默认用 `hetero_heavy` 减弱碳骨架主导 |

**实现要点**：

- DBSCAN：`eps=1.5` Å，`min_samples=3`；簇数越少基础分越高（≤1 → 1.0；≤5 线性降；>5 继续降）。  
- 轮廓系数 / 紧凑度做乘性修正。  
- **默认** `atom_coord_subset=hetero_heavy`（非 H、非 C 重原子）。  
- KMeans 仅用于可视化质心，**不进分数**。

### C — 配体效率 LE（`score_c` / `ligand_efficiency`）

| | |
|--|--|
| **测什么** | \(LE = -\Delta G / N_{\mathrm{heavy}}\)（kcal·mol⁻¹·重原子⁻¹），ΔG 用 Vina 亲和力近似 |
| **为何合理** | Hopkins 等强调效率而非绝对亲和力；药物样分子典型 LE 约 0.2–0.4 |
| **局限** | 依赖 A 的下标对齐；大分子、弱对接会压低 LE |

**实现**：

- 仅当 `vina < 0`、分子非空、`N_heavy > 0` 时计入（失败对接跳过，避免负 LE）。  
- 映射：LE 均值从 **0.18 → 0** 分，到 **0.45 → 1** 分（线性 clip）。  
- 标签：≥0.6 / ≥0.3。

### D — 药物相似性（`score_d` / `drug_likeness`）

| | |
|--|--|
| **测什么** | QED、SA（易合成，越高越好）、Lipinski 合规比例、Veber 通过率、PAINS 命中率 |
| **为何合理** | 成药性筛选的标准组合；PAINS 控制假阳性风险 |
| **局限** | 本仓库 Lipinski 自定义为 5 条（含可旋转键），与 Veber（可旋转键 + TPSA）有轻度重叠 |

**实现加权**（和=1）：

\[
\begin{align*}
\text{score}_D &= 0.35\,\overline{\mathrm{QED}} + 0.25\,\overline{\mathrm{SA}} \\
&\quad + 0.15\,\overline{\mathrm{Lip}/5} + 0.10\,\mathrm{Veber率} + 0.15\,(1-\mathrm{PAINS率})
\end{align*}
\]

标签：≥0.6 / ≥0.3。

### E — 完整性（`score_e` / `completeness`）

| | |
|--|--|
| **测什么** | SMILES **不含** `.` 的单组分完整分子数 / 分母 |
| **为何合理** | 作为**数据质量 / 生成成功率**门控；完整分子太少则下游不可信 |
| **局限** | **不是**口袋固有物理属性；分母选错会扭曲分数 |

**分母**：`--idea_e_expected_n_molecules N` 优先；否则 `.pt` 的 `len(pred_ligand_pos)`；无 .pt 且无 N 则跳过（不参与加权）。

**分段映射**（对 clip 到 [0,1] 的比率）：

| rate | score |
|------|-------|
| ≥0.95 | 1.0 |
| [0.80, 0.95) | 0.6 → 1.0 线性 |
| [0.60, 0.80) | 0.3 → 0.6 线性 |
| <0.60 | 线性压到 [0, 0.3] |

概念上属 **generation_quality**；当前返回值**未**单独写 `role` 字段。

### F — 唯一性（`score_f` / `diversity`）

| | |
|--|--|
| **测什么** | `unique_ratio = n_unique_SMILES / n_complete` |
| **为何合理** | 过低 → mode collapse；约 100 个生成分子中至少约 95 个不同才给满分，提高区分度 |
| **局限** | 唯一性 ≠ 化学多样性；指纹项不进总分 |

**实现（满分门槛 unique_full_at = 0.95）**：

| unique_ratio | score |
|--------------|-------|
| ≥0.95 | 1.0 |
| [0.70, 0.95) | 0.70 → 1.0 线性 |
| [0.40, 0.70) | 0.30 → 0.70 线性 |
| <0.40 | → [0, 0.30] |

参考指标：`fingerprint_dissimilarity`（Morgan 指纹**余弦**相异度；历史别名 `tanimoto_diversity`，**不是**真 Tanimoto，且**不进总分**）。

### G — 分子量一致性（`score_g` / `size_consistency`）

| | |
|--|--|
| **测什么** | \(CV_{MW} = \sigma(MW)/\mu(MW)\) |
| **为何合理** | 口袋可约束配体尺寸；CV 低表示尺寸稳定 |
| **局限** | 也可反映 mode collapse；属生成质量相关维 |

| \(CV_{MW}\) | score |
|-------------|-------|
| <0.15 | 1.0 |
| <0.25 | 0.8 |
| <0.4 | 0.5 |
| ≥0.4 | 继续下降 |

DB 指标名：`size_consistency`（勿与旧误名 `interaction` 混淆）。

### H — 口袋体积（`score_h` / `pocket_volume`）

| | |
|--|--|
| **测什么** | 口袋有效体积是否落在药物样区间 |
| **为何合理** | 过小难容纳药物样配体，过大结合熵/特异性差；文献常以数百 Å³ 为药物样口袋量级 |
| **局限** | 默认 **MC**：质心均值周围 **12 Å** 球内配体 vdW 占据 → 体积估计依赖生成分子，存在**循环论证**。有蛋白时应用 **FPocket** |

**默认 MC 满分带**：**300–800** Å³（归零尾约 80 / 2000）。  
配体 FPocket 成功时用同一 300–800 带；仅蛋白回退时用宽区间 300–2200（零尾 80 / 4500）。  
标签：≥0.8 high，≥0.5 medium。  
DB 指标名：`pocket_volume`（勿与旧误名 `stability` 混淆）。

---

## 3. 综合分与校准

### 3.1 聚合

对所有 `success=True` 且权重 >0 的维：

\[
\mathrm{overall} = \frac{\sum_i w_i s_i}{\sum_i w_i}
\]

### 3.2 默认权重（和 = 1.00）

| A | B | C | D | E | F | G | H |
|---|---|---|---|---|---|---|---|
| 0.28 | 0.15 | 0.15 | 0.15 | 0.05 | 0.08 | 0.07 | 0.07 |

依据：

- **A** 与 overall 相关最强，保持较高权。  
- **E** 易饱和（完整率高时常满分），降权。  
- **F** 以 ≥95% unique 为满分，可区分，权重 0.08。  
- 区分力主要来自 **A / C / F**（以及未饱和时的 B/D/H）。

### 3.3 Overall 标签

| overall | label |
|---------|-------|
| ≥ **0.65** | high |
| ≥ 0.3 | medium |
| < 0.3 | low |

### 3.4 各维标签阈值（实现如此，非 bug）

| 维 | high | medium |
|----|------|--------|
| A | ≥0.5 | ≥0.2 |
| H | ≥0.8 | ≥0.5 |
| 其余 | ≥0.6 | ≥0.3 |

---

## 4. 实现一致性检查表

对照日期：与仓库内当前 `evaluate_pocket_quality.py` / `server/job_helpers.py` 一致。

| 声明 | 代码核对 | 状态 |
|------|----------|------|
| LE 跳过失败对接（vina≥0） | `evaluate_idea_c_ligand_efficiency` | OK |
| B 默认 `hetero_heavy` | 函数默认、CLI、`evaluate_pocket_quality` 一致 | OK |
| F：`unique_ratio ≥ 0.95` → 1.0（分段见上） | `evaluate_idea_f_uniqueness` | OK |
| H：12 Å、满分 300–800 | `_evaluate_idea_h_mc_only`；模块 docstring | OK |
| 权重和 1.0（A0.28…F0.08…） | 函数默认与 CLI `--weight_*` | OK |
| overall high ≥ 0.65 | `evaluate_pocket_quality` 聚合处 | OK |
| CSV 含 `vis_dir` | `EVAL_RECORD_FIELDS` / `append_evaluation_record` | OK |
| DB：G=`size_consistency`，H=`pocket_volume` | `store_pocket_evaluations` | OK |
| D：QED35/SA25/Lip15/Veber10/PAINS15 | `evaluate_idea_d_druglikeness` | OK |
| A：0.5 mean + 0.3 best + 0.2 hit | `evaluate_idea_a_vina` | OK |
| 指纹多样性不进总分；余弦非 Tanimoto | `fingerprint_dissimilarity` | OK |
| E/G 为生成质量相关维 | 科学归类成立；返回值**无** `role` 字段 | 文档按代码表述 |
| 雷达标签 F Unique；参考带 0.65/0.35 | `visualize_radar_summary` | OK |
| B 主图为示意 2D（非打分路径） | `visualize_clustering_2d` 标题注明 | OK |

---

## 5. 与其他评分体系

| 体系 | 入口 | 量纲 | 用途 |
|------|------|------|------|
| 口袋 8 维 | `evaluate_pocket_quality.py` / `/api/pocket-eval` | 0–1 | 口袋/生成批次可药性报告 |
| 分子综合分 | `evaluate_pt_with_correct_reconstruct.py` / `/api/evaluate` | 0–100 | 单分子排序与 Excel |
| Prudent 综合分 | `sample_diffusion.py` + `sampling.yml` | 内部筛选 | 多轮生成门控 |

---

## 6. 用法速查

```bash
# 单个 .pt（Vina 输出与 .pt 同目录时可省略 --vina_outputs_dir）
python3 evaluate_pocket_quality.py \
  --pt_file jsdpt3010/result_0_*.pt \
  --vina_outputs_dir jsdpt3010 \
  --idea_e_expected_n_molecules 100 \
  --visualize

# 目录内 data_id 批量（不重新生成）
python3 evaluate_pocket_quality.py \
  --pt_dir jsdpt3010 --start 0 --end 99 \
  --vina_outputs_dir jsdpt3010 \
  --idea_e_expected_n_molecules 100 \
  --num_cpu_cores 40 --cores_per_task 2 \
  --vis_dir pocket_quality_vis/jsdpt3010_run5_f95

# 想法 H 启用 FPocket（推荐有蛋白结构时）
python3 evaluate_pocket_quality.py \
  --pt_file path/to/result.pt \
  --fpocket_protein_pdb path/to/receptor.pdb \
  --visualize
```

说明：

- `--idea_e_expected_n_molecules` 应等于该次生成规模（如每口袋 100），**不是**冒烟用的 `batch_size=5`。  
- 记录追加到可视化根目录下的 `evaluation_records.csv`（含 `pt_path`、`vis_dir`）。  
- Web：`POST /api/pocket-eval`；结果图在 `pocket_quality_vis/`。  
- **可视化 vs 打分**：PNG 消费 `idea_*` 分数，不另算一套；B 聚类主图是 2D 示意（全原子、eps≈0.8），打分用 3D `hetero_heavy` eps=1.5。

---

## 7. 参考文献与概念锚点（非穷尽）

- Ligand efficiency：Hopkins et al., 配体效率作为优化度量的常用讨论。  
- QED：Bickerton et al., 定量估计药物相似性。  
- Lipinski Ro5 / Veber：口服药物样经验规则。  
- PAINS：Baell & Holloway，泛干扰化合物警示结构。  
- 口袋体积与可药性：药物样口袋体积量级的经验区间（实现取 300–800 Å³ 为满分带）。

具体阈值与权重以本仓库代码与 `jsdpt3010` 经验校准为准，而非单一文献公式。
