# Scaffold-Prudent vs Dynamic-Locked 对比报告

生成时间: 2026-06-29 | 最终版 (Full Run, 全量评估)

## 方法

- **Normal (dynamic_locked)**: 骨架位置锁定，一次性 large_step + refine，生成后不迭代筛选
- **Prudent**: 骨架位置锁定 + 迭代重筛选（Gen 0 生成 → QED/SA/Vina 打分 → 选 top-2 幸存者 → forward-diffuse 加噪 → re-refine × 5 代）
- 参数: num_samples=200, n_generations=5, advance_top_k=2, n_chains_per_seed=4, renoise_t=600
- Grow gate: 关闭 (min_qed=0.0, min_sa=0.0)

---

## 最终对比 (Full Run, 全量评估)

### 6W63 (SARS-CoV-2 Mpro)

| 指标 | Normal (n=380) | Prudent (n=219) | 差异 |
|------|:---:|:---:|:---:|
| 平均 Vina | **-8.87** | -8.83 | -0.04 |
| 中位 Vina | **-8.80** | -8.74 | -0.06 |
| 最佳 Vina | **-12.95** | -12.80 | +0.15 |
| Top-10% Vina | **-11.48** | -11.25 | +0.23 |
| 平均 QED | 0.250 | **0.263** | +5.2% ✓ |
| 平均 SA | 0.457 | 0.455 | ≈ |

**结论**: 6W63 上 Prudent ≈ Normal（基本持平），QED 略优。全量评估后差异缩小（Vina mean -8.83 vs -8.87），Prudent 在 6W63 无显著优势。

---

### 8RI2 (NLRP3)

| 指标 | Normal (n=136) | Prudent (n=221) | 提升 |
|------|:---:|:---:|:---:|
| 平均 Vina | -5.77 | **-10.73** | **+86%** ✓✓✓ |
| 中位 Vina | -6.13 | **-11.03** | **+80%** ✓✓✓ |
| 最佳 Vina | -13.95 | **-14.41** | +0.46 ✓ |
| Top-10% Vina | -11.57 | **-13.67** | +2.10 ✓✓ |
| 平均 QED | 0.264 | **0.318** | +20% ✓✓ |
| 平均 SA | 0.369 | **0.439** | +19% ✓✓ |

**结论**: 8RI2 上 Prudent **碾压 Normal**。Vina 均值 -5.77→-10.73 (+86%)，QED +20%，SA +19%。全量评估 (n=221) 验证了 50-sample 的结论，且样本量充足。

---

## 综合结论

1. **Prudent 效果高度依赖靶点**：NLRP3 (8RI2) 上巨大成功（Vina +86%），Mpro (6W63) 上基本持平
2. **Grow gate (QED/SA) 非必要**：关掉门控对 8RI2 无影响，6W63 仅增加样本量
3. **Prudent 迭代筛选对 NLRP3 类靶点最有效**：Gen0→Gen4 的迭代优化显著提升 Vina 亲和力
4. **分子数量**：Prudent 输出 219-221 分子（vs Normal 136-380），但 8RI2 上质量远超 Normal
5. **全量评估价值**：6W63 50-sample (n=47) 时 min=-11.10，全量 (n=219) 时 min=-12.80 — 小样本会漏掉优质分子

---

## 输出文件

### 6W63 Full
- 生成: `6w63prudent_full/result_custom_20260629_194004.pt` (232 分子)
- 评估: `6w63prudent_full/eval_20260629_194538_.../evaluation_results_20260629_194538.xlsx` (220 行)
- CSV: `6w63prudent_full/generated_dock_qed_sa.csv`
- 图表: `6w63prudent_full/plots/` (8 张)

### 8RI2 Full
- 生成: `8ri2prudent_full/result_custom_20260629_194016.pt` (232 分子)
- 评估: `8ri2prudent_full/eval_20260629_194756_.../evaluation_results_20260629_194756.xlsx` (228 行)
- CSV: `8ri2prudent_full/generated_dock_qed_sa.csv`
- 图表: `8ri2prudent_full/plots/` (8 张)

### 对比基准
- 6W63 Normal: `/data/ye/protein-ligand/6W63/generated_dock_qed_sa.csv` (380 分子)
- 8RI2 Normal: `/data/ye/protein-ligand/8RI2/generated_dock_qed_sa.csv` (136 分子)

### 图表清单 (每个蛋白 8 张)
- `*_full_comparison.png` — Prudent vs Normal vs Known Actives 三色 violin
- `*_full_violin.png` — Prudent vs Known Actives 双色 violin
- `*_full_tsne.png` — t-SNE 化学空间分布
- `*_full_logp_vs_molweight.png` — LogP vs MW 类药性空间
