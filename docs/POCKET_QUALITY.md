# DiffDynamic 口袋质量评估（三层八维）

**实现**：[`evaluate_pocket_quality.py`](../evaluate_pocket_quality.py)  
**Web**：`POST /api/pocket-eval`

## 公式

层内加权几何平均（失败维剔除后重归一，\(\varepsilon=10^{-3}\)）：

- \(S_{\mathrm{ligand}}=\mathrm{wgeo}(A,B,C;\ w=0.45,0.25,0.30)\)
- \(S_{\mathrm{compatibility}}=\mathrm{wgeo}(D,E;\ w=0.45,0.55)\)
- \(S_{\mathrm{pocket}}=\mathrm{wgeo}(F,G,H;\ w=0.20,0.55,0.25)\)

层间等权立方根乘积：

\[
S_{\mathrm{overall}}=(S_p\cdot S_c\cdot S_l)^{1/3}
\]

| overall | label |
|---------|-------|
| ≥ 0.70 | high |
| ≥ 0.45 | medium |
| &lt; 0.45 | low |

## 八维

| 槽 | 含义 | 层 |
|----|------|----|
| A | Affinity + LE（0.55 Vina + 0.45 LE） | ligand |
| B | Binding-mode clustering | ligand |
| C | Druglikeness（PAINS&gt;5% ×0.85） | ligand |
| D | Uniqueness（≥95% 满分） | compatibility |
| E | Completeness（≥98% 满分） | compatibility |
| F | Pocket chemistry（配体邻域疏水/极性/封闭度） | pocket |
| G | P4 Anchor richness（0.35/0.35/0.30） | pocket |
| H | Geometry volume（满分 350–750 Å³） | pocket |

旧「尺寸 CV」**不进总分**（可作辅图）。

## F / G / H 简述

- **F**：配体邻域（默认 **10 Å**）残基疏水/极性比例与封闭度（生物学代理）；**不调用 FPocket**。
- **G**：口袋残基供受体/离子/芳香锚点的丰富度、空间覆盖、类别平衡；需蛋白 PDB + 配体质心。
- **H**：体积；**仅 MC**（配体质心默认 **10 Å** 球内配体占据，满分 350–750 Å³）；不用 FPocket Volume。

## CSV

`evaluation_records.csv` 含 `score_a..h`、`s_pocket`、`s_compatibility`、`s_ligand`、`overall_score`。
