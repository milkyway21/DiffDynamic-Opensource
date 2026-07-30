# Paper plot scripts

论文 demo 用的绘图脚本（冰蓝 Known / `#F56E1A` Generated），已集中到本目录。

## 文件

| 脚本 | 作用 |
|------|------|
| `plot_ridgeline_iceblue_f56e1a.py` | 3D 透视山脊图（4×2） |
| `plot_violin_iceblue_f56e1a.py` | Violin（core + expanded） |
| `select_n100_valid_and_plot.py` | 抽 n=100 valid Vina 并调 violin |
| `plot_generic_results.py` | t-SNE + LogP–MW |
| `plot_generic_violin.py` | Violin 底层（core 三面板） |
| `plot_generic_violin_expanded.py` | Violin 底层（expanded）+ 数据加载 |
| `plot_property_violins.py` | Violin 样式 / KDE 绘制 |

同目录内互相 import，无需再依赖 `/data/ye/protein-ligand` 路径。

## 环境

```bash
conda activate diffdynamic
cd /data/ye/DiffDynamic
export PYTHONPATH=/data/ye/DiffDynamic:$PYTHONPATH
```

## 示例

```bash
# 山脊图
python3 paper_plot_scripts/plot_ridgeline_iceblue_f56e1a.py \
  --gen_csv experiments/<job>/generated_dock_qed_sa_n100.csv \
  --act /data/ye/protein-ligand/<TARGET>/active_dock_qed_sa.csv \
  --xlsx experiments/<job>/evaluation_results_n100_enriched.xlsx \
  --out_dir experiments/<job>/plots --tag <TAG>

# Violin
python3 paper_plot_scripts/plot_violin_iceblue_f56e1a.py \
  --xlsx experiments/<job>/evaluation_results_n100.xlsx \
  --gen_csv experiments/<job>/generated_dock_qed_sa_n100.csv \
  --act /data/ye/protein-ligand/<TARGET>/active_dock_qed_sa.csv \
  --out_dir experiments/<job>/plots --tag <TAG>

# t-SNE
python3 paper_plot_scripts/plot_generic_results.py \
  --gen_sdf experiments/<job>/reconstructed_molecules_n100 \
  --gen_csv experiments/<job>/generated_dock_qed_sa_n100.csv \
  --act_sdf /data/ye/protein-ligand/<TARGET>/active_ligands/sdf \
  --out_dir experiments/<job>/plots --tag <TAG>
```

原始位置仍保留：`scripts/plot_*.py`、`/data/ye/protein-ligand/plot_*.py`（本目录为拷贝汇总）。
