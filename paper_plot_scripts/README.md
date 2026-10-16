# Paper plot scripts

可复用的分子评估绘图入口，配色为冰蓝 Known / `#F56E1A` Generated。
实现位于 `src/diffdynamic/analysis/plots/`，本目录保留兼容命令入口。

## 文件

- `plot_ridgeline_iceblue_f56e1a.py`：3D 透视山脊图（4×2）。
- `plot_violin_iceblue_f56e1a.py`：核心指标与扩展指标 violin 图。
- `select_n100_valid_and_plot.py`：抽取 n=100 个有效 Vina 分子并绘图。
- `plot_generic_results.py`：t-SNE 与 LogP–MW 图。
- `plot_generic_violin.py`、`plot_generic_violin_expanded.py`：violin 图和数据加载。
- `plot_property_violins.py`：共用绘图样式与 KDE。

## 环境

在项目根目录使用现有科学计算环境，或先完成项目的可编辑安装：

```bash
conda activate diffdynamic
python3 -m pip install --no-deps -e .
```

## 示例

以下输入由调用者提供，仓库不分发实验结果和参考分子集合。

```bash
# 山脊图
python3 paper_plot_scripts/plot_ridgeline_iceblue_f56e1a.py \
  --gen_csv outputs/example/generated_dock_qed_sa_n100.csv \
  --act inputs/reference/active_dock_qed_sa.csv \
  --xlsx outputs/example/evaluation_results_n100_enriched.xlsx \
  --out_dir outputs/example/plots --tag example

# Violin
python3 paper_plot_scripts/plot_violin_iceblue_f56e1a.py \
  --xlsx outputs/example/evaluation_results_n100.xlsx \
  --gen_csv outputs/example/generated_dock_qed_sa_n100.csv \
  --act inputs/reference/active_dock_qed_sa.csv \
  --out_dir outputs/example/plots --tag example

# t-SNE
python3 paper_plot_scripts/plot_generic_results.py \
  --gen_sdf outputs/example/reconstructed_molecules_n100 \
  --gen_csv outputs/example/generated_dock_qed_sa_n100.csv \
  --act_sdf inputs/reference/active_ligands/sdf \
  --out_dir outputs/example/plots --tag example
```
