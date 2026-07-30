# DiffDynamic

**DiffDynamic** 是面向 3D 结构化药物设计（SBDD）的扩散模型推理框架：在 [TargetDiff](https://github.com/DeepGraphLearning/TargetDiff) 骨干上，提供两阶段动态跳步采样、梯度融合、Prudent 多轮过滤与骨架约束生成，并附带 Web 演示界面。

- 技术原理（英文）：[README.en.md](README.en.md)
- 中文补充：[README.zh-CN.md](README.zh-CN.md)
- 许可：[LICENSE](LICENSE)

## 仓库内容

```
DiffDynamic-Opensource/
├── README.md                 # 本使用引导（仓库主页）
├── README.en.md              # 英文技术说明
├── README.zh-CN.md           # 中文说明
├── pretrained_models/        # 预训练权重（已入库）
├── configs/sampling.yml      # 采样主配置
├── scripts/sample_diffusion.py
├── evaluate_pt_with_correct_reconstruct.py
├── evaluate_pocket_quality.py
├── extract_pt_to_sdf_excel.py
├── server/ + ui/             # FastAPI + Web 前端
└── start_server.sh
```

## 环境准备

- Python 3.8+
- Conda 环境建议命名为 `diffdynamic`（PyTorch 1.12+、CUDA、RDKit、AutoDock Vina）
- NVIDIA GPU（生成与对接评估）

```bash
git clone https://github.com/milkyway21/DiffDynamic-Opensource.git
cd DiffDynamic-Opensource

conda activate diffdynamic
pip install -r requirements.txt
# Web 服务额外依赖（可选）
pip install -r requirements-web.txt

export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

## 预训练权重

Clone 后权重已在 `pretrained_models/`：

| 文件 | 用途 |
|------|------|
| `pretrained_models/pretrained_diffusion.pt` | 默认采样检查点（必需） |
| `pretrained_models/pretrained_GlintDM.pt` | 相关对照权重 |

`configs/sampling.yml` 中默认指向：

```yaml
model:
  checkpoint: ./pretrained_models/pretrained_diffusion.pt
```

无需再单独下载权重即可开始采样。

## 快速开始

### 1. 单口袋采样（测试集 data_id）

```bash
python scripts/sample_diffusion.py configs/sampling.yml \
    --data_id 0 --device cuda:0
```

自定义蛋白 / 参考配体：

```bash
python scripts/sample_diffusion.py configs/sampling.yml \
    --protein_path /path/to/protein.pdb \
    --ligand_path /path/to/reference.sdf \
    --device cuda:0
```

结果默认写入 `outputs/`（运行时生成，不在仓库中）。

### 2. 评估与提取

```bash
# 重建 + Vina / QED / SA 等评估
python evaluate_pt_with_correct_reconstruct.py \
    outputs/result_0_*.pt \
    --protein_root ./data/crossdocked_v1.1_rmsd1.0_pocket10

# .pt → SDF / Excel
python extract_pt_to_sdf_excel.py outputs/result_0_*.pt
```

口袋质量评估：

```bash
python evaluate_pocket_quality.py --help
```

> CrossDocked 测试口袋数据（`data/`）需自行准备；仓库不包含该数据集。

### 3. Web 演示界面

```bash
bash start_server.sh
```

- 界面：http://localhost:7860/
- API 文档：http://localhost:7860/docs

## 配置入口

所有生成行为以 [`configs/sampling.yml`](configs/sampling.yml) 为准：

| 字段 | 说明 |
|------|------|
| `sample.mode` | `baseline` / `dynamic` / `prudent` / `optimization` |
| `sample.dynamic.time_boundary` | 大步跳跃与精修分界（默认约 650） |
| `model.use_grad_fusion` | 是否启用梯度融合 |
| `sample.scaffold` | 骨架约束（grow / evolve） |

更多说明见 [README.en.md](README.en.md)、[SEEDFORGE.md](SEEDFORGE.md)。

## 数据说明：仓库内 vs 自备

| 内容 | 是否入库 | 说明 |
|------|----------|------|
| 源码、配置、脚本、Web | 是 | 本仓库主体 |
| `pretrained_models/*.pt` | 是 | 预训练权重 ~64MB |
| `data/`（CrossDocked 等） | 否 | 需自行准备测试口袋 |
| `outputs/`、`experiments/` | 否 | 运行时生成 |

## 致谢

本项目基于 [TargetDiff](https://github.com/DeepGraphLearning/TargetDiff)（Guan et al., ICLR 2023）构建。感谢其在扩散模型结构化药物设计方面的奠基性工作。
