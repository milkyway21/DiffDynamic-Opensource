# /data/ye 工作区

本目录包含多个独立项目，每个子目录对应一个项目：

| 目录 | 说明 |
|------|------|
| **DiffDynamic/** | 扩散模型 SBDD 框架 + Web 演示平台（前后端） |
| DiffSBDD/ | DiffSBDD 基线 |
| diffgui/ | DiffGUI 相关 |
| e-drug-lab/ | 药物发现实验室 |
| omicos/ | 组学分析 |
| pocket_quality_vis/ | 口袋质量可视化 |
| protein-ligand/ | 蛋白-配体数据 |
| pt/ | 生成结果 (.pt) 存储 |

## DiffDynamic 快速启动

```bash
cd DiffDynamic
bash start_server.sh
```

Web UI: http://localhost:7860/  
API 文档: http://localhost:7860/docs

详细说明见 [DiffDynamic/README.zh-CN.md](DiffDynamic/README.zh-CN.md)
