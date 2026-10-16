# DiffDynamic

面向蛋白口袋的三维分子生成与评估平台，提供 baseline、dynamic、Prudent、
骨架约束生成、分子优化、批量生成、口袋分析和 Web 界面。

[架构说明](docs/ARCHITECTURE.md) · [药化模块](docs/MEDCHEM.md) · [GitHub 上传分类](docs/PUBLICATION.md)

## 代码放在哪里

常用实现统一在 `src/diffdynamic/`，按 models、data、sampling、chemistry、
evaluation、analysis、batch、io、cli 分类；原入口仅保留兼容适配。
`server/` 管理 API、请求模型、任务配置、调度与执行器，`ui/` 保存前端及资源。

一次性脚本、专用配置和科研测试放在 `local/archive/`，历史研究数据放在
`local/research/`，新旧位置表、回滚清单和验证证据放在 `local/inventory/`。
数据库、上传文件及仍被前端或任务引用的数据保持原位。

## 安装与权重

沿用现有 conda 环境及 Python 3.8，本轮不升级科学计算依赖。
已验证版本记录在 `requirements-validated.txt`，不会自动修改现用环境。

```bash
conda activate diffdynamic
python -m pip install --no-deps --no-build-isolation -e .
python tools/project.py weights --download
```

默认权重位置仍为 `pretrained_models/pretrained_diffusion.pt`。
下载工具使用固定版本地址并检查 SHA256，不覆盖校验不同的现有模型。
需要 GlintDM 对照权重时添加 `--include-optional`。
新机器需先准备科学计算依赖、匹配的 PyTorch 图扩展、OpenBabel 和 AutoDock Vina。

## 常用命令

```bash
# 测试集或自定义口袋生成
diffdynamic sample configs/sampling.yml --data_id 0 --device cuda:0
diffdynamic sample configs/sampling.yml --protein_path pocket.pdb --ligand_path ligand.sdf --device cuda:0
# 批量生成与评估
diffdynamic batch --start 0 --end 9 --gpus 0,1
# 评估、提取与其他接口
diffdynamic evaluate outputs/result.pt --protein_root data/crossdocked_pocket10_test_only
diffdynamic extract outputs/result.pt
diffdynamic pocket --help
diffdynamic cascade --help
diffdynamic plot --help
```

旧命令 `python scripts/sample_diffusion.py`、
`python evaluate_pt_with_correct_reconstruct.py` 等继续可用。
原参数、默认值、进度日志、PT 字段、SDF 属性及评估表格式保持兼容。
跨模型 benchmark 需另外准备相应模型安装。

## Web 与检查

```bash
bash start_server.sh
# 仅供回环地址的本机开发
DD_AUTH_MODE=off DD_HOST=127.0.0.1 bash start_server.sh

python tools/checks.py
node ui/tests/check-contracts.cjs
python tools/project.py audit
python tools/project.py export --destination /path/to/new-release-candidate
```

默认地址为 `127.0.0.1:7860`，保留平台认证及中英文代理前缀。
实际配置 `server/server_config.json` 不上传，示例见同目录的 example 文件。
显式的 DD 路径环境变量优先于保存的路径。

保留生成、投放、优化、评估、分子下载、排队、取消、历史重跑、权限和配额。
测试使用临时数据库与输出目录；发布工具默认只预览，不自动暂存、提交或推送。
历史脚本仅归档并记录位置，不再维护旧绝对路径。
来源与许可见 `LICENSE`、源文件头及 `THIRD_PARTY_NOTICES.md`。
