# DiffDynamic

面向蛋白口袋的三维分子生成与评估平台，提供 baseline、dynamic、Prudent、
骨架约束生成、分子优化、批量生成、口袋分析和 Web 界面。

[English](README.md) · [项目说明](PROJECT.md) · [代理规范](AGENTS.md) · [架构说明](docs/ARCHITECTURE.md) · [药化模块](docs/MEDCHEM.md) · [GitHub 上传分类](docs/PUBLICATION.md)

DiffDynamic 是仅推理框架：仓库不含训练链路，运行
`python tools/project.py weights --download` 获取预训练扩散模型权重。

## 代码放在哪里

常用实现统一在 `src/diffdynamic/`，按 models、data、sampling、chemistry、
evaluation、analysis、batch、io、cli 分类；原入口仅保留兼容适配。
`server/` 管理 API、请求模型、任务配置、调度与执行器，`ui/` 保存前端及资源。

一次性脚本、专用配置和科研测试放在 `local/archive/`，历史研究数据放在
`local/research/`，新旧位置表、回滚清单和验证证据放在 `local/inventory/`。
数据库、上传文件及仍被前端或任务引用的数据保持原位。

## 从零搭建环境

已验证环境为 Python 3.8 + PyTorch 2.4.1（CUDA 11.8），需要 CUDA GPU。
必须先装 torch 栈——图扩展 wheel 必须匹配 PyTorch/CUDA 版本——再装固定版本
依赖和本包：

```bash
conda create -n diffdynamic python=3.8 -y
conda activate diffdynamic

# 1. PyTorch（CUDA 11.8 构建）
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# 2. 图扩展 —— wheel 必须匹配 torch 2.4.1 + cu118
pip install torch_geometric==2.6.1
pip install torch_scatter==2.1.2 torch_sparse==0.6.18 \
    -f https://data.pyg.org/whl/torch-2.4.1+cu118.html

# 3. OpenBabel Python 绑定（重建与对接使用）
conda install -c conda-forge openbabel=3.1.0 -y

# 4. 其余固定版本依赖，然后安装本包
pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .

# 5. 预训练权重（URL 与 SHA256 固定于 weights-manifest.json）
python tools/project.py weights --download
# 可选 GlintDM 对照权重：
python tools/project.py weights --download --include-optional
```

`requirements-validated.txt` 是已验证环境的精确版本记录，不是自动迁移方案。
对接使用 `vina` Python 包；可选 QVina 基准另需外部 `qvina2` 二进制。
下载工具不覆盖校验不符的现有权重文件。

已运行验证环境的机器只需可编辑安装和权重：

```bash
conda activate diffdynamic
python -m pip install --no-deps --no-build-isolation -e .
python tools/project.py weights --download
```

## 数据准备

采样有两种模式，第一种不需要任何数据集：

- **自定义口袋**：`--protein_path pocket.pdb --ligand_path ligand.sdf`，
  装好环境后即可对任意蛋白/配体对运行。仓库自带一对开箱即用的演示结构
  （官网首页素材），可直接首次运行：

  ```bash
  diffdynamic sample configs/sampling.yml \
      --protein_path ui/static/data/landing/protein.pdb \
      --ligand_path ui/static/data/landing/reference-ligand.sdf --device cuda:0
  ```

- **测试集口袋**：`--data_id N` 索引 CrossDocked 测试划分（100 个口袋）。
  checkpoint 内嵌了期望路径，文件必须按下述布局放置：

  ```text
  data/
  ├── crossdocked_pocket10_test_only/                    # 口袋 PDB + 配体 SDF
  │   ├── index.pkl                                      # (pocket, ligand, ...) 条目
  │   └── ..._processed.lmdb                             # 首次使用时自动构建
  └── crossdocked_pocket10_test_only_split.pt            # 子集名 -> data id 列表
  ```

  数据管线沿用上游扩散项目 **TargetDiff**（来源声明见 `THIRD_PARTY_NOTICES.md`）。
  下列下载资源由 TargetDiff / CrossDocked2020 作者提供，遵循其许可条款
  （仅限研究使用），DiffDynamic 仅复用：

  | 资源 | 链接 | 内容 |
  |---|---|---|
  | TargetDiff 仓库 | https://github.com/guanjq/targetdiff | 上游代码及构建 pocket10 数据集的 `scripts/data_preparation/` |
  | TargetDiff 处理好的数据（Google Drive） | https://drive.google.com/drive/folders/1j21cc7-97TedKh_El5E34yI8o5ckI7eK?usp=share_link | `crossdocked_pocket10_pose_split.pt`（train 99,990 / test 100 划分）、`crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb`（完整训练 LMDB）、`test_set.zip`（测试结构）、`split_by_name.pt` |
  | CrossDocked2020 v1.1 原始数据 | https://bits.csb.pitt.edu/files/crossdock2020/ | 原始数据集，仅在需要从头重建时使用 |

  复现上述布局：把 100 个测试口袋（结构见 TargetDiff 的 `test_set.zip`，
  口袋/配体组织方式与其 `extract_pockets.py` 产物一致）放入
  `data/crossdocked_pocket10_test_only/` 并配 `index.pkl`；把其 pose 划分文件
  复制为 `data/crossdocked_pocket10_test_only_split.pt`（对测试专用目录只有
  `test` 子集有效）。DiffDynamic 仅推理：训练 LMDB 与原始 CrossDocked2020
  仅为完整性列出，非必需。LMDB 缓存首次使用时遍历全部索引条目构建，第一次
  跑数据集会明显偏慢。评估时需显式传入同一目录：
  `--protein_root data/crossdocked_pocket10_test_only`。

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
# 首次运行：从示例创建本地配置，再按本机修改
# gpu_ids / default_gpu_id / max_concurrent_jobs
cp server/server_config.example.json server/server_config.json

bash start_server.sh
# 仅供回环地址的本机开发
DD_AUTH_MODE=off DD_HOST=127.0.0.1 bash start_server.sh

python tools/checks.py
node --experimental-vm-modules ui/tests/check-contracts.cjs
python tools/project.py audit
python tools/project.py export --destination /path/to/new-release-candidate
```

认证模式：

- `DD_AUTH_MODE=platform`（默认）把登录委托给生产部署前置的外部平台网关
  （URL 可用 `DD_PLATFORM_SESSION_URL` 覆盖），并隔离每个用户的上传与输出；
  该网关必须可达。
- `DD_AUTH_MODE=off` 仅限单用户本机开发且必须绑定回环地址；访问页面会种下
  内部 cookie，`server/.internal_api_key` 在首次启动时自动生成。

服务器必须单进程运行（内存态调度器、队列上限与限流器无法跨 uvicorn worker
共享）；扩容请改 `gpu_ids` 与 `max_concurrent_jobs`。用户级上限在
`server_config.json` 中：`max_generate_per_day` / `max_eval_per_day`
（0 关闭，管理员豁免）以及上传/队列限制；管理员可通过 `PUT /api/config`
运行时调整。单用户本机环境可将日配额设为 0，或启动时加 `DD_ADMIN_USERS=web_ui`。

默认地址为 `127.0.0.1:7860`，保留中英文代理前缀。
实际配置 `server/server_config.json` 不上传，示例见同目录的 example 文件。
显式的 DD 路径环境变量优先于保存的路径。

保留生成、投放、优化、评估、分子下载、排队、取消、历史重跑、权限和配额。
测试使用临时数据库与输出目录；发布工具默认只预览，不自动暂存、提交或推送。
历史脚本仅归档并记录位置，不再维护旧绝对路径。
来源与许可见 `LICENSE`、源文件头及 `THIRD_PARTY_NOTICES.md`。
