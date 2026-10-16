# GitHub 上传分类与本地整理

## 必须上传

计算库、兼容入口、完整Web/前端资源、公共配置、生产测试、结构修复/药化优化运行库及其YAML规则、安装文件、文档和来源说明。
`publish-manifest.json` 是具体文件清单来源，不能只按扩展名或 Git tracked 状态选择。

## 精选上传

通用绘图与可选 benchmark 演示配置。正式批处理模块、结构修复库和药化优化规则属于必须上传的运行依赖。
外部程序、模型安装和研究结果不随源码发布。

## 本地保留

- `local/archive`：一次性脚本、专用配置及科研测试。
- `local/research`：按课题保留的历史结果树。
- `local/inventory`：源码快照、位置表、迁移/回滚记录和验证证据。
- 数据库、上传、生成结果、缓存、环境、密钥、实际配置和模型二进制默认不上传。

前端分子演示资产及 SA 评分资源是运行资源，按明确清单包含。
前端只选取声明的页面、脚本、样式、供应库与六份演示结构；不递归收集UI目录下的任意文件。
历史消融配置不发布；算法测试使用`tests/fixtures/`中的便携参数，不依赖本地研究目录。

## 权重与导出

权重使用固定 Git 版本下载地址及 SHA256 清单，保持本地原位置。
已有校验不同的文件不会覆盖；不重写 Git 历史，旧克隆历史仍可能包含权重。

```bash
python tools/project.py weights --download
python tools/project.py audit
python tools/project.py export --destination /path/to/new-release-candidate
```

候选目录直接对应 GitHub 仓库根并附带 SOURCE_HASHES.json。
工具可读取远端跟踪快照比较文件，不 fetch、暂存、提交、推送或修改 remote。
发布前审查文件变更，再执行独立 Git 发布操作。

## 本地迁移

```bash
python -m tools.maintenance.cli plan --root /path/to/project --manifest /path/to/new-plan.json
python -m tools.maintenance.cli apply --manifest /path/to/plan.json
python -m tools.maintenance.cli rollback --manifest /path/to/plan.json
```

清单提供完整CSV、简表和可阅读版本，含迁移项及原位保留项。
应用前重查数据库、进程及代码引用；源变化、目标冲突、跨文件系统和发现失败阻止对应项。
回滚也拒绝覆盖后来修改的内容，不删除数据或跟随链接搬运外部目标。
