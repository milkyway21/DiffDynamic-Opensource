// ── i18n ────────────────────────────────────────────────────────────────────

let currentLang = localStorage.getItem('dd_lang') || (
  (typeof window !== 'undefined' && /\/en(\/|$)/.test(window.location.pathname)) ? 'en' : 'zh'
);
if (!['zh', 'en'].includes(currentLang)) currentLang = 'zh';
const localizedContent = new Map();
const idleTranslations = new WeakMap();

const T = {
  'nav.project':        { zh: '项目情况', en: 'Overview' },
  'nav.generate':       { zh: '分子生成', en: 'Generate' },
  'nav.evaluate':       { zh: '评估提取', en: 'Evaluate' },
  'nav.molecules':      { zh: '分子数据库', en: 'Molecules' },
  'nav.history':        { zh: '历史记录', en: 'History' },
  'nav.config':         { zh: '系统配置', en: 'Config' },
  'nav.pocket_eval':    { zh: '口袋评估', en: 'Pocket Eval' },

  'hero.desc':          { zh: '基于扩散模型的 3D 结构化药物设计框架。在 TargetDiff 基础上引入动态跳步采样、梯度融合、Prudent 多轮过滤等推理阶段优化技术，显著提升生成分子质量与多样性。', en: 'A diffusion-based 3D structure-based drug design framework. Built on TargetDiff, it introduces dynamic skip-step sampling, gradient fusion, and Prudent multi-round filtering to significantly improve molecular quality and diversity.' },
  'hero.eyebrow':       { zh: '论文演示平台', en: 'Paper demo platform' },
  'hero.cta':           { zh: '开始生成', en: 'Start generating' },
  'hero.live_running':  { zh: '运行中', en: 'Running' },
  'hero.live_queued':   { zh: '排队', en: 'Queued' },
  'hero.live_mols':     { zh: '分子', en: 'Molecules' },
  'hero.live_done':     { zh: '已完成', en: 'Completed' },

  'gen.title':          { zh: '测试集生成', en: 'Test Set Generation' },
  'gen.unified_title':  { zh: '从头生成分子', en: 'De Novo Generation' },
  'gen.unified_hint':   { zh: '选择生成模式后，界面会自动切换对应的参数。', en: 'Pick a generation mode and the form adjusts automatically.' },
  'gen.source':         { zh: '生成模式', en: 'Generation mode' },
  'gen.src_custom':     { zh: '自定义口袋（上传蛋白 PDB）', en: 'Custom pocket (upload protein PDB)' },
  'gen.src_testset':    { zh: '测试集口袋（data_id）', en: 'Test-set pocket (data_id)' },
  'gen.src_batch':      { zh: '批量范围（多个 data_id）', en: 'Batch range (multiple data_id)' },
  'gen.sampling_mode':  { zh: '采样模式', en: 'Sampling mode' },
  'gen.mode':           { zh: '生成模式', en: 'Generation Mode' },
  'gen.mode_dynamic':   { zh: 'Dynamic (快速)', en: 'Dynamic (Fast)' },
  'gen.mode_prudent':   { zh: 'Prudent (多轮优化)', en: 'Prudent (Multi-round)' },
  'gen.dataid':         { zh: 'data_id（测试集索引）', en: 'data_id (Test set index)' },
  'gen.gpu_ph':         { zh: '留空自动分配', en: 'Auto-assign if empty' },
  'gen.batch':          { zh: 'batch_size（每批分子数）', en: 'batch_size (molecules per batch)' },
  'gen.batch_ph':       { zh: '默认 5', en: 'Default 5' },
  'gen.max_samples':    { zh: '评估分子数上限 (max_samples)', en: 'Eval max samples' },
  'gen.vina_timeout':   { zh: 'Vina 超时 (秒)', en: 'Vina timeout (sec)' },
  'gen.prudent_title':  { zh: 'Prudent 参数（10 选 2）', en: 'Prudent params (pick 2 of 10)' },
  'gen.prudent_n_sampling': { zh: 'n_sampling（每种子候选链数）', en: 'n_sampling (chains per seed)' },
  'gen.prudent_advance_topk': { zh: 'advance_top_k（晋级数）', en: 'advance_top_k (survivors)' },
  'gen.prudent_refine_batch': { zh: 'refine_batch_size', en: 'refine_batch_size' },
  'gen.prudent_workers': { zh: 'scoring_workers（打分线程）', en: 'scoring_workers' },
  'gen.config':         { zh: '配置文件路径', en: 'Config file path' },
  'gen.config_ph':      { zh: '留空使用默认 sampling.yml', en: 'Leave empty for default sampling.yml' },
  'gen.auto':           { zh: '生成后自动操作', en: 'Auto actions after generation' },
  'gen.auto_eval':      { zh: '自动评估主结果（化学指标与结合评分）', en: 'Evaluate primary results (chemistry and binding scores)' },
  'gen.auto_extract':   { zh: '自动提取（.pt → SDF + Excel）', en: 'Auto extract (.pt → SDF + Excel)' },
  'gen.auto_eval_short':{ zh: '自动评估', en: 'Auto evaluate' },
  'gen.auto_extract_short':{ zh: '自动提取', en: 'Auto extract' },
  'gen.remove_fragments':{ zh: '对接前去除小碎片（仅保留最大片段）', en: 'Remove fragments before docking (keep largest)' },
  'eval.remove_fragments':{ zh: '去除小碎片（仅保留最大片段）', en: 'Remove fragments (keep largest)' },
  'gen.start':          { zh: '开始生成', en: 'Start Generation' },
  'gen.cancel':         { zh: '取消任务', en: 'Cancel task' },
  'quota.header':       { zh: '今日剩余 生成 {gen} · 评估 {eval}', en: 'Today left: gen {gen} · eval {eval}' },
  'quota.gen_left':     { zh: '今日生成/骨架剩余 {n} 次', en: 'Generation/scaffold left today: {n}' },
  'quota.eval_left':    { zh: '今日评估剩余 {n} 次', en: 'Evaluation left today: {n}' },
  'quota.gen_empty':    { zh: '今日生成次数已用完，北京时间 0:00 重置', en: 'Daily generation used up. Resets at 00:00 CST.' },
  'quota.eval_empty':   { zh: '今日评估次数已用完，北京时间 0:00 重置', en: 'Daily evaluation used up. Resets at 00:00 CST.' },
  'quota.unlimited':    { zh: '管理员不受每日次数限制', en: 'Admins have no daily cap' },
  'mol.dl_fail':        { zh: '文件不存在或无权下载', en: 'File missing or access denied' },
  'mol.need_run':       { zh: '请先填写 Run ID', en: 'Enter a Run ID first' },
  'gen.extracting':     { zh: '生成完成，正在提取分子…', en: 'Generation done, extracting molecules…' },
  'gen.ready_dl':       { zh: '可下载', en: 'Ready to download' },
  'gen.open_mols':      { zh: '打开分子库', en: 'Open molecule table' },
  'gen.extract_wait':   { zh: '提取超时，请稍后在分子库查看', en: 'Extract timed out; check Molecules later' },
  'mol.dl_csv':         { zh: '下载 CSV', en: 'Download CSV' },
  'mol.dl_xlsx':        { zh: '下载 Excel', en: 'Download Excel' },
  'mol.dl_zip':         { zh: '下载 SDF 包', en: 'Download SDF zip' },
  'mol.redock':         { zh: '重新Dock', en: 'Redock' },
  'mol.th_redock':      { zh: '重新Dock', en: 'Redock' },
  'mol.redock_hint':    { zh: 'Vina 全局对接搜索（慢），补齐精确对接姿态与打分', en: 'Full Vina docking search (slow); precise poses & scores' },
  'mol.redock_confirm': { zh: '重新 Dock 为全局搜索，单分子可达数十秒，确认继续？', en: 'Redocking runs full global search (tens of seconds per molecule). Continue?' },
  'mol.redock_started': { zh: '已提交重新 Dock 任务 {job}，完成后可在分子库查看新分数', en: 'Redock job {job} submitted; check Molecules for updated scores' },
  'mol.redock_no_pt':   { zh: '该 run 没有 .pt 输出，无法重新对接', en: 'No .pt output for this run; cannot redock' },
  'mol.redock_failed':  { zh: '提交重新 Dock 失败', en: 'Failed to submit redock' },
  'hist.dl_xlsx':       { zh: 'Excel', en: 'Excel' },
  'hist.dl_csv':        { zh: 'CSV', en: 'CSV' },
  'hist.dl_zip':        { zh: 'SDF包', en: 'SDF zip' },
  'hist.dl_pt':         { zh: '下载PT', en: 'PT' },
  'gen.progress':       { zh: '任务进度', en: 'Task Progress' },
  'gen.idle':           { zh: '空闲 — 尚未启动任务', en: 'Idle — No task started' },
  'gen.custom_title':   { zh: '自定义口袋生成', en: 'Custom Pocket Generation' },
  'gen.custom_hint':    { zh: '上传蛋白 PDB 文件和可选的参考配体 SDF 文件，直接生成分子。', en: 'Upload protein PDB and optional reference ligand SDF to generate molecules.' },
  'gen.protein':        { zh: '蛋白 PDB 文件路径', en: 'Protein PDB file path' },
  'gen.ligand':         { zh: '参考配体 SDF（可选）', en: 'Reference ligand SDF (optional)' },
  'gen.gpu_auto':       { zh: '自动分配', en: 'Auto-assign' },
  'gen.radius':         { zh: '口袋半径 (Å)', en: 'Pocket radius (Å)' },
  'gen.samples':        { zh: '采样数量', en: 'Num samples' },
  'gen.structure_title': { zh: '配体分子与口袋预览', en: 'Ligand and pocket preview' },
  'gen.load_ligand': { zh: '预览配体分子', en: 'Preview ligand' },
  'gen.structure_hint': { zh: '蛋白以条带显示，参考配体用于定位口袋；从头生成不锁定配体骨架。', en: 'View the protein as ribbons and the reference ligand in its pocket. De novo generation does not lock the ligand scaffold.' },
  'gen.fit_view': { zh: '重置视角', en: 'Reset view' },
  'gen.preview_dataid': { zh: '预览 data_id', en: 'Preview data_id' },
  'gen.batch_preview_hint': { zh: '每次预览批量范围内的一个口袋，不改变任务的批量范围。', en: 'Preview one pocket in the selected range without changing the batch.' },
  'gen.viewer_label': { zh: '蛋白条带和参考配体三维视图', en: '3D protein ribbon and reference ligand view' },
  'gen.protein_ribbon': { zh: '蛋白条带', en: 'Protein ribbons' },
  'gen.reference_ligand': { zh: '参考配体（可选）', en: 'Reference ligand (optional)' },
  'gen.structure_empty': { zh: '选择输入后，点击「预览配体分子」查看配体及蛋白口袋。', en: 'Select the input, then preview the ligand and protein pocket.' },
  'gen.structure_changed': { zh: '输入已改变，请重新加载三维结构。', en: 'Input changed. Reload the 3D structure.' },
  'gen.structure_loading': { zh: '正在加载口袋三维结构…', en: 'Loading the 3D pocket…' },
  'gen.structure_ready': { zh: '已加载蛋白条带和参考配体（{atoms} 个重原子）。拖动旋转，滚轮缩放。', en: 'Loaded protein ribbons and a reference ligand with {atoms} heavy atoms. Drag to rotate; scroll to zoom.' },
  'gen.structure_protein_only': { zh: '已加载蛋白条带。未提供参考配体，将按输入蛋白或口袋生成。', en: 'Loaded protein ribbons without a reference ligand. Generation uses the input protein or pocket.' },
  'gen.radius_invalid': { zh: '口袋半径须为 4–20 Å。', en: 'Pocket radius must be between 4 and 20 Å.' },
  'gen.preview_id_invalid': { zh: '请选择有效的 data_id；批量预览须在所选范围内。', en: 'Select a valid data_id within the selected batch range.' },

  'batch.title':        { zh: '批量生成（data_id 范围）', en: 'Batch generation (data_id range)' },
  'batch.start':        { zh: '起始 data_id', en: 'Start data_id' },
  'batch.end':          { zh: '结束 data_id', en: 'End data_id' },
  'batch.size':         { zh: '每口袋分子数', en: 'Molecules per pocket' },
  'batch.gpus':         { zh: 'GPU 列表', en: 'GPU list' },
  'batch.auto_eval':    { zh: '采样后自动评估', en: 'Auto-evaluate after sampling' },
  'batch.start_btn':    { zh: '开始批量任务', en: 'Start batch job' },

  'eval.title':         { zh: 'PT 文件评估', en: 'PT File Evaluation' },
  'eval.pt_path':       { zh: '.pt 文件路径', en: '.pt file path' },
  'eval.pt_upload_hint': {
    zh: '可上传兼容的 PT 结果文件（最大 50MB）；本平台生成的结果也可通过「浏览」选择。',
    en: 'Upload a compatible PT result file (up to 50MB), or use Browse to select results generated here.',
  },
  'eval.prot_root':     { zh: '蛋白根目录', en: 'Protein root directory' },
  'eval.vina_mode':     { zh: 'Vina 模式', en: 'Vina Mode' },
  'eval.start':         { zh: '开始评估', en: 'Start Evaluation' },
  'eval.extract_title': { zh: '分子提取', en: 'Molecule Extraction' },
  'eval.extract_btn':   { zh: '开始提取', en: 'Start Extraction' },
  'eval.status_title':  { zh: '任务状态', en: 'Task Status' },
  'eval.idle':          { zh: '空闲', en: 'Idle' },

  'mol.title':          { zh: '分子数据库', en: 'Molecule Database' },
  'mol.pocket_ph':      { zh: '蛋白口袋搜索', en: 'Search pocket' },
  'mol.smiles_ph':      { zh: 'SMILES 搜索', en: 'Search SMILES' },
  'mol.vina_min':       { zh: 'Vina 最小', en: 'Vina min' },
  'mol.vina_max':       { zh: 'Vina 最大', en: 'Vina max' },
  'mol.sort_default':   { zh: '默认排序', en: 'Default sort' },
  'mol.sort_vina_asc':  { zh: 'Vina ↑ (越低越好)', en: 'Vina ↑ (lower is better)' },
  'mol.sort_vina_desc': { zh: 'Vina ↓', en: 'Vina ↓' },
  'mol.sort_score':     { zh: '综合分 ↓', en: 'Score ↓' },
  'mol.sort_qed':       { zh: 'QED ↓', en: 'QED ↓' },
  'mol.sort_sa':        { zh: 'SA ↓', en: 'SA ↓' },
  'mol.search':         { zh: '搜索', en: 'Search' },
  'mol.th_pocket':      { zh: '口袋', en: 'Pocket' },
  'mol.th_score':       { zh: '综合分', en: 'Score' },
  'mol.th_lilly':       { zh: 'Lilly通过', en: 'Lilly Pass' },
  'mol.th_demerit':     { zh: 'Lilly扣分', en: 'Lilly Demerit' },
  'mol.th_energy':      { zh: '能量', en: 'Energy' },
  'mol.th_stable':      { zh: '稳定', en: 'Stable' },
  'mol.th_sdf':         { zh: 'SDF文件', en: 'SDF File' },

  'hist.title':         { zh: '操作历史', en: 'Action History' },
  'hist.all':           { zh: '全部操作', en: 'All actions' },
  'hist.act_gen':       { zh: '开始生成', en: 'Start generation' },
  'hist.act_eval':      { zh: '开始评估', en: 'Start evaluation' },
  'hist.act_ext':       { zh: '开始提取', en: 'Start extraction' },
  'hist.act_gen_done':  { zh: '生成完成', en: 'Generation completed' },
  'hist.act_gen_fail':  { zh: '生成失败', en: 'Generation failed' },
  'hist.act_cfg':       { zh: '配置更新', en: 'Config updated' },
  'hist.th_action':     { zh: '操作', en: 'Action' },
  'hist.th_user':       { zh: '用户', en: 'User' },
  'hist.th_time':       { zh: '时间', en: 'Time' },
  'hist.th_detail':     { zh: '详情', en: 'Details' },
  'hist.runs_title':    { zh: '运行记录', en: 'Run Records' },
  'hist.all_types':     { zh: '全部类型', en: 'All types' },
  'hist.type_eval':     { zh: '评估', en: 'Evaluate' },
  'hist.type_extract':  { zh: '提取', en: 'Extract' },
  'hist.all_status':    { zh: '全部状态', en: 'All statuses' },
  'hist.st_pending':    { zh: '等待中', en: 'Pending' },
  'hist.st_running':    { zh: '运行中', en: 'Running' },
  'hist.st_completed':  { zh: '已完成', en: 'Completed' },
  'hist.st_failed':     { zh: '失败', en: 'Failed' },
  'hist.th_type':       { zh: '类型', en: 'Type' },
  'hist.th_status':     { zh: '状态', en: 'Status' },
  'hist.th_created':    { zh: '创建时间', en: 'Created' },
  'hist.th_finished':   { zh: '完成时间', en: 'Finished' },
  'hist.th_progress':   { zh: '进度', en: 'Progress' },
  'hist.th_log':        { zh: '日志', en: 'Log' },

  'cfg.title':          { zh: '默认采样配置（只读）', en: 'Default sampling configuration (read-only)' },
  'cfg.load':           { zh: '加载', en: 'Load' },
  'cfg.save':           { zh: '保存', en: 'Save' },
  'cfg.runtime':        { zh: '运行时配置', en: 'Runtime Config' },
  'cfg.gpu':            { zh: 'GPU 状态', en: 'GPU Status' },

  'footer.desc':        { zh: '基于扩散模型的结构化药物设计平台', en: 'Diffusion-based Structure-based Drug Design Platform' },
  'modal.log_title':    { zh: '运行日志', en: 'Run Log' },
  'common.goto':        { zh: '前往', en: 'Open' },
  'common.browse':      { zh: '浏览', en: 'Browse' },
  'common.upload':      { zh: '上传', en: 'Upload' },
  'common.basic':       { zh: '基本', en: 'Basic' },
  'common.advanced':    { zh: '高级', en: 'Advanced' },
  'common.cancel':      { zh: '取消', en: 'Cancel' },
  'common.error':       { zh: '错误', en: 'Error' },
  'common.started':     { zh: '已启动', en: 'Started' },
  'common.custom_started': { zh: '自定义生成已启动', en: 'Custom generation started' },
  'common.extract_started': { zh: '提取已启动', en: 'Extraction started' },
  'common.submitting':  { zh: '正在提交…', en: 'Submitting…' },
  'common.cancelled':   { zh: '已取消', en: 'Cancelled' },
  'common.cancel_failed': { zh: '取消失败', en: 'Cancellation failed' },
  'common.progress':    { zh: '进度', en: 'Progress' },
  'common.queue_position': { zh: '排队中（位置 {n}）', en: 'Queued (position {n})' },
  'common.language':    { zh: '切换到英文', en: 'Switch to Chinese' },
  'common.home':        { zh: '← BianLab 首页', en: '← BianLab Home' },
  'common.home_title':  { zh: '返回 BianLab 首页', en: 'Back to BianLab Home' },
  'common.admin':       { zh: '管理员 · 全库可见', en: 'Admin · All records visible' },
  'common.log':         { zh: '日志', en: 'Log' },
  'common.parameters':  { zh: '参数', en: 'Parameters' },
  'common.results':     { zh: '结果', en: 'Results' },
  'common.rerun':       { zh: '重跑', en: 'Rerun' },
  'common.actions':     { zh: '操作', en: 'Actions' },
  'common.select_file': { zh: '选择文件', en: 'Select file' },
  'common.all_files':   { zh: '所有文件', en: 'All files' },
  'common.file_browser': { zh: '文件浏览', en: 'File browser' },
  'common.parent':      { zh: '↑ 上级', en: '↑ Parent' },
  'common.file_search': { zh: '搜索文件名…', en: 'Search filenames…' },
  'common.empty_dir':   { zh: '空目录', en: 'Empty directory' },
  'common.selected':    { zh: '已选择', en: 'Selected' },
  'common.uploading':   { zh: '上传中', en: 'Uploading' },
  'common.upload_done': { zh: '上传完成', en: 'Upload complete' },
  'common.upload_failed': { zh: '上传失败', en: 'Upload failed' },
  'common.file_too_large': { zh: '文件过大：{name} ({size})，上限 {limit}', en: 'File too large: {name} ({size}); limit {limit}' },
  'common.need_protein': { zh: '请填写蛋白 PDB 路径。', en: 'Enter a protein PDB path.' },
  'common.need_pt':     { zh: '请输入 .pt 文件路径', en: 'Enter a .pt file path' },
  'common.molecule_view': { zh: '3D 分子查看', en: '3D molecule viewer' },
  'common.result_summary': { zh: '运行结果摘要', en: 'Run result summary' },
  'common.load_failed': { zh: '加载失败', en: 'Load failed' },
  'common.no_log':      { zh: '暂无日志', en: 'No log available' },
  'common.no_files':    { zh: '无文件', en: 'No files' },
  'common.file_prompt': { zh: '选择文件（输入序号）：', en: 'Select a file (enter its number):' },
  'common.directory':   { zh: '目录', en: 'Directory' },
  'hist.rerun_confirm': { zh: '重跑 Run #{id}？', en: 'Rerun Run #{id}?' },
  'hist.submitted':     { zh: '已提交', en: 'Submitted' },
  'hist.need_two':      { zh: '请至少选择 2 个 Run', en: 'Select at least 2 runs' },
  'hist.molecules':     { zh: '分子', en: 'Molecules' },
  'hist.mean_score':    { zh: '综合分均值', en: 'Mean score' },
  'mol.result_count':   { zh: '{n} / {total} 条结果', en: '{n} / {total} results' },
  'result.run_missing': { zh: '运行不存在', en: 'Run not found' },
  'result.title':       { zh: '运行结果 #{id}', en: 'Run results #{id}' },
  'result.generated':   { zh: '生成分子', en: 'Generated molecules' },
  'result.valid':       { zh: '有效分子', en: 'Valid molecules' },
  'result.qed':         { zh: 'QED 均值', en: 'Mean QED' },
  'result.sa':          { zh: 'SA 均值', en: 'Mean SA' },
  'result.vina':        { zh: 'Vina 均值', en: 'Mean Vina' },
  'result.lipinski':    { zh: 'Lipinski 通过', en: 'Lipinski passed' },
  'result.metrics':     { zh: '评估指标', en: 'Evaluation metrics' },
  'result.no_eval':     { zh: '该运行暂无评估数据', en: 'No evaluation data for this run yet' },
  'common.preset':      { zh: '预设配置', en: 'Preset configuration' },
  'common.preset_select': { zh: '— 选择预设配置 —', en: '— Select a preset —' },
  'common.preset_loaded': { zh: '已加载预设', en: 'Preset loaded' },
  'common.limit_reached': { zh: '已达使用上限', en: 'Usage limit reached' },
  'global.monitor':     { zh: '任务监控', en: 'Job monitor' },
  'global.expand':      { zh: '展开任务面板', en: 'Expand job panel' },
  'global.collapse':    { zh: '收起', en: 'Collapse' },
  'global.counts':      { zh: '{running} 运行中 · {pending} 排队', en: '{running} running · {pending} queued' },
  'global.cancel_confirm': { zh: '取消任务 {id}？', en: 'Cancel job {id}?' },
  'hist.compare':       { zh: '对比选中', en: 'Compare selected' },
  'cfg.user':           { zh: '用户标识', en: 'User identifier' },
  'cfg.user_hint':      { zh: '用于操作审计记录', en: 'Used for action audit records' },
  'cfg.saved':          { zh: '已保存', en: 'Saved' },
  'cfg.save_failed':    { zh: '保存失败', en: 'Save failed' },
  'cfg.gpu_name':       { zh: '名称', en: 'Name' },
  'cfg.gpu_memory':     { zh: '显存', en: 'Memory' },
  'cfg.gpu_utilization': { zh: '利用率', en: 'Utilization' },

  // Dashboard
  'dash.status_idle':   { zh: '系统就绪 · 等待任务', en: 'System ready · No active jobs' },
  'dash.status_active': { zh: '任务运行中', en: 'Jobs running' },
  'dash.hero_title':    { zh: '从噪声到分子：动态扩散的配体生成', en: 'From Noise to Molecules: Dynamic Diffusion Ligand Design' },
  'dash.section_innovation': { zh: '核心创新', en: 'Key Innovations' },
  'dash.section_cap':   { zh: '平台能力', en: 'Capabilities' },
  'dash.section_score': { zh: '八维评估体系', en: '8-Dimension Scoring' },
  'dash.inn1_title':    { zh: '两阶段动态跳步采样', en: 'Two-Stage Dynamic Skip-Step' },
  'dash.inn1_desc':     { zh: '按信息密度分配计算：高噪声区大步跳跃探索拓扑，低噪声区小步精修几何，配合结构修复，减少约 60–70% 网络评估。', en: 'Allocates compute proportionally to information density: large strides explore topology at high noise, dense steps refine geometry at low noise, with structural repair — saving ~60–70% of network evaluations.' },
  'dash.inn2_title':    { zh: '梯度融合', en: 'Gradient Fusion' },
  'dash.inn2_desc':     { zh: '在每次去噪中混合全局 x₀ 预测与局部后验均值方向，λ(t) 随时间衰减，兼顾构象方向与键级精度。', en: 'Blends the global x₀ prediction and the local posterior-mean direction at every denoising step, with a time-decaying λ(t) balancing conformation and geometry.' },
  'dash.inn3_title':    { zh: 'Prudent 多轮过滤', en: 'Prudent Multi-Round Filtering' },
  'dash.inn3_desc':     { zh: '将属性过滤嵌入扩散过程：生成 → QED/SA 门控 → Vina 对接 → Top-K 晋级，由对接打分驱动迭代优化。', en: 'Embeds property filtering inside the diffusion loop: generate → QED/SA gate → Vina docking → Top-K advance, with docking scores driving iterative refinement.' },
  'dash.inn4_title':    { zh: '骨架约束生成', en: 'Scaffold-Constrained Generation' },
  'dash.inn4_desc':     { zh: 'Grow / Evolve / Dynamic Locked / Scaffold-Prudent 四种模式，支持 Murcko 骨架自动识别与自适应门控。', en: 'Four modes — Grow, Evolve, Dynamic Locked, Scaffold-Prudent — with automatic Murcko scaffold detection and adaptive gating.' },
  'dash.cap_gen_title': { zh: '分子生成', en: 'Generate' },
  'dash.cap_gen_desc':  { zh: 'Dynamic 快速采样 / Prudent 多轮优化，支持批量与自定义口袋。', en: 'Dynamic fast sampling and Prudent multi-round optimization, with batch and custom pocket support.' },
  'dash.cap_sc_title':  { zh: '骨架优化', en: 'Scaffold' },
  'dash.cap_sc_desc':   { zh: '骨架约束生成与进化，保留药效团核心。', en: 'Scaffold-constrained generation and evolution that preserve the pharmacophore core.' },
  'dash.cap_eval_title':{ zh: '评估提取', en: 'Evaluate & Extract' },
  'dash.cap_eval_desc': { zh: 'Vina 对接、化学指标评估，导出 SDF / Excel。', en: 'Vina docking, chemical metrics, and SDF / Excel export.' },
  'dash.cap_pe_title':  { zh: '口袋评估', en: 'Pocket Eval' },
  'dash.cap_pe_desc':   { zh: '八维口袋质量评分与可视化图表。', en: '8-dimension pocket quality scoring with visualization charts.' },
  'dash.cap_hist_title':{ zh: '历史记录', en: 'History' },
  'dash.cap_hist_desc': { zh: '完整操作审计，支持 Run 对比与一键重跑。', en: 'Full audit trail with run comparison and one-click rerun.' },
  'dash.cap_cfg_title': { zh: '系统配置', en: 'Config' },
  'dash.cap_cfg_desc':  { zh: '管理员配置查看与 GPU 状态监控。', en: 'Administrator configuration and GPU monitoring.' },
  'dash.score_lead':    { zh: '生成结果通过 8 维综合评估输出 0–100 评分：亲和力、聚类、配体效率、药物相似性、完整性、唯一性、尺寸一致性与口袋体积。', en: 'Generated molecules are scored 0–100 across eight dimensions: affinity, clustering, ligand efficiency, drug-likeness, completeness, uniqueness, size consistency, and pocket volume.' },

  'common.loading':     { zh: '加载中...', en: 'Loading...' },
  'common.optional':    { zh: '可选', en: 'Optional' },
  'common.refresh':     { zh: '刷新', en: 'Refresh' },
  'global.tasks':       { zh: '任务队列', en: 'Job Queue' },
  'global.no_tasks':    { zh: '暂无运行任务', en: 'No active jobs' },
  'global.toast_start_title': { zh: '任务已启动', en: 'Job started' },
  'global.toast_done_title': { zh: '任务完成', en: 'Job completed' },
  'global.toast_fail_title': { zh: '任务失败', en: 'Job failed' },
  'global.toast_cancel_title': { zh: '任务已取消', en: 'Job cancelled' },
  'global.toast_job':   { zh: '任务', en: 'Job' },

  'pe.title':           { zh: '口袋质量评估', en: 'Pocket Quality Evaluation' },
  'pe.mode':            { zh: '评估模式', en: 'Evaluation Mode' },
  'pe.mode_existing':   { zh: '评估已有 .pt 文件', en: 'Evaluate existing .pt' },
  'pe.mode_generate':   { zh: '先生成再评估', en: 'Generate then evaluate' },
  'pe.protein':         { zh: '蛋白质 PDB 路径（必须）', en: 'Protein PDB path (required)' },
  'pe.ligand':          { zh: '配体 SDF 路径（必须）', en: 'Ligand SDF path (required)' },
  'pe.pt_path':         { zh: '.pt 文件路径', en: '.pt file path' },
  'pe.start':           { zh: '开始评估', en: 'Start Evaluation' },
  'pe.progress':        { zh: '评估进度', en: 'Evaluation Progress' },
  'pe.scores':          { zh: '八维评分结果', en: '8-Dimension Scores' },
  'pe.images':          { zh: '可视化图表', en: 'Visualization Charts' },
  'pe.idle':            { zh: '等待提交评估任务', en: 'Waiting to submit evaluation' },
  'pe.no_images':       { zh: '暂无图表', en: 'No charts yet' },
  'pe.loading_results': { zh: '正在加载结果...', en: 'Loading results...' },
  'pe.dim_a': { zh: 'A · Vina对接+LE', en: 'A · Vina+LE' },
  'pe.dim_b': { zh: 'B · 原子聚类', en: 'B · Clustering' },
  'pe.dim_c': { zh: 'C · 配体效率', en: 'C · Ligand Eff.' },
  'pe.dim_d': { zh: 'D · 药物相似性', en: 'D · Drug-likeness' },
  'pe.dim_e': { zh: 'E · 完整分子比例', en: 'E · Completeness' },
  'pe.dim_f': { zh: 'F · 分子唯一性', en: 'F · Uniqueness' },
  'pe.dim_g': { zh: 'G · 尺寸一致性', en: 'G · Size Consist.' },
  'pe.dim_h': { zh: 'H · 口袋体积', en: 'H · Pocket Vol.' },

  // Scaffold optimization (merged tab: SDEdit / Scaffold / Cascade)
  'so.title':           { zh: '骨架优化', en: 'Scaffold Optimization' },
  'so.hint':            { zh: '选择优化功能，下方表单随之切换。', en: 'Pick a function; the form below switches accordingly.' },
  'so.func_opt':        { zh: '分子优化 (SDEdit)', en: 'Molecule Optimization (SDEdit)' },
  'so.func_sc':         { zh: '骨架生成与优化', en: 'Scaffold Generation & Optimization' },
  'so.func_cascade':    { zh: '骨架级联', en: 'Scaffold Cascade' },

  // Optimization tab
  'opt.title':          { zh: '分子优化 (SDEdit)', en: 'Molecular Optimization (SDEdit)' },
  'opt.hint':           { zh: '对已有分子进行扩散模型优化，通过多轮筛选提升分子质量。需提供测试集 data_id 或分子 SDF 路径。', en: 'Optimize existing molecules via diffusion refinement with multi-round filtering. Provide test set data_id or molecule SDF path.' },
  'opt.dataid':         { zh: 'data_id（测试集索引）', en: 'data_id (Test set index)' },
  'opt.molecule_path':  { zh: '分子 SDF 路径（可选）', en: 'Molecule SDF path (optional)' },
  'opt.molecule_ph':    { zh: '留空则使用测试集配体', en: 'Uses test set ligand if empty' },
  'opt.advanced':       { zh: '高级参数', en: 'Advanced Parameters' },
  'opt.num_samples':    { zh: '每轮采样数', en: 'Samples per cycle' },
  'opt.start_t':        { zh: '起始时间步 (start_t)', en: 'Start timestep (start_t)' },
  'opt.cycles':         { zh: '优化轮数 (cycles)', en: 'Optimization cycles' },
  'opt.stride':         { zh: '步长 (stride)', en: 'Stride' },
  'opt.step_size':      { zh: '步大小 (step_size)', en: 'Step size' },
  'opt.schedule':       { zh: '调度策略', en: 'Schedule' },
  'opt.min_qed':        { zh: '最低 QED', en: 'Min QED' },
  'opt.min_sa':         { zh: '最低 SA', en: 'Min SA' },
  'opt.max_survivors':  { zh: '每轮最多保留', en: 'Max survivors/cycle' },
  'opt.start':          { zh: '开始优化', en: 'Start Optimization' },
  'opt.progress':       { zh: '任务进度', en: 'Task Progress' },

  // Scaffold tab
  'nav.scaffold':       { zh: '骨架优化', en: 'Scaffold' },
  'sc.title':           { zh: '骨架约束生成/优化', en: 'Scaffold-Constrained Generation' },
  'sc.hint':            { zh: '基于骨架约束的分子生成：Grow / Evolve / Dynamic Locked / Scaffold-Prudent。需提供测试集 data_id 或参考分子 SDF。', en: 'Scaffold-constrained generation: Grow, Evolve, Dynamic Locked, or Scaffold-Prudent. Provide test set data_id or reference molecule SDF.' },
  'sc.mode':            { zh: '骨架模式', en: 'Scaffold Mode' },
  'sc.mode_grow':       { zh: 'Grow（骨架固定，生成侧链）', en: 'Grow (fix scaffold, generate side chains)' },
  'sc.mode_evolve':     { zh: 'Evolve（骨架固定，进化非骨架原子）', en: 'Evolve (fix scaffold, evolve non-scaffold atoms)' },
  'sc.mode_dynamic_locked': { zh: 'Dynamic Locked（完整 dynamic，锁骨架位置）', en: 'Dynamic Locked (full dynamic, lock scaffold position)' },
  'sc.mode_prudent':    { zh: 'Scaffold-Prudent（锁骨架 + 迭代筛选）', en: 'Scaffold-Prudent (locked scaffold + iterative selection)' },
  'sc.note_dynamic_locked': { zh: '跑完整 dynamic 两阶段（large_step + refine），仅锁定骨架位置，元素类型由模型自由预测。', en: 'Runs full dynamic two-stage (large_step + refine); locks scaffold position only; atom types are predicted freely.' },
  'sc.note_prudent':    { zh: 'Gen 0 用 dynamic_locked 生成初始池，之后多代 Vina/QED/SA 打分选优并 renoise→refine 迭代。', en: 'Gen 0 uses dynamic_locked; subsequent generations score (Vina/QED/SA), select top-K, then renoise→refine.' },
  'sc.sample_title':    { zh: '采样参数', en: 'Sampling Parameters' },
  'sc.desc_grow':       { zh: '从固定骨架生长侧链', en: 'Grow side chains from a fixed scaffold' },
  'sc.desc_evolve':     { zh: '逐步进化整个分子', en: 'Iteratively evolve the molecule' },
  'sc.desc_dynamic':    { zh: '完整 dynamic 与骨架约束', en: 'Full dynamic sampling with scaffold constraints' },
  'sc.desc_prudent':    { zh: '多轮谨慎生成筛选', en: 'Generate and select over multiple rounds' },
  'sc.input':           { zh: '输入来源', en: 'Input source' },
  'sc.input_dataset':   { zh: '测试集', en: 'Test set' },
  'sc.input_custom':    { zh: '自定义蛋白与参考配体', en: 'Custom protein and reference ligand' },
  'sc.protein':         { zh: '蛋白 PDB', en: 'Protein PDB' },
  'sc.workflow':        { zh: '采样流程', en: 'Sampling workflow' },
  'sc.workflow_legacy': { zh: '现有流程', en: 'Existing workflow' },
  'sc.workflow_repaint': { zh: '最新 RePaint', en: 'Latest RePaint' },
  'sc.seed':            { zh: '随机种子', en: 'Random seed' },
  'sc.placement':       { zh: '原子投放', en: 'Atom placement' },
  'sc.placement_auto':  { zh: '自动位点', en: 'Automatic sites' },
  'sc.placement_custom': { zh: '自定义区域', en: 'Custom regions' },
  'sc.repaint_hint':    { zh: 'RePaint：锁定骨架位置和类型，t=999 加噪，refine 最多 30 次；投放只决定扩散前位置。', en: 'RePaint locks scaffold positions and types, adds noise at t=999, and refines up to 30 times. Placement sets positions before diffusion.' },
  'sc.repaint_note':    { zh: '完整 dynamic：锁定骨架位置和类型，从 t=999 前向加噪，refine 最多 30 次。', en: 'Full dynamic sampling: lock scaffold positions and types, add forward noise from t=999, and refine up to 30 times.' },
  'sc.custom_count_note': { zh: '新增原子数由所有区域数量之和决定。', en: 'The extra atom count is the sum of all region counts.' },
  'sc.load_3d':         { zh: '加载三维结构', en: 'Load 3D structure' },
  'sc.add_region':      { zh: '添加投放区域', en: 'Add region' },
  'sc.preview':         { zh: '预览初始投放', en: 'Preview initial placement' },
  'sc.viewer_label':    { zh: '蛋白、参考分子与投放区域三维视图', en: '3D view of protein, reference ligand, and placement regions' },
  'sc.regions_label':   { zh: '投放区域列表', en: 'Placement regions' },
  'sc.pick_hint':       { zh: '选择区域后点击骨架原子设锚点，点击蛋白或配体原子设中心参考，再编辑 XYZ（Å）精调。', en: 'Select a region, click a scaffold atom to set its anchor, or click a protein or ligand atom to set its center reference. Fine-tune XYZ (Å) afterwards.' },
  'sc.placement_idle':  { zh: '请先加载结构。预览显示扩散前投放，后续扩散会优化新增原子。', en: 'Load a structure first. The preview shows initial placement before diffusion, which will optimize the added atoms.' },
  'sc.input_changed':   { zh: '输入结构已改变，请重新加载三维结构。', en: 'Input structure changed. Reload the 3D structure.' },
  'sc.preview_invalid': { zh: '参数已改变，初始投放预览已失效。', en: 'Parameters changed. The initial placement preview is no longer valid.' },
  'sc.loading_context': { zh: '正在加载结构与骨架索引…', en: 'Loading the structure and scaffold indices…' },
  'sc.context_loaded':  { zh: '已加载 {atoms} 个重原子，骨架 {scaffold} 个。绿色为骨架；索引从 0 开始。', en: 'Loaded {atoms} heavy atoms with {scaffold} scaffold atoms. The scaffold is green; indices start at 0.' },
  'sc.need_regions':    { zh: '请加载结构并添加投放区域。', en: 'Load a structure and add a placement region.' },
  'sc.seed_invalid':    { zh: '种子须为 0–4294967295 的整数。', en: 'The seed must be an integer from 0 to 4294967295.' },
  'sc.region_invalid':  { zh: '每个区域需要有效骨架锚点、XYZ 坐标、正整数原子数和正数宽度。', en: 'Each region needs a valid scaffold anchor, finite XYZ coordinates, a positive integer atom count, and positive widths.' },
  'sc.count_invalid':   { zh: '所有投放区域合计最多 50 个新增原子。', en: 'All regions combined may add at most 50 atoms.' },
  'sc.width_invalid':   { zh: '高斯宽度须大于 0 且不超过 10 Å。', en: 'Gaussian widths must be greater than 0 and no more than 10 Å.' },
  'sc.pick_region_first': { zh: '先选择“自定义区域”，再添加一个区域。', en: 'Select Custom regions and add a region first.' },
  'sc.pick_scaffold':   { zh: '锚点须选绿色骨架原子。', en: 'Choose a green scaffold atom as the anchor.' },
  'sc.anchor_selected': { zh: '区域 {region} 锚点：{index}；可继续选中心。', en: 'Region {region} anchor: {index}. You can now select a center.' },
  'sc.center_selected': { zh: '已设置中心参考，请按需编辑 XYZ 将中心移入空腔。', en: 'Center reference set. Edit XYZ to move the center into the cavity as needed.' },
  'sc.pick_anchor_prompt': { zh: '区域 {region}：点击三维结构选择骨架锚点。', en: 'Region {region}: click the 3D structure to select a scaffold anchor.' },
  'sc.pick_center_prompt': { zh: '区域 {region}：点击三维结构选择中心参考。', en: 'Region {region}: click the 3D structure to select a center reference.' },
  'sc.region':          { zh: '区域 {n}', en: 'Region {n}' },
  'sc.select_anchor':   { zh: '选锚点', en: 'Select anchor' },
  'sc.select_center':   { zh: '选中心', en: 'Select center' },
  'sc.delete':          { zh: '删除', en: 'Delete' },
  'sc.anchor_index':    { zh: '骨架锚点索引', en: 'Scaffold anchor index' },
  'sc.atom_count':      { zh: '新增原子数', en: 'Extra atom count' },
  'sc.atom_count_placeholder': { zh: '填写 1–50', en: 'Enter 1–50' },
  'sc.atom_count_hint': { zh: '先填写 1–50 个新增原子再预览，生成时使用同一数量。', en: 'Enter 1–50 extra atoms before previewing. Generation uses the same count.' },
  'sc.need_atom_count': { zh: '预览前请填写 1–50 的整数新增原子数；自定义区域须填写各区域数量。', en: 'Before previewing, enter an integer from 1 to 50, or set the count for each custom region.' },
  'sc.method':          { zh: '投放方式', en: 'Placement method' },
  'sc.gaussian':        { zh: '中心高斯', en: 'Center Gaussian' },
  'sc.corridor':        { zh: '界面走廊', en: 'Interface corridor' },
  'sc.axial_width':     { zh: '轴向宽度 (Å)', en: 'Axial width (Å)' },
  'sc.lateral_width':   { zh: '侧向宽度 (Å)', en: 'Lateral width (Å)' },
  'sc.anchor_label':    { zh: 'R{region} · 锚点 {index}', en: 'R{region} · anchor {index}' },
  'sc.preview_loading': { zh: '正在计算扩散前投放…', en: 'Computing initial placement before diffusion…' },
  'sc.preview_ready':   { zh: '扩散前初始投放：{atoms} 个原子，种子 {seed}，第 1 个样本。后续扩散会优化这些位置。', en: 'Initial placement before diffusion: {atoms} atoms, seed {seed}, sample 1. Diffusion will optimize these positions.' },
  'sc.canvas_label':    { zh: '三维投放视图；拖动或方向键旋转，滚轮或加减键缩放，点击选择原子', en: '3D placement view; drag or use arrow keys to rotate, scroll or use plus and minus to zoom, and click to select an atom' },
  'sc.canvas_hint':     { zh: '拖动旋转 · 滚轮缩放 · 点击选点', en: 'Drag to rotate · Scroll to zoom · Click to select' },
  'sc.need_ligand':     { zh: '请先填写参考配体 SDF 路径', en: 'Enter the reference ligand SDF path first' },
  'sc.molecule_ph':     { zh: '留空则使用测试集配体', en: 'Uses test set ligand if empty' },
  'sc.source':          { zh: '骨架识别方式', en: 'Scaffold Source' },
  'sc.source_murcko':   { zh: '自动 Murcko', en: 'Auto Murcko' },
  'sc.source_generic':  { zh: '通用 Murcko', en: 'Generic Murcko' },
  'sc.source_atoms':    { zh: '指定原子索引', en: 'Atom indices' },
  'sc.source_smarts':   { zh: 'SMARTS 匹配', en: 'SMARTS match' },
  'sc.source_none':     { zh: '无约束', en: 'None' },
  'sc.atom_indices':    { zh: '原子索引（逗号分隔）', en: 'Atom indices (comma-separated)' },
  'sc.smarts':          { zh: 'SMARTS 模式', en: 'SMARTS pattern' },
  'sc.weights_title':   { zh: '权重与过滤', en: 'Weights & Filters' },
  'sc.qed_weight':      { zh: 'QED 权重', en: 'QED weight' },
  'sc.sa_weight':       { zh: 'SA 权重', en: 'SA weight' },
  'sc.diversity_weight':{ zh: '多样性权重', en: 'Diversity weight' },
  'sc.max_tanimoto':    { zh: '最大 Tanimoto 相似度', en: 'Max Tanimoto similarity' },
  'sc.fix_pos':         { zh: '固定骨架位置', en: 'Fix scaffold position' },
  'sc.fix_type':        { zh: '固定骨架原子类型', en: 'Fix scaffold atom types' },
  'sc.div_filter':      { zh: '多样性过滤', en: 'Diversity filter' },
  'sc.enable_refine':   { zh: 'TargetDiff 结构修复（独立后处理）', en: 'TargetDiff structure repair (separate post-processing)' },
  'sc.grow_title':      { zh: 'Grow 参数', en: 'Grow Parameters' },
  'sc.lambda_a':        { zh: 'Lambda a', en: 'Lambda a' },
  'sc.lambda_b':        { zh: 'Lambda b', en: 'Lambda b' },
  'sc.n_extra_mode':    { zh: '新原子数策略', en: 'Extra atoms strategy' },
  'sc.n_extra_prior_minus': { zh: '口袋先验 - 骨架', en: 'Pocket prior - scaffold' },
  'sc.n_extra_pocket':  { zh: '口袋先验', en: 'Pocket prior' },
  'sc.n_extra_fixed':   { zh: '固定数量', en: 'Fixed count' },
  'sc.n_extra_range':   { zh: '范围随机', en: 'Random range' },
  'sc.n_extra_min':     { zh: '最少新原子', en: 'Min extra atoms' },
  'sc.n_extra_max':     { zh: '最多新原子', en: 'Max extra atoms' },
  'sc.evolve_title':    { zh: 'Evolve 参数', en: 'Evolve Parameters' },
  'sc.population':      { zh: '种群大小', en: 'Population size' },
  'sc.generations':     { zh: '进化代数', en: 'Generations' },
  'sc.children':        { zh: '每代子代数', en: 'Children per parent' },
  'sc.start_t_high':    { zh: '起始时间步（高）', en: 'Start t (high)' },
  'sc.start_t_low':     { zh: '起始时间步（低）', en: 'Start t (low)' },
  'sc.prudent_title':   { zh: 'Prudent 迭代参数', en: 'Prudent Iteration Parameters' },
  'sc.prudent_chains':  { zh: '每种子链数', en: 'Chains per seed' },
  'sc.prudent_top_k':   { zh: '每代晋级数', en: 'Advance top-K' },
  'sc.prudent_renoise_t': { zh: '重加噪时间步', en: 'Renoise t' },
  'sc.prudent_qed_w':   { zh: 'QED 打分权重', en: 'QED score weight' },
  'sc.prudent_sa_w':    { zh: 'SA 打分权重', en: 'SA score weight' },
  'sc.prudent_vina_w':  { zh: 'Vina 打分权重', en: 'Vina score weight' },
  'sc.prudent_vina_exh': { zh: 'Vina exhaustiveness', en: 'Vina exhaustiveness' },
  'sc.prudent_min_qed_dock': { zh: '对接最低 QED', en: 'Min QED for docking' },
  'sc.prudent_min_sa_dock': { zh: '对接最低 SA', en: 'Min SA for docking' },
  'sc.gate_mode': { zh: '门控模式', en: 'Gate mode' },
  'sc.gate_adaptive': { zh: '自适应（参考配体）', en: 'Adaptive (reference ligand)' },
  'sc.gate_manual': { zh: '人工控制', en: 'Manual' },
  'sc.gate_mode_hint': { zh: '自适应：按任务参考配体性质放宽 QED/SA/LogP/尺寸；人工：使用下方固定阈值', en: 'Adaptive: relax gates from reference ligand; Manual: fixed thresholds below' },
  'sc.adaptive_preview': { zh: '预览自适应门控', en: 'Preview adaptive gates' },
  'sc.manual_max_logp': { zh: '人工 max LogP', en: 'Manual max LogP' },
  'sc.manual_max_molwt': { zh: '人工 max MW', en: 'Manual max MW' },
  'sc.manual_max_heavy': { zh: '人工 max 重原子', en: 'Manual max heavy atoms' },
  'sc.manual_max_rings': { zh: '人工 max 环数', en: 'Manual max rings' },
  'sc.manual_gates_hint': { zh: '人工模式下关闭自适应，直接使用上述阈值（对接最低 QED/SA 仍用上方字段）', en: 'Manual disables adaptive gates and uses the fixed thresholds above' },

  // Scaffold Cascade (Innovation)
  'cascade.title':          { zh: 'ScaffoldCascade 骨架级联优化', en: 'ScaffoldCascade — Dual-Seed Scaffold Grow' },
  'cascade.hint':           { zh: '创新流程：使用不同随机种子和参数多次运行骨架固定生成（Grow），合并所有结果以获得更高质量、更多样化的分子。每次运行使用不同的采样策略和起始时间步，最大化结果多样性。', en: 'Innovation pipeline: run scaffold-constrained generation (Grow) multiple times with different random seeds and parameters, merging all results for higher quality and more diverse molecules. Each round uses different sampling strategies and start timesteps to maximize diversity.' },
  'cascade.params_title':   { zh: '级联参数', en: 'Cascade Parameters' },
  'cascade.samples_per_round': { zh: '每轮采样数', en: 'Samples per round' },
  'cascade.rounds':         { zh: '运行轮数', en: 'Number of rounds' },
  'cascade.start':          { zh: '开始级联生成', en: 'Start Cascade' },
  'cascade.progress':       { zh: '级联进度', en: 'Cascade Progress' },
};

function t(key) {
  const entry = T[key];
  if (!entry) return key;
  return entry[currentLang] || entry.zh || key;
}

function tf(key, vars) {
  let s = t(key);
  Object.keys(vars || {}).forEach((k) => {
    s = s.replace(new RegExp('\\{' + k + '\\}', 'g'), String(vars[k]));
  });
  return s;
}

function setLocalizedContent(id, render) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = render();
  localizedContent.set(id, { render, html: el.innerHTML });
}

function translateElements(root = document) {
  root.querySelectorAll('[data-i18n]').forEach(el => {
    if (el.id && localizedContent.has(el.id)) return;
    if (!['INPUT', 'TEXTAREA'].includes(el.tagName)) el.textContent = t(el.dataset.i18n);
  });
  ['title', 'aria-label'].forEach(attribute => {
    root.querySelectorAll(`[data-i18n-${attribute}]`).forEach(el => {
      el.setAttribute(attribute, t(el.getAttribute(`data-i18n-${attribute}`)));
    });
  });
  root.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    const text = t(el.dataset.i18nPlaceholder);
    if (['INPUT', 'TEXTAREA'].includes(el.tagName)) el.placeholder = text;
    else if (!el.textContent.trim() || el.textContent === idleTranslations.get(el)) {
      // Translate the idle label only while the panel is still idle.
      el.textContent = text;
      idleTranslations.set(el, text);
    }
  });
}

function jobStatusText(status) {
  return t(({ pending: 'hist.st_pending', queued: 'hist.st_pending', running: 'hist.st_running',
    completed: 'hist.st_completed', failed: 'hist.st_failed', cancelled: 'common.cancelled' })[status] || status);
}

function setJobStatus(id, data) {
  setLocalizedContent(id, () => {
    const status = data.status || 'running';
    let detail = data.queue_position != null && status === 'pending'
      ? ` · ${tf('common.queue_position', { n: data.queue_position })}` : '';
    if (data.progress != null) detail += ` · ${Math.round(data.progress * 100)}%`;
    const reference = data.run_id != null ? data.run_id : data.job_id;
    if (reference != null) detail += ` (${escHtml(reference)})`;
    if (data.error) detail += `: ${escHtml(data.error)}`;
    return `<span class="${statusClass(status)}">${escHtml(jobStatusText(status))}${detail}</span>`;
  });
}

function setLang(lang) {
  currentLang = ['zh', 'en'].includes(lang) ? lang : 'zh';
  lang = currentLang;
  localStorage.setItem('dd_lang', lang);
  document.getElementById('lang-toggle').textContent = lang === 'zh' ? 'EN' : '中文';
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
  translateElements();
  localizedContent.forEach((content, id) => {
    const el = document.getElementById(id);
    // Do not restore an older status if another operation replaced its content.
    if (!el || el.innerHTML !== content.html) { localizedContent.delete(id); return; }
    el.innerHTML = content.render();
    content.html = el.innerHTML;
  });

  // Update page title
  document.title = lang === 'zh' ? 'DiffDynamic — 分子生成评估平台' : 'DiffDynamic — Drug Design Platform';

  // Refresh overview content when language changes
  const dash = document.querySelector('#tab-dashboard');
  if (dash && dash.classList.contains('active')) refreshDashboard();

  const mode = document.getElementById('sc-scaffold-mode').value;
  document.getElementById('sc-grow-summary').textContent = t(mode === 'grow' ? 'sc.grow_title' : 'sc.sample_title');
  if (window.ScaffoldPlacement) ScaffoldPlacement.refreshLanguage();
  if (window.GenerationViewer) GenerationViewer.refreshLanguage();
  if (typeof applyDailyQuota === 'function') applyDailyQuota();
  window.dispatchEvent(new CustomEvent('dd:language', { detail: { language: lang } }));
}

function toggleLang() {
  setLang(currentLang === 'zh' ? 'en' : 'zh');
}

