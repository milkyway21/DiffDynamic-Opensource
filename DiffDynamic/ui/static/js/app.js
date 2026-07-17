/* DiffDynamic Web UI — API client + UI logic */

const API = '';

// ── i18n ────────────────────────────────────────────────────────────────────

let currentLang = localStorage.getItem('dd_lang') || 'zh';

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
  'gen.mode':           { zh: '生成模式', en: 'Generation Mode' },
  'gen.mode_dynamic':   { zh: 'Dynamic (快速)', en: 'Dynamic (Fast)' },
  'gen.mode_prudent':   { zh: 'Prudent (多轮优化)', en: 'Prudent (Multi-round)' },
  'gen.dataid':         { zh: 'data_id（测试集索引）', en: 'data_id (Test set index)' },
  'gen.gpu_ph':         { zh: '留空自动分配', en: 'Auto-assign if empty' },
  'gen.batch':          { zh: 'batch_size（每批分子数）', en: 'batch_size (molecules per batch)' },
  'gen.batch_ph':       { zh: '默认 5', en: 'Default 5' },
  'gen.max_samples':    { zh: '评估分子数上限 (max_samples)', en: 'Eval max samples' },
  'gen.vina_timeout':   { zh: 'Vina 超时 (秒)', en: 'Vina timeout (sec)' },
  'gen.config':         { zh: '配置文件路径', en: 'Config file path' },
  'gen.config_ph':      { zh: '留空使用默认 sampling.yml', en: 'Leave empty for default sampling.yml' },
  'gen.auto':           { zh: '生成后自动操作', en: 'Auto actions after generation' },
  'gen.auto_eval':      { zh: '自动评估（Vina docking + 化学指标）', en: 'Auto evaluate (Vina docking + chemistry)' },
  'gen.auto_extract':   { zh: '自动提取（.pt → SDF + Excel）', en: 'Auto extract (.pt → SDF + Excel)' },
  'gen.auto_eval_short':{ zh: '自动评估', en: 'Auto evaluate' },
  'gen.auto_extract_short':{ zh: '自动提取', en: 'Auto extract' },
  'gen.remove_fragments':{ zh: '对接前去除小碎片（仅保留最大片段）', en: 'Remove fragments before docking (keep largest)' },
  'eval.remove_fragments':{ zh: '去除小碎片（仅保留最大片段）', en: 'Remove fragments (keep largest)' },
  'gen.start':          { zh: '开始生成', en: 'Start Generation' },
  'gen.cancel':         { zh: '取消任务', en: 'Cancel task' },
  'gen.progress':       { zh: '任务进度', en: 'Task Progress' },
  'gen.idle':           { zh: '空闲 — 尚未启动任务', en: 'Idle — No task started' },
  'gen.custom_title':   { zh: '自定义口袋生成', en: 'Custom Pocket Generation' },
  'gen.custom_hint':    { zh: '上传蛋白 PDB 文件和可选的参考配体 SDF 文件，直接生成分子。', en: 'Upload protein PDB and optional reference ligand SDF to generate molecules.' },
  'gen.protein':        { zh: '蛋白 PDB 文件路径', en: 'Protein PDB file path' },
  'gen.ligand':         { zh: '参考配体 SDF（可选）', en: 'Reference ligand SDF (optional)' },
  'gen.gpu_auto':       { zh: '自动分配', en: 'Auto-assign' },
  'gen.radius':         { zh: '口袋半径 (Å)', en: 'Pocket radius (Å)' },
  'gen.samples':        { zh: '采样数量', en: 'Num samples' },

  'batch.title':        { zh: '批量生成（data_id 范围）', en: 'Batch generation (data_id range)' },
  'batch.start':        { zh: '起始 data_id', en: 'Start data_id' },
  'batch.end':          { zh: '结束 data_id', en: 'End data_id' },
  'batch.size':         { zh: '每口袋分子数', en: 'Molecules per pocket' },
  'batch.gpus':         { zh: 'GPU 列表', en: 'GPU list' },
  'batch.auto_eval':    { zh: '采样后自动评估', en: 'Auto-evaluate after sampling' },
  'batch.start_btn':    { zh: '开始批量任务', en: 'Start batch job' },

  'eval.title':         { zh: 'PT 文件评估', en: 'PT File Evaluation' },
  'eval.pt_path':       { zh: '.pt 文件路径', en: '.pt file path' },
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

  'cfg.title':          { zh: '采样配置 (sampling.yml)', en: 'Sampling Config (sampling.yml)' },
  'cfg.load':           { zh: '加载', en: 'Load' },
  'cfg.save':           { zh: '保存', en: 'Save' },
  'cfg.runtime':        { zh: '运行时配置', en: 'Runtime Config' },
  'cfg.gpu':            { zh: 'GPU 状态', en: 'GPU Status' },

  'footer.desc':        { zh: '基于扩散模型的结构化药物设计平台', en: 'Diffusion-based Structure-based Drug Design Platform' },
  'modal.log_title':    { zh: '运行日志', en: 'Run Log' },

  'common.loading':     { zh: '加载中...', en: 'Loading...' },
  'common.optional':    { zh: '可选', en: 'Optional' },
  'common.refresh':     { zh: '刷新', en: 'Refresh' },

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

  // Optimization tab
  'nav.optimization':   { zh: '分子优化', en: 'Optimization' },
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
  'sc.enable_refine':   { zh: 'TargetDiff 结构修复（10步，t=9）', en: 'TargetDiff structure repair (10 steps, t=9)' },
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
  'nav.cascade':            { zh: '骨架级联', en: 'Cascade' },
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

function setLang(lang) {
  currentLang = lang;
  localStorage.setItem('dd_lang', lang);
  document.getElementById('lang-toggle').textContent = lang === 'zh' ? 'EN' : '中文';

  // Update all data-i18n elements
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    const text = t(key);
    if (el.tagName === 'OPTION') {
      el.textContent = text;
    } else if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
      // skip
    } else {
      el.textContent = text;
    }
  });

  // Update placeholders
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    const key = el.getAttribute('data-i18n-placeholder');
    if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
      el.placeholder = t(key);
    } else {
      el.textContent = t(key);
    }
  });

  // Update page title
  document.title = lang === 'zh' ? 'DiffDynamic — 分子生成评估平台' : 'DiffDynamic — Drug Design Platform';

  // Refresh overview content when language changes
  const dash = document.querySelector('#tab-dashboard');
  if (dash && dash.classList.contains('active')) refreshDashboard();

  if (typeof toggleScaffoldMode === 'function') toggleScaffoldMode();
}

function toggleLang() {
  setLang(currentLang === 'zh' ? 'en' : 'zh');
}

// ── Helpers ────────────────────────────────────────────────────────────────

async function api(path, opts = {}) {
  const user = localStorage.getItem('dd_user') || 'web_ui';
  const headers = { 'Content-Type': 'application/json', 'X-DD-User': user, ...(opts.headers || {}) };
  const res = await fetch(API + path, { ...opts, headers });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

function getUser() { return localStorage.getItem('dd_user') || 'web_ui'; }
function setUser(name) { localStorage.setItem('dd_user', name || 'web_ui'); }

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function statusClass(s) {
  if (s === 'completed') return 'status-completed';
  if (s === 'running') return 'status-running';
  if (s === 'failed') return 'status-failed';
  return 'status-pending';
}

function setProgressRunning(barEl, running) {
  if (!barEl) return;
  barEl.classList.toggle('is-running', !!running);
}

function switchToTab(tabName) {
  const btn = document.querySelector(`.nav-btn[data-tab="${tabName}"]`);
  if (btn) btn.click();
}

function prefersReducedMotion() {
  return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** Signature: light particle field in the header (diffusion metaphor). */
function initDiffusionField() {
  const canvas = document.getElementById('diffusion-field');
  if (!canvas || prefersReducedMotion()) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  let particles = [];
  let raf = 0;
  let w = 0;
  let h = 0;

  function resize() {
    const rect = canvas.parentElement.getBoundingClientRect();
    w = Math.max(1, Math.floor(rect.width));
    h = Math.max(1, Math.floor(rect.height));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.min(40, Math.max(24, Math.floor(w / 28)));
    particles = Array.from({ length: n }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      r: 1.2 + Math.random() * 2.2,
      vx: (Math.random() - 0.5) * 0.35,
      vy: (Math.random() - 0.5) * 0.25,
      a: 0.15 + Math.random() * 0.35,
    }));
  }

  function tick() {
    ctx.clearRect(0, 0, w, h);
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < -10) p.x = w + 10;
      if (p.x > w + 10) p.x = -10;
      if (p.y < -10) p.y = h + 10;
      if (p.y > h + 10) p.y = -10;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(34, 211, 238, ${p.a})`;
      ctx.fill();
    }
    // soft links between nearby particles
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const a = particles[i];
        const b = particles[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const d2 = dx * dx + dy * dy;
        if (d2 < 90 * 90) {
          const alpha = 0.12 * (1 - Math.sqrt(d2) / 90);
          ctx.strokeStyle = `rgba(126, 176, 255, ${alpha})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }
    raf = requestAnimationFrame(tick);
  }

  resize();
  tick();
  window.addEventListener('resize', () => {
    cancelAnimationFrame(raf);
    resize();
    tick();
  });
}

function fmtTime(t) {
  if (!t) return '—';
  return new Date(t).toLocaleString('zh-CN');
}

function escHtml(s) {
  if (!s) return '';
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Tab Navigation ─────────────────────────────────────────────────────────

let currentJobId = null;
let pollTimer = null;

document.addEventListener('DOMContentLoaded', () => {
  // Init language
  setLang(currentLang);
  initDiffusionField();

  const heroCta = $('#hero-cta-generate');
  if (heroCta) heroCta.addEventListener('click', () => switchToTab('generate'));

  $$('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      $$('.nav-btn').forEach(b => b.classList.remove('active'));
      $$('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const panel = $(`#tab-${btn.dataset.tab}`);
      if (panel) {
        panel.classList.add('active');
      }
      // Auto-refresh on tab switch
      const tab = btn.dataset.tab;
      if (tab === 'dashboard') refreshDashboard();
      if (tab === 'history') { refreshHistory(); refreshRuns(); }
      if (tab === 'config') { loadConfig(); loadGPU(); }
      if (tab === 'pocket-eval') { /* nothing to auto-load */ }
    });
  });

  // Form handlers
  $('#gen-form').addEventListener('submit', handleGenerate);
  $('#gen-cancel-btn').addEventListener('click', handleCancel);
  $('#custom-form').addEventListener('submit', handleCustomGenerate);
  $('#eval-form').addEventListener('submit', handleEvaluate);
  $('#extract-form').addEventListener('submit', handleExtract);
  $('#mol-form').addEventListener('submit', handleMolSearch);
  $('#hist-form').addEventListener('submit', e => { e.preventDefault(); refreshHistory(); });
  $('#runs-form').addEventListener('submit', e => { e.preventDefault(); refreshRuns(); });
  $('#cfg-load-btn').addEventListener('click', loadConfig);
  $('#cfg-save-btn').addEventListener('click', saveConfig);

  // Pocket eval handlers
  $('#pe-form').addEventListener('submit', handlePocketEval);
  $('#pe-cancel-btn').addEventListener('click', handlePocketEvalCancel);
  $('#pe-mode').addEventListener('change', togglePocketEvalMode);
  $('#opt-form').addEventListener('submit', handleOptimization);
  $('#opt-cancel-btn').addEventListener('click', handleOptCancel);
  $('#sc-form').addEventListener('submit', handleScaffold);
  $('#sc-cancel-btn').addEventListener('click', handleScaffoldCancel);
  $('#sc-source').addEventListener('change', toggleScaffoldSource);
  const gateModeEl = $('#sc-gate-mode');
  if (gateModeEl) {
    gateModeEl.addEventListener('change', toggleGateMode);
    toggleGateMode();
  }
  toggleScaffoldMode();
  $('#cascade-form').addEventListener('submit', handleCascade);
  $('#cascade-cancel-btn').addEventListener('click', handleCascadeCancel);
  const batchForm = $('#batch-form');
  if (batchForm) batchForm.addEventListener('submit', handleBatchGenerate);
  const compareBtn = $('#compare-runs-btn');
  if (compareBtn) compareBtn.addEventListener('click', compareSelectedRuns);
  const userInput = $('#cfg-user');
  if (userInput) {
    userInput.value = getUser();
    userInput.addEventListener('change', () => setUser(userInput.value));
  }

  // Initial load
  refreshDashboard();
});

// ── Dashboard ──────────────────────────────────────────────────────────────

async function refreshDashboard() {
  try {
    const [readme, stats] = await Promise.all([
      api(`/api/readme?lang=${currentLang}`),
      api('/api/stats').catch(() => null),
    ]);
    $('#readme-content').innerHTML = mdToHtml(readme.content || '');
    if (stats) {
      const bar = $('#dashboard-stats');
      if (bar) {
        bar.innerHTML = `
          <span class="hero-live-chip"><strong>${stats.active_jobs || 0}</strong> ${t('hero.live_running')}</span>
          <span class="hero-live-chip"><strong>${stats.pending_jobs || 0}</strong> ${t('hero.live_queued')}</span>
          <span class="hero-live-chip"><strong>${stats.total_molecules || 0}</strong> ${t('hero.live_mols')}</span>
          <span class="hero-live-chip"><strong>${stats.completed_runs || 0}</strong> ${t('hero.live_done')}</span>`;
      }
    }
  } catch (e) {
    $('#readme-content').innerHTML = `<p class="status-failed">加载失败: ${e.message}</p>`;
  }
}

// Simple markdown → HTML converter
function mdToHtml(md) {
  let html = escHtml(md);
  // Blockquotes (must be before paragraph wrapping)
  html = html.replace(/^&gt;\s+(.+)$/gm, '<blockquote><p>$1</p></blockquote>');
  html = html.replace(/<\/blockquote>\n<blockquote>/g, '\n');
  // Headings
  html = html.replace(/^######\s+(.+)$/gm, '<h6>$1</h6>');
  html = html.replace(/^#####\s+(.+)$/gm, '<h5>$1</h5>');
  html = html.replace(/^####\s+(.+)$/gm, '<h4>$1</h4>');
  html = html.replace(/^###\s+(.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^##\s+(.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^#\s+(.+)$/gm, '<h1>$1</h1>');
  // Horizontal rule
  html = html.replace(/^---+$/gm, '<hr>');
  // Bold + italic
  html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Links
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
  // Tables
  html = html.replace(/^(\|.+\|)\s*\n(\|[-| :]+\|)\s*\n((?:\|.+\|\s*\n)*)/gm, function(match, header, sep, body) {
    const ths = header.split('|').filter(c => c.trim()).map(c => `<th>${c.trim()}</th>`).join('');
    const rows = body.trim().split('\n').map(row => {
      const tds = row.split('|').filter(c => c.trim()).map(c => `<td>${c.trim()}</td>`).join('');
      return `<tr>${tds}</tr>`;
    }).join('');
    return `<table><thead><tr>${ths}</tr></thead><tbody>${rows}</tbody></table>`;
  });
  // Code blocks
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, function(match, lang, code) {
    return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
  });
  // Unordered lists
  html = html.replace(/^(?:- |\* )(.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');
  // Ordered lists
  html = html.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');
  // Paragraphs (lines not already wrapped)
  html = html.replace(/^(?!<[a-z]|$)(.+)$/gm, '<p>$1</p>');
  // Clean up double p wrapping
  html = html.replace(/<p><(h[1-6]|ul|ol|table|pre|hr|li)/g, '<$1');
  html = html.replace(/<\/(h[1-6]|ul|ol|table|pre|hr|li)><\/p>/g, '</$1>');
  return html;
}

// ── Generation ─────────────────────────────────────────────────────────────

async function handleGenerate(e) {
  e.preventDefault();
  const body = {
    mode: $('#gen-mode').value,
    data_id: parseInt($('#gen-dataid').value) || 0,
    use_test_set: true,
    auto_evaluate: $('#gen-auto-eval').checked,
    auto_extract: $('#gen-auto-extract').checked,
    remove_fragments: $('#gen-remove-fragments').checked,
    max_samples: parseInt($('#gen-max-samples').value) || 5,
    vina_timeout: parseInt($('#gen-vina-timeout').value) || 20,
    batch_size: parseInt($('#gen-batch').value) || 5,
  };
  const gpu = $('#gen-gpu').value;
  if (gpu) body.gpu_id = parseInt(gpu);
  const cfg = $('#gen-config').value;
  if (cfg) body.config_path = cfg;

  try {
    const result = await api('/api/generate', { method: 'POST', body: JSON.stringify(body) });
    currentJobId = result.job_id;
    const q = result.status === 'queued' ? ` (排队 #${result.queue_position})` : '';
    $('#gen-status').innerHTML = `<span class="status-running">已启动</span> — Job: <code>${result.job_id}</code>${q}`;
    $('#gen-cancel-btn').disabled = false;
    const bar = $('#gen-progress-bar');
    bar.style.display = 'block';
    setProgressRunning(bar, true);
    startPolling();
  } catch (e) {
    $('#gen-status').innerHTML = `<span class="status-failed">错误: ${e.message}</span>`;
  }
}

async function handleCancel() {
  if (!currentJobId) return;
  try {
    await api(`/api/jobs/${currentJobId}`, { method: 'DELETE' });
    $('#gen-status').innerHTML = '<span class="status-pending">已取消</span>';
    setProgressRunning($('#gen-progress-bar'), false);
    stopPolling();
  } catch (e) {
    $('#gen-status').innerHTML = `<span class="status-failed">取消失败: ${e.message}</span>`;
  }
}

async function handleCustomGenerate(e) {
  e.preventDefault();
  const protein = $('#custom-protein').value.trim();
  if (!protein) { alert('请输入蛋白 PDB 文件路径'); return; }
  const body = {
    protein_path: protein,
    ligand_path: $('#custom-ligand').value.trim() || null,
    pocket_radius: parseFloat($('#custom-radius').value) || 10.0,
    num_samples: parseInt($('#custom-samples').value) || 5,
    config_path: $('#custom-config').value.trim() || null,
    auto_evaluate: $('#custom-auto-eval').checked,
    auto_extract: $('#custom-auto-extract').checked,
    remove_fragments: $('#custom-remove-fragments').checked,
  };
  const gpu = $('#custom-gpu').value;
  if (gpu) body.gpu_id = parseInt(gpu);
  try {
    const result = await api('/api/generate/custom', { method: 'POST', body: JSON.stringify(body) });
    currentJobId = result.job_id;
    $('#gen-status').innerHTML = `<span class="status-running">自定义生成已启动</span> — Job: <code>${result.job_id}</code>, GPU: ${result.gpu_id}`;
    $('#gen-cancel-btn').disabled = false;
    const bar = $('#gen-progress-bar');
    bar.style.display = 'block';
    setProgressRunning(bar, true);
    startPolling();
  } catch (e) {
    $('#gen-status').innerHTML = `<span class="status-failed">错误: ${e.message}</span>`;
  }
}

function startPolling() {
  stopPolling();
  pollTimer = setInterval(pollJob, 2000);
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

async function pollJob() {
  if (!currentJobId) return;
  try {
    const job = await api(`/api/jobs/${currentJobId}`);
    const pct = Math.round((job.progress || 0) * 100);
    const qInfo = job.status === 'pending' ? ` (排队 #${job.queue_position || '?'})` : '';
    const bar = $('#gen-progress-bar');
    $('#gen-progress-fill').style.width = pct + '%';
    setProgressRunning(bar, job.status === 'running' || job.status === 'pending');
    let statusHtml = `<span class="${statusClass(job.status)}">${job.status.toUpperCase()}</span>${qInfo} — 进度: ${pct}%`;
    $('#gen-status').innerHTML = statusHtml;
    if (job.status === 'completed') {
      const el = $('#gen-status').querySelector('.status-completed');
      if (el) el.classList.add('flash');
    }
    $('#gen-log').textContent = (job.log_tail || []).slice(-30).join('\n');
    if (['completed', 'failed', 'cancelled'].includes(job.status)) {
      stopPolling();
      setProgressRunning(bar, false);
      $('#gen-cancel-btn').disabled = true;
    }
  } catch (e) { /* ignore */ }
}

// ── Evaluation ─────────────────────────────────────────────────────────────

async function handleEvaluate(e) {
  e.preventDefault();
  const pt = $('#eval-pt').value.trim();
  if (!pt) { alert('请输入 .pt 文件路径'); return; }
  const modes = [...$$('input[name="vina"]:checked')].map(c => c.value).join(',');
  try {
    const result = await api('/api/evaluate', {
      method: 'POST',
      body: JSON.stringify({
        pt_path: pt, protein_root: $('#eval-prot').value || null,
        vina_modes: modes || 'auto',
        max_samples: parseInt($('#eval-max-samples').value) || 5,
        vina_timeout: parseInt($('#eval-vina-timeout').value) || 20,
      }),
    });
    $('#eval-status').innerHTML = `<span class="status-running">已启动</span> — Job: <code>${result.job_id}</code>`;
    pollEvalJob(result.job_id);
  } catch (e) {
    $('#eval-status').innerHTML = `<span class="status-failed">错误: ${e.message}</span>`;
  }
}

async function handleExtract(e) {
  e.preventDefault();
  const pt = $('#extract-pt').value.trim();
  if (!pt) { alert('请输入 .pt 文件路径'); return; }
  try {
    const result = await api('/api/extract', {
      method: 'POST',
      body: JSON.stringify({ pt_path: pt, protein_root: $('#extract-prot').value || null, remove_fragments: $('#extract-remove-fragments').checked }),
    });
    $('#eval-status').innerHTML = `<span class="status-running">提取已启动</span> — Job: <code>${result.job_id}</code>`;
    pollEvalJob(result.job_id);
  } catch (e) {
    $('#eval-status').innerHTML = `<span class="status-failed">错误: ${e.message}</span>`;
  }
}

async function pollEvalJob(jobId) {
  const poll = setInterval(async () => {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      const pct = Math.round((job.progress || 0) * 100);
      $('#eval-status').innerHTML = `<span class="${statusClass(job.status)}">${job.status.toUpperCase()}</span> — 进度: ${pct}%`;
      $('#eval-log').textContent = (job.log_tail || []).slice(-20).join('\n');
      if (job.status !== 'running' && job.status !== 'pending') clearInterval(poll);
    } catch (e) { clearInterval(poll); }
  }, 3000);
}

// ── Molecules ──────────────────────────────────────────────────────────────

async function handleMolSearch(e) {
  e.preventDefault();
  const params = new URLSearchParams();
  const run = $('#mol-run').value;
  if (run) params.set('run_id', run);
  const prot = $('#mol-protein').value;
  if (prot) params.set('protein', prot);
  const smi = $('#mol-smiles').value;
  if (smi) params.set('smiles', smi);
  const minV = $('#mol-min-vina').value;
  if (minV) params.set('min_vina', minV);
  const maxV = $('#mol-max-vina').value;
  if (maxV) params.set('max_vina', maxV);
  if ($('#mol-lipinski').checked) params.set('lipinski', 'true');
  params.set('limit', '500');
  const sortBy = ($('#mol-sort') || {}).value || '';

  try {
    const data = await api(`/api/molecules?${params}`);
    let mols = data.items || data;
    const total = data.total != null ? data.total : mols.length;
    // Client-side sorting
    if (sortBy) {
      const [key, dir] = sortBy.split('_');
      const field = key === 'score' ? 'comprehensive_score' : key;
      mols.sort((a, b) => {
        const va = a[field] ?? (dir === 'asc' ? Infinity : -Infinity);
        const vb = b[field] ?? (dir === 'asc' ? Infinity : -Infinity);
        return dir === 'asc' ? va - vb : vb - va;
      });
    }
    $('#mol-count').textContent = `${mols.length} / ${total} 条结果`;
    const tbody = $('#mol-table tbody');
    tbody.innerHTML = mols.map(m => `<tr>
      <td>${m.id}</td>
      <td class="col-pocket" title="${escHtml(m.pocket_id)}">${escHtml(m.pocket_id||'')}</td>
      <td title="${escHtml(m.smiles)}">${escHtml((m.smiles||'').slice(0,30))}${(m.smiles||'').length>30?'...':''}</td>
      <td>${m.vina_score != null ? m.vina_score.toFixed(2) : '—'}</td>
      <td>${m.qed != null ? m.qed.toFixed(3) : '—'}</td>
      <td>${m.sa != null ? m.sa.toFixed(3) : '—'}</td>
      <td>${m.logp != null ? m.logp.toFixed(2) : '—'}</td>
      <td>${m.tpsa != null ? m.tpsa.toFixed(1) : '—'}</td>
      <td>${m.comprehensive_score != null ? m.comprehensive_score.toFixed(1) : '—'}</td>
      <td>${m.lipinski_pass != null ? m.lipinski_pass + '/5' : '—'}</td>
      <td>${m.lilly_passed != null ? (m.lilly_passed ? '✓' : '✗') : '—'}</td>
      <td>${m.lilly_demerit != null ? m.lilly_demerit : '—'}</td>
      <td>${m.conformer_energy != null ? m.conformer_energy.toFixed(1) : '—'}</td>
      <td>${m.molecule_stable != null ? (m.molecule_stable ? '✓' : '✗') : '—'}</td>
      <td>${m.tanimoto != null ? m.tanimoto.toFixed(3) : '—'}</td>
      <td>${m.sdf_path ? `<a href="/api/sdf/${encodeURIComponent(m.sdf_path)}" target="_blank" title="${escHtml(m.sdf_path)}" style="color:var(--primary);font-size:0.8rem;text-decoration:none">SDF</a>` : '—'}</td>
    </tr>`).join('');
  } catch (e) {
    $('#mol-count').textContent = `错误: ${e.message}`;
  }
}

// ── History ────────────────────────────────────────────────────────────────

async function refreshHistory() {
  const action = $('#hist-action').value;
  const params = action ? `?action=${action}` : '';
  try {
    const records = await api(`/api/history${params}`);
    const tbody = $('#hist-table tbody');
    tbody.innerHTML = records.map(r => `<tr>
      <td>${r.id}</td><td>${escHtml(r.action)}</td><td>${escHtml(r.user)}</td>
      <td>${fmtTime(r.created_at)}</td>
      <td title="${escHtml(JSON.stringify(r.details))}">${escHtml(JSON.stringify(r.details || '')).slice(0, 60)}</td>
    </tr>`).join('');
  } catch (e) {
    $('#hist-table tbody').innerHTML = `<tr><td colspan="5" class="status-failed">${e.message}</td></tr>`;
  }
}

async function refreshRuns() {
  const params = new URLSearchParams();
  const type = $('#runs-type').value;
  if (type) params.set('run_type', type);
  const status = $('#runs-status').value;
  if (status) params.set('status', status);
  try {
    const runs = await api(`/api/runs?${params}`);
    const tbody = $('#runs-table tbody');
    tbody.innerHTML = runs.map(r => `<tr data-run-id="${r.id}">
      <td><input type="checkbox" class="run-compare-cb" value="${r.id}"> ${r.id}</td>
      <td>${escHtml(r.run_type)}</td>
      <td><span class="${statusClass(r.status)}">${r.status}</span></td>
      <td>${fmtTime(r.created_at)}</td><td>${fmtTime(r.finished_at)}</td>
      <td>${r.progress != null ? Math.round(r.progress * 100) + '%' : '—'}</td>
      <td>
        ${r.has_log ? `<button class="btn btn-secondary btn-sm" onclick="viewLog(${r.id})">日志</button> ` : ''}
        <button class="btn btn-primary btn-sm" onclick="viewRunDetail(${r.id})">参数</button>
        <button class="btn btn-secondary btn-sm" onclick="rerunRun(${r.id})">重跑</button>
      </td>
    </tr>`).join('');
  } catch (e) {
    $('#runs-table tbody').innerHTML = `<tr><td colspan="7" class="status-failed">${e.message}</td></tr>`;
  }
}

async function viewLog(runId) {
  try {
    const data = await api(`/api/runs/${runId}/log`);
    const modal = $('#log-modal');
    const content = $('#log-modal-content');
    content.textContent = data.log || '暂无日志';
    modal.style.display = 'flex';
  } catch (e) {
    alert(`获取日志失败: ${e.message}`);
  }
}

function closeLogModal() {
  $('#log-modal').style.display = 'none';
}

async function viewRunDetail(runId) {
  try {
    const d = await api(`/api/runs/${runId}`);
    const modal = $('#log-modal');
    $('#log-modal-content').textContent = JSON.stringify(d.parameters, null, 2);
    modal.style.display = 'flex';
  } catch (e) { alert(e.message); }
}

async function rerunRun(runId) {
  if (!confirm(`重跑 Run #${runId}？`)) return;
  try {
    const r = await api(`/api/runs/${runId}/rerun`, { method: 'POST', body: JSON.stringify({}) });
    alert(`已提交: ${r.job_id}${r.status === 'queued' ? ' (排队中)' : ''}`);
    refreshRuns();
  } catch (e) { alert(e.message); }
}

async function compareSelectedRuns() {
  const ids = [...$$('.run-compare-cb:checked')].map(c => c.value);
  if (ids.length < 2) { alert('请至少选择 2 个 Run'); return; }
  try {
    const data = await api(`/api/runs/compare?ids=${ids.join(',')}`);
    const el = $('#compare-result');
    if (!el) return;
    el.innerHTML = data.map(r => `
      <div class="card" style="margin-bottom:0.8rem">
        <h3>Run #${r.id} — ${escHtml(r.run_type)} <span class="${statusClass(r.status)}">${r.status}</span></h3>
        <p>分子: ${r.molecule_count} | Vina min: ${r.vina_min ?? '—'} | Vina avg: ${r.vina_avg?.toFixed(2) ?? '—'} | QED avg: ${r.qed_avg?.toFixed(3) ?? '—'} | 综合分 avg: ${r.score_avg?.toFixed(1) ?? '—'}</p>
      </div>`).join('');
  } catch (e) { alert(e.message); }
}

async function browseFiles(root, targetInputId) {
  try {
    const data = await api(`/api/browse/files?root=${root}`);
    const pick = data.filter(f => f.type === 'file').slice(0, 20);
    const name = pick.length ? pick.map((f, i) => `${i + 1}. ${f.name}`).join('\n') : '无文件';
    const idx = prompt(`选择文件 (输入序号):\n${name}`);
    if (idx && pick[parseInt(idx) - 1]) {
      $(targetInputId).value = pick[parseInt(idx) - 1].path;
    }
  } catch (e) { alert(e.message); }
}

async function handleBatchGenerate(e) {
  e.preventDefault();
  const body = {
    start_id: parseInt($('#batch-start').value) || 0,
    end_id: parseInt($('#batch-end').value) || 0,
    batch_size: parseInt($('#batch-size').value) || 5,
    auto_evaluate: $('#batch-auto-eval').checked,
    sample_only: !$('#batch-auto-eval').checked,
  };
  const gpus = $('#batch-gpus').value.trim();
  if (gpus) body.gpus = gpus;
  try {
    const r = await api('/api/generate/batch', { method: 'POST', body: JSON.stringify(body) });
    $('#gen-status').innerHTML = `<span class="status-running">批量任务 ${r.job_id}</span>${r.status === 'queued' ? ' (排队)' : ''}`;
    currentJobId = r.job_id;
    const bar = $('#gen-progress-bar');
    bar.style.display = 'block';
    setProgressRunning(bar, true);
    startPolling();
  } catch (e) { $('#gen-status').innerHTML = `<span class="status-failed">${e.message}</span>`; }
}

// ── Config ─────────────────────────────────────────────────────────────────

async function loadConfig() {
  try {
    const cfg = await api('/api/config');
    $('#cfg-yaml').value = cfg.sampling_yml ? jsYamlDump(cfg.sampling_yml) : '# 加载失败';
    $('#cfg-runtime').textContent = JSON.stringify(cfg.runtime || {}, null, 2);
  } catch (e) {
    $('#cfg-status').textContent = `错误: ${e.message}`;
  }
}

async function saveConfig() {
  try {
    // Simple YAML validation: just send as text to backend
    const text = $('#cfg-yaml').value;
    await api('/api/config', {
      method: 'PUT',
      body: JSON.stringify({ sampling_yml_text: text }),
    });
    $('#cfg-status').textContent = '已保存';
  } catch (e) {
    $('#cfg-status').textContent = `保存失败: ${e.message}`;
  }
}

async function loadGPU() {
  try {
    const gpus = await api('/api/gpus');
    if (gpus.length) {
      $('#cfg-gpu').innerHTML = '<table><thead><tr><th>GPU</th><th>名称</th><th>显存</th><th>利用率</th></tr></thead><tbody>' +
        gpus.map(g => `<tr><td>${g.index}</td><td>${g.name}</td><td>${g.memory_used_mb}/${g.memory_total_mb} MB</td><td>${g.utilization_pct}%</td></tr>`).join('') + '</tbody></table>';
    }
  } catch (e) { /* ignore */ }
}

// Minimal YAML-like dump (for display only)
function jsYamlDump(obj, indent = 0) {
  if (obj === null || obj === undefined) return 'null';
  if (typeof obj === 'string') return obj;
  if (typeof obj === 'number' || typeof obj === 'boolean') return String(obj);
  if (Array.isArray(obj)) {
    return obj.map(item => '  '.repeat(indent) + '- ' + jsYamlDump(item, indent + 1)).join('\n');
  }
  if (typeof obj === 'object') {
    return Object.entries(obj).map(([k, v]) => {
      if (typeof v === 'object' && v !== null) {
        return '  '.repeat(indent) + k + ':\n' + jsYamlDump(v, indent + 1);
      }
      return '  '.repeat(indent) + k + ': ' + jsYamlDump(v, indent + 1);
    }).join('\n');
  }
  return String(obj);
}

// ── Pocket Evaluation ──────────────────────────────────────────────────

let peJobId = null;
let pePollTimer = null;

function togglePocketEvalMode() {
  const mode = $('#pe-mode').value;
  const ptGroup = $('#pe-pt-group');
  const genOpts = $('#pe-gen-options');
  if (mode === 'generate') {
    ptGroup.style.display = 'none';
    genOpts.style.display = 'block';
  } else {
    ptGroup.style.display = 'block';
    genOpts.style.display = 'none';
  }
}

async function handlePocketEval(e) {
  e.preventDefault();
  const protein = $('#pe-protein').value.trim();
  const ligand = $('#pe-ligand').value.trim();
  if (!protein) { alert(t('pe.protein')); return; }
  if (!ligand) { alert(t('pe.ligand')); return; }

  const mode = $('#pe-mode').value;
  const body = {
    protein_path: protein,
    ligand_path: ligand,
    generate_first: mode === 'generate',
  };

  if (mode === 'existing') {
    const pt = $('#pe-pt').value.trim();
    if (!pt) { alert(t('pe.pt_path')); return; }
    body.pt_path = pt;
  } else {
    const gpu = $('#pe-gpu').value;
    if (gpu) body.gpu_id = parseInt(gpu);
    body.batch_size = parseInt($('#pe-batch').value) || 5;
    body.pocket_radius = parseFloat($('#pe-radius').value) || 10.0;
  }

  try {
    const result = await api('/api/pocket-eval', { method: 'POST', body: JSON.stringify(body) });
    peJobId = result.job_id;
    $('#pe-status').innerHTML = `<span class="status-running">${t('pe.progress')}</span> — Job: <code>${result.job_id}</code>, GPU: ${result.gpu_id}`;
    $('#pe-cancel-btn').disabled = false;
    const peBar = $('#pe-progress-bar');
    peBar.style.display = 'block';
    setProgressRunning(peBar, true);
    $('#pe-scores').innerHTML = '';
    $('#pe-gallery').innerHTML = '';
    startPePolling();
  } catch (e) {
    $('#pe-status').innerHTML = `<span class="status-failed">Error: ${e.message}</span>`;
  }
}

async function handlePocketEvalCancel() {
  if (!peJobId) return;
  try {
    await api(`/api/jobs/${peJobId}`, { method: 'DELETE' });
    $('#pe-status').innerHTML = '<span class="status-pending">Cancelled</span>';
    stopPePolling();
  } catch (e) {
    $('#pe-status').innerHTML = `<span class="status-failed">Cancel failed: ${e.message}</span>`;
  }
}

function startPePolling() {
  stopPePolling();
  pePollTimer = setInterval(pollPocketEval, 3000);
}

function stopPePolling() {
  if (pePollTimer) { clearInterval(pePollTimer); pePollTimer = null; }
}

async function pollPocketEval() {
  if (!peJobId) return;
  try {
    const jobs = await api('/api/jobs');
    const job = jobs.find(j => j.job_id === peJobId);
    if (!job) return;

    const pct = Math.round((job.progress || 0) * 100);
    $('#pe-progress-fill').style.width = pct + '%';
    setProgressRunning($('#pe-progress-bar'), job.status === 'running' || job.status === 'pending');
    $('#pe-status').innerHTML = `<span class="${statusClass(job.status)}">${job.status.toUpperCase()}</span> — ${pct}%`;

    const log = (job.log_tail || []).slice(-25).join('\n');
    $('#pe-log').textContent = log;

    if (job.status === 'completed') {
      stopPePolling();
      setProgressRunning($('#pe-progress-bar'), false);
      $('#pe-cancel-btn').disabled = true;
      loadPocketEvalResults(peJobId);
    } else if (job.status === 'failed' || job.status === 'cancelled') {
      stopPePolling();
      setProgressRunning($('#pe-progress-bar'), false);
      $('#pe-cancel-btn').disabled = true;
    }
  } catch (e) { /* ignore poll errors */ }
}

async function loadPocketEvalResults(jobId) {
  try {
    const data = await api(`/api/pocket-eval/results/${jobId}`);
    renderPeScores(data.scores);
    renderPeGallery(data.images, data.dir_name);
  } catch (e) {
    $('#pe-scores').innerHTML = `<span class="status-failed">${e.message}</span>`;
  }
}

function renderPeScores(scores) {
  if (!scores || Object.keys(scores).length === 0) {
    $('#pe-scores').innerHTML = '<p style="color:var(--text-secondary)">No scores available</p>';
    return;
  }
  const dims = [
    { key: 'score_a', label: 'A · Vina' },
    { key: 'score_b', label: 'B · Clustering' },
    { key: 'score_c', label: 'C · LE' },
    { key: 'score_d', label: 'D · Druglike' },
    { key: 'score_e', label: 'E · Complete' },
    { key: 'score_f', label: 'F · Unique' },
    { key: 'score_g', label: 'G · Size' },
    { key: 'score_h', label: 'H · Volume' },
  ];
  let html = '';
  for (const d of dims) {
    const raw = scores[d.key];
    const val = raw === 'N/A' ? '—' : raw;
    const numVal = parseFloat(val);
    let cls = 'pe-score-na';
    let label = 'N/A';
    if (!isNaN(numVal)) {
      if (numVal >= 0.6) { cls = 'pe-score-high'; label = 'HIGH'; }
      else if (numVal >= 0.3) { cls = 'pe-score-medium'; label = 'MEDIUM'; }
      else { cls = 'pe-score-low'; label = 'LOW'; }
    }
    html += `<div class="pe-score-card ${cls}">
      <div class="pe-dim">${d.label}</div>
      <div class="pe-val">${val}</div>
      <div class="pe-label">${label}</div>
    </div>`;
  }
  // Overall
  const ov = scores.overall_score || '—';
  const ovLabel = scores.overall_label || '';
  let ovCls = 'pe-score-na';
  const ovNum = parseFloat(ov);
  if (!isNaN(ovNum)) {
    if (ovNum >= 0.6) ovCls = 'pe-score-high';
    else if (ovNum >= 0.3) ovCls = 'pe-score-medium';
    else ovCls = 'pe-score-low';
  }
  html += `<div class="pe-score-card ${ovCls}" style="grid-column: span 2">
    <div class="pe-dim">OVERALL</div>
    <div class="pe-val">${ov}</div>
    <div class="pe-label">${ovLabel.toUpperCase()}</div>
  </div>`;
  $('#pe-scores').innerHTML = html;
}

function renderPeGallery(images, dirName) {
  if (!images || images.length === 0) {
    $('#pe-gallery').innerHTML = `<p style="color:var(--text-secondary)">${t('pe.no_images')}</p>`;
    return;
  }
  dirName = dirName || '';
  const html = images.map(img => {
    const src = `/api/pocket-eval/images/${dirName}/${img}`;
    const label = img.replace('.png', '').replace(/_/g, ' ');
    return `<div class="pe-gallery-item">
      <img src="${src}" alt="${escHtml(img)}" onclick="openPeLightbox('${src}')" loading="lazy">
      <div class="pe-img-label">${escHtml(label)}</div>
    </div>`;
  }).join('');
  $('#pe-gallery').innerHTML = html;
}

function openPeLightbox(src) {
  const overlay = document.createElement('div');
  overlay.className = 'pe-lightbox';
  overlay.onclick = () => overlay.remove();
  const img = document.createElement('img');
  img.src = src;
  overlay.appendChild(img);
  document.body.appendChild(overlay);
}

// ── Optimization Tab ────────────────────────────────────────────────────

let optJobId = null;
let optPollTimer = null;

async function handleOptimization(e) {
  e.preventDefault();
  const body = {
    data_id: parseInt($('#opt-dataid').value) || 0,
    molecule_path: $('#opt-molecule').value || null,
    gpu_id: $('#opt-gpu').value ? parseInt($('#opt-gpu').value) : null,
    auto_evaluate: $('#opt-auto-eval').checked,
    auto_extract: $('#opt-auto-extract').checked,
    remove_fragments: $('#opt-remove-fragments').checked,
    num_samples: parseInt($('#opt-num-samples').value) || 2,
    start_t: parseInt($('#opt-start-t').value) || 16,
    stride: parseInt($('#opt-stride').value) || 2,
    step_size: parseFloat($('#opt-step-size').value) || 0.2,
    cycles: parseInt($('#opt-cycles').value) || 5,
    schedule: $('#opt-schedule').value,
    min_qed: parseFloat($('#opt-min-qed').value) || 0.3,
    min_sa: parseFloat($('#opt-min-sa').value) || 0.3,
    max_survivors_per_cycle: parseInt($('#opt-max-survivors').value) || 10,
  };
  try {
    const data = await api('/api/optimization', { method: 'POST', body: JSON.stringify(body) });
    optJobId = data.job_id;
    $('#opt-status').innerHTML = `<span class="status-running">${t('hist.st_running')} (${data.job_id})</span>`;
    const optBar = $('#opt-progress-bar');
    optBar.style.display = 'block';
    setProgressRunning(optBar, true);
    $('#opt-cancel-btn').disabled = false;
    $('#opt-log').textContent = '';
    startOptPolling();
  } catch (err) {
    $('#opt-status').innerHTML = `<span class="status-failed">${escHtml(err.message)}</span>`;
  }
}

async function handleOptCancel() {
  if (!optJobId) return;
  try { await api(`/api/jobs/${optJobId}`, { method: 'DELETE' }); } catch {}
  stopOptPolling();
  $('#opt-cancel-btn').disabled = true;
  $('#opt-status').innerHTML = `<span class="status-failed">Cancelled</span>`;
}

function startOptPolling() {
  stopOptPolling();
  optPollTimer = setInterval(pollOptimization, 2000);
}
function stopOptPolling() {
  if (optPollTimer) { clearInterval(optPollTimer); optPollTimer = null; }
}

async function pollOptimization() {
  if (!optJobId) return;
  try {
    const data = await api(`/api/optimization/${optJobId}`);
    const pct = Math.round((data.progress || 0) * 100);
    $('#opt-progress-fill').style.width = pct + '%';
    setProgressRunning($('#opt-progress-bar'), !['completed', 'failed', 'cancelled'].includes(data.status));
    if (data.log_tail && data.log_tail.length) {
      $('#opt-log').textContent = data.log_tail.join('\n');
      $('#opt-log').scrollTop = $('#opt-log').scrollHeight;
    }
    if (data.status === 'completed') {
      stopOptPolling();
      setProgressRunning($('#opt-progress-bar'), false);
      $('#opt-cancel-btn').disabled = true;
      $('#opt-status').innerHTML = `<span class="status-completed flash">Completed (${data.run_id})</span>`;
    } else if (data.status === 'failed') {
      stopOptPolling();
      setProgressRunning($('#opt-progress-bar'), false);
      $('#opt-cancel-btn').disabled = true;
      $('#opt-status').innerHTML = `<span class="status-failed">Failed: ${escHtml(data.error || '')}</span>`;
    } else if (data.status === 'cancelled') {
      stopOptPolling();
      setProgressRunning($('#opt-progress-bar'), false);
      $('#opt-cancel-btn').disabled = true;
      $('#opt-status').innerHTML = `<span class="status-failed">Cancelled</span>`;
    }
  } catch {}
}

// ── Scaffold Tab ────────────────────────────────────────────────────────

let scJobId = null;
let scPollTimer = null;

function toggleScaffoldMode() {
  const mode = $('#sc-scaffold-mode').value;
  const isGrow = mode === 'grow';
  const isDynamic = mode === 'dynamic_locked';
  const isPrudent = mode === 'prudent';
  const isGrowLike = isGrow || isDynamic || isPrudent;

  $('#sc-grow-section').style.display = isGrowLike ? '' : 'none';
  $('#sc-grow-diffusion').style.display = isGrow ? '' : 'none';
  $('#sc-evolve-section').style.display = mode === 'evolve' ? '' : 'none';
  $('#sc-prudent-section').style.display = isPrudent ? '' : 'none';

  const summary = $('#sc-grow-summary');
  if (summary) {
    summary.textContent = isGrow ? t('sc.grow_title') : t('sc.sample_title');
  }

  const note = $('#sc-mode-note');
  if (note) {
    if (isDynamic) {
      note.textContent = t('sc.note_dynamic_locked');
      note.style.display = '';
    } else if (isPrudent) {
      note.textContent = t('sc.note_prudent');
      note.style.display = '';
    } else {
      note.style.display = 'none';
    }
  }

  const fixPos = $('#sc-fix-pos');
  const fixType = $('#sc-fix-type');
  const lockPosType = isDynamic || isPrudent;
  if (fixPos) {
    fixPos.checked = true;
    fixPos.disabled = lockPosType;
  }
  if (fixType) {
    fixType.checked = !lockPosType;
    fixType.disabled = lockPosType;
  }
}

function toggleScaffoldSource() {
  const src = $('#sc-source').value;
  $('#sc-atom-indices-group').style.display = src === 'atom_indices' ? '' : 'none';
  $('#sc-smarts-group').style.display = src === 'smarts' ? '' : 'none';
}

function toggleGateMode() {
  const mode = ($('#sc-gate-mode') || {}).value || 'adaptive';
  const adaptivePanel = $('#sc-adaptive-panel');
  const manualPanel = $('#sc-manual-gates-panel');
  if (adaptivePanel) adaptivePanel.style.display = mode === 'adaptive' ? '' : 'none';
  if (manualPanel) manualPanel.style.display = mode === 'manual' ? '' : 'none';
}

async function previewAdaptiveGates() {
  const lig = ($('#sc-molecule').value || '').trim();
  const pre = $('#sc-adaptive-preview');
  if (!lig) {
    pre.textContent = '请先填写参考配体 SDF 路径';
    return;
  }
  pre.textContent = 'loading...';
  try {
    const q = new URLSearchParams({ ligand_path: lig });
    const data = await api('/api/adaptive_gates/preview?' + q.toString());
    pre.textContent = JSON.stringify({
      ligand: data.ligand,
      gates: data.gates,
      rationale: data.rationale,
    }, null, 2);
  } catch (err) {
    pre.textContent = String(err);
  }
}

async function handleScaffold(e) {
  e.preventDefault();
  const body = {
    data_id: parseInt($('#sc-dataid').value) || 0,
    molecule_path: $('#sc-molecule').value || null,
    gpu_id: $('#sc-gpu').value ? parseInt($('#sc-gpu').value) : null,
    auto_evaluate: $('#sc-auto-eval').checked,
    auto_extract: $('#sc-auto-extract').checked,
    remove_fragments: $('#sc-remove-fragments').checked,
    scaffold_mode: $('#sc-scaffold-mode').value,
    scaffold_source: $('#sc-source').value,
    scaffold_atom_indices: $('#sc-atom-indices').value || null,
    scaffold_smarts: $('#sc-smarts').value || null,
    fix_scaffold_pos: $('#sc-fix-pos').checked,
    fix_scaffold_type: $('#sc-fix-type').checked,
    scaffold_enable_refine: $('#sc-enable-refine').checked,
    qed_weight: parseFloat($('#sc-qed-w').value) || 1.0,
    sa_weight: parseFloat($('#sc-sa-w').value) || 1.0,
    diversity_weight: parseFloat($('#sc-div-w').value) || 0.5,
    min_qed: parseFloat($('#sc-min-qed').value) || 0.15,
    min_sa: parseFloat($('#sc-min-sa').value) || 0.15,
    max_tanimoto: parseFloat($('#sc-max-tanimoto').value) || 0.9,
    diversity_filter_enable: $('#sc-div-filter').checked,
    grow_num_samples: parseInt($('#sc-grow-samples').value) || 200,
    grow_start_t: parseInt($('#sc-grow-start-t').value) || 450,
    grow_stride: parseInt($('#sc-grow-stride').value) || 15,
    grow_step_size: parseFloat($('#sc-grow-step').value) || 0.33,
    grow_lambda_a: parseInt($('#sc-grow-lambda-a').value) || 40,
    grow_lambda_b: parseInt($('#sc-grow-lambda-b').value) || 5,
    grow_n_extra_mode: $('#sc-n-extra-mode').value,
    grow_n_extra_fixed: parseInt($('#sc-n-extra-fixed').value) || 8,
    grow_n_extra_min: parseInt($('#sc-n-extra-min').value) || 3,
    grow_n_extra_max: parseInt($('#sc-n-extra-max').value) || 20,
    evolve_population_size: parseInt($('#sc-evo-pop').value) || 50,
    evolve_n_generations: parseInt($('#sc-evo-gen').value) || 5,
    evolve_children_per_parent: parseInt($('#sc-evo-children').value) || 10,
    evolve_start_t_high: parseInt($('#sc-evo-t-high').value) || 200,
    evolve_start_t_low: parseInt($('#sc-evo-t-low').value) || 30,
    evolve_stride: parseInt($('#sc-evo-stride').value) || 2,
    evolve_step_size: parseFloat($('#sc-evo-step').value) || 0.2,
    evolve_lambda_a: parseInt($('#sc-evo-lambda-a').value) || 10,
    evolve_lambda_b: parseInt($('#sc-evo-lambda-b').value) || 1,
    prudent_n_generations: parseInt($('#sc-prudent-gen').value) || 5,
    prudent_n_chains_per_seed: parseInt($('#sc-prudent-chains').value) || 4,
    prudent_advance_top_k: parseInt($('#sc-prudent-top-k').value) || 2,
    prudent_renoise_t: parseInt($('#sc-prudent-renoise-t').value) || 600,
    prudent_qed_weight: parseFloat($('#sc-prudent-qed-w').value) || 0.2,
    prudent_sa_weight: parseFloat($('#sc-prudent-sa-w').value) || 0.2,
    prudent_vina_weight: parseFloat($('#sc-prudent-vina-w').value) || 0.6,
    prudent_vina_exhaustiveness: parseInt($('#sc-prudent-vina-exh').value) || 8,
    prudent_min_qed_for_docking: parseFloat($('#sc-prudent-min-qed-dock').value) || 0.15,
    prudent_min_sa_for_docking: parseFloat($('#sc-prudent-min-sa-dock').value) || 0.15,
    gate_mode: ($('#sc-gate-mode') || {}).value || 'adaptive',
    adaptive_gates_enable: (($('#sc-gate-mode') || {}).value || 'adaptive') === 'adaptive',
    manual_max_logp: parseFloat(($('#sc-manual-max-logp') || {}).value) || 5.0,
    manual_max_molwt: parseFloat(($('#sc-manual-max-molwt') || {}).value) || 850,
    manual_max_heavy_atoms: parseInt(($('#sc-manual-max-heavy') || {}).value, 10) || 60,
    manual_max_rings: parseInt(($('#sc-manual-max-rings') || {}).value, 10) || 7,
    // 任务原始参考配体：UI 的 molecule 即配体；同时写入 ligand_path 供后端自适应门控
    ligand_path: ($('#sc-molecule').value || '').trim() || null,
  };
  if (!body.molecule_path && body.ligand_path) {
    body.molecule_path = body.ligand_path;
  }
  try {
    const data = await api('/api/scaffold', { method: 'POST', body: JSON.stringify(body) });
    scJobId = data.job_id;
    $('#sc-status').innerHTML = `<span class="status-running">${t('hist.st_running')} (${data.job_id})</span>`;
    const scBar = $('#sc-progress-bar');
    scBar.style.display = 'block';
    setProgressRunning(scBar, true);
    $('#sc-cancel-btn').disabled = false;
    $('#sc-log').textContent = '';
    startScPolling();
  } catch (err) {
    $('#sc-status').innerHTML = `<span class="status-failed">${escHtml(err.message)}</span>`;
  }
}

async function handleScaffoldCancel() {
  if (!scJobId) return;
  try { await api(`/api/jobs/${scJobId}`, { method: 'DELETE' }); } catch {}
  stopScPolling();
  $('#sc-cancel-btn').disabled = true;
  $('#sc-status').innerHTML = `<span class="status-failed">Cancelled</span>`;
}

function startScPolling() {
  stopScPolling();
  scPollTimer = setInterval(pollScaffold, 2000);
}
function stopScPolling() {
  if (scPollTimer) { clearInterval(scPollTimer); scPollTimer = null; }
}

async function pollScaffold() {
  if (!scJobId) return;
  try {
    const data = await api(`/api/scaffold/${scJobId}`);
    const pct = Math.round((data.progress || 0) * 100);
    $('#sc-progress-fill').style.width = pct + '%';
    setProgressRunning($('#sc-progress-bar'), !['completed', 'failed', 'cancelled'].includes(data.status));
    if (data.log_tail && data.log_tail.length) {
      $('#sc-log').textContent = data.log_tail.join('\n');
      $('#sc-log').scrollTop = $('#sc-log').scrollHeight;
    }
    if (data.status === 'completed') {
      stopScPolling();
      setProgressRunning($('#sc-progress-bar'), false);
      $('#sc-cancel-btn').disabled = true;
      $('#sc-status').innerHTML = `<span class="status-completed flash">Completed (${data.run_id})</span>`;
    } else if (data.status === 'failed') {
      stopScPolling();
      setProgressRunning($('#sc-progress-bar'), false);
      $('#sc-cancel-btn').disabled = true;
      $('#sc-status').innerHTML = `<span class="status-failed">Failed: ${escHtml(data.error || '')}</span>`;
    } else if (data.status === 'cancelled') {
      stopScPolling();
      setProgressRunning($('#sc-progress-bar'), false);
      $('#sc-cancel-btn').disabled = true;
      $('#sc-status').innerHTML = `<span class="status-failed">Cancelled</span>`;
    }
  } catch {}
}

// ── Scaffold Cascade Tab ──────────────────────────────────────────────────

let cascadeJobId = null;
let cascadePollTimer = null;

async function handleCascade(e) {
  e.preventDefault();
  const body = {
    data_id: parseInt($('#cascade-dataid').value) || 0,
    gpu_id: $('#cascade-gpu').value ? parseInt($('#cascade-gpu').value) : null,
    samples_per_round: parseInt($('#cascade-samples').value) || 5,
    rounds: parseInt($('#cascade-rounds').value) || 2,
    auto_evaluate: $('#cascade-auto-eval').checked,
    auto_extract: $('#cascade-auto-extract').checked,
    remove_fragments: $('#cascade-remove-fragments').checked,
  };
  try {
    const data = await api('/api/scaffold-cascade', { method: 'POST', body: JSON.stringify(body) });
    cascadeJobId = data.job_id;
    $('#cascade-status').innerHTML = `<span class="status-running">${t('hist.st_running')} (${data.job_id})</span>`;
    const cascadeBar = $('#cascade-progress-bar');
    cascadeBar.style.display = 'block';
    setProgressRunning(cascadeBar, true);
    $('#cascade-cancel-btn').disabled = false;
    $('#cascade-log').textContent = '';
    startCascadePolling();
  } catch (err) {
    $('#cascade-status').innerHTML = `<span class="status-failed">${escHtml(err.message)}</span>`;
  }
}

async function handleCascadeCancel() {
  if (!cascadeJobId) return;
  try { await api(`/api/jobs/${cascadeJobId}`, { method: 'DELETE' }); } catch {}
  stopCascadePolling();
  $('#cascade-cancel-btn').disabled = true;
  $('#cascade-status').innerHTML = `<span class="status-failed">Cancelled</span>`;
}

function startCascadePolling() {
  stopCascadePolling();
  cascadePollTimer = setInterval(pollCascade, 2000);
}
function stopCascadePolling() {
  if (cascadePollTimer) { clearInterval(cascadePollTimer); cascadePollTimer = null; }
}

async function pollCascade() {
  if (!cascadeJobId) return;
  try {
    const data = await api(`/api/scaffold-cascade/${cascadeJobId}`);
    const pct = Math.round((data.progress || 0) * 100);
    $('#cascade-progress-fill').style.width = pct + '%';
    setProgressRunning($('#cascade-progress-bar'), !['completed', 'failed', 'cancelled'].includes(data.status));
    if (data.log_tail && data.log_tail.length) {
      $('#cascade-log').textContent = data.log_tail.join('\n');
      $('#cascade-log').scrollTop = $('#cascade-log').scrollHeight;
    }
    if (data.status === 'completed') {
      stopCascadePolling();
      setProgressRunning($('#cascade-progress-bar'), false);
      $('#cascade-cancel-btn').disabled = true;
      $('#cascade-status').innerHTML = `<span class="status-completed flash">Completed (${data.run_id})</span>`;
    } else if (data.status === 'failed') {
      stopCascadePolling();
      setProgressRunning($('#cascade-progress-bar'), false);
      $('#cascade-cancel-btn').disabled = true;
      $('#cascade-status').innerHTML = `<span class="status-failed">Failed: ${escHtml(data.error || '')}</span>`;
    } else if (data.status === 'cancelled') {
      stopCascadePolling();
      setProgressRunning($('#cascade-progress-bar'), false);
      $('#cascade-cancel-btn').disabled = true;
      $('#cascade-status').innerHTML = `<span class="status-failed">Cancelled</span>`;
    }
  } catch {}
}
