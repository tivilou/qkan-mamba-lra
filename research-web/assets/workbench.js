'use strict';
const content = document.querySelector('#content');
const element = (tag, text, cls) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (cls) node.className = cls; return node; };
const tooltip = document.querySelector('#tooltip');
function help(text) {
  const button = element('button', '?', 'help'); button.type = 'button'; button.dataset.tooltip = text; button.setAttribute('aria-label', text);
  const show = () => {
    tooltip.textContent = text; tooltip.hidden = false;
    const rect = button.getBoundingClientRect();
    tooltip.style.left = Math.max(12, Math.min(rect.left, innerWidth - tooltip.offsetWidth - 12)) + 'px';
    tooltip.style.top = (rect.bottom + tooltip.offsetHeight + 14 < innerHeight ? rect.bottom + 8 : Math.max(10, rect.top - tooltip.offsetHeight - 8)) + 'px';
  };
  button.addEventListener('pointerenter', show); button.addEventListener('focus', show);
  button.addEventListener('pointerleave', () => { tooltip.hidden = true; }); button.addEventListener('blur', () => { tooltip.hidden = true; });
  return button;
}
function table(title, columns, rows, emptyText) {
  const section = element('section', undefined, 'table-section'); section.append(element('h2', title));
  const top = element('div', undefined, 'top-scroll'); const topWidth = element('div'); topWidth.style.height = '1px'; top.append(topWidth);
  const wrap = element('div', undefined, 'table-wrap'); wrap.tabIndex = 0; wrap.setAttribute('aria-label', title + ' 可滚动表格');
  const grid = element('table'); const group = element('colgroup');
  for (const col of columns) { const node = element('col'); node.style.width = col.width + 'px'; group.append(node); }
  const head = element('thead'); const headRow = element('tr'); const body = element('tbody');
  let sortId = null, direction = 1;
  const value = (row, col) => col.get ? col.get(row) : row[col.id];
  const paint = () => {
    body.replaceChildren();
    const sorted = [...rows];
    if (sortId) {
      const col = columns.find(c => c.id === sortId);
      sorted.sort((a, b) => {
        let av = value(a, col), bv = value(b, col);
        if (av == null && bv == null) return 0;
        if (av == null) return 1; if (bv == null) return -1;
        if (col.type === 'metric') return direction * (parseFloat(av) - parseFloat(bv));
        return direction * String(av).localeCompare(String(bv), 'zh-CN', {numeric: true});
      });
    }
    if (!sorted.length) { const row = element('tr'); const cell = element('td', emptyText, 'empty'); cell.colSpan = columns.length; row.append(cell); body.append(row); }
    sorted.forEach(row => {
      const tr = element('tr'); tr.dataset.rowId = row.id || row.idea_id;
      columns.forEach((col, index) => { const td = element('td', value(row, col) ?? '未提供'); td.dataset.column = col.id;
        if (!index) td.classList.add('frozen'); if (col.type === 'metric') td.classList.add('metric'); tr.append(td); });
      body.append(tr);
    });
  };
  columns.forEach((col, index) => {
    const th = element('th'); th.scope = 'col'; th.dataset.column = col.id; if (!index) th.className = 'frozen';
    const sort = element('button', col.label, 'sort'); sort.type = 'button'; sort.setAttribute('aria-label', '按' + col.label + '排序');
    sort.addEventListener('click', () => { direction = sortId === col.id ? -direction : 1; sortId = col.id;
      headRow.querySelectorAll('th').forEach(h => h.removeAttribute('aria-sort')); th.setAttribute('aria-sort', direction === 1 ? 'ascending' : 'descending'); paint(); });
    th.append(sort, help(col.help)); headRow.append(th);
  });
  head.append(headRow); grid.append(group, head, body); wrap.append(grid); section.append(top, wrap);
  top.addEventListener('scroll', () => { wrap.scrollLeft = top.scrollLeft; });
  wrap.addEventListener('scroll', () => { top.scrollLeft = wrap.scrollLeft; tooltip.hidden = true; });
  new ResizeObserver(() => { topWidth.style.width = grid.scrollWidth + 'px'; }).observe(wrap);
  paint(); return section;
}
function reader(documents) {
  const outer = element('section', undefined, 'reader'); const index = element('nav', undefined, 'reader-index'); index.setAttribute('aria-label', '文档索引');
  const detail = element('article', undefined, 'reader-detail');
  const choose = (doc, button) => {
    index.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    detail.replaceChildren(element('h2', doc.title), element('p', doc.source, 'source'), element('pre', doc.text)); detail.scrollTop = 0;
  };
  documents.forEach((doc, n) => { const button = element('button', doc.title); button.addEventListener('click', () => choose(doc, button)); index.append(button); if (!n) choose(doc, button); });
  outer.append(index, detail); return outer;
}
function results(bundle) {
  content.append(element('p', '历史初测：当前可取得的是 GitHub 报告，原运行 JSON、权重和轨迹未找回。没有新增 QKAN-Mamba 正式比较；无已登记的量子基线结果。', 'notice'));
  const filters = element('div', undefined, 'filters'); const label = element('label', '数据集 '); const select = element('select'); select.setAttribute('aria-label', '筛选数据集');
  const all = element('option', '全部数据集'); all.value = ''; select.append(all);
  bundle.datasets.forEach(d => { const option = element('option', d.name); option.value = d.id; select.append(option); }); label.append(select); filters.append(label); content.append(filters);
  const tables = element('div'); content.append(tables);
  const columns = [
    {id:'model',label:'模型',width:165,help:'运行所使用的模型名称。当前仅有纯 Mamba 历史初测。'},
    {id:'family',label:'类型',width:115,help:'classical、quantum 或 hybrid 模型类型；类型本身不表示优势。'},
    {id:'dataset_id',label:'数据集',width:130,help:'任务命名空间；不同数据处理版本不能直接混合比较。'},
    {id:'status',label:'状态',width:145,help:'historical 表示旧初测，not_completed 表示尚无完成结果。'},
    {id:'test_accuracy',label:'Test Acc ↑',width:145,type:'metric',get:r=>r.metrics.test_accuracy,help:'报告所载测试集分类准确率，单位百分比，越高越好；未完成为未提供。'},
    {id:'seed_protocol',label:'运行协议',width:310,help:'报告可证实的轮数、种子及原件可得性；未提供不做推断。'},
    {id:'provenance',label:'证据来源',width:410,get:r=>r.provenance.source_ref+' · SHA-256 '+r.provenance.source_sha256,help:'原始报告路径及其完整 SHA-256；仅证明报告来源，不固定实际执行代码版本。'},
    {id:'claim_ceiling',label:'结论边界',width:320,help:'当前证据允许表达的最高结论，不能由单次初测推导机制或量子优势。'}];
  const render = () => {
    tables.replaceChildren();
    tables.append(table('01 / Idea', [
      {id:'title',label:'研究方案',width:240,help:'独立登记的候选或已测试 Idea，默认按加入顺序。'},
      {id:'status',label:'状态',width:160,help:'计划、预检、正式或失败状态；没有结果时不填造数值。'},
      {id:'evidence_tier',label:'证据',width:220,help:'经审计的证据状态；当前没有正式 Idea 结果。'}
    ], bundle.datasets.filter(d=>!select.value || d.id===select.value).flatMap(d=>d.rows), '暂无已登记的正式 Idea 结果。QKAN 门控仍是待验证候选。'));
    tables.append(table('02 / Baseline', columns, bundle.baselines.filter(r=>!select.value || r.dataset_id===select.value), '当前筛选下没有基线记录。'));
  }; select.addEventListener('change', render); render();
  const section = element('section', undefined, 'table-section'); section.id = 'case-study'; section.append(element('h2', '03 / Case Study'), element('p', 'unavailable · trace_incomplete · provenance_incomplete', 'notice'), element('p', bundle.extensions.find(e=>e.id==='case-study').data.reason));
  section.append(element('p', '预先固定样本、真实采集关键表示、通过结构与语义及同样本重放检查后，才展示新实验轨迹。历史缺失内容不会补造。', 'small'));
  const report = element('a', '查看原始历史报告（Markdown）'); report.href = 'historical-report.md'; section.append(report); content.append(section);
}
(async () => {
  const response = await fetch('data.json'); if (!response.ok) throw new Error('missing data'); const data = await response.json(); content.replaceChildren();
  const app = document.querySelector('main').dataset.app;
  if (app === 'experiment-results') return results(data);
  if (app === 'plan') {
    content.append(element('p', `${data.plan.plan_id} v${data.plan.plan_version} · ${data.plan.status} · 当前步骤 ${data.plan.focus_step_id}`, 'notice'));
    content.append(element('p', '下一步：' + data.plan.next_action));
    content.append(table('完整研究路线', [
      {id:'id',label:'步骤',width:85,help:'稳定步骤编号，跨会话保留。'}, {id:'status',label:'状态',width:105,help:'done已完成；active当前；todo待执行；blocked有明确阻塞。'},
      {id:'title',label:'工作包',width:260,help:'此步骤交付的研究或协作工作。'}, {id:'depends_on',label:'依赖',width:150,get:r=>r.depends_on.join(', ')||'无',help:'完成该步骤之前需满足的上游步骤。'},
      {id:'acceptance_checks',label:'完成条件',width:430,get:r=>r.acceptance_checks.join('；'),help:'只有这些条件有证据支持时才能标记done。'}, {id:'owner',label:'执行方',width:140,help:'我们开发分析，合作者执行正式训练。'}
    ], data.plan.steps, '没有计划步骤')); return;
  }
  content.append(reader(data.documents));
})().catch(() => { content.replaceChildren(element('p', '证据投影不可用，请刷新来源记录后重试。', 'notice')); });
