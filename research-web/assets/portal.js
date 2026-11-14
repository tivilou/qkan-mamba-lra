'use strict';
(async () => {
  const response = await fetch('/registry.json');
  if (!response.ok) throw new Error('registry unavailable');
  const registry = await response.json();
  const cards = document.querySelector('#cards');
  function render() {
    const query = document.querySelector('#search').value.trim().toLowerCase();
    cards.replaceChildren();
    registry.apps.forEach((app, index) => {
      if (![app.title, app.summary, app.skill_id, app.instance_id, ...(app.meta || [])].join(' ').toLowerCase().includes(query)) return;
      const card = document.createElement('article'); card.className = 'card';
      const meta = document.createElement('div'); meta.className = 'card-meta';
      const order = document.createElement('span'); order.textContent = String(index + 1).padStart(2, '0');
      const badge = document.createElement('span'); badge.className = 'badge'; badge.textContent = app.status;
      meta.append(order, badge);
      const heading = document.createElement('h3'); const link = document.createElement('a');
      link.href = app.entry; link.textContent = app.title; heading.append(link);
      const summary = document.createElement('p'); summary.textContent = app.summary;
      const owner = document.createElement('code'); owner.textContent = app.skill_id;
      card.append(meta, heading, summary, owner); cards.append(card);
    });
    if (!cards.children.length) cards.textContent = '没有匹配的工作台。';
  }
  document.querySelector('#search').addEventListener('input', render);
  document.querySelector('#registry-state').textContent = `${registry.apps.length} 个已登记工作台 · 登记状态不代表外部服务或实验在线`;
  render();
})().catch(() => { document.querySelector('#cards').textContent = '无法读取工作台登记，请检查本地门户服务。'; });
