"""
build_html(sections) — собирает итоговый HTML из базового каркаса и секций.

Каркас содержит:
    - общий CSS (layout, cards, tables, tokens, pagination)
    - глобальное JS-состояние S
    - инфраструктуру роутинга/вкладок
    - утилиты esc(), fmtNum(), uiEmpty()

Каждая секция вносит:
    - вкладку в <nav>
    - <div id="{name}-panel"> в контент
    - дополнительный HTML (extra_html) — сайдбары и т.д.
    - свой JS-блок (js_function)
    - опциональный CSS (css)
"""

from __future__ import annotations

import json
from typing import List, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .sections.base import BaseSection

# ── Базовый CSS (не зависит от секций) ──────────────────────────────
_BASE_CSS = """
:root {
  --bg:      #f4f6fb;
  --surface: #ffffff;
  --border:  #e2e6f0;
  --text:    #1e2035;
  --muted:   #7a82a6;
  --accent:  #5469d4;
  --accent2: #eef0ff;
  --del:     #c0392b; --del-bg: #fdf0ef;
  --ins:     #1a7a1a; --ins-bg: #edfaed;
  --sub:     #a05a00; --sub-bg: #fff8e6;
  --radius:  10px;
  --shadow:  0 1px 4px rgba(0,0,0,.08);
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     background:var(--bg);color:var(--text);font-size:14px;min-height:100vh}
#app{display:flex;flex-direction:column;height:100vh;overflow:hidden}

/* header */
header{padding:0 24px;background:var(--surface);border-bottom:1px solid var(--border);
       display:flex;align-items:center;gap:16px;height:52px;flex-shrink:0;
       box-shadow:var(--shadow)}
.logo{font-size:15px;font-weight:700;color:var(--accent);display:flex;align-items:center;gap:6px}
#model-select{margin-left:auto;display:flex;align-items:center;gap:8px}
#model-select label{color:var(--muted);font-size:12px;font-weight:500}
#model-select select{background:var(--bg);color:var(--text);border:1px solid var(--border);
  border-radius:8px;padding:5px 28px 5px 12px;font-size:13px;cursor:pointer;min-width:180px;
  appearance:none;-webkit-appearance:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%237a82a6' d='M6 8L1 3h10z'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 10px center}
#model-select select:focus{outline:none;border-color:var(--accent)}

/* nav */
nav{display:flex;padding:0 24px;background:var(--surface);
    border-bottom:1px solid var(--border);flex-shrink:0}
.tab-btn{padding:12px 18px;cursor:pointer;border:none;background:none;
         color:var(--muted);font-size:13px;font-weight:500;
         border-bottom:2px solid transparent;transition:color .15s,border-color .15s;
         white-space:nowrap}
.tab-btn:hover{color:var(--text)}
.tab-btn.active{color:var(--accent);border-bottom-color:var(--accent)}

/* content */
#content{flex:1;overflow:hidden;display:flex}
.tab-panel{display:none;width:100%;overflow:auto;padding:24px;
           flex-direction:column;gap:16px}
.tab-panel.active{display:flex}

/* cards */
.card{background:var(--surface);border:1px solid var(--border);
      border-radius:var(--radius);box-shadow:var(--shadow);overflow:hidden}
.card-title{padding:13px 18px;font-size:12px;font-weight:600;color:var(--muted);
            text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border)}

/* tables */
table{width:100%;border-collapse:collapse}
thead{background:var(--bg)}
th{padding:10px 16px;text-align:left;font-size:11px;font-weight:600;color:var(--muted);
   text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid var(--border);
   cursor:pointer;user-select:none;white-space:nowrap}
th:hover{color:var(--text)}
th.sorted{color:var(--accent)}
.sort-icon{margin-left:4px}
td{padding:10px 16px;border-bottom:1px solid var(--border);vertical-align:middle}
tr:last-child td{border-bottom:none}
tr:hover td{background:var(--accent2)}
.num{font-variant-numeric:tabular-nums;font-size:13px}
.badge{display:inline-block;padding:3px 9px;border-radius:20px;font-size:12px;font-weight:600}
.badge-del{background:var(--del-bg);color:var(--del)}
.badge-ins{background:var(--ins-bg);color:var(--ins)}
.badge-sub{background:var(--sub-bg);color:var(--sub)}
.c-good{color:#1a7a1a;font-weight:600}
.c-ok{color:#a05a00;font-weight:600}
.c-bad{color:#c0392b;font-weight:600}
.clickable-row{cursor:pointer}
.row-arrow{color:var(--muted);font-size:16px}
tr.clickable-row:hover .row-arrow{color:var(--accent)}

/* metrics bars */
.bar-wrap{display:flex;gap:14px;flex-wrap:wrap}
.bar-card{flex:1;min-width:220px}
.bar-row{display:flex;gap:8px;align-items:center;padding:6px 18px}
.bar-label{width:140px;overflow:hidden;text-overflow:ellipsis;
           white-space:nowrap;font-size:12px;color:var(--muted)}
.bar-track{flex:1;height:7px;background:var(--bg);border-radius:4px;overflow:hidden}
.bar-fill{height:100%;background:var(--accent);border-radius:4px}
.bar-val{width:44px;text-align:right;font-size:12px;font-variant-numeric:tabular-nums}

/* diff */
.diff-toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:12px 16px;box-shadow:var(--shadow)}
.search-box{display:flex;align-items:center;gap:6px;flex:1;min-width:200px;
  background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:6px 12px}
.search-box input{border:none;background:none;color:var(--text);font-size:13px;
  outline:none;flex:1;min-width:0}
.search-box input::placeholder{color:var(--muted)}
.wer-range{display:flex;align-items:center;gap:6px;color:var(--muted);
  font-size:12px;white-space:nowrap}
.wer-range input[type=number]{width:56px;padding:5px 8px;background:var(--bg);
  border:1px solid var(--border);border-radius:6px;color:var(--text);
  font-size:12px;text-align:center}
.wer-range input[type=number]:focus{outline:none;border-color:var(--accent)}
.diff-list{display:flex;flex-direction:column;gap:10px}
.s-card{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);box-shadow:var(--shadow);overflow:hidden}
.s-head{padding:9px 14px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:10px;background:var(--bg);font-size:12px}
.s-id{font-family:"SF Mono","Consolas",monospace;font-size:11px;color:var(--muted);
  flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.s-audio audio{height:26px}
.s-ref{padding:7px 14px;font-size:12px;color:var(--muted);
  border-bottom:1px solid var(--border);line-height:1.6}
.s-ref b{color:var(--text);font-weight:500}
.s-models{padding:10px 14px;display:flex;flex-direction:column;gap:8px}
.m-row{display:flex;gap:10px;align-items:baseline}
.m-name{font-size:11px;color:var(--muted);width:150px;flex-shrink:0;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.m-wer{font-size:11px;font-weight:600;width:46px;flex-shrink:0;text-align:right}
.m-toks{flex:1;font-size:13px;line-height:1.9;word-break:break-word}
.tok-eq{color:var(--text)}
.tok-del{background:var(--del-bg);color:var(--del);border-radius:3px;
  padding:1px 3px;margin:0 1px;text-decoration:line-through;font-size:12px}
.tok-ins{background:var(--ins-bg);color:var(--ins);border-radius:3px;
  padding:1px 3px;margin:0 1px;font-size:12px}
.tok-sub{background:var(--sub-bg);color:var(--sub);border-radius:3px;
  padding:1px 3px;margin:0 1px;font-size:12px}
.tok-sub::after{content:" → " attr(data-h);font-size:10px;opacity:.8}

/* pagination */
.pager{display:flex;gap:4px;justify-content:center;padding:8px 0}
.pg{padding:5px 11px;background:var(--surface);border:1px solid var(--border);
  color:var(--text);border-radius:6px;cursor:pointer;font-size:12px}
.pg:hover{border-color:var(--accent);color:var(--accent)}
.pg.active{background:var(--accent);border-color:var(--accent);color:#fff}
.pg-ellipsis{padding:5px 6px;color:var(--muted);font-size:12px}

/* legend */
.legend{display:flex;gap:14px;align-items:center;flex-wrap:wrap;
  font-size:11px;color:var(--muted)}
.leg-item{display:flex;align-items:center;gap:4px}
.leg-dot{width:10px;height:10px;border-radius:2px;flex-shrink:0}

/* states */
.spinner-wrap{padding:48px;text-align:center;color:var(--muted)}
.spinner{display:inline-block;width:28px;height:28px;border:3px solid var(--border);
  border-top-color:var(--accent);border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
"""

# ── Базовый JS (инфраструктура) ──────────────────────────────────────
_BASE_JS = """
// ════════════════════════════════════════════════════════
// Global state
// ════════════════════════════════════════════════════════
const S = {
  data:        {},
  activeTab:   null,
  activeModel: '__all__',
  errSubTab:   'deletions',
  diffQ:       '',
  diffWerMin:  0,
  diffWerMax:  200,
  diffPage:    0,
  PAGE:        30,
  sortKey:     null,
  sortDir:     1,
};

// ════════════════════════════════════════════════════════
// Tab routing
// ════════════════════════════════════════════════════════
document.querySelectorAll('.tab-btn').forEach(btn =>
  btn.addEventListener('click', () => switchTab(btn.dataset.tab))
);

async function switchTab(tab) {
  S.activeTab = tab;
  document.querySelectorAll('.tab-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === tab)
  );
  document.querySelectorAll('.tab-panel').forEach(p =>
    p.classList.toggle('active', p.id === tab + '-panel')
  );
  await ensureLoaded(tab);
  render(tab);
}

async function ensureLoaded(tab) {
  if (S.data[tab]) return;
  if (window.__PLANTAIN_STATIC__ && window.__PLANTAIN_STATIC_DATA__) {
    S.data[tab] = window.__PLANTAIN_STATIC_DATA__[tab] || null;
    return;
  }
  const panel = document.getElementById(tab + '-panel');
  // Для errors-panel сохраняем структуру (left + sidebar), спиннер внутри errors-left
  const target = document.getElementById(tab + '-left') || panel;
  target.innerHTML = '<div class="spinner-wrap"><div class="spinner"></div></div>';
  try {
    const r = await fetch('/api/' + tab);
    if (!r.ok) throw new Error(r.statusText);
    S.data[tab] = await r.json();
    populateSelector();
  } catch (e) {
    panel.innerHTML = `<div style="padding:48px;text-align:center;color:var(--muted)">
      Failed to load: ${e.message}</div>`;
  }
}

function render(tab) {
  const fn = RENDERERS[tab];
  if (fn) fn();
}

function onModelChange() {
  S.activeModel = document.getElementById('global-model').value;
  S.diffPage = 0;
  if (S.activeTab) render(S.activeTab);
}

function populateSelector() {
  const sel = document.getElementById('global-model');
  const existing = new Set(Array.from(sel.options).map(o => o.value));
  const models = allModels();
  models.forEach(m => {
    if (!existing.has(m)) {
      const o = document.createElement('option');
      o.value = m; o.textContent = m;
      sel.appendChild(o);
    }
  });
}

function allModels() {
  const seen = new Set();
  const result = [];
  const sources = [
    S.data.metrics?.models || [],
    Object.keys(S.data.errors || {}),
    S.data.diff?.models    || [],
  ];
  sources.flat().forEach(m => { if (m && !seen.has(m)) { seen.add(m); result.push(m); } });
  return result;
}

// ════════════════════════════════════════════════════════
// Shared utilities
// ════════════════════════════════════════════════════════
function esc(s) {
  if (!s) return '';
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function fmtNum(v) {
  if (typeof v !== 'number') return String(v ?? '');
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
}
function uiEmpty(msg) {
  return `<div style="padding:48px;text-align:center;color:var(--muted)">${msg}</div>`;
}

function audioPathToUrl(path) {
  if (!path) return '';
  if (path.startsWith('file://') || path.startsWith('http://') || path.startsWith('https://')) {
    return path;
  }
  if (path.startsWith('/')) {
    return 'file://' + encodeURI(path);
  }
  return encodeURI(path);
}

function audioSrc(audioId, audioPath) {
  if (window.__PLANTAIN_STATIC__) {
    return audioPathToUrl(audioPath);
  }
  return '/audio/' + encodeURIComponent(audioId);
}
"""


def build_html(
    sections: 'List[BaseSection]',
    *,
    initial_data: Optional[dict] = None,
) -> str:
    """
    Собирает финальный HTML из базового каркаса и списка секций.
    Вызывается один раз при старте ReportServer.
    """
    # 1. Nav tabs (первая секция — активная по умолчанию)
    tabs_html = "\n".join(
        f'<button class="tab-btn{" active" if i == 0 else ""}" data-tab="{s.name}">'
        f'{s.icon} {s.title}</button>'
        for i, s in enumerate(sections)
    )

    # 2. Content panels (каждая секция предоставляет начальное содержимое)
    panels_html = "\n".join(
        f'<div class="tab-panel{" active" if i == 0 else ""}" id="{s.name}-panel">'
        f'{s.panel_html()}'
        f'</div>'
        for i, s in enumerate(sections)
    )

    extra_html = ""  # больше не используется (sidebar теперь внутри panel_html)

    # 4. CSS от секций
    section_css = "\n".join(s.css() for s in sections)

    # 5. JS-функции секций
    js_functions = "\n\n".join(s.js_function() for s in sections)

    # 6. RENDERERS dict (роутинг вкладок)
    renderers = "{\n" + ",\n".join(
        f"  '{s.name}': render_{s.name}"
        for s in sections
    ) + "\n}"

    # 7. Первая секция по умолчанию
    first_tab = sections[0].name if sections else ""
    static_data = json.dumps(initial_data, ensure_ascii=False) if initial_data is not None else "null"
    static_flag = "true" if initial_data is not None else "false"

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>plantain2asr · Report</title>
<style>
{_BASE_CSS}
{section_css}
</style>
</head>
<body>
<div id="app">
  <header>
    <div class="logo">🌱 plantain2asr</div>
    <span style="color:var(--border)">|</span>
    <span style="color:var(--muted);font-size:12px">ASR Evaluation Report</span>
    <div id="model-select">
      <label>Filter model:</label>
      <select id="global-model" onchange="onModelChange()">
        <option value="__all__">All models</option>
      </select>
    </div>
  </header>

  <nav>
    {tabs_html}
  </nav>

  <div id="content">
    {panels_html}
    {extra_html}
  </div>
</div>

<script>
window.__PLANTAIN_STATIC__ = {static_flag};
window.__PLANTAIN_STATIC_DATA__ = {static_data};

{_BASE_JS}

// ════════════════════════════════════════════════════════
// Section renderers (injected by sections)
// ════════════════════════════════════════════════════════
{js_functions}

// ════════════════════════════════════════════════════════
// Routing table (auto-generated)
// ════════════════════════════════════════════════════════
const RENDERERS = {renderers};

// ════════════════════════════════════════════════════════
// Boot: load first tab
// ════════════════════════════════════════════════════════
switchTab('{first_tab}');
</script>
</body>
</html>"""
