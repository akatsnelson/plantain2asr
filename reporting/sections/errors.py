from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from .base import BaseSection

if TYPE_CHECKING:
    from ...dataloaders.base import BaseASRDataset

try:
    import jiwer
    _HAS_JIWER = True
except ImportError:
    _HAS_JIWER = False

_TOP_N = 100
_MAX_EX = 10


class ErrorFrequencySection(BaseSection):
    """
    Частотный анализ ошибок: deletions, insertions, substitutions.
    Клик на строку → боковая панель с примерами + аудио-плеер.

    Нормализация применяется к датасету ДО передачи сюда:
        dataset >> DagrusNormalizer() >> server
    """

    def __init__(self, top_n: int = _TOP_N, max_examples: int = _MAX_EX):
        self.top_n        = top_n
        self.max_examples = max_examples

    @property
    def name(self) -> str:
        return "errors"

    @property
    def title(self) -> str:
        return "Error Frequency"

    @property
    def icon(self) -> str:
        return "🔍"

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def compute(self, dataset: 'BaseASRDataset') -> dict:
        if not _HAS_JIWER:
            return {"_error": "jiwer not installed: pip install jiwer"}

        models: set = set()
        for s in dataset:
            models.update(s.asr_results.keys())

        result = {}
        for model in sorted(models):
            dels:  Counter = Counter()
            ins:   Counter = Counter()
            subs:  Counter = Counter()
            del_ex:  dict = defaultdict(list)
            ins_ex:  dict = defaultdict(list)
            sub_ex:  dict = defaultdict(list)

            for sample in dataset:
                if not sample.text or model not in sample.asr_results:
                    continue

                ref = sample.text or ""
                hyp = sample.asr_results[model].get("hypothesis", "")

                if not ref.strip():
                    continue

                try:
                    out = jiwer.process_words(ref, hyp)
                    ref_words = out.references[0]
                    hyp_words = out.hypotheses[0]

                    for chunk in out.alignments[0]:
                        t  = chunk.type
                        ex = {
                            "ref": ref,
                            "hyp": hyp,
                            "audio": sample.id,
                            "audio_path": sample.audio_path,
                            "model": model,
                        }

                        if t == "delete":
                            for w in ref_words[chunk.ref_start_idx:chunk.ref_end_idx]:
                                dels[w] += 1
                                if len(del_ex[w]) < self.max_examples:
                                    del_ex[w].append({**ex, "ref_word": w, "hyp_word": ""})

                        elif t == "insert":
                            for w in hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]:
                                ins[w] += 1
                                if len(ins_ex[w]) < self.max_examples:
                                    ins_ex[w].append({**ex, "ref_word": "", "hyp_word": w})

                        elif t == "substitute":
                            rws = ref_words[chunk.ref_start_idx:chunk.ref_end_idx]
                            hws = hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]
                            for rw, hw in zip(rws, hws):
                                key = f"{rw}→{hw}"
                                subs[key] += 1
                                if len(sub_ex[key]) < self.max_examples:
                                    sub_ex[key].append({**ex, "ref_word": rw, "hyp_word": hw})
                except Exception:
                    continue

            result[model] = {
                "deletions": [
                    {"word": w, "count": c, "examples": del_ex[w]}
                    for w, c in dels.most_common(self.top_n)
                ],
                "insertions": [
                    {"word": w, "count": c, "examples": ins_ex[w]}
                    for w, c in ins.most_common(self.top_n)
                ],
                "substitutions": [
                    {"pair": k, "count": c, "examples": sub_ex[k]}
                    for k, c in subs.most_common(self.top_n)
                ],
            }

        return result

    # ------------------------------------------------------------------
    # Panel HTML: flex-row (список слева + сайдбар справа)
    # Оба элемента — прямые дети панели, поэтому flex-лейаут работает.
    # ------------------------------------------------------------------

    def panel_html(self) -> str:
        return """
<div id="errors-left">
  <div class="spinner-wrap"><div class="spinner"></div></div>
</div>
<div id="errors-sidebar" class="hidden">
  <div class="sidebar-head">
    <h3 id="sidebar-title">Examples</h3>
    <button onclick="closeSidebar()" title="Close">✕</button>
  </div>
  <div id="examples-list"></div>
</div>"""

    # ------------------------------------------------------------------
    # JS
    # ------------------------------------------------------------------

    def js_function(self) -> str:
        return r"""
// ── Error Frequency section ──────────────────────────────────────────
var _errItems = [];

function render_errors() {
  const d = S.data['errors'];
  if (!d) return;
  const left = document.getElementById('errors-left');

  const models = Object.keys(d).filter(k => k !== '_error');
  const model  = (S.activeModel !== '__all__' && d[S.activeModel])
    ? S.activeModel : models[0];

  if (!model || d['_error']) {
    left.innerHTML = uiEmpty(d['_error'] || 'No error data.');
    return;
  }

  const mData = d[model];
  const types = ['deletions', 'insertions', 'substitutions'];

  left.innerHTML =
    `<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding-bottom:4px">
      <div class="err-subtabs">
        ${types.map(t => `
          <button class="err-tab ${t===S.errSubTab?'active':''}"
                  onclick="switchErrTab('${t}')">${errLabel(t)}</button>`).join('')}
      </div>
      ${S.activeModel === '__all__'
        ? `<span style="color:var(--muted);font-size:12px">Showing: <strong>${esc(model)}</strong></span>`
        : ''}
    </div>
    ${types.map(t => renderErrTable(mData[t], t)).join('')}`;
}

function errLabel(t) {
  return t === 'deletions'  ? '📉 Deletions'
       : t === 'insertions' ? '📈 Insertions'
                            : '🔄 Substitutions';
}

function switchErrTab(t) { S.errSubTab = t; render_errors(); }

function renderErrTable(items, type) {
  const active = type === S.errSubTab;
  const isSub  = type === 'substitutions';
  const badge  = type === 'deletions' ? 'del' : type === 'insertions' ? 'ins' : 'sub';
  const total  = items.reduce((s, x) => s + x.count, 0);

  if (active) _errItems = items;

  return `
    <div class="err-section ${active ? 'active' : ''} card">
      <table>
        <thead><tr>
          <th>${isSub ? 'Ref → Hyp' : 'Word'}</th>
          <th>Count</th>
          <th>Share</th>
          <th></th>
        </tr></thead>
        <tbody>
          ${items.map((item, i) => {
            const label = isSub ? item.pair : item.word;
            const pct   = total > 0 ? (item.count / total * 100).toFixed(1) : '0';
            return `<tr class="clickable-row" onclick="openExamples(${i})">
              <td><span class="badge badge-${badge}">${esc(label)}</span></td>
              <td class="num">${item.count}</td>
              <td class="num">${pct}%</td>
              <td class="row-arrow">›</td>
            </tr>`;
          }).join('')}
        </tbody>
      </table>
    </div>`;
}

// ── Sidebar ───────────────────────────────────────────────────────────
function openExamples(idx) {
  const item    = _errItems[idx];
  if (!item) return;
  const key     = item.pair || item.word;
  const sidebar = document.getElementById('errors-sidebar');
  const title   = document.getElementById('sidebar-title');
  const list    = document.getElementById('examples-list');

  sidebar.classList.remove('hidden');
  title.textContent = `"${key}" · ${item.examples.length} examples`;

  list.innerHTML = item.examples.map((ex, i) => {
    const tokens = wordDiff(ex.ref, ex.hyp);
    const refHtml = renderDiffLine(tokens, 'ref', ex.ref_word, ex.hyp_word);
    const hypHtml = renderDiffLine(tokens, 'hyp', ex.ref_word, ex.hyp_word);
    const modelLabel = ex.model
      ? `<span class="ex-model">${esc(ex.model)}</span>`
      : '';

    return `
    <div class="ex-card">
      <div class="ex-meta">
        <span class="ex-num">Example ${i + 1}</span>
        ${modelLabel}
      </div>
      <div class="ex-diff-wrap">
        <div class="ex-diff-line">
          <span class="lbl lbl-ref">Ref</span>
          <span class="ex-toks">${refHtml}</span>
        </div>
        <div class="ex-diff-line">
          <span class="lbl lbl-hyp">Hyp</span>
          <span class="ex-toks">${hypHtml}</span>
        </div>
      </div>
      ${ex.audio ? `<div class="ex-audio">
        <audio src="${audioSrc(ex.audio, ex.audio_path)}" controls preload="none"></audio>
      </div>` : ''}
    </div>`;
  }).join('');
}

// Простой word-diff через LCS (без jiwer в браузере)
function wordDiff(ref, hyp) {
  const rw = ref ? ref.split(/\s+/).filter(Boolean) : [];
  const hw = hyp ? hyp.split(/\s+/).filter(Boolean) : [];
  const m = rw.length, n = hw.length;

  // DP LCS
  const dp = Array.from({length: m + 1}, () => new Int32Array(n + 1));
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      dp[i][j] = rw[i-1] === hw[j-1]
        ? dp[i-1][j-1] + 1
        : Math.max(dp[i-1][j], dp[i][j-1]);

  // Backtrack
  const ops = [];
  let i = m, j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && rw[i-1] === hw[j-1])
      ops.push({t: 'eq', r: rw[--i], h: hw[--j]});
    else if (j > 0 && (i === 0 || dp[i][j-1] >= dp[i-1][j]))
      ops.push({t: 'ins', r: '', h: hw[--j]});
    else
      ops.push({t: 'del', r: rw[--i], h: ''});
  }
  return ops.reverse();
}

// Рендерит одну строку (ref или hyp) с подсветкой
// primaryR/primaryH — ключевые слова из клика (выделяем ярче)
function renderDiffLine(tokens, side, primaryR, primaryH) {
  return tokens.map(op => {
    const word  = side === 'ref' ? op.r : op.h;
    if (!word) return '';   // пропускаем пустые слоты

    const isPrimary = side === 'ref'
      ? (op.t === 'del' && op.r === primaryR)
      : (op.t === 'ins' && op.h === primaryH);
    const isSubPrimary = op.t === 'del' && side === 'ref'
      ? op.r === primaryR
      : op.t === 'ins' && side === 'hyp'
      ? op.h === primaryH
      : false;

    if (op.t === 'eq')
      return `<span class="sd-eq">${esc(word)}</span> `;

    const cls = op.t === 'del' ? 'sd-del' : 'sd-ins';
    const prime = isPrimary || isSubPrimary ? ' sd-primary' : '';
    return `<span class="${cls}${prime}">${esc(word)}</span> `;
  }).join('');
}

function closeSidebar() {
  document.getElementById('errors-sidebar').classList.add('hidden');
}
"""

    def css(self) -> str:
        return """
/* ── errors panel: два столбца ── */
#errors-panel { flex-direction:row !important; padding:0 !important;
  gap:0 !important; overflow:hidden !important; align-items:stretch; }
/*
  display:block (не flex!) — иначе .err-section.active становится flex-элементом
  с flex-shrink:1 и сжимается вместо переполнения → scroll не работает.
*/
#errors-left { flex:1; min-height:0; overflow-y:auto; padding:20px; min-width:0; }

/* ── sidebar ── */
#errors-sidebar { width:460px; flex-shrink:0; min-height:0; border-left:1px solid var(--border);
  background:var(--surface); display:flex; flex-direction:column;
  overflow:hidden; transition:width .2s ease; }
#errors-sidebar.hidden { width:0; border:none; }
.sidebar-head { padding:14px 18px; border-bottom:1px solid var(--border);
  display:flex; align-items:center; justify-content:space-between; flex-shrink:0; }
.sidebar-head h3 { font-size:13px; font-weight:600; }
.sidebar-head button { background:none; border:none; cursor:pointer;
  color:var(--muted); font-size:20px; line-height:1;
  border-radius:4px; padding:2px 6px; }
.sidebar-head button:hover { background:var(--bg); color:var(--text); }
#examples-list { flex:1; overflow-y:auto; padding:14px;
  display:flex; flex-direction:column; gap:10px; }

/* ── example card ── */
.ex-card { border:1px solid var(--border); border-radius:8px;
  padding:12px 14px; background:var(--bg); }
.ex-meta { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
.ex-num { font-size:10px; font-weight:700; text-transform:uppercase;
  letter-spacing:.06em; color:var(--muted); }
.ex-model { font-size:10px; font-weight:600; background:var(--accent2);
  color:var(--accent); border-radius:10px; padding:1px 8px; }
.ex-diff-wrap { display:flex; flex-direction:column; gap:4px; }
.ex-diff-line { display:flex; gap:6px; align-items:baseline; font-size:12px;
  line-height:1.8; word-break:break-word; }
.lbl { font-size:10px; font-weight:700; text-transform:uppercase;
  letter-spacing:.05em; flex-shrink:0; width:28px; padding-top:2px; }
.lbl-ref { color:var(--muted); }
.lbl-hyp { color:var(--muted); }
.ex-toks { flex:1; }
.ex-audio { margin-top:8px; }
.ex-audio audio { width:100%; height:28px; }

/* ── diff tokens в сайдбаре ── */
.sd-eq  { color:var(--text); }
.sd-del { background:var(--del-bg); color:var(--del); border-radius:3px;
  padding:1px 3px; font-size:11px; }
.sd-ins { background:var(--ins-bg); color:var(--ins); border-radius:3px;
  padding:1px 3px; font-size:11px; }
/* Ключевое слово (то, по которому кликнули) — ярче и жирнее */
.sd-primary { font-weight:700; outline:2px solid currentColor;
  outline-offset:1px; border-radius:3px; }

/* ── subtabs ── */
.err-subtabs { display:flex; gap:6px; }
.err-tab { padding:7px 16px; cursor:pointer; border:1px solid var(--border);
  background:var(--bg); color:var(--muted); border-radius:20px;
  font-size:12px; font-weight:500; transition:all .15s; }
.err-tab:hover { border-color:var(--accent); color:var(--accent); }
.err-tab.active { background:var(--accent); border-color:var(--accent); color:#fff; }
.err-section { display:none; }
.err-section.active { display:block; margin-top:14px; }
"""
