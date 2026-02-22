from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseSection

if TYPE_CHECKING:
    from ...dataloaders.base import BaseASRDataset

try:
    import jiwer
    _HAS_JIWER = True
except ImportError:
    _HAS_JIWER = False

_MAX_SAMPLES = 500


class DiffSection(BaseSection):
    """
    Пословный дифф Reference vs Hypothesis.
    Фильтрация по тексту, WER-диапазону, модели. Аудио-плеер на каждом сэмпле.

    Нормализация применяется к датасету ДО передачи сюда:
        dataset >> DagrusNormalizer() >> server
    """

    def __init__(self, max_samples: int = _MAX_SAMPLES):
        self.max_samples = max_samples

    @property
    def name(self) -> str:
        return "diff"

    @property
    def title(self) -> str:
        return "Diff View"

    @property
    def icon(self) -> str:
        return "📝"

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def compute(self, dataset: 'BaseASRDataset') -> dict:
        models: set = set()
        for s in dataset:
            models.update(s.asr_results.keys())
        models = sorted(models)

        samples_out = []
        for sample in dataset:
            if len(samples_out) >= self.max_samples:
                break
            if not sample.text or not sample.asr_results:
                continue

            models_data = {}
            for model in models:
                if model not in sample.asr_results:
                    continue
                res = sample.asr_results[model]
                hyp = res.get("hypothesis", "")
                ref = sample.text or ""

                wer = (
                    res.get("metrics", {}).get("WER")
                    or res.get("metrics", {}).get("wer")
                )
                models_data[model] = {
                    "tokens": self._diff_tokens(ref, hyp),
                    "wer":    round(wer, 1) if wer is not None else None,
                    "hyp":    hyp,
                }

            if not models_data:
                continue

            samples_out.append({
                "id":     sample.id,
                "audio":  sample.id,
                "ref":    ref,
                "models": models_data,
            })

        return {"models": models, "samples": samples_out}

    @staticmethod
    def _diff_tokens(ref: str, hyp: str) -> list:
        if not _HAS_JIWER or not ref.strip():
            return [{"t": "eq", "w": ref}]
        try:
            out = jiwer.process_words(ref, hyp)
            ref_words = out.references[0]
            hyp_words = out.hypotheses[0]
            tokens = []
            for chunk in out.alignments[0]:
                t = chunk.type
                if t == "equal":
                    for w in ref_words[chunk.ref_start_idx:chunk.ref_end_idx]:
                        tokens.append({"t": "eq", "w": w})
                elif t == "delete":
                    for w in ref_words[chunk.ref_start_idx:chunk.ref_end_idx]:
                        tokens.append({"t": "del", "w": w})
                elif t == "insert":
                    for w in hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]:
                        tokens.append({"t": "ins", "w": w})
                elif t == "substitute":
                    rws = ref_words[chunk.ref_start_idx:chunk.ref_end_idx]
                    hws = hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]
                    for rw, hw in zip(rws, hws):
                        tokens.append({"t": "sub", "w": rw, "h": hw})
            return tokens
        except Exception:
            return [{"t": "eq", "w": ref}]

    # ------------------------------------------------------------------
    # JS
    # ------------------------------------------------------------------

    def js_function(self) -> str:
        return r"""
// ── Diff section ────────────────────────────────────────────────────
function render_diff() {
  const d = S.data['diff'];
  if (!d) return;
  const panel = document.getElementById('diff-panel');

  const activeModels = S.activeModel === '__all__' ? d.models : [S.activeModel];

  let samples = d.samples.filter(s => {
    const wers = activeModels.map(m => s.models[m]?.wer).filter(v => v != null);
    if (!wers.length) return false;
    const avg  = wers.reduce((a, b) => a + b, 0) / wers.length;
    if (avg < S.diffWerMin || avg > S.diffWerMax) return false;
    if (S.diffQ) {
      const q = S.diffQ.toLowerCase();
      return s.ref.toLowerCase().includes(q)
        || s.id.toLowerCase().includes(q)
        || activeModels.some(m => s.models[m]?.hyp?.toLowerCase().includes(q));
    }
    return true;
  });

  const total = samples.length;
  const pages = Math.max(1, Math.ceil(total / S.PAGE));
  S.diffPage  = Math.min(S.diffPage, pages - 1);
  const slice = samples.slice(S.diffPage * S.PAGE, (S.diffPage + 1) * S.PAGE);

  const wc = v => v < 30 ? 'c-good' : v < 60 ? 'c-ok' : 'c-bad';

  const toolbar =
    `<div class="diff-toolbar">
      <div class="search-box">
        <span style="color:var(--muted)">🔍</span>
        <input type="text" placeholder="Search in ref / hyp / filename…"
               value="${esc(S.diffQ)}"
               oninput="S.diffQ=this.value;S.diffPage=0;render_diff()">
      </div>
      <div class="wer-range">
        WER ≥
        <input type="number" value="${S.diffWerMin}" min="0" max="200"
               oninput="S.diffWerMin=+this.value||0;S.diffPage=0;render_diff()">
        ≤
        <input type="number" value="${S.diffWerMax}" min="0" max="500"
               oninput="S.diffWerMax=+this.value||200;S.diffPage=0;render_diff()">
      </div>
      <div style="display:flex;gap:10px;align-items:center">
        <div class="legend">
          <div class="leg-item"><div class="leg-dot" style="background:var(--del-bg);border:1px solid var(--del)"></div> Deletion</div>
          <div class="leg-item"><div class="leg-dot" style="background:var(--ins-bg);border:1px solid var(--ins)"></div> Insertion</div>
          <div class="leg-item"><div class="leg-dot" style="background:var(--sub-bg);border:1px solid var(--sub)"></div> Substitution</div>
        </div>
        <span style="color:var(--muted);font-size:12px">${total} samples</span>
      </div>
    </div>`;

  const cards = slice.map(s => {
    const modelRows = activeModels.filter(m => s.models[m]).map(m => {
      const md   = s.models[m];
      const toks = (md.tokens || []).map(tok => {
        if (tok.t === 'eq')  return `<span class="tok-eq">${esc(tok.w)} </span>`;
        if (tok.t === 'del') return `<span class="tok-del">${esc(tok.w)}</span> `;
        if (tok.t === 'ins') return `<span class="tok-ins">${esc(tok.w)}</span> `;
        if (tok.t === 'sub') return `<span class="tok-sub" data-h="${esc(tok.h)}">${esc(tok.w)}</span> `;
        return '';
      }).join('');
      const werLabel = md.wer != null ? md.wer.toFixed(1) + '%' : '—';
      return `<div class="m-row">
        <span class="m-name" title="${esc(m)}">${esc(m)}</span>
        <span class="m-wer ${md.wer != null ? wc(md.wer) : ''}">${werLabel}</span>
        <div class="m-toks">${toks}</div>
      </div>`;
    }).join('');

    const audioTag = s.audio
      ? `<div class="s-audio"><audio src="/audio/${encodeURIComponent(s.audio)}" controls preload="none"></audio></div>`
      : '';

    return `<div class="s-card">
      <div class="s-head">
        <span class="s-id">${esc(s.id)}</span>
        ${audioTag}
      </div>
      <div class="s-ref"><b>Ref:</b> ${esc(s.ref)}</div>
      <div class="s-models">${modelRows}</div>
    </div>`;
  }).join('');

  panel.innerHTML = toolbar
    + `<div class="diff-list">${cards}</div>`
    + diffPager(pages);
}

function diffPager(pages) {
  if (pages <= 1) return '';
  const cur = S.diffPage;
  let html  = '';
  for (let i = 0; i < pages; i++) {
    if (i === 0 || i === pages - 1 || Math.abs(i - cur) <= 2) {
      html += `<button class="pg ${i===cur?'active':''}" onclick="goPage(${i})">${i + 1}</button>`;
    } else if (Math.abs(i - cur) === 3) {
      html += `<span class="pg-ellipsis">…</span>`;
    }
  }
  return `<div class="pager">${html}</div>`;
}

function goPage(p) { S.diffPage = p; render_diff(); window.scrollTo(0, 0); }
"""
