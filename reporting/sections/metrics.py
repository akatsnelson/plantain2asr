from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from .base import BaseSection

if TYPE_CHECKING:
    from ...dataloaders.base import BaseASRDataset


class MetricsSection(BaseSection):
    """Сводная таблица метрик по моделям с мини-барами."""

    @property
    def name(self) -> str:
        return "metrics"

    @property
    def title(self) -> str:
        return "Metrics"

    @property
    def icon(self) -> str:
        return "📊"

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def compute(self, dataset: 'BaseASRDataset') -> dict:
        models: set = set()
        for s in dataset:
            models.update(s.asr_results.keys())

        rows = []
        for model in sorted(models):
            agg: dict = defaultdict(list)
            for s in dataset:
                if model not in s.asr_results:
                    continue
                for k, v in s.asr_results[model].get("metrics", {}).items():
                    if isinstance(v, (int, float)):
                        agg[k].append(v)
            if not agg:
                continue
            row = {"model": model}
            for k, vals in agg.items():
                row[k] = round(sum(vals) / len(vals), 3)
            row["_count"] = len(next(iter(agg.values())))
            rows.append(row)

        metric_keys: list = []
        seen: set = set()
        for r in rows:
            for k in r:
                if k not in ("model", "_count") and k not in seen:
                    metric_keys.append(k)
                    seen.add(k)

        return {
            "models": sorted(models),
            "metric_keys": metric_keys,
            "rows": rows,
        }

    # ------------------------------------------------------------------
    # JS
    # ------------------------------------------------------------------

    def js_function(self) -> str:
        return r"""
// ── Metrics section ──────────────────────────────────────────────────
function render_metrics() {
  const d = S.data['metrics'];
  if (!d) return;
  const panel = document.getElementById('metrics-panel');

  let rows = d.rows;
  if (S.activeModel !== '__all__')
    rows = rows.filter(r => r.model === S.activeModel);

  if (!rows.length) {
    panel.innerHTML = uiEmpty('No data for selected model.');
    return;
  }

  const keys = d.metric_keys;
  if (S.sortKey) {
    rows = [...rows].sort((a, b) => {
      const av = a[S.sortKey] ?? (S.sortDir > 0 ? 1e9 : -1e9);
      const bv = b[S.sortKey] ?? (S.sortDir > 0 ? 1e9 : -1e9);
      return (typeof av === 'string' ? av.localeCompare(bv) : av - bv) * S.sortDir;
    });
  }

  const wc  = v => v == null ? '' : v < 30 ? 'c-good' : v < 60 ? 'c-ok' : 'c-bad';
  const si  = k => {
    const active = k === S.sortKey;
    const arrow  = active ? (S.sortDir > 0 ? '▲' : '▼') : '⇅';
    const op     = active ? '1' : '.25';
    return `<span class="sort-icon" style="opacity:${op}">${arrow}</span>`;
  };

  const barKeys = keys.filter(k => ['WER','CER','wer','cer'].includes(k));

  panel.innerHTML =
    `<div class="card">
      <div class="card-title">Model Comparison</div>
      <table>
        <thead><tr>
          <th onclick="sortBy('model')" class="${S.sortKey==='model'?'sorted':''}">Model ${si('model')}</th>
          <th onclick="sortBy('_count')" class="${S.sortKey==='_count'?'sorted':''}">Samples ${si('_count')}</th>
          ${keys.map(k => `
            <th onclick="sortBy('${k}')" class="${S.sortKey===k?'sorted':''}">
              ${esc(k)} ${si(k)}
            </th>`).join('')}
        </tr></thead>
        <tbody>
          ${rows.map(r => `<tr>
            <td><strong>${esc(r.model)}</strong></td>
            <td class="num">${r._count ?? '—'}</td>
            ${keys.map(k => `
              <td class="num ${(k==='WER'||k==='wer') ? wc(r[k]) : ''}">
                ${r[k] != null ? fmtNum(r[k]) : '—'}
              </td>`).join('')}
          </tr>`).join('')}
        </tbody>
      </table>
    </div>`
    + (barKeys.length ? `<div class="bar-wrap">${barKeys.map(k => renderMetricBar(rows, k)).join('')}</div>` : '');
}

function renderMetricBar(rows, key) {
  const max = Math.max(...rows.map(r => r[key] ?? 0));
  return `<div class="card bar-card">
    <div class="card-title">${esc(key)}</div>
    ${rows.map(r => {
      const v   = r[key] ?? 0;
      const pct = max > 0 ? (v / max * 100) : 0;
      return `<div class="bar-row">
        <span class="bar-label" title="${esc(r.model)}">${esc(r.model)}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
        <span class="bar-val">${fmtNum(v)}</span>
      </div>`;
    }).join('')}
  </div>`;
}

function sortBy(key) {
  if (S.sortKey === key) S.sortDir *= -1;
  else { S.sortKey = key; S.sortDir = 1; }
  render(S.activeTab);
}
"""
