"""
DiffVisualizer — генератор статичного HTML-отчёта с диффом ASR-ошибок.

.. deprecated::
    Используйте :class:`plantain2asr.ReportServer` вместо этого класса.
    ReportServer предоставляет интерактивный живой отчёт с аудио-плеером,
    фильтрами и вкладками прямо в браузере.

    Вместо::
        DiffVisualizer(output_file="diff.html").apply_to(dataset)

    Используйте::
        ReportServer(dataset, audio_dir="...").serve()

    DiffVisualizer остаётся для случаев, когда нужен offline-файл без сервера.

Нормализация применяется к датасету до передачи сюда:
    dataset >> DagrusNormalizer() >> DiffVisualizer(output_file="diff.html")
"""

import html
import json
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Dict

try:
    import jiwer
    _HAS_JIWER = True
except ImportError:
    _HAS_JIWER = False

from ..dataloaders.base import BaseASRDataset
from ..core.processor import Processor


@dataclass
class DiffReport:
    output_path: str
    sample_count: int
    model_count: int

    def print(self):
        print(f"\n🌐 HTML diff report → {self.output_path}")
        print(f"   Samples: {self.sample_count}, Models: {self.model_count}")
        print("   Открой в браузере для анализа ошибок.")


# ---------------------------------------------------------------------------
# Цвета для каждого типа ошибки
# ---------------------------------------------------------------------------
_COLORS = {
    "delete":    ("#ffd5d5", "#c0392b"),   # bg, text
    "insert":    ("#d5f5d5", "#1a7a1a"),
    "substitute": ("#fff3cc", "#8a6800"),
    "equal":     ("transparent", "inherit"),
}


class DiffVisualizer(Processor):
    """
    Генерирует статичный HTML с визуальным сравнением Reference vs Hypothesis.

    .. deprecated::
        Используйте :class:`plantain2asr.ReportServer` для интерактивного отчёта
        с аудио-плеером и живыми фильтрами.

    Нормализация применяется к данным заранее:
        dataset >> DagrusNormalizer() >> DiffVisualizer(output_file="diff.html")
    """

    def __init__(
        self,
        output_file: str = "diff_report.html",
        max_samples: int = 300,
        models: Optional[List[str]] = None,
        page_size: int = 50,
    ):
        """
        Args:
            output_file:  Путь к выходному HTML.
            max_samples:  Максимальное число сэмплов в отчёте.
            models:       Список моделей для отчёта. None → все.
            page_size:    Число сэмплов на страницу (JS-пагинация).
        """
        warnings.warn(
            "DiffVisualizer устарел. Используйте ReportServer для интерактивного отчёта:\n"
            "    from plantain2asr import ReportServer\n"
            "    ReportServer(dataset, audio_dir='...').serve()",
            DeprecationWarning,
            stacklevel=2,
        )
        self.output_file = output_file
        self.max_samples = max_samples
        self.models = models
        self.page_size = page_size

    # ------------------------------------------------------------------
    # Diff engine
    # ------------------------------------------------------------------

    def _word_diff(self, ref: str, hyp: str) -> List[dict]:
        """
        Возвращает список токенов с типом ошибки.
        Использует jiwer если доступен, иначе difflib.
        """
        if _HAS_JIWER:
            return self._jiwer_diff(ref, hyp)
        return self._difflib_diff(ref, hyp)

    @staticmethod
    def _jiwer_diff(ref: str, hyp: str) -> List[dict]:
        out = jiwer.process_words(ref, hyp)
        ref_words = out.references[0]
        hyp_words = out.hypotheses[0]
        tokens = []

        for chunk in out.alignments[0]:
            t = chunk.type
            if t == "equal":
                for w in ref_words[chunk.ref_start_idx:chunk.ref_end_idx]:
                    tokens.append({"type": "equal", "text": w})
            elif t == "delete":
                for w in ref_words[chunk.ref_start_idx:chunk.ref_end_idx]:
                    tokens.append({"type": "delete", "ref": w, "hyp": ""})
            elif t == "insert":
                for w in hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]:
                    tokens.append({"type": "insert", "ref": "", "hyp": w})
            elif t == "substitute":
                ref_chunk = ref_words[chunk.ref_start_idx:chunk.ref_end_idx]
                hyp_chunk = hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]
                for rw, hw in zip(ref_chunk, hyp_chunk):
                    tokens.append({"type": "substitute", "ref": rw, "hyp": hw})
                # Если длины не совпадают — остаток как delete/insert
                for rw in ref_chunk[len(hyp_chunk):]:
                    tokens.append({"type": "delete", "ref": rw, "hyp": ""})
                for hw in hyp_chunk[len(ref_chunk):]:
                    tokens.append({"type": "insert", "ref": "", "hyp": hw})
        return tokens

    @staticmethod
    def _difflib_diff(ref: str, hyp: str) -> List[dict]:
        import difflib
        matcher = difflib.SequenceMatcher(None, ref.split(), hyp.split())
        ref_w = ref.split()
        hyp_w = hyp.split()
        tokens = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for w in ref_w[i1:i2]:
                    tokens.append({"type": "equal", "text": w})
            elif tag == "delete":
                for w in ref_w[i1:i2]:
                    tokens.append({"type": "delete", "ref": w, "hyp": ""})
            elif tag == "insert":
                for w in hyp_w[j1:j2]:
                    tokens.append({"type": "insert", "ref": "", "hyp": w})
            elif tag == "replace":
                for rw, hw in zip(ref_w[i1:i2], hyp_w[j1:j2]):
                    tokens.append({"type": "substitute", "ref": rw, "hyp": hw})
        return tokens

    # ------------------------------------------------------------------
    # HTML rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _render_tokens(tokens: List[dict]) -> str:
        parts = []
        for tok in tokens:
            t = tok["type"]
            bg, fg = _COLORS[t]

            if t == "equal":
                parts.append(
                    f'<span class="tok-equal">{html.escape(tok["text"])}</span> '
                )
            elif t == "delete":
                parts.append(
                    f'<span class="tok-del" title="Пропущено">'
                    f'<s>{html.escape(tok["ref"])}</s></span> '
                )
            elif t == "insert":
                parts.append(
                    f'<span class="tok-ins" title="Вставлено моделью">'
                    f'{html.escape(tok["hyp"])}</span> '
                )
            elif t == "substitute":
                ref_e = html.escape(tok["ref"])
                hyp_e = html.escape(tok["hyp"])
                parts.append(
                    f'<span class="tok-sub" title="Ref: {ref_e}">'
                    f'{hyp_e}</span> '
                )
        return "".join(parts)

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def apply_to(self, dataset: BaseASRDataset) -> BaseASRDataset:
        os.makedirs(os.path.dirname(self.output_file) or ".", exist_ok=True)
        print(f"🌐 Building HTML diff report → {self.output_file} ...")

        # Собираем модели
        all_models: set = set()
        for s in dataset:
            all_models.update(s.asr_results.keys())
        target_models = sorted(self.models or all_models)

        # Собираем данные сэмплов
        rows = []
        for sample in dataset:
            if len(rows) >= self.max_samples:
                break
            if not sample.text or not sample.asr_results:
                continue

            ref_norm = sample.text

            models_data = []
            for model_name in target_models:
                res = sample.asr_results.get(model_name)
                if not res:
                    continue
                hyp_norm = res.get("hypothesis", "")
                metrics = res.get("metrics", {})
                wer = metrics.get("WER", metrics.get("wer"))

                tokens = self._word_diff(ref_norm, hyp_norm) if ref_norm else []
                models_data.append({
                    "model": model_name,
                    "hyp": hyp_norm,
                    "wer": wer,
                    "tokens": self._render_tokens(tokens),
                    "n_del": sum(1 for t in tokens if t["type"] == "delete"),
                    "n_ins": sum(1 for t in tokens if t["type"] == "insert"),
                    "n_sub": sum(1 for t in tokens if t["type"] == "substitute"),
                })

            if not models_data:
                continue

            rows.append({
                "id": sample.id,
                "duration": sample.duration,
                "ref": html.escape(sample.text),
                "ref_norm": html.escape(ref_norm),
                "models": models_data,
            })

        # Сводная статистика
        summary = self._build_summary(dataset, target_models)

        html_out = self._render_page(rows, summary, target_models)

        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(html_out)

        self.report = DiffReport(self.output_file, len(rows), len(target_models))
        self.report.print()
        return dataset

    def _build_summary(self, dataset, models):
        stats = {m: {"wer": [], "cer": []} for m in models}
        for s in dataset:
            for m in models:
                res = s.asr_results.get(m, {})
                metrics = res.get("metrics", {})
                wer = metrics.get("WER", metrics.get("wer"))
                cer = metrics.get("CER", metrics.get("cer"))
                if wer is not None:
                    stats[m]["wer"].append(wer)
                if cer is not None:
                    stats[m]["cer"].append(cer)

        rows = []
        for m in models:
            wers = stats[m]["wer"]
            cers = stats[m]["cer"]
            rows.append({
                "model": m,
                "wer": f'{sum(wers)/len(wers):.2f}' if wers else "—",
                "cer": f'{sum(cers)/len(cers):.2f}' if cers else "—",
                "n": len(wers),
            })
        rows.sort(key=lambda r: float(r["wer"]) if r["wer"] != "—" else 999)
        return rows

    # ------------------------------------------------------------------
    # HTML template
    # ------------------------------------------------------------------

    def _render_page(self, rows, summary, models) -> str:
        rows_json = json.dumps(rows, ensure_ascii=False)
        summary_rows_html = "".join(
            f"<tr><td>{r['model']}</td><td>{r['wer']}</td>"
            f"<td>{r['cer']}</td><td>{r['n']}</td></tr>"
            for r in summary
        )
        model_options = "".join(
            f'<option value="{m}">{m}</option>' for m in ["(все)"] + list(models)
        )
        page_size = self.page_size

        return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ASR Diff Report</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg: #f0f2f5; --card: #fff; --border: #e0e0e0;
    --del-bg: #fde8e8; --del-fg: #b71c1c; --del-border: #ef9a9a;
    --ins-bg: #e8f5e9; --ins-fg: #1b5e20; --ins-border: #a5d6a7;
    --sub-bg: #fff8e1; --sub-fg: #5d4037; --sub-border: #ffe082;
    --accent: #3f51b5; --text: #212121; --muted: #757575;
    --radius: 8px; --shadow: 0 1px 4px rgba(0,0,0,.1);
  }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg);
         color: var(--text); line-height: 1.5; }}

  /* Layout */
  .layout {{ display: flex; min-height: 100vh; }}
  .sidebar {{ width: 280px; flex-shrink: 0; background: var(--card);
              border-right: 1px solid var(--border); padding: 20px;
              position: sticky; top: 0; height: 100vh; overflow-y: auto; }}
  .main {{ flex: 1; padding: 24px; overflow: hidden; }}

  /* Sidebar */
  .sidebar h2 {{ font-size: 1.1rem; color: var(--accent); margin-bottom: 16px; }}
  .filter-group {{ margin-bottom: 16px; }}
  .filter-group label {{ display: block; font-size: .82rem; font-weight: 600;
                         color: var(--muted); margin-bottom: 4px; text-transform: uppercase; }}
  .filter-group select,
  .filter-group input {{ width: 100%; padding: 7px 10px; border: 1px solid var(--border);
                         border-radius: 6px; font-size: .9rem; background: var(--bg); }}
  .filter-group input[type=range] {{ padding: 0; }}
  .wer-val {{ font-size: .8rem; color: var(--muted); text-align: right; }}
  .btn {{ display: block; width: 100%; padding: 9px; background: var(--accent);
          color: #fff; border: none; border-radius: 6px; cursor: pointer;
          font-size: .95rem; margin-top: 8px; }}
  .btn:hover {{ opacity: .9; }}

  /* Summary table */
  .summary {{ background: var(--card); border-radius: var(--radius);
              box-shadow: var(--shadow); padding: 16px; margin-bottom: 24px; }}
  .summary h2 {{ font-size: 1rem; margin-bottom: 10px; color: var(--accent); }}
  table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
  th {{ background: var(--bg); padding: 8px 12px; text-align: left;
        font-weight: 600; border-bottom: 2px solid var(--border); }}
  td {{ padding: 7px 12px; border-bottom: 1px solid var(--border); }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #fafafa; }}

  /* Samples */
  .header-bar {{ display: flex; justify-content: space-between; align-items: center;
                 margin-bottom: 16px; }}
  .counter {{ font-size: .9rem; color: var(--muted); }}
  .sample {{ background: var(--card); border-radius: var(--radius);
             box-shadow: var(--shadow); margin-bottom: 16px; overflow: hidden; }}
  .sample-header {{ padding: 10px 16px; background: #f8f9fa;
                    border-bottom: 1px solid var(--border);
                    display: flex; gap: 16px; font-size: .82rem; color: var(--muted); }}
  .sample-header strong {{ color: var(--text); }}
  .ref-block {{ padding: 10px 16px; border-bottom: 1px solid var(--border);
                font-size: .88rem; color: var(--muted); }}
  .ref-block span {{ color: var(--text); font-family: monospace; }}
  .model-block {{ padding: 10px 16px; border-bottom: 1px solid var(--border); }}
  .model-block:last-child {{ border-bottom: none; }}
  .model-name {{ font-weight: 600; font-size: .85rem; margin-bottom: 6px;
                 display: flex; justify-content: space-between; }}
  .model-name .wer-badge {{ font-weight: normal; font-size: .8rem;
    background: var(--bg); border: 1px solid var(--border);
    padding: 1px 8px; border-radius: 12px; color: var(--muted); }}
  .diff-line {{ font-family: monospace; font-size: .95rem; line-height: 1.8;
                word-break: break-word; }}

  /* Token styles */
  .tok-equal {{ }}
  .tok-del {{ background: var(--del-bg); color: var(--del-fg);
              border: 1px solid var(--del-border); border-radius: 3px;
              padding: 0 3px; text-decoration: line-through; }}
  .tok-ins {{ background: var(--ins-bg); color: var(--ins-fg);
              border: 1px solid var(--ins-border); border-radius: 3px;
              padding: 0 3px; }}
  .tok-sub {{ background: var(--sub-bg); color: var(--sub-fg);
              border: 1px solid var(--sub-border); border-radius: 3px;
              padding: 0 3px; cursor: help; }}

  /* Legend */
  .legend {{ display: flex; gap: 12px; flex-wrap: wrap;
             font-size: .82rem; margin-bottom: 16px; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; }}

  /* Pagination */
  .pagination {{ display: flex; gap: 6px; flex-wrap: wrap;
                 margin-top: 20px; justify-content: center; }}
  .pg-btn {{ padding: 6px 12px; border: 1px solid var(--border);
             border-radius: 5px; cursor: pointer; background: var(--card);
             font-size: .85rem; }}
  .pg-btn.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
  .pg-btn:hover:not(.active) {{ background: var(--bg); }}

  .hidden {{ display: none !important; }}
</style>
</head>
<body>
<div class="layout">

<!-- Sidebar -->
<aside class="sidebar">
  <h2>🔍 Фильтры</h2>

  <div class="filter-group">
    <label>Модель</label>
    <select id="f-model">
      {model_options}
    </select>
  </div>

  <div class="filter-group">
    <label>WER ≤ <span id="wer-label">100</span>%</label>
    <input type="range" id="f-wer" min="0" max="100" value="100" step="1"
           oninput="document.getElementById('wer-label').textContent=this.value">
  </div>

  <div class="filter-group">
    <label>Поиск в тексте</label>
    <input type="text" id="f-search" placeholder="слово или фраза...">
  </div>

  <button class="btn" onclick="applyFilters()">Применить</button>
  <button class="btn" style="background:#757575; margin-top:6px"
          onclick="resetFilters()">Сбросить</button>

  <hr style="margin: 20px 0; border-color: var(--border)">

  <div class="legend">
    <div class="legend-item"><span class="tok-del">слово</span> Пропуск</div>
    <div class="legend-item"><span class="tok-ins">слово</span> Вставка</div>
    <div class="legend-item"><span class="tok-sub" title="Ref: оригинал">замена</span> Замена</div>
  </div>
</aside>

<!-- Main -->
<main class="main">
  <h1 style="margin-bottom:8px; color: var(--accent)">ASR Diff Report</h1>
  <p style="color:var(--muted); margin-bottom:20px; font-size:.9rem">
    Визуальное сравнение Reference vs Hypothesis · {len(rows)} сэмплов · {len(models)} моделей
  </p>

  <!-- Summary table -->
  <div class="summary">
    <h2>Сводка по моделям</h2>
    <table>
      <thead><tr><th>Модель</th><th>WER (%)</th><th>CER (%)</th><th>Сэмплов</th></tr></thead>
      <tbody>{summary_rows_html}</tbody>
    </table>
  </div>

  <div class="header-bar">
    <div class="counter" id="counter"></div>
  </div>

  <div id="samples-container"></div>
  <div class="pagination" id="pagination"></div>
</main>
</div>

<script>
const ALL_ROWS = {rows_json};
const PAGE_SIZE = {page_size};

let filtered = ALL_ROWS;
let currentPage = 0;

function applyFilters() {{
  const model  = document.getElementById('f-model').value;
  const werMax = parseFloat(document.getElementById('f-wer').value);
  const search = document.getElementById('f-search').value.toLowerCase().trim();

  filtered = ALL_ROWS.filter(row => {{
    // Model filter
    if (model !== '(все)') {{
      if (!row.models.some(m => m.model === model)) return false;
    }}
    // WER filter (for selected model or min across models)
    const wers = row.models
      .filter(m => model === '(все)' || m.model === model)
      .map(m => m.wer)
      .filter(w => w !== null && w !== undefined);
    if (wers.length && Math.min(...wers) > werMax) return false;
    // Search filter
    if (search) {{
      const haystack = (row.ref_norm + ' ' + row.models.map(m=>m.hyp).join(' ')).toLowerCase();
      if (!haystack.includes(search)) return false;
    }}
    return true;
  }});

  currentPage = 0;
  render();
}}

function resetFilters() {{
  document.getElementById('f-model').value = '(все)';
  document.getElementById('f-wer').value = 100;
  document.getElementById('wer-label').textContent = '100';
  document.getElementById('f-search').value = '';
  filtered = ALL_ROWS;
  currentPage = 0;
  render();
}}

function renderSample(row) {{
  const modelsHtml = row.models.map(m => {{
    const werBadge = m.wer !== null && m.wer !== undefined
      ? `<span class="wer-badge">WER ${{m.wer.toFixed(1)}}% &nbsp;·&nbsp; del ${{m.n_del}} ins ${{m.n_ins}} sub ${{m.n_sub}}</span>`
      : '';
    return `
      <div class="model-block">
        <div class="model-name">${{m.model}}${{werBadge}}</div>
        <div class="diff-line">${{m.tokens}}</div>
      </div>`;
  }}).join('');

  return `
    <div class="sample">
      <div class="sample-header">
        <span><strong>${{row.id}}</strong></span>
        <span>⏱ ${{row.duration.toFixed(2)}}s</span>
      </div>
      <div class="ref-block">Ref: <span>${{row.ref_norm}}</span></div>
      ${{modelsHtml}}
    </div>`;
}}

function render() {{
  const total = filtered.length;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  currentPage = Math.min(currentPage, pages - 1);

  const start = currentPage * PAGE_SIZE;
  const slice = filtered.slice(start, start + PAGE_SIZE);

  document.getElementById('counter').textContent =
    `Показано: ${{slice.length}} из ${{total}} сэмплов`;

  document.getElementById('samples-container').innerHTML =
    slice.map(renderSample).join('');

  // Pagination
  const pg = document.getElementById('pagination');
  pg.innerHTML = '';
  if (pages <= 1) return;
  for (let i = 0; i < pages; i++) {{
    const btn = document.createElement('button');
    btn.className = 'pg-btn' + (i === currentPage ? ' active' : '');
    btn.textContent = i + 1;
    btn.onclick = (() => {{ const p = i; return () => {{ currentPage = p; render(); window.scrollTo(0,0); }}; }})();
    pg.appendChild(btn);
  }}
}}

// Init
render();
</script>
</body>
</html>"""
