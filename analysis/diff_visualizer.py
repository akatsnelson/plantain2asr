import html
import difflib
import os
from dataclasses import dataclass
from typing import List, Dict, Optional
from ..dataloaders.base import BaseASRDataset

@dataclass
class DiffReport:
    output_path: str
    sample_count: int

    def print(self):
        print("\n👁️ Diff Visualizer Report")
        print("-" * 60)
        print(f"✅ HTML report generated: {self.output_path}")
        print(f"   Samples included: {self.sample_count}")
        print("   Open this file in your browser to inspect errors visually.")

class DiffVisualizer:
    """
    Генерирует HTML отчет с визуальным сравнением Reference vs Hypothesis.
    Подсвечивает ошибки цветами:
    - Красный: Удалено (было в Ref, нет в Hyp)
    - Зеленый: Вставлено (нет в Ref, есть в Hyp)
    - Желтый: Замена (Ref -> Hyp)
    """
    def __init__(self, output_file: str = "diff_report.html", max_samples: int = 100):
        self.output_file = output_file
        self.max_samples = max_samples

    def _generate_diff_html(self, ref: str, hyp: str) -> str:
        """Создает HTML фрагмент с подсветкой diff"""
        matcher = difflib.SequenceMatcher(None, ref.split(), hyp.split())
        html_parts = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            ref_chunk = html.escape(" ".join(ref.split()[i1:i2]))
            hyp_chunk = html.escape(" ".join(hyp.split()[j1:j2]))
            
            if tag == 'equal':
                html_parts.append(f"<span>{ref_chunk} </span>")
            elif tag == 'delete':
                html_parts.append(f"<span style='background-color: #ffcccc; text-decoration: line-through; color: #cc0000;'>{ref_chunk} </span>")
            elif tag == 'insert':
                html_parts.append(f"<span style='background-color: #ccffcc; color: #006600;'>{hyp_chunk} </span>")
            elif tag == 'replace':
                html_parts.append(f"<span style='background-color: #ffffcc;' title='Was: {ref_chunk}'>{hyp_chunk} </span>")
                
        return "".join(html_parts)

    def apply_to(self, dataset: BaseASRDataset) -> DiffReport:
        print(f"👁️ Generating HTML Diff Report ({self.output_file})...")
        
        html_content = ["""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>ASR Diff Visualization</title>
            <style>
                body { font-family: sans-serif; padding: 20px; background: #f5f5f5; }
                .sample { background: white; padding: 15px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .meta { color: #666; font-size: 0.9em; margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 5px; }
                .diff { font-family: monospace; font-size: 1.1em; line-height: 1.6; }
                h1 { color: #333; }
                .legend { margin-bottom: 20px; padding: 10px; background: #fff; border-radius: 4px; }
                .del { background-color: #ffcccc; text-decoration: line-through; padding: 2px 4px; border-radius: 3px; }
                .ins { background-color: #ccffcc; padding: 2px 4px; border-radius: 3px; }
                .sub { background-color: #ffffcc; padding: 2px 4px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>ASR Diff Visualization</h1>
            <div class="legend">
                <strong>Legend:</strong> 
                <span class="del">Deleted (Missed)</span>
                <span class="ins">Inserted (Hallucination)</span>
                <span class="sub">Substituted (Confusion)</span>
            </div>
        """]
        
        count = 0
        for sample in dataset:
            if count >= self.max_samples: break
            
            # Пропускаем, если нет текста
            if not sample.text: continue
            
            # Находим модели с результатами
            models_res = {k: v for k, v in sample.asr_results.items() if 'hypothesis' in v}
            if not models_res: continue
            
            count += 1
            audio_name = os.path.basename(sample.audio_path)
            
            html_content.append(f"<div class='sample'>")
            html_content.append(f"<div class='meta'><strong>Audio:</strong> {audio_name} | <strong>Duration:</strong> {sample.duration:.2f}s</div>")
            
            html_content.append(f"<div style='margin-bottom: 8px;'><strong>Reference:</strong> {html.escape(sample.text)}</div>")
            
            for model_name, res in models_res.items():
                hyp = res['hypothesis']
                wer = res.get('metrics', {}).get('wer', 0) * 100
                diff_html = self._generate_diff_html(sample.text, hyp)
                
                html_content.append(f"<div style='margin-top: 10px; border-left: 3px solid #ddd; padding-left: 10px;'>")
                html_content.append(f"<div style='font-size: 0.9em; color: #555;'><strong>{model_name}</strong> (WER: {wer:.1f}%)</div>")
                html_content.append(f"<div class='diff'>{diff_html}</div>")
                html_content.append(f"</div>")
                
            html_content.append(f"</div>")
            
        html_content.append("</body></html>")
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(html_content))
            
        report = DiffReport(self.output_file, count)
        report.print()
        return report
