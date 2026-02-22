"""
ReportServer — локальный HTTP-сервер для интерактивного отчёта.

Что сервер знает:
    - Какие секции показывать (sections) → генерирует HTML при старте
    - Где взять аудиофайлы (audio_dir)
    - Как получить данные секций (ReportBuilder)

Архитектурно:
    - Реализует Processor → вписывается в пайплайн: dataset >> normalizer >> server
    - HTML генерируется один раз из build_html(sections)
    - Каждый запрос /api/{name} отдаёт данные конкретной секции
    - /audio/{id} стримит файл с поддержкой Range-заголовков

Быстрый старт:
    server = ReportServer(dataset, audio_dir="/path/to/wav", normalizer=n)
    server.serve()  →  http://localhost:8765

В пайплайне:
    server = ReportServer(audio_dir="...", normalizer=n)
    dataset >> normalizer >> server
    server.serve()
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, List, Optional
from urllib.parse import unquote

from ..core.processor import Processor
from .builder import ReportBuilder
from .template import build_html

if TYPE_CHECKING:
    from ..dataloaders.base import BaseASRDataset
    from .sections.base import BaseSection


def _default_sections() -> List['BaseSection']:
    from .sections import MetricsSection, ErrorFrequencySection, DiffSection
    return [MetricsSection(), ErrorFrequencySection(), DiffSection()]


class ReportServer(Processor):
    """
    Локальный сервер интерактивного ASR-отчёта.

    Args:
        dataset:      Датасет (уже нормализованный, если нужно).
                      Можно передать позже через .serve(dataset) или pipeline >>.
        audio_dir:    Папка с аудиофайлами для стриминга по id.
        sections:     Список секций. По умолчанию: Metrics + Errors + Diff.
        port:         Порт HTTP-сервера (по умолчанию 8765).
        open_browser: Открывать ли браузер автоматически.

    Нормализация применяется ДО сервера:
        norm_ds = dataset >> DagrusNormalizer()
        ReportServer(norm_ds, audio_dir=...).serve()

        # или в пайплайне:
        dataset >> DagrusNormalizer() >> server
        server.serve()
    """

    def __init__(
        self,
        dataset: Optional['BaseASRDataset'] = None,
        audio_dir: Optional[str] = None,
        sections: Optional[List['BaseSection']] = None,
        port: int = 8765,
        open_browser: bool = True,
    ):
        self.dataset      = dataset
        self.audio_dir    = audio_dir
        self.sections     = sections or _default_sections()
        self.port         = port
        self.open_browser = open_browser

        self._html        = build_html(self.sections)   # генерируется один раз
        self._builder: Optional[ReportBuilder] = None
        self._server: Optional[HTTPServer]     = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Processor interface
    # ------------------------------------------------------------------

    def apply_to(self, dataset: 'BaseASRDataset') -> 'BaseASRDataset':
        """Позволяет включить сервер в пайплайн: dataset >> server."""
        self.dataset = dataset
        return dataset

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def serve(self, dataset: Optional['BaseASRDataset'] = None) -> 'ReportServer':
        """Запускает сервер. Если dataset передан, обновляет self.dataset."""
        if dataset is not None:
            self.dataset = dataset

        if self.dataset is None:
            raise ValueError("dataset is required; pass it to ReportServer() or serve()")

        self._builder = ReportBuilder(self.dataset, sections=self.sections)

        print(f"🌱 plantain2asr report server starting at http://localhost:{self.port}")

        server = self       # захват для замыкания
        html   = self._html

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass  # тихий режим

            def do_GET(self):
                path = self.path.split("?")[0]

                # ── Главная страница ───────────────────────────────────
                if path in ("/", "/index.html"):
                    body = html.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                # ── API секций ─────────────────────────────────────────
                elif path.startswith("/api/"):
                    name = path[5:]
                    if server._builder is None:
                        self._json_error(503, "Builder not ready")
                        return
                    data = server._builder.build()
                    if name not in data:
                        self._json_error(404, f"Section '{name}' not found")
                        return
                    body = json.dumps(data[name], ensure_ascii=False).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                # ── Аудио ─────────────────────────────────────────────
                elif path.startswith("/audio/"):
                    self._serve_audio(path[7:])

                else:
                    self.send_error(404)

            def _json_error(self, code, msg):
                body = json.dumps({"error": msg}).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _serve_audio(self, filename):
                filename = unquote(filename)
                audio_dir = server.audio_dir

                # Поиск файла по имени в audio_dir
                filepath = None
                if audio_dir:
                    candidate = os.path.join(audio_dir, filename)
                    if os.path.isfile(candidate):
                        filepath = candidate
                    else:
                        # Ищем без расширения (filename может быть без .wav)
                        for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
                            c2 = os.path.join(audio_dir, filename + ext)
                            if os.path.isfile(c2):
                                filepath = c2
                                break
                        if not filepath:
                            # Рекурсивный поиск в поддиректориях
                            for root, _, files in os.walk(audio_dir):
                                for f in files:
                                    base = os.path.splitext(f)[0]
                                    if base == filename or f == filename:
                                        filepath = os.path.join(root, f)
                                        break
                                if filepath:
                                    break

                if not filepath:
                    self.send_error(404, f"Audio not found: {filename}")
                    return

                size = os.path.getsize(filepath)
                mime = self._audio_mime(filepath)

                # Range-заголовок для перемотки браузером
                rng = self.headers.get("Range")
                if rng and rng.startswith("bytes="):
                    parts = rng[6:].split("-")
                    start = int(parts[0]) if parts[0] else 0
                    end   = int(parts[1]) if parts[1] else size - 1
                    end   = min(end, size - 1)
                    length = end - start + 1
                    self.send_response(206)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.send_header("Content-Length", str(length))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    with open(filepath, "rb") as fh:
                        fh.seek(start)
                        self.wfile.write(fh.read(length))
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(size))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    with open(filepath, "rb") as fh:
                        self.wfile.write(fh.read())

            @staticmethod
            def _audio_mime(path: str) -> str:
                ext = os.path.splitext(path)[1].lower()
                return {
                    ".wav":  "audio/wav",
                    ".mp3":  "audio/mpeg",
                    ".flac": "audio/flac",
                    ".ogg":  "audio/ogg",
                    ".m4a":  "audio/mp4",
                }.get(ext, "audio/octet-stream")

        self._server = HTTPServer(("", self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

        url = f"http://localhost:{self.port}"
        if self.open_browser:
            webbrowser.open(url)

        print(f"✅ Serving at {url}  (Ctrl+C or .stop() to quit)")
        return self

    def stop(self) -> None:
        """Останавливает сервер."""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None
            print("⛔ Server stopped.")

    def restart(self) -> 'ReportServer':
        """Перезапускает сервер (удобно при обновлении датасета)."""
        self.stop()
        return self.serve()

    # ------------------------------------------------------------------
    # Добавление/замена секций (можно вызвать до .serve())
    # ------------------------------------------------------------------

    def add_section(self, section: 'BaseSection') -> 'ReportServer':
        """Добавляет секцию и пересобирает HTML. Вызывать до .serve()."""
        self.sections.append(section)
        self._html = build_html(self.sections)
        return self

    def replace_section(self, section: 'BaseSection') -> 'ReportServer':
        """Заменяет секцию с тем же name. Вызывать до .serve()."""
        self.sections = [s if s.name != section.name else section for s in self.sections]
        self._html = build_html(self.sections)
        return self
