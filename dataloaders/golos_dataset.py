"""
GolosDataset — загрузчик датасета GOLOS (тестовая часть).

Оба сабсета (crowd и farfield) загружаются в один датасет.
Принадлежность к сабсету хранится в ``sample.meta["subset"]``,
что позволяет фильтровать стандартным pipeline-методом::

    ds = GolosDataset("data/golos")

    crowd    = ds.filter(lambda s: s.meta["subset"] == "crowd")
    farfield = ds.filter(lambda s: s.meta["subset"] == "farfield")

Если директория не существует — датасет скачивается автоматически::

    ds = GolosDataset("data/golos")   # скачает и распакует при первом запуске

Ожидаемая структура директории (после распаковки)::

    <root>/
        crowd/
            manifest.jsonl      # NeMo-формат, ~9 994 записей
            files/
                *.wav
        farfield/               # опционально
            manifest.jsonl      # если есть
            files/
                *.wav           # ~1 547 записей

Если для farfield нет ``manifest.jsonl``, аудио-файлы загружаются
без reference-текста (только для инференса, метрики не считаются).
"""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

from .base import BaseASRDataset
from .types import AudioSample

try:
    from tqdm.auto import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

#: URL тестового архива GOLOS (~2.5 GB)
_DOWNLOAD_URL = "https://cdn.chatwm.opensmodel.sberdevices.ru/golos/test.tar"


class GolosDataset(BaseASRDataset):
    """
    GOLOS speech corpus – test split.

    Оба сабсета (``crowd`` и ``farfield``) объединяются в один датасет.
    Принадлежность к сабсету хранится в ``sample.meta["subset"]``.

    Если директория ``root_dir`` не существует или пуста, датасет
    скачивается автоматически и распаковывается на месте.

    Args:
        root_dir:     Путь к корню GOLOS (или куда скачать).
        limit:        Ограничение общего числа сэмплов (для быстрых тестов).
        auto_download: Автоматически скачивать при отсутствии данных
                       (по умолчанию ``True``).

    Pipeline::

        from plantain2asr import GolosDataset, SimpleNormalizer, Metrics, Models

        ds = GolosDataset("data/golos")          # скачает если нет

        crowd = ds.filter(lambda s: s.meta["subset"] == "crowd")
        crowd >> Models.GigaAM_v3()

        norm = crowd >> SimpleNormalizer()
        norm >> Metrics.composite()
        norm.to_pandas()
    """

    SUBSETS = ("crowd", "farfield")

    def __init__(
        self,
        root_dir: str,
        limit: Optional[int] = None,
        auto_download: bool = True,
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.name = "Golos"
        self.limit = limit

        if auto_download and not self._looks_valid():
            self._download_and_extract()

        for subset in self.SUBSETS:
            if self.limit and len(self._samples) >= self.limit:
                break
            self._load_subset(subset)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _looks_valid(self) -> bool:
        """True если root_dir уже содержит хотя бы один сабсет."""
        for subset in self.SUBSETS:
            sub = self.root_dir / subset
            if sub.exists() and any(sub.iterdir()):
                return True
        return False

    def _download_and_extract(self) -> None:
        """Скачивает test.tar и распаковывает в root_dir."""
        self.root_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Golos] Датасет не найден в {self.root_dir}")
        print(f"[Golos] Скачиваем {_DOWNLOAD_URL} …")

        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            self._download(tmp_path)
            print(f"[Golos] Распаковываем в {self.root_dir} …")
            self._extract(tmp_path)
            print(f"[Golos] ✅ Готово")
        finally:
            tmp_path.unlink(missing_ok=True)

    def _download(self, dest: Path) -> None:
        if _HAS_TQDM:
            self._download_with_progress(dest)
        else:
            urllib.request.urlretrieve(_DOWNLOAD_URL, dest)

    def _download_with_progress(self, dest: Path) -> None:
        response = urllib.request.urlopen(_DOWNLOAD_URL)
        total = int(response.headers.get("Content-Length", 0))
        block = 1 << 20  # 1 MB

        bar = _tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading GOLOS test",
        )
        with open(dest, "wb") as f:
            while True:
                chunk = response.read(block)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
        bar.close()

    def _extract(self, archive: Path) -> None:
        with tarfile.open(archive, "r:*") as tar:
            members = tar.getmembers()

            # Определяем общий префикс (если архив вложен в папку)
            top_dirs = {m.name.split("/")[0] for m in members if m.name}
            strip = (
                len(top_dirs) == 1
                and next(iter(top_dirs)) not in self.SUBSETS
            )

            if _HAS_TQDM:
                members = list(_tqdm(members, desc="Extracting", unit="file"))

            if strip:
                # Убираем верхний уровень (e.g. "test/crowd/..." → "crowd/...")
                prefix = next(iter(top_dirs)) + "/"
                for member in members:
                    if not member.name.startswith(prefix):
                        continue
                    member.name = member.name[len(prefix):]
                    tar.extract(member, path=self.root_dir, set_attrs=False)
            else:
                tar.extractall(path=self.root_dir)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_subset(self, subset: str) -> None:
        sub_dir = self.root_dir / subset

        if not sub_dir.exists():
            return

        manifest = sub_dir / "manifest.jsonl"
        if manifest.exists():
            self._load_from_manifest(sub_dir, manifest, subset)
        else:
            self._load_from_audio_files(sub_dir, subset)

    def _load_from_manifest(
        self, sub_dir: Path, manifest_path: Path, subset: str
    ) -> None:
        seen: set = set()
        loaded = 0

        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                if self.limit and len(self._samples) >= self.limit:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    rel = data.get("audio_filepath", "")
                    if not rel or rel in seen:
                        continue
                    seen.add(rel)

                    audio_path = sub_dir / rel
                    sid = audio_path.name

                    sample = AudioSample(
                        id=sid,
                        audio_path=str(audio_path),
                        text=data.get("text", ""),
                        duration=float(data.get("duration", 0.0)),
                        meta={
                            "subset": subset,
                            "golos_id": data.get("id", ""),
                        },
                    )
                    self._samples.append(sample)
                    self._id_map[sid] = sample
                    loaded += 1

                except (json.JSONDecodeError, Exception):
                    continue

        print(f"[{self.name}] {subset}: {loaded} samples loaded")

    def _load_from_audio_files(self, sub_dir: Path, subset: str) -> None:
        """Загружает аудио без манифеста (нет reference → только инференс)."""
        files_dir = sub_dir / "files"
        if not files_dir.exists():
            return

        wavs = sorted(files_dir.glob("*.wav"))
        if self.limit:
            remaining = self.limit - len(self._samples)
            wavs = wavs[:remaining]

        for wav in wavs:
            sid = wav.name
            if sid in self._id_map:
                continue

            sample = AudioSample(
                id=sid,
                audio_path=str(wav),
                text="",
                duration=0.0,
                meta={"subset": subset, "no_reference": True},
            )
            self._samples.append(sample)
            self._id_map[sid] = sample

        print(
            f"[{self.name}] {subset}: {len(wavs)} files loaded "
            f"(⚠️  no reference text — inference only)"
        )
