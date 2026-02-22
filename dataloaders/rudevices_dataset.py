"""
RuDevicesDataset — загрузчик датасета SOVA RuDevices.

Структура директории::

    <root>/
        0/
            0/
                <uuid>.wav
                <uuid>.txt   ← reference text (одна строка)
            1/
                ...
        1/
            ...
        f/
            f/
                ...

Каждый сэмпл — пара файлов с одинаковым UUID.
Текст хранится в UTF-8 txt-файле рядом с аудио.

Использование::

    from plantain2asr import RuDevicesDataset, Models, SimpleNormalizer, Metrics

    ds   = RuDevicesDataset("data/RuDevices")
    ds   >> Models.GigaAM_v3()
    norm = ds >> SimpleNormalizer()
    norm >> Metrics.composite()

Ссылки:
    https://github.com/sovaai/sova-dataset
    https://huggingface.co/datasets/bond005/sova_rudevices
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import BaseASRDataset
from .types import AudioSample

try:
    from tqdm.auto import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


class RuDevicesDataset(BaseASRDataset):
    """
    SOVA RuDevices speech corpus.

    ~10 000 сэмплов живой речи на русском языке, записанной на мобильных
    устройствах с ручной разметкой.

    Args:
        root_dir: Путь к корневой директории датасета (содержит папки 0–f).
        limit:    Максимальное число загружаемых сэмплов (для быстрых тестов).

    Структура файлов::

        <root>/<hex1>/<hex2>/<uuid>.wav
        <root>/<hex1>/<hex2>/<uuid>.txt

    Pipeline::

        ds   = RuDevicesDataset("data/RuDevices")
        ds   >> Models.GigaAM_v3()
        norm = ds >> SimpleNormalizer()
        norm >> Metrics.composite()
        norm.to_pandas()
    """

    def __init__(self, root_dir: str, limit: Optional[int] = None):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.name = "RuDevices"
        self.limit = limit

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"[RuDevices] Directory not found: {self.root_dir}\n"
                f"Download from: https://github.com/sovaai/sova-dataset"
            )

        self._load()

    def _load(self) -> None:
        wav_files = sorted(self.root_dir.rglob("*.wav"))

        if _HAS_TQDM:
            wav_files_iter = _tqdm(wav_files, desc="Loading RuDevices", unit="file")
        else:
            wav_files_iter = wav_files

        loaded = 0
        missing_txt = 0

        for wav_path in wav_files_iter:
            if self.limit and loaded >= self.limit:
                break

            txt_path = wav_path.with_suffix(".txt")

            if txt_path.exists():
                try:
                    text = txt_path.read_text(encoding="utf-8").strip()
                except Exception:
                    text = ""
            else:
                text = ""
                missing_txt += 1

            uid = wav_path.stem
            sample = AudioSample(
                id=uid,
                audio_path=str(wav_path),
                text=text,
                duration=0.0,
                meta={"rudevices_id": uid},
            )
            self._samples.append(sample)
            self._id_map[uid] = sample
            loaded += 1

        print(f"[{self.name}] {loaded} samples loaded", end="")
        if missing_txt:
            print(f"  (⚠️  {missing_txt} missing .txt → no reference)", end="")
        print()
