import contextlib
from pathlib import Path
import wave

import pytest

from plantain2asr import Metrics, Models
from plantain2asr.dataloaders.base import BaseASRDataset
from plantain2asr.dataloaders.types import AudioSample
from tests.fake_backends import (
    import_fresh,
    install_fake_gigaam,
    install_fake_tone,
    install_fake_vosk,
    install_fake_whisper,
)


class SmallDataset(BaseASRDataset):
    def __init__(self, root: Path):
        super().__init__()
        self.name = "SmallDataset"
        self.cache_dir = root / "cache"
        self._samples = [
            AudioSample(
                id="sample.wav",
                audio_path=str(root / "sample.wav"),
                text="sample wav",
                duration=1.0,
            )
        ]
        self._id_map = {sample.id: sample for sample in self._samples}


def _write_wav(path: Path):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 160)


@pytest.mark.integration
def test_multiple_models_work_in_one_environment(monkeypatch, tmp_path):
    _write_wav(tmp_path / "sample.wav")

    install_fake_tone(monkeypatch, providers=["CPUExecutionProvider"])
    install_fake_gigaam(monkeypatch, cuda_available=False)
    install_fake_vosk(monkeypatch)

    import_fresh("plantain2asr.models.local.gigaam_v2")
    gigaam_v3 = import_fresh("plantain2asr.models.local.gigaam_v3")
    monkeypatch.setattr(gigaam_v3, "_gigaam_v3_compat", contextlib.nullcontext)

    dataset = SmallDataset(tmp_path)
    dataset >> Models.Tone(device="cpu")
    dataset >> Models.GigaAM_v2(model_name="v2_ctc", device="cpu")
    dataset >> Models.GigaAM_v3(model_name="ctc", device="cpu")
    install_fake_whisper(monkeypatch, cuda_available=False)
    import_fresh("plantain2asr.models.local.whisper_model")
    import_fresh("plantain2asr.models.local.vosk_model")
    dataset >> Models.Whisper(device="cpu")
    model_dir = tmp_path / "vosk-model"
    model_dir.mkdir()
    dataset >> Models.Vosk(model_path=str(model_dir))
    dataset >> Metrics.WER()

    model_names = set(dataset[0].asr_results)
    assert model_names == {
        "T-One",
        "GigaAM-v2_ctc",
        "GigaAM-v3-ctc",
        "Whisper-whisper-large-v3-ru-podlodka",
        "Vosk",
    }

    for result in dataset[0].asr_results.values():
        assert "WER" in result["metrics"]
