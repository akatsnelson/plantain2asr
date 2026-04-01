from pathlib import Path

import pytest

from plantain2asr import Metrics
from plantain2asr.dataloaders.base import BaseASRDataset
from plantain2asr.dataloaders.types import AudioSample
from plantain2asr.models.base import BaseASRModel


class DummyDataset(BaseASRDataset):
    def __init__(self, root: Path):
        super().__init__()
        self.name = "DummyDataset"
        self.cache_dir = root / "cache"
        self._samples = [
            AudioSample(
                id="sample-1.wav",
                audio_path=str(root / "sample-1.wav"),
                text="privet mir",
                duration=1.0,
            ),
            AudioSample(
                id="sample-2.wav",
                audio_path=str(root / "sample-2.wav"),
                text="kak dela",
                duration=2.0,
            ),
        ]
        self._id_map = {sample.id: sample for sample in self._samples}


class EchoModel(BaseASRModel):
    @property
    def name(self) -> str:
        return "EchoModel"

    def transcribe(self, audio_path):
        if str(audio_path).endswith("sample-1.wav"):
            return "privet mir"
        return "kak dela"


@pytest.mark.core
def test_core_pipeline_save_and_reload(tmp_path):
    for file_name in ("sample-1.wav", "sample-2.wav"):
        (tmp_path / file_name).write_bytes(b"")

    dataset = DummyDataset(tmp_path)
    filtered = dataset.filter(lambda sample: sample.duration >= 1.0)
    assert len(filtered.take(1)) == 1

    dataset >> EchoModel()
    dataset >> Metrics.WER()

    first_result = dataset[0].asr_results["EchoModel"]
    assert first_result["hypothesis"] == "privet mir"
    assert first_result["metrics"]["WER"] == 0.0

    output_path = tmp_path / "results.jsonl"
    dataset.save_results(str(output_path))
    assert output_path.exists()

    reloaded = DummyDataset(tmp_path)
    reloaded.load_results(str(output_path))
    assert "EchoModel" in reloaded[0].asr_results
    assert reloaded[1].asr_results["EchoModel"]["hypothesis"] == "kak dela"
