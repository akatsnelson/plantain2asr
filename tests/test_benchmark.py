from pathlib import Path

import pytest

from plantain2asr import ModelBenchmark as RootModelBenchmark
from plantain2asr.analysis.benchmark import ModelBenchmark
from plantain2asr.dataloaders.base import BaseASRDataset
from plantain2asr.dataloaders.types import AudioSample
from plantain2asr.models.base import BaseASRModel


class BenchmarkDataset(BaseASRDataset):
    def __init__(self, root: Path):
        super().__init__()
        self.name = "BenchmarkDataset"
        self.cache_dir = root / "cache"
        self._samples = [
            AudioSample(
                id="a.wav",
                audio_path=str(root / "a.wav"),
                text="a",
                duration=1.0,
            ),
            AudioSample(
                id="b.wav",
                audio_path=str(root / "b.wav"),
                text="b",
                duration=2.0,
            ),
        ]
        self._id_map = {sample.id: sample for sample in self._samples}


class FastEchoModel(BaseASRModel):
    def __init__(self, device="cpu"):
        self.device = device
        self.batch_size = 2

    @property
    def name(self) -> str:
        return f"FastEcho-{self.device}"

    def transcribe(self, audio_path):
        return Path(audio_path).stem

    def transcribe_batch(self, audio_paths):
        return [Path(path).stem for path in audio_paths]


@pytest.mark.core
def test_model_benchmark_reports_basic_metrics(tmp_path):
    for name in ("a.wav", "b.wav"):
        (tmp_path / name).write_bytes(b"")

    dataset = BenchmarkDataset(tmp_path)
    benchmark = ModelBenchmark(sample_limit=2, warmup_samples=1, batch_size=2, devices=["cpu"])

    report = benchmark.run(dataset, model_factory=lambda device: FastEchoModel(device=device))

    assert len(report.results) == 1
    result = report.results[0]
    assert result.requested_device == "cpu"
    assert result.actual_device == "cpu"
    assert result.samples == 2
    assert result.successes == 2
    assert result.failures == 0
    assert result.audio_seconds_per_sec >= 0.0

    csv_path = tmp_path / "benchmark.csv"
    saved = report.save_csv(str(csv_path))
    assert saved == str(csv_path)
    assert csv_path.exists()


@pytest.mark.core
def test_model_benchmark_is_exported_from_root():
    assert RootModelBenchmark is ModelBenchmark


@pytest.mark.core
def test_model_benchmark_falls_back_to_cpu_for_factories_without_device(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"")
    dataset = BenchmarkDataset(tmp_path)

    class CpuOnlyModel(FastEchoModel):
        def __init__(self):
            super().__init__(device="cpu")

    benchmark = ModelBenchmark(sample_limit=1)
    devices = benchmark.available_devices(lambda: CpuOnlyModel())
    report = benchmark.run(dataset.take(1), model_factory=lambda: CpuOnlyModel())

    assert devices == ["cpu"]
    assert report.results[0].requested_device == "cpu"
