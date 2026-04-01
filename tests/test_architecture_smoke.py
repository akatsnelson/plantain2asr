from pathlib import Path

import pytest

import plantain2asr as plantain
from plantain2asr import Models
from plantain2asr.dataloaders.base import BaseASRDataset
from plantain2asr.dataloaders.types import AudioSample
from plantain2asr.metrics.factory import Metrics
from plantain2asr.reporting.server import ReportServer


class EmptyDataset(BaseASRDataset):
    def __init__(self, name: str = "EmptyDataset"):
        super().__init__()
        self.name = name


class TinyDataset(BaseASRDataset):
    def __init__(self, root: Path):
        super().__init__()
        self.name = "TinyDataset"
        self._samples = [
            AudioSample(
                id="sample.wav",
                audio_path=str(root / "sample.wav"),
                text="privet",
                duration=1.0,
            )
        ]
        self._id_map = {sample.id: sample for sample in self._samples}


@pytest.mark.core
def test_models_create_accepts_case_and_separator_variants(monkeypatch):
    from tests.fake_backends import install_fake_tone

    install_fake_tone(monkeypatch, providers=["CPUExecutionProvider"])

    model = Models.create("tone", device="auto")
    assert model.name == "T-One"


@pytest.mark.core
def test_models_create_suggests_nearest_backend_name():
    with pytest.raises(ValueError, match="Did you mean"):
        Models.create("Gigam_v3")


@pytest.mark.core
def test_metrics_composite_fails_fast_on_unknown_metric():
    with pytest.raises(ValueError, match="Unknown metric 'werr'"):
        Metrics.composite(["werr"])


@pytest.mark.core
def test_empty_dataset_fails_fast_for_model_and_report():
    dataset = EmptyDataset()
    dummy_model = type("DummyModel", (), {"name": "DummyModel"})()

    with pytest.raises(ValueError, match="Dataset 'EmptyDataset' is empty"):
        dataset.run_model(dummy_model)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="empty dataset"):
        ReportServer(dataset=dataset, open_browser=False).serve()


@pytest.mark.core
def test_filter_no_longer_prints_to_stdout(tmp_path, capsys):
    dataset = TinyDataset(tmp_path)
    dataset.filter(lambda sample: sample.duration > 0)

    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.core
def test_optional_exports_raise_helpful_import_error(monkeypatch):
    monkeypatch.setattr(
        plantain,
        "_OPTIONAL_EXPORTS",
        {"TrainingConfig": (".definitely_missing_module", "Install plantain2asr[train].")},
    )

    with pytest.raises(ImportError, match="Install plantain2asr\\[train\\]"):
        plantain.__getattr__("TrainingConfig")
