import pytest

from plantain2asr import Experiment, ModelBenchmark, Models
from tests.fake_backends import import_fresh, install_fake_tone


@pytest.mark.core
def test_models_factory_lists_expected_backends():
    available = set(Models.list())
    assert {"GigaAM_v2", "GigaAM_v3", "Whisper", "Vosk", "Tone"}.issubset(available)


@pytest.mark.core
def test_models_create_returns_requested_backend(monkeypatch):
    install_fake_tone(monkeypatch, providers=["CPUExecutionProvider"])
    import_fresh("plantain2asr.models.local.tone_model")
    tone = Models.create("Tone", device="auto")
    assert tone.name == "T-One"


@pytest.mark.core
def test_root_exports_model_benchmark():
    assert ModelBenchmark.__name__ == "ModelBenchmark"


@pytest.mark.core
def test_root_exports_experiment():
    assert Experiment.__name__ == "Experiment"
