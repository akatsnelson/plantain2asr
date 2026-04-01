from pathlib import Path

import pytest

from tests.fake_backends import import_fresh, install_fake_tone


@pytest.mark.tone
def test_tone_model_uses_cuda_provider_when_available(monkeypatch, tmp_path):
    install_fake_tone(monkeypatch, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    tone_module = import_fresh("plantain2asr.models.local.tone_model")

    model = tone_module.ToneModel(device="auto")
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"")

    assert model.device == "cuda"
    assert model.transcribe(audio_path) == "tone:default"


@pytest.mark.tone
def test_tone_model_loads_local_pipeline_with_cpu_provider(monkeypatch, tmp_path):
    ort_module = install_fake_tone(monkeypatch, providers=["CPUExecutionProvider"])
    tone_module = import_fresh("plantain2asr.models.local.tone_model")

    model_dir = tmp_path / "tone-model"
    model_dir.mkdir()

    model = tone_module.ToneModel(model_name=str(model_dir), device="cpu")
    assert model.device == "cpu"
    assert model.pipeline.source == str(model_dir)


@pytest.mark.tone
def test_tone_model_errors_when_cuda_is_requested_but_missing(monkeypatch):
    install_fake_tone(monkeypatch, providers=["CPUExecutionProvider"])
    tone_module = import_fresh("plantain2asr.models.local.tone_model")

    with pytest.raises(RuntimeError, match="CUDA provider requested"):
        tone_module.ToneModel(device="cuda")
