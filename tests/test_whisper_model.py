import pytest

from tests.fake_backends import import_fresh, install_fake_whisper


@pytest.mark.whisper
def test_whisper_model_auto_device_prefers_cuda(monkeypatch):
    install_fake_whisper(monkeypatch, cuda_available=True)
    whisper_module = import_fresh("plantain2asr.models.local.whisper_model")

    model = whisper_module.WhisperModel(device="auto")

    assert model.device == "cuda"
    assert model.name.startswith("Whisper-")


@pytest.mark.whisper
def test_whisper_model_rejects_missing_cuda_for_explicit_request(monkeypatch):
    install_fake_whisper(monkeypatch, cuda_available=False, mps_available=False)
    whisper_module = import_fresh("plantain2asr.models.local.whisper_model")

    with pytest.raises(RuntimeError, match="CUDA requested for Whisper"):
        whisper_module.WhisperModel(device="cuda")


@pytest.mark.whisper
def test_whisper_model_transcribes_batch(monkeypatch, tmp_path):
    install_fake_whisper(monkeypatch, cuda_available=False)
    whisper_module = import_fresh("plantain2asr.models.local.whisper_model")

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"")

    model = whisper_module.WhisperModel(device="cpu")
    result = model.transcribe(audio_path)

    assert "russian" in result
