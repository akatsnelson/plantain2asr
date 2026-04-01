import wave

import pytest

from tests.fake_backends import import_fresh, install_fake_vosk


def _write_wav(path, framerate=16000, channels=1, sampwidth=2):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(b"\x00\x00" * 160)


@pytest.mark.vosk
def test_vosk_model_transcribes_native_wav_without_ffmpeg(monkeypatch, tmp_path):
    install_fake_vosk(monkeypatch)
    vosk_module = import_fresh("plantain2asr.models.local.vosk_model")

    model_dir = tmp_path / "vosk-model"
    model_dir.mkdir()
    audio_path = tmp_path / "sample.wav"
    _write_wav(audio_path)

    model = vosk_module.VoskModel(model_path=str(model_dir))
    result = model.transcribe(audio_path)

    assert result == "vosk partial vosk final"


@pytest.mark.vosk
def test_vosk_model_requires_ffmpeg_for_conversion(monkeypatch, tmp_path):
    install_fake_vosk(monkeypatch)
    vosk_module = import_fresh("plantain2asr.models.local.vosk_model")
    monkeypatch.setattr(vosk_module.shutil, "which", lambda command: None)

    model_dir = tmp_path / "vosk-model"
    model_dir.mkdir()
    audio_path = tmp_path / "needs-convert.wav"
    _write_wav(audio_path, framerate=8000)

    model = vosk_module.VoskModel(model_path=str(model_dir))

    with pytest.raises(RuntimeError, match="ffmpeg is required"):
        model._convert_audio(str(audio_path))


@pytest.mark.vosk
def test_vosk_model_missing_model_dir_raises(monkeypatch, tmp_path):
    install_fake_vosk(monkeypatch)
    vosk_module = import_fresh("plantain2asr.models.local.vosk_model")

    with pytest.raises(FileNotFoundError, match="Vosk model not found"):
        vosk_module.VoskModel(model_path=str(tmp_path / "missing-model"))
