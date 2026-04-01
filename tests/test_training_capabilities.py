import contextlib

import pytest

from tests.fake_backends import (
    import_fresh,
    install_fake_gigaam,
    install_fake_tone,
    install_fake_vosk,
    install_fake_whisper,
)


@pytest.mark.train
def test_gigaam_v3_rnnt_variant_fails_fast_for_training(monkeypatch):
    install_fake_gigaam(monkeypatch, cuda_available=False)
    gigaam_v3 = import_fresh("plantain2asr.models.local.gigaam_v3")
    monkeypatch.setattr(gigaam_v3, "_gigaam_v3_compat", contextlib.nullcontext)

    model = gigaam_v3.GigaAMv3(model_name="rnnt", device="cpu")

    assert model.supports_training is False
    with pytest.raises(NotImplementedError, match="first supported training backend is GigaAM v3 CTC"):
        model.get_training_components()


@pytest.mark.train
def test_whisper_reports_seq2seq_training_capability_gap(monkeypatch):
    install_fake_whisper(monkeypatch, cuda_available=False)
    whisper_model = import_fresh("plantain2asr.models.local.whisper_model")

    model = whisper_model.WhisperModel(model_name="fake/whisper", device="cpu")

    assert "seq2seq training backend" in model.training_not_supported_reason()


@pytest.mark.train
def test_tone_reports_inference_only_training_status(monkeypatch):
    install_fake_tone(monkeypatch, providers=["CPUExecutionProvider"])
    tone_model = import_fresh("plantain2asr.models.local.tone_model")

    model = tone_model.ToneModel(device="cpu")

    assert "inference-only backend" in model.training_not_supported_reason()


@pytest.mark.train
def test_vosk_reports_training_capability_gap(monkeypatch, tmp_path):
    install_fake_vosk(monkeypatch)
    vosk_model = import_fresh("plantain2asr.models.local.vosk_model")
    model_dir = tmp_path / "vosk-model"
    model_dir.mkdir()

    model = vosk_model.VoskModel(model_path=str(model_dir))

    assert "available only for inference" in model.training_not_supported_reason()
