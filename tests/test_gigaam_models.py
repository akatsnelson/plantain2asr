import contextlib

import pytest

from tests.fake_backends import import_fresh, install_fake_gigaam


@pytest.mark.gigaam
def test_gigaam_v2_auto_device_prefers_cuda(monkeypatch):
    install_fake_gigaam(monkeypatch, cuda_available=True)
    gigaam_v2 = import_fresh("plantain2asr.models.local.gigaam_v2")

    model = gigaam_v2.GigaAMv2(model_name="v2_ctc", device="auto")

    assert model.device == "cuda"
    assert model.name == "GigaAM-v2_ctc"


@pytest.mark.gigaam
def test_gigaam_v2_rejects_unknown_variant(monkeypatch):
    install_fake_gigaam(monkeypatch, cuda_available=False)
    gigaam_v2 = import_fresh("plantain2asr.models.local.gigaam_v2")

    with pytest.raises(ValueError, match="Unsupported GigaAM v2 variant"):
        gigaam_v2.GigaAMv2(model_name="bad", device="cpu")


@pytest.mark.gigaam
def test_gigaam_v3_has_stable_unique_names(monkeypatch):
    install_fake_gigaam(monkeypatch, cuda_available=False)
    gigaam_v3 = import_fresh("plantain2asr.models.local.gigaam_v3")
    monkeypatch.setattr(gigaam_v3, "_gigaam_v3_compat", contextlib.nullcontext)

    first = gigaam_v3.GigaAMv3(model_name="e2e_rnnt", device="cpu")
    second = gigaam_v3.GigaAMv3(model_name="ctc", device="cpu")

    assert first.name == "GigaAM-v3-e2e_rnnt"
    assert second.name == "GigaAM-v3-ctc"
    assert first.name != second.name
