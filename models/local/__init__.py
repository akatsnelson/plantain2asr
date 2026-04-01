from ...utils.optional_imports import resolve_optional_export

_LOCAL_EXPORTS = {
    "GigaAMv3": (".gigaam_v3", "Install the matching GigaAM dependencies to use GigaAM v3."),
    "GigaAMv2": (".gigaam_v2", "Install the matching GigaAM dependencies to use GigaAM v2."),
    "WhisperModel": (".whisper_model", "Install plantain2asr[whisper] to use Whisper."),
    "VoskModel": (".vosk_model", "Install plantain2asr[vosk] to use Vosk."),
    "CanaryModel": (".canary_model", "Install NeMo dependencies to use Canary."),
    "ToneModel": (".tone_model", "Install plantain2asr[tone] or plantain2asr[tone-gpu], then install the T-One package from its source archive."),
}


def __getattr__(name):
    return resolve_optional_export(__name__, name, _LOCAL_EXPORTS)


__all__ = list(_LOCAL_EXPORTS.keys())
