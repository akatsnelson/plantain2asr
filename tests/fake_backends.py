from __future__ import annotations

import contextlib
import importlib
import sys
import types
from pathlib import Path


class _FakePhrase:
    def __init__(self, text: str):
        self.text = text


class _FakeTonePipeline:
    last_source = None

    def __init__(self, source: str):
        self.source = source

    @classmethod
    def from_local(cls, model_path):
        cls.last_source = ("local", str(model_path))
        return cls(str(model_path))

    @classmethod
    def from_hugging_face(cls, *args, **kwargs):
        repo_id = kwargs.get("repo_id") or kwargs.get("model_name")
        if args:
            repo_id = args[0]
        cls.last_source = ("hf", repo_id or "default")
        return cls(repo_id or "default")

    def forward_offline(self, _audio):
        return [_FakePhrase(f"tone:{self.source}")]


class _FakeStreamingCTCModel:
    @classmethod
    def from_local(cls, model_path):
        return cls()

    def __init__(self, session=None):
        self.session = session


class _FakeOnnxRuntime(types.SimpleNamespace):
    def __init__(self, providers):
        super().__init__()
        self._providers = list(providers)
        self.sessions = []

    def get_available_providers(self):
        return list(self._providers)

    def InferenceSession(self, model_path, providers=None):
        session = {"model_path": str(model_path), "providers": providers or []}
        self.sessions.append(session)
        return session


class _FakeTorchModule(types.SimpleNamespace):
    def __init__(self, cuda_available=True, mps_available=False):
        cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
        mps = types.SimpleNamespace(is_available=lambda: mps_available)
        cudnn = types.SimpleNamespace(enabled=True)
        backends = types.SimpleNamespace(mps=mps, cudnn=cudnn)
        super().__init__(
            cuda=cuda,
            backends=backends,
            zeros=lambda *args, **kwargs: [],
            tensor=lambda *args, **kwargs: [],
            no_grad=contextlib.nullcontext,
            load=lambda *args, **kwargs: {},
        )


class _FakeGigaAMModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = None
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def transcribe(self, path):
        return f"{self.model_name}:{Path(path).name}"

    def transcribe_longform(self, path):
        return f"{self.model_name}:long:{Path(path).name}"


class _FakeAutoModelWrapper:
    def __init__(self, model_name):
        inner = types.SimpleNamespace(
            prepare_wav=lambda path: ([0.0], types.SimpleNamespace(item=lambda: 1)),
            forward=lambda wav, lengths: ([], []),
            head=object(),
            decoding=types.SimpleNamespace(decode=lambda head, encoded, encoded_len: []),
        )
        self.model_name = model_name
        self.model = inner
        self.device = None
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def transcribe(self, path):
        return f"{self.model_name}:{Path(path).name}"


class _FakeWhisperProcessor:
    @classmethod
    def from_pretrained(cls, model_name):
        instance = cls()
        instance.model_name = model_name
        return instance

    def __call__(self, audios, sampling_rate, return_tensors, padding, return_attention_mask):
        return types.SimpleNamespace(
            input_features=types.SimpleNamespace(to=lambda device, dtype=None: f"features:{len(audios)}:{device}"),
            attention_mask=types.SimpleNamespace(to=lambda device: f"mask:{len(audios)}:{device}"),
        )

    def batch_decode(self, predicted_ids, skip_special_tokens=True):
        return list(predicted_ids)


class _FakeWhisperModel:
    @classmethod
    def from_pretrained(cls, model_name, torch_dtype):
        instance = cls()
        instance.model_name = model_name
        instance.torch_dtype = torch_dtype
        instance.device = None
        return instance

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def generate(self, input_features, attention_mask, language):
        return [f"{self.model_name}:{language}:sample-{idx}" for idx in range(1, 1 + 1)]


class _FakeVoskModel:
    def __init__(self, model_path):
        self.model_path = model_path


class _FakeKaldiRecognizer:
    def __init__(self, model, framerate):
        self.model = model
        self.framerate = framerate
        self.words_enabled = False
        self._accepted = False

    def SetWords(self, enabled):
        self.words_enabled = enabled

    def AcceptWaveform(self, data):
        if not self._accepted and data:
            self._accepted = True
            return True
        return False

    def Result(self):
        return '{"text": "vosk partial"}'

    def FinalResult(self):
        return '{"text": "vosk final"}'


def install_fake_tone(monkeypatch, providers=None):
    providers = providers or ["CPUExecutionProvider"]

    tone_module = types.ModuleType("tone")
    tone_module.StreamingCTCPipeline = _FakeTonePipeline
    tone_module.read_audio = lambda path: f"audio:{Path(path).name}"

    wrapper_module = types.ModuleType("tone.onnx_wrapper")
    wrapper_module.StreamingCTCModel = _FakeStreamingCTCModel

    ort_module = _FakeOnnxRuntime(providers)

    monkeypatch.setitem(sys.modules, "tone", tone_module)
    monkeypatch.setitem(sys.modules, "tone.onnx_wrapper", wrapper_module)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort_module)
    return ort_module


def install_fake_gigaam(monkeypatch, cuda_available=True, mps_available=False):
    torch_module = _FakeTorchModule(
        cuda_available=cuda_available,
        mps_available=mps_available,
    )
    gigaam_module = types.ModuleType("gigaam")
    gigaam_module.load_model = lambda model_name: _FakeGigaAMModel(model_name)

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoProcessor = object
    transformers_module.AutoModel = types.SimpleNamespace(
        from_pretrained=lambda repo_id, revision, trust_remote_code, low_cpu_mem_usage, device_map: _FakeAutoModelWrapper(revision)
    )

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "gigaam", gigaam_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    return torch_module


def install_fake_whisper(monkeypatch, cuda_available=True, mps_available=False):
    torch_module = _FakeTorchModule(
        cuda_available=cuda_available,
        mps_available=mps_available,
    )
    torch_module.bfloat16 = "bf16"
    torch_module.float16 = "fp16"
    torch_module.float32 = "fp32"
    torch_module.cuda.is_bf16_supported = lambda: True

    transformers_module = types.ModuleType("transformers")
    transformers_module.WhisperProcessor = _FakeWhisperProcessor
    transformers_module.WhisperForConditionalGeneration = _FakeWhisperModel

    librosa_module = types.ModuleType("librosa")
    librosa_module.load = lambda path, sr=16000: ([0.0, 0.1], sr)

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "librosa", librosa_module)
    return torch_module


def install_fake_vosk(monkeypatch):
    vosk_module = types.ModuleType("vosk")
    vosk_module.Model = _FakeVoskModel
    vosk_module.KaldiRecognizer = _FakeKaldiRecognizer
    vosk_module.SetLogLevel = lambda level: None
    monkeypatch.setitem(sys.modules, "vosk", vosk_module)
    return vosk_module


def import_fresh(module_name: str):
    sys.modules.pop(module_name, None)
    sys.modules.pop("plantain2asr.utils.device", None)
    return importlib.import_module(module_name)
