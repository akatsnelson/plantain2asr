from __future__ import annotations

import contextlib
import os
import sys
import types

import pytest

np = pytest.importorskip("numpy")

from plantain2asr.dataloaders.base import BaseASRDataset
from plantain2asr.dataloaders.types import AudioSample
from plantain2asr.models.base import BaseASRModel
from tests.fake_backends import import_fresh


class _FakeTensor:
    def __init__(self, data):
        self.data = np.array(data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def to(self, *_args, **_kwargs):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.data)

    def mean(self, dim=None, keepdim=False):
        return _FakeTensor(self.data.mean(axis=dim, keepdims=keepdim))

    def squeeze(self, axis=None):
        return _FakeTensor(np.squeeze(self.data, axis=axis))

    def unsqueeze(self, axis):
        return _FakeTensor(np.expand_dims(self.data, axis))

    def item(self):
        return float(self.data.reshape(-1)[0])

    def numel(self):
        return int(self.data.size)

    def tolist(self):
        return self.data.tolist()

    def backward(self):
        return None

    def __len__(self):
        return len(self.data)

    def __float__(self):
        return self.item()


class _FakeAdamW:
    def __init__(self, params, lr, weight_decay):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self):
        self.zero_grad_calls += 1

    def step(self):
        self.step_calls += 1


class _FakeDatasetBase:
    pass


class _FakeDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, collate_fn=None, **_kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            indices = list(reversed(indices))
        for start in range(0, len(indices), self.batch_size):
            items = [self.dataset[idx] for idx in indices[start : start + self.batch_size]]
            yield self.collate_fn(items) if self.collate_fn else items


def _pad_sequence(items, batch_first=True, padding_value=0.0):
    arrays = [item.data for item in items]
    max_len = max(arr.shape[0] for arr in arrays)
    padded = []
    for arr in arrays:
        pad_width = max_len - arr.shape[0]
        padded.append(np.pad(arr, (0, pad_width), constant_values=padding_value))
    stacked = np.stack(padded, axis=0 if batch_first else 1)
    return _FakeTensor(stacked)


def install_fake_train_runtime(monkeypatch):
    for module_name in (
        "plantain2asr",
        "plantain2asr.train",
        "plantain2asr.train.base_trainer",
        "plantain2asr.train.dataset",
        "plantain2asr.train.gigaam_trainer",
    ):
        sys.modules.pop(module_name, None)

    torch_module = types.ModuleType("torch")
    torch_module.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_module.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False),
        cudnn=types.SimpleNamespace(enabled=True),
    )
    torch_module.optim = types.SimpleNamespace(AdamW=_FakeAdamW)
    torch_module.nn = types.SimpleNamespace(
        Module=object,
        utils=types.SimpleNamespace(
            clip_grad_norm_=lambda *_args, **_kwargs: None,
            rnn=types.SimpleNamespace(pad_sequence=_pad_sequence),
        ),
    )
    torch_module.long = "long"
    torch_module.Tensor = _FakeTensor
    torch_module.no_grad = contextlib.nullcontext
    torch_module.device = lambda name: name
    torch_module.from_numpy = lambda arr: _FakeTensor(arr)
    torch_module.tensor = lambda data, dtype=None: _FakeTensor(data)
    torch_module.argmax = lambda tensor, dim=-1: _FakeTensor(np.argmax(tensor.data, axis=dim))
    torch_module.zeros = lambda *shape, **_kwargs: _FakeTensor(np.zeros(shape))

    torch_utils_module = types.ModuleType("torch.utils")
    torch_utils_data_module = types.ModuleType("torch.utils.data")
    torch_utils_data_module.Dataset = _FakeDatasetBase
    torch_utils_data_module.DataLoader = _FakeDataLoader

    torch_nn_module = types.ModuleType("torch.nn")
    torch_nn_utils_module = types.ModuleType("torch.nn.utils")
    torch_nn_utils_rnn_module = types.ModuleType("torch.nn.utils.rnn")
    torch_nn_utils_rnn_module.pad_sequence = _pad_sequence

    torchaudio_module = types.ModuleType("torchaudio")
    torchaudio_module.load = lambda _path: (_FakeTensor([[0.1, 0.2, 0.3]]), 16000)
    torchaudio_module.transforms = types.SimpleNamespace(
        Resample=lambda _src, _dst: (lambda tensor: tensor)
    )

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils_module)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_utils_data_module)
    monkeypatch.setitem(sys.modules, "torch.nn", torch_nn_module)
    monkeypatch.setitem(sys.modules, "torch.nn.utils", torch_nn_utils_module)
    monkeypatch.setitem(sys.modules, "torch.nn.utils.rnn", torch_nn_utils_rnn_module)
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio_module)


class _TinyDataset(BaseASRDataset):
    def __init__(self):
        super().__init__()
        self.name = "tiny-train"
        self._samples = [
            AudioSample(id="s1", audio_path="/tmp/s1.wav", text="аб"),
            AudioSample(id="s2", audio_path="/tmp/s2.wav", text="аб"),
        ]


class _FakeProcessor:
    def __init__(self):
        self.saved_paths = []

    def save_pretrained(self, path):
        self.saved_paths.append(path)
        with open(os.path.join(path, "processor.txt"), "w", encoding="utf-8") as handle:
            handle.write("processor")


class _FakeTrainModule:
    def __init__(self):
        self.config = types.SimpleNamespace(
            cfg={
                "model": {
                    "cfg": {
                        "decoding": {"vocabulary": [" ", "а", "б"]},
                        "head": {"num_classes": 4},
                    }
                }
            }
        )
        self.saved_paths = []
        self.freeze_calls = 0
        self.train_calls = 0
        self.eval_calls = 0
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def train(self):
        self.train_calls += 1
        return self

    def eval(self):
        self.eval_calls += 1
        return self

    def parameters(self):
        return [object()]

    def freeze_feature_encoder(self):
        self.freeze_calls += 1

    def save_pretrained(self, path):
        self.saved_paths.append(path)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "weights.txt"), "w", encoding="utf-8") as handle:
            handle.write("weights")

    def __call__(self, input_values, labels):
        batch_size = input_values.shape[0]
        logits = np.zeros((batch_size, 4, 4), dtype=float)
        logits[:, 0, 1] = 1.0
        logits[:, 1, 1] = 1.0
        logits[:, 2, 3] = 1.0
        logits[:, 3, 2] = 1.0
        return types.SimpleNamespace(loss=_FakeTensor(0.25), logits=_FakeTensor(logits))


class _TrainableCTCModel(BaseASRModel):
    def __init__(self):
        self._name = "Fake-GigaAM-v3-ctc"
        self.model = _FakeTrainModule()
        self._inner_model = self.model
        self.processor = _FakeProcessor()

    @property
    def name(self):
        return self._name

    @property
    def supports_training(self):
        return True

    @property
    def training_backend(self):
        return "ctc"

    def transcribe(self, audio_path):
        return str(audio_path)

    def get_training_components(self):
        return self.model, self.processor, None


class _UnsupportedTrainModel(BaseASRModel):
    def __init__(self):
        self._name = "Fake-RNNT"
        self.model = _FakeTrainModule()

    @property
    def name(self):
        return self._name

    def transcribe(self, audio_path):
        return str(audio_path)

    def training_not_supported_reason(self):
        return "RNNT training is not available in the current plantain pipeline."


def _make_config(tmp_path):
    train = import_fresh("plantain2asr.train")
    return train.TrainingConfig(
        output_dir=str(tmp_path),
        project_name=None,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_steps=1,
        fp16=False,
        warmup_steps=0,
        group_by_length=False,
        gradient_checkpointing=False,
        save_steps=0,
        eval_steps=0,
        save_total_limit=0,
    )


@pytest.mark.train
def test_dataset_shift_trainer_returns_original_dataset(monkeypatch, tmp_path):
    install_fake_train_runtime(monkeypatch)
    train_module = import_fresh("plantain2asr.train.gigaam_trainer")
    dataset = _TinyDataset()
    model = _TrainableCTCModel()
    trainer = train_module.GigaAMTrainer(model=model, config=_make_config(tmp_path), eval_dataset=dataset)

    result = dataset >> trainer

    assert result is dataset
    assert trainer.trained_model is model
    assert trainer.last_train_dataset is dataset
    assert trainer.training_summary["epochs"] == 1
    assert any(path.endswith("best_model") for path in model.model.saved_paths)
    assert any(path.endswith("checkpoint-epoch-1") for path in model.model.saved_paths)


@pytest.mark.train
def test_gigaam_trainer_uses_blank_id_for_ctc_decode(monkeypatch, tmp_path):
    install_fake_train_runtime(monkeypatch)
    train_module = import_fresh("plantain2asr.train.gigaam_trainer")
    dataset = _TinyDataset()
    trainer = train_module.GigaAMTrainer(
        model=_TrainableCTCModel(),
        config=_make_config(tmp_path),
        eval_dataset=dataset,
    )

    trainer.prepare_training(dataset)
    batch = next(iter(trainer.val_loader))
    result = trainer.eval_step(batch)

    assert result["preds"] == ["аб", "аб"]
    assert result["refs"] == ["аб", "аб"]


@pytest.mark.train
def test_gigaam_trainer_fails_fast_for_unsupported_backend(monkeypatch, tmp_path):
    install_fake_train_runtime(monkeypatch)
    train_module = import_fresh("plantain2asr.train.gigaam_trainer")
    dataset = _TinyDataset()
    trainer = train_module.GigaAMTrainer(
        model=_UnsupportedTrainModel(),
        config=_make_config(tmp_path),
        eval_dataset=dataset,
    )

    with pytest.raises(NotImplementedError, match="RNNT training is not available"):
        trainer.prepare_training(dataset)


@pytest.mark.train
def test_root_exports_train_api_when_runtime_is_available(monkeypatch):
    install_fake_train_runtime(monkeypatch)
    root = import_fresh("plantain2asr")

    assert root.BaseTrainer.__name__ == "BaseTrainer"
    assert root.GigaAMTrainer.__name__ == "GigaAMTrainer"
