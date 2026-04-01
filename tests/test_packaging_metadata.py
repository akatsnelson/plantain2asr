import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TONE_ARCHIVE = "https://github.com/voicekit-team/T-one/archive/3c5b6c015038173840e62cea99e10cdb1c759116.tar.gz"


@pytest.mark.core
def test_optional_dependency_profiles_cover_single_environment_use_case():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    for extra_name in (
        "tone",
        "tone-cpu",
        "tone-gpu",
        "gigaam-v2",
        "gigaam-v3",
        "gigaam",
        "whisper",
        "vosk",
        "asr-cpu",
        "asr-gpu",
        "all",
        "all-gpu",
    ):
        assert extra_name in extras

    tone_cpu = extras["tone-cpu"]
    tone_gpu = extras["tone-gpu"]
    asr_cpu = extras["asr-cpu"]
    asr_gpu = extras["asr-gpu"]
    all_cpu = extras["all"]
    all_gpu = extras["all-gpu"]

    assert any(TONE_ARCHIVE in dep for dep in tone_cpu)
    assert any(dep.startswith("onnxruntime>=") for dep in tone_cpu)
    assert any(dep.startswith("onnxruntime-gpu>=") for dep in tone_gpu)
    assert any(TONE_ARCHIVE in dep for dep in extras["tone"])

    assert "gigaam>=0.1.0" in asr_cpu
    assert "gigaam>=0.1.0" in asr_gpu
    assert "gigaam>=0.1.0" in all_cpu
    assert "gigaam>=0.1.0" in all_gpu
    assert "librosa>=0.10" in extras["whisper"]
    assert "vosk>=0.3.45" in asr_cpu
    assert "vosk>=0.3.45" in asr_gpu
    assert "vosk>=0.3.45" in extras["vosk"]
    assert "transformers>=4.40,<5" in extras["gigaam-v3"]
    assert any(dep.startswith("torch>=") for dep in extras["gigaam"])
    assert any(dep.startswith("torchaudio>=") for dep in extras["gigaam"])
    assert any(dep.startswith("torch>=") for dep in asr_cpu)
    assert any(dep.startswith("torchaudio>=") for dep in asr_cpu)
    assert any(dep.startswith("torch>=") for dep in asr_gpu)
    assert any(dep.startswith("torchaudio>=") for dep in asr_gpu)
    assert any(dep.startswith("torch>=") for dep in all_cpu)
    assert any(dep.startswith("torchaudio>=") for dep in all_cpu)
    assert any(dep.startswith("torch>=") for dep in all_gpu)
    assert any(dep.startswith("torchaudio>=") for dep in all_gpu)
    assert all(not dep.startswith("torchaudio") for dep in extras["gigaam-v2"])
    assert any(TONE_ARCHIVE in dep for dep in asr_cpu)
    assert any(TONE_ARCHIVE in dep for dep in asr_gpu)
    assert any(TONE_ARCHIVE in dep for dep in all_cpu)
    assert any(TONE_ARCHIVE in dep for dep in all_gpu)
    assert all("git+" not in dep for deps in extras.values() for dep in deps)


@pytest.mark.core
def test_requirements_file_tracks_development_profile():
    requirements_text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "-e .[asr-cpu,analysis,train]" in requirements_text
    assert "pytest>=" in requirements_text


@pytest.mark.core
def test_release_workflows_gate_publish_and_validate_public_profiles():
    ci_text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    publish_text = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")

    assert 'python -m pip install ".[tone-cpu]" pytest' in ci_text
    assert 'python -m pip install ".[gigaam]" pytest' in ci_text
    assert 'python -m pip install ".[whisper]" pytest' in ci_text
    assert 'python -m pip install ".[vosk]" pytest' in ci_text
    assert 'python -m pip install ".[train]" pytest' in ci_text
    assert 'python -m pip install ".[asr-cpu]" pytest' in ci_text
    assert 'extra: ["analysis", "asr-cpu", "all"]' in ci_text
    assert 'python -m pip install ".[${{ matrix.extra }}]"' in ci_text
    assert "mkdocs build --strict" in ci_text

    assert "python -m build" in publish_text
    assert "twine check dist/*" in publish_text
    assert "python -m pytest -m core" in publish_text
    assert "mkdocs build --strict" in publish_text


@pytest.mark.core
def test_release_metadata_is_no_longer_alpha():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    assert project["version"] == "1.0.0"
    assert "Development Status :: 5 - Production/Stable" in project["classifiers"]
    assert all("Alpha" not in classifier for classifier in project["classifiers"])
