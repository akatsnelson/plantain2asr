__version__ = "1.0.2"
__author__  = "Artem Katsnelson"

from .utils.logging import configure_logging
from .utils.optional_imports import resolve_optional_export

# ── Core data types ──────────────────────────────────────────────────
from .dataloaders import BaseASRDataset, NeMoDataset, DagrusDataset, GolosDataset, RuDevicesDataset, AudioSample

# ── Pipeline building blocks ─────────────────────────────────────────
from .models import Models
from .metrics import Metrics
from .normalization import BaseNormalizer, SimpleNormalizer, DagrusNormalizer, GolosNormalizer
from .utils.functional import Filter, Sort, Take, Split
from .experiment import Experiment, ExperimentModelSpec
from .utils import get_logger

# ── Reporting (interactive server + extensible sections) ─────────────
from .reporting import (
    ReportServer,
    ReportBuilder,
    BaseSection,
    MetricsSection,
    ErrorFrequencySection,
    DiffSection,
)

_OPTIONAL_EXPORTS = {
    "ModelBenchmark": (".analysis", "Install plantain2asr[analysis] to use benchmark helpers."),
    "BenchmarkReport": (".analysis", "Install plantain2asr[analysis] to use benchmark helpers."),
    "BenchmarkResult": (".analysis", "Install plantain2asr[analysis] to use benchmark helpers."),
    "WordErrorAnalyzer": (".analysis", "Install plantain2asr[analysis] to use analysis modules."),
    "DiffVisualizer": (".analysis", "Install plantain2asr[analysis] to use analysis modules."),
    "PerformanceAnalyzer": (".analysis", "Install plantain2asr[analysis] to use analysis modules."),
    "BootstrapAnalyzer": (".analysis", "Install plantain2asr[analysis] to use analysis modules."),
    "AgreementAnalyzer": (".analysis", "Install plantain2asr[analysis] to use analysis modules."),
    "TopicAnalyzer": (".analysis", "Install plantain2asr[analysis] to use analysis modules."),
    "HallucinationAnalyzer": (".analysis", "Install plantain2asr[analysis] to use analysis modules."),
    "DurationAnalyzer": (".analysis", "Install plantain2asr[analysis] to use analysis modules."),
    "NgramErrorAnalyzer": (".analysis", "Install plantain2asr[analysis] to use analysis modules."),
    "CalibrationAnalyzer": (".analysis", "Install plantain2asr[analysis] to use analysis modules."),
    "CorpusStatsAnalyzer": (".analysis", "Install plantain2asr[analysis] to use analysis modules."),
    "CorpusComparison": (".analysis", "Install plantain2asr[analysis] to use analysis modules."),
    "CorpusReport": (".analysis", "Install plantain2asr[analysis] to use analysis modules."),
    "VocabIntersectionAnalyzer": (".analysis", "Install plantain2asr[analysis] to use analysis modules."),
    "PosErrorAnalyzer": (".analysis", "Install plantain2asr[analysis] to use analysis modules."),
    "PosErrorComparison": (".analysis", "Install plantain2asr[analysis] to use analysis modules."),
    "PosErrorReport": (".analysis", "Install plantain2asr[analysis] to use analysis modules."),
    "TrainingConfig": (".train", "Install plantain2asr[train] and compatible model extras to use training."),
    "BaseTrainer": (".train", "Install plantain2asr[train] and compatible model extras to use training."),
    "GigaAMTrainer": (".train", "Install plantain2asr[train] and compatible model extras to use training."),
    "CTCCharTokenizer": (".train", "Install plantain2asr[train] and compatible model extras to use training."),
    "CharCTCAudioDataset": (".train", "Install plantain2asr[train] and compatible model extras to use training."),
    "ctc_audio_collate_fn": (".train", "Install plantain2asr[train] and compatible model extras to use training."),
}


def __getattr__(name):
    return resolve_optional_export(__name__, name, _OPTIONAL_EXPORTS)


# ── Public API surface ───────────────────────────────────────────────
__all__ = [
    # metadata
    "__version__",
    "__author__",
    "configure_logging",
    "get_logger",
    # data
    "AudioSample",
    "BaseASRDataset",
    "NeMoDataset",
    "DagrusDataset",
    "GolosDataset",
    "RuDevicesDataset",
    # pipeline
    "Models",
    "Metrics",
    "Filter",
    "Sort",
    "Take",
    "Split",
    # normalization
    "BaseNormalizer",
    "SimpleNormalizer",
    "DagrusNormalizer",
    "GolosNormalizer",
    # experiment
    "Experiment",
    "ExperimentModelSpec",
    # reporting
    "ReportServer",
    "ReportBuilder",
    "BaseSection",
    "MetricsSection",
    "ErrorFrequencySection",
    "DiffSection",
    # benchmark (optional)
    "ModelBenchmark",
    "BenchmarkReport",
    "BenchmarkResult",
    # analysis (optional)
    "WordErrorAnalyzer",
    "DiffVisualizer",
    "PerformanceAnalyzer",
    "BootstrapAnalyzer",
    "AgreementAnalyzer",
    "TopicAnalyzer",
    "HallucinationAnalyzer",
    "DurationAnalyzer",
    "NgramErrorAnalyzer",
    "CalibrationAnalyzer",
    "CorpusStatsAnalyzer",
    "CorpusComparison",
    "CorpusReport",
    "VocabIntersectionAnalyzer",
    "PosErrorAnalyzer",
    "PosErrorComparison",
    "PosErrorReport",
    # training (optional)
    "TrainingConfig",
    "BaseTrainer",
    "GigaAMTrainer",
    "CTCCharTokenizer",
    "CharCTCAudioDataset",
    "ctc_audio_collate_fn",
]
