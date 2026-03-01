__version__ = "0.1.5"
__author__  = "Artem Katsnelson"

# ── Core data types ──────────────────────────────────────────────────
from .dataloaders import BaseASRDataset, NeMoDataset, DagrusDataset, GolosDataset, RuDevicesDataset, AudioSample

# ── Pipeline building blocks ─────────────────────────────────────────
from .models import Models
from .metrics import Metrics
from .normalization import BaseNormalizer, SimpleNormalizer, DagrusNormalizer, GolosNormalizer
from .utils.functional import Filter, Sort, Take, Split

# ── Reporting (interactive server + extensible sections) ─────────────
from .reporting import (
    ReportServer,
    ReportBuilder,
    BaseSection,
    MetricsSection,
    ErrorFrequencySection,
    DiffSection,
)

# ── Analysis tools (optional — requires pandas, scikit-learn, etc.) ──
try:
    from .analysis import (
        WordErrorAnalyzer,
        DiffVisualizer,
        PerformanceAnalyzer,
        BootstrapAnalyzer,
        AgreementAnalyzer,
        TopicAnalyzer,
        HallucinationAnalyzer,
        DurationAnalyzer,
        NgramErrorAnalyzer,
        CalibrationAnalyzer,
        CorpusStatsAnalyzer,
        CorpusComparison,
        CorpusReport,
        VocabIntersectionAnalyzer,
        PosErrorAnalyzer,
        PosErrorComparison,
        PosErrorReport,
    )
except ImportError:
    pass

# ── Training (optional — requires torch + transformers) ──────────────
try:
    from .train import TrainingConfig, GigaAMTrainer
except ImportError:
    pass
