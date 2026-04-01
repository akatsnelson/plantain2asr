from .benchmark import ModelBenchmark, BenchmarkReport, BenchmarkResult
from ..utils.optional_imports import resolve_optional_export

_OPTIONAL_EXPORTS = {
    "WordErrorAnalyzer": (".word_error_rate", "Install plantain2asr[analysis] to use word-level analyzers."),
    "PerformanceAnalyzer": (".performance", "Install plantain2asr[analysis] to use performance analyzers."),
    "BootstrapAnalyzer": (".statistics", "Install plantain2asr[analysis] to use bootstrap analyzers."),
    "AgreementAnalyzer": (".agreement", "Install plantain2asr[analysis] to use agreement analyzers."),
    "TopicAnalyzer": (".topic_modeling", "Install plantain2asr[analysis] to use topic analyzers."),
    "HallucinationAnalyzer": (".hallucinations", "Install plantain2asr[analysis] to use hallucination analyzers."),
    "DurationAnalyzer": (".duration", "Install plantain2asr[analysis] to use duration analyzers."),
    "DiffVisualizer": (".diff_visualizer", "Install plantain2asr[analysis] to use diff visualizers."),
    "NgramErrorAnalyzer": (".ngram_errors", "Install plantain2asr[analysis] to use n-gram analyzers."),
    "CalibrationAnalyzer": (".calibration", "Install plantain2asr[analysis] to use calibration analyzers."),
    "CorpusStatsAnalyzer": (".corpus_stats", "Install plantain2asr[analysis] to use corpus analyzers."),
    "CorpusComparison": (".corpus_stats", "Install plantain2asr[analysis] to use corpus analyzers."),
    "CorpusReport": (".corpus_stats", "Install plantain2asr[analysis] to use corpus analyzers."),
    "VocabIntersectionAnalyzer": (".vocab_intersection", "Install plantain2asr[analysis] to use vocabulary analyzers."),
    "PosErrorAnalyzer": (".pos_errors", "Install plantain2asr[analysis] to use POS analyzers."),
    "PosErrorComparison": (".pos_errors", "Install plantain2asr[analysis] to use POS analyzers."),
    "PosErrorReport": (".pos_errors", "Install plantain2asr[analysis] to use POS analyzers."),
}


def __getattr__(name):
    return resolve_optional_export(__name__, name, _OPTIONAL_EXPORTS)


__all__ = [
    "ModelBenchmark",
    "BenchmarkReport",
    "BenchmarkResult",
    *list(_OPTIONAL_EXPORTS.keys()),
]
