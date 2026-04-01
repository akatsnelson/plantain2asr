import csv
from pathlib import Path

import pytest

from plantain2asr import Experiment
from plantain2asr.dataloaders.base import BaseASRDataset
from plantain2asr.dataloaders.types import AudioSample
from plantain2asr.metrics.base import BaseMetric
from plantain2asr.models.base import BaseASRModel
from plantain2asr.normalization.base import BaseNormalizer


class ResearchDataset(BaseASRDataset):
    def __init__(self, root: Path):
        super().__init__()
        self.name = "ResearchDataset"
        self.cache_dir = root / "cache"
        self._samples = [
            AudioSample(
                id="sample-1.wav",
                audio_path=str(root / "sample-1.wav"),
                text="Alpha",
                duration=1.0,
                meta={"speaker": "A"},
            ),
            AudioSample(
                id="sample-2.wav",
                audio_path=str(root / "sample-2.wav"),
                text="Beta",
                duration=2.0,
                meta={"speaker": "B"},
            ),
        ]
        self._id_map = {sample.id: sample for sample in self._samples}


class CountingModel(BaseASRModel):
    def __init__(self, device="cpu"):
        self.calls = 0
        self.device = device

    @property
    def name(self) -> str:
        return f"CountingModel-{self.device}"

    def transcribe(self, audio_path):
        self.calls += 1
        stem = Path(audio_path).stem
        return stem.replace("sample-1", "ALPHA").replace("sample-2", "BETA")


class ExactMatchMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "ExactMatch"

    def calculate(self, reference: str, hypothesis: str) -> float:
        return 0.0 if reference == hypothesis else 100.0


class LowerNormalizer(BaseNormalizer):
    def normalize_ref(self, text: str) -> str:
        return (text or "").lower()


class BenchmarkEchoModel(BaseASRModel):
    def __init__(self, device="cpu"):
        self.device = device
        self.batch_size = 2

    @property
    def name(self) -> str:
        return f"BenchmarkEcho-{self.device}"

    def transcribe(self, audio_path):
        return Path(audio_path).stem

    def transcribe_batch(self, audio_paths):
        return [Path(path).stem for path in audio_paths]


@pytest.mark.core
def test_experiment_runs_pipeline_and_exports_research_friendly_artifacts(tmp_path):
    for file_name in ("sample-1.wav", "sample-2.wav"):
        (tmp_path / file_name).write_bytes(b"")

    dataset = ResearchDataset(tmp_path)
    experiment = Experiment(
        dataset=dataset,
        models=[lambda: CountingModel()],
        normalizer=LowerNormalizer(),
        metrics=[ExactMatchMetric()],
    )

    result_dataset = experiment.run()
    summary = experiment.summary(metrics=["ExactMatch"])
    csv_path = Path(experiment.save_csv(str(tmp_path / "flat.csv")))
    summary_path = Path(experiment.save_summary_csv(str(tmp_path / "summary.csv")))

    assert result_dataset is experiment.analysis_dataset
    assert result_dataset is not dataset
    assert dataset[0].asr_results["CountingModel-cpu"]["hypothesis"] == "ALPHA"
    assert result_dataset[0].asr_results["CountingModel-cpu"]["hypothesis"] == "alpha"
    assert summary[0]["avg_ExactMatch"] == 0.0
    assert csv_path.exists()
    assert summary_path.exists()

    with open(csv_path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["model"] == "CountingModel-cpu"
    assert rows[0]["speaker"] == "A"


@pytest.mark.core
def test_experiment_is_itself_pipeline_compatible(tmp_path):
    for file_name in ("sample-1.wav", "sample-2.wav"):
        (tmp_path / file_name).write_bytes(b"")

    dataset = ResearchDataset(tmp_path)
    experiment = Experiment(
        dataset=dataset,
        models=[lambda: CountingModel()],
        normalizer=LowerNormalizer(),
        metrics=[ExactMatchMetric()],
    )

    result = dataset >> experiment

    assert result is experiment.analysis_dataset
    assert result[0].asr_results["CountingModel-cpu"]["metrics"]["ExactMatch"] == 0.0


@pytest.mark.core
def test_experiment_force_recompute_controls_cache_behavior(tmp_path):
    for file_name in ("sample-1.wav", "sample-2.wav"):
        (tmp_path / file_name).write_bytes(b"")

    dataset = ResearchDataset(tmp_path)
    model = CountingModel()
    experiment = Experiment(dataset=dataset, models=[model])

    experiment.run()
    assert model.calls == 2

    experiment.run()
    assert model.calls == 2

    experiment.run(force_recompute=True)
    assert model.calls == 4


@pytest.mark.core
def test_experiment_benchmark_reuses_modular_model_factories(tmp_path):
    for file_name in ("sample-1.wav", "sample-2.wav"):
        (tmp_path / file_name).write_bytes(b"")

    dataset = ResearchDataset(tmp_path)
    experiment = Experiment(dataset=dataset, models=[lambda device="cpu": BenchmarkEchoModel(device=device)])

    report = experiment.benchmark(devices=["cpu"], sample_limit=2, batch_size=2)
    output_path = Path(report.save_csv(str(tmp_path / "benchmark.csv")))

    assert len(report.results) == 1
    assert report.results[0].requested_device == "cpu"
    assert output_path.exists()


@pytest.mark.core
def test_experiment_exposes_research_friendly_compare_and_error_case_scenarios(tmp_path):
    for file_name in ("sample-1.wav", "sample-2.wav"):
        (tmp_path / file_name).write_bytes(b"")

    dataset = ResearchDataset(tmp_path)
    experiment = Experiment(
        dataset=dataset,
        models=[lambda: CountingModel()],
        normalizer=LowerNormalizer(),
        metrics=[ExactMatchMetric()],
    )
    experiment.evaluate()

    comparison = experiment.compare_models(metrics=["ExactMatch"])
    leaderboard = experiment.leaderboard(primary_metric="ExactMatch", metrics=["ExactMatch"])
    error_rows = experiment.error_cases(metric="ExactMatch", min_value=0.0)
    error_csv = Path(experiment.export_error_cases(str(tmp_path / "errors.csv"), metric="ExactMatch", min_value=0.0))
    leaderboard_csv = Path(
        experiment.save_leaderboard_csv(
            str(tmp_path / "leaderboard.csv"),
            primary_metric="ExactMatch",
            metrics=["ExactMatch"],
        )
    )

    assert comparison[0]["avg_ExactMatch"] == 0.0
    assert leaderboard[0]["rank"] == 1
    assert error_rows == []
    assert error_csv.exists()
    assert leaderboard_csv.exists()


@pytest.mark.core
def test_experiment_error_cases_capture_failures_and_static_report_export(tmp_path):
    for file_name in ("sample-1.wav", "sample-2.wav"):
        (tmp_path / file_name).write_bytes(b"")

    class PartialFailureModel(BaseASRModel):
        @property
        def name(self) -> str:
            return "PartialFailure"

        def transcribe(self, audio_path):
            if str(audio_path).endswith("sample-2.wav"):
                raise RuntimeError("boom")
            return "Alpha"

    dataset = ResearchDataset(tmp_path)
    experiment = Experiment(
        dataset=dataset,
        models=[PartialFailureModel()],
        normalizer=LowerNormalizer(),
        metrics=[ExactMatchMetric()],
    )
    experiment.run()

    error_rows = experiment.error_cases(include_failures=True)
    html_path = Path(experiment.save_report_html(str(tmp_path / "report.html")))

    assert len(error_rows) == 1
    assert error_rows[0]["model"] == "PartialFailure"
    assert error_rows[0]["error"] == "boom"
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "window.__PLANTAIN_STATIC__ = true;" in html
    assert "window.__PLANTAIN_STATIC_DATA__" in html


@pytest.mark.core
def test_experiment_presets_prepare_comparison_and_thesis_tables(tmp_path):
    for file_name in ("sample-1.wav", "sample-2.wav"):
        (tmp_path / file_name).write_bytes(b"")

    dataset = ResearchDataset(tmp_path)
    experiment = Experiment(
        dataset=dataset,
        models=[lambda: CountingModel()],
        normalizer=LowerNormalizer(),
        metrics=[ExactMatchMetric()],
    )

    comparison_payload = experiment.compare_on_corpus(
        metrics="ExactMatch",
        primary_metric="ExactMatch",
    )
    thesis_payload = experiment.prepare_thesis_tables(
        output_dir=str(tmp_path / "tables"),
        metrics="ExactMatch",
        primary_metric="ExactMatch",
    )

    assert comparison_payload["leaderboard"][0]["rank"] == 1
    assert comparison_payload["comparison"][0]["avg_ExactMatch"] == 0.0
    assert Path(thesis_payload["results_csv"]).exists()
    assert Path(thesis_payload["summary_csv"]).exists()
    assert Path(thesis_payload["leaderboard_csv"]).exists()
    assert Path(thesis_payload["error_cases_csv"]).exists()


@pytest.mark.core
def test_experiment_appendix_bundle_can_include_static_report_and_benchmark(tmp_path):
    for file_name in ("sample-1.wav", "sample-2.wav"):
        (tmp_path / file_name).write_bytes(b"")

    dataset = ResearchDataset(tmp_path)
    experiment = Experiment(
        dataset=dataset,
        models=[lambda device="cpu": BenchmarkEchoModel(device=device)],
        metrics=[],
    )

    bundle = experiment.export_appendix_bundle(
        output_dir=str(tmp_path / "bundle"),
        include_benchmark=True,
        benchmark_devices=["cpu"],
        benchmark_sample_limit=2,
        benchmark_batch_size=2,
    )

    assert Path(bundle["results_csv"]).exists()
    assert Path(bundle["summary_csv"]).exists()
    assert Path(bundle["leaderboard_csv"]).exists()
    assert Path(bundle["error_cases_csv"]).exists()
    assert Path(bundle["report_html"]).exists()
    assert Path(bundle["benchmark_csv"]).exists()
    assert len(bundle["benchmark_rows"]) == 1
