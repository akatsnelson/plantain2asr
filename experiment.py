from __future__ import annotations

import csv
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Union

from .analysis import BenchmarkReport, ModelBenchmark
from .core.processor import Processor
from .dataloaders.base import BaseASRDataset
from .metrics.base import BaseMetric
from .models.base import BaseASRModel
from .reporting import ReportBuilder, ReportServer

try:
    from .normalization.base import BaseNormalizer
except ImportError:  # pragma: no cover - defensive import only
    BaseNormalizer = object


ModelLike = Union[BaseASRModel, Callable[..., BaseASRModel], "ExperimentModelSpec"]
MetricLike = Union[BaseMetric, Sequence[BaseMetric]]


def _flatten_metrics(metrics: Optional[MetricLike]) -> List[BaseMetric]:
    if metrics is None:
        return []
    if isinstance(metrics, BaseMetric):
        return [metrics]
    return list(metrics)


@dataclass
class ExperimentModelSpec:
    """
    Регистрирует модель для исследовательского сценария, не ломая modular pipeline.

    Можно передать:
    - готовый инстанс модели
    - factory/callable, который строит модель лениво
    """

    source: ModelLike
    label: Optional[str] = None
    _model: Optional[BaseASRModel] = None

    def resolve_model(self) -> BaseASRModel:
        if self._model is not None:
            return self._model

        if isinstance(self.source, BaseASRModel):
            self._model = self.source
            return self._model

        factory = self.source
        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):
            signature = None

        if signature is not None and (
            "device" in signature.parameters
            or any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in signature.parameters.values()
            )
        ):
            self._model = factory(device="auto")
        else:
            self._model = factory()
        return self._model

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        if self._model is not None:
            return self._model.name
        if isinstance(self.source, BaseASRModel):
            return self.source.name
        return getattr(self.source, "__name__", type(self.source).__name__)

    def benchmark_factory(self) -> Callable[..., BaseASRModel]:
        if not isinstance(self.source, BaseASRModel):
            return self.source

        model = self.resolve_model()

        def _existing_instance_factory(device: str = "cpu"):
            actual = getattr(model, "device", None)
            if actual and actual != device:
                raise ValueError(
                    f"Model '{model.name}' is already instantiated on '{actual}'. "
                    "Register it via a factory/callable to benchmark multiple devices."
                )
            return model

        return _existing_instance_factory


class Experiment(Processor):
    """
    Верхнеуровневый исследовательский façade поверх pipeline API.

    Он не заменяет `dataset >> model >> metric`, а композиционно собирает
    этот сценарий в повторяемый и дружелюбный workflow.
    """

    def __init__(
        self,
        dataset: BaseASRDataset,
        models: Optional[Sequence[ModelLike]] = None,
        normalizer: Optional[BaseNormalizer] = None,
        metrics: Optional[MetricLike] = None,
        name: Optional[str] = None,
    ):
        self.name = name or getattr(dataset, "name", "experiment")
        self.dataset = dataset
        self.analysis_dataset = dataset
        self.normalizer = normalizer
        self.metrics = _flatten_metrics(metrics)
        self.model_specs: List[ExperimentModelSpec] = []
        self.benchmark_report: Optional[BenchmarkReport] = None
        self.report_server: Optional[ReportServer] = None

        for model in models or []:
            self.add_model(model)

    @staticmethod
    def _metric_prefers_lower(metric_name: str) -> bool:
        key = metric_name.lower()
        return any(token in key for token in ("wer", "cer", "mer", "wil", "loss", "error", "ratio"))

    @staticmethod
    def _rows_to_csv(rows: List[dict], output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not rows:
            path.write_text("", encoding="utf-8")
            return str(path)

        fieldnames = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)

        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return str(path)

    @staticmethod
    def _normalise_metric_names(metrics: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
        if metrics is None:
            return None
        if isinstance(metrics, str):
            return [metrics]
        return list(metrics)

    def apply_to(self, dataset: BaseASRDataset) -> BaseASRDataset:
        self.dataset = dataset
        self.analysis_dataset = dataset
        return self.run()

    def add_model(self, model: ModelLike, label: Optional[str] = None) -> "Experiment":
        if isinstance(model, ExperimentModelSpec):
            self.model_specs.append(model)
        else:
            self.model_specs.append(ExperimentModelSpec(source=model, label=label))
        return self

    def add_metric(self, metric: BaseMetric) -> "Experiment":
        self.metrics.append(metric)
        return self

    def set_normalizer(self, normalizer: Optional[BaseNormalizer]) -> "Experiment":
        self.normalizer = normalizer
        return self

    def resolved_models(self) -> List[BaseASRModel]:
        return [spec.resolve_model() for spec in self.model_specs]

    def run_inference(
        self,
        force_recompute: bool = False,
        batch_size: Optional[int] = None,
        save_step: Optional[int] = None,
    ) -> BaseASRDataset:
        for model in self.resolved_models():
            self.dataset.run_model(
                model,
                batch_size=batch_size,
                save_step=save_step,
                force_process=force_recompute,
            )
        self.analysis_dataset = self.dataset
        return self.dataset

    def run_metrics(self, force: bool = False) -> BaseASRDataset:
        target = self.analysis_dataset
        for metric in self.metrics:
            target.evaluate_metric(metric, force=force)
        return target

    def normalize(self) -> BaseASRDataset:
        if self.normalizer is None:
            self.analysis_dataset = self.dataset
        else:
            self.analysis_dataset = self.dataset >> self.normalizer
        return self.analysis_dataset

    def run(
        self,
        force_recompute: bool = False,
        metric_force: bool = False,
        batch_size: Optional[int] = None,
        save_step: Optional[int] = None,
    ) -> BaseASRDataset:
        self.run_inference(
            force_recompute=force_recompute,
            batch_size=batch_size,
            save_step=save_step,
        )
        self.normalize()
        self.run_metrics(force=metric_force)
        return self.analysis_dataset

    def benchmark(
        self,
        devices: Optional[List[str]] = None,
        sample_limit: int = 20,
        warmup_samples: int = 1,
        batch_size: Optional[int] = None,
        models: Optional[Sequence[Union[ExperimentModelSpec, ModelLike]]] = None,
    ) -> BenchmarkReport:
        specs: List[ExperimentModelSpec] = []
        if models is None:
            specs = list(self.model_specs)
        else:
            for model in models:
                if isinstance(model, ExperimentModelSpec):
                    specs.append(model)
                else:
                    specs.append(ExperimentModelSpec(source=model))

        results = []
        for spec in specs:
            bench = ModelBenchmark(
                sample_limit=sample_limit,
                warmup_samples=warmup_samples,
                batch_size=batch_size,
                devices=devices,
            )
            report = bench.run(self.dataset, model_factory=spec.benchmark_factory(), devices=devices)
            results.extend(report.results)

        self.benchmark_report = BenchmarkReport(results)
        return self.benchmark_report

    def summary(self, metrics: Optional[List[str]] = None) -> List[dict]:
        return self.analysis_dataset.summarize_by_model(metrics=metrics)

    def compare_models(
        self,
        metrics: Optional[List[str]] = None,
        sort_by: Optional[str] = None,
        ascending: Optional[bool] = None,
    ) -> List[dict]:
        rows = list(self.summary(metrics=metrics))
        if not rows:
            return rows

        resolved_sort = sort_by
        if resolved_sort is None and metrics:
            first_metric = metrics[0]
            candidate = f"avg_{first_metric}"
            if candidate in rows[0]:
                resolved_sort = candidate

        if resolved_sort is None:
            return rows

        if ascending is None:
            ascending = self._metric_prefers_lower(resolved_sort)

        return sorted(
            rows,
            key=lambda row: (row.get(resolved_sort) is None, row.get(resolved_sort)),
            reverse=not ascending,
        )

    def leaderboard(
        self,
        primary_metric: Optional[str] = None,
        ascending: Optional[bool] = None,
        metrics: Optional[List[str]] = None,
    ) -> List[dict]:
        sort_by = f"avg_{primary_metric}" if primary_metric else None
        ordered = self.compare_models(metrics=metrics, sort_by=sort_by, ascending=ascending)
        leaderboard_rows = []
        for idx, row in enumerate(ordered, start=1):
            enriched = dict(row)
            enriched["rank"] = idx
            leaderboard_rows.append(enriched)
        return leaderboard_rows

    def save_leaderboard_csv(
        self,
        output_path: str,
        primary_metric: Optional[str] = None,
        ascending: Optional[bool] = None,
        metrics: Optional[List[str]] = None,
    ) -> str:
        rows = self.leaderboard(primary_metric=primary_metric, ascending=ascending, metrics=metrics)
        return self._rows_to_csv(rows, output_path)

    def results_table(self) -> List[dict]:
        return self.analysis_dataset.iter_results_rows()

    def error_cases(
        self,
        model: Optional[str] = None,
        metric: Optional[str] = None,
        min_value: float = 0.0,
        include_failures: bool = True,
        limit: Optional[int] = None,
    ) -> List[dict]:
        rows: List[dict] = []
        for sample in self.analysis_dataset:
            for model_name, result in sample.asr_results.items():
                if model is not None and model_name != model:
                    continue

                metrics_dict = result.get("metrics") or {}
                has_failure = bool(result.get("error"))
                has_metric_issue = False

                if metric is not None:
                    value = metrics_dict.get(metric)
                    if isinstance(value, (int, float)) and value > min_value:
                        has_metric_issue = True
                else:
                    has_metric_issue = any(
                        isinstance(value, (int, float)) and value > min_value
                        for value in metrics_dict.values()
                    )

                if not has_metric_issue and not (include_failures and has_failure):
                    continue

                row = {
                    "id": sample.id,
                    "audio_path": sample.audio_path,
                    "reference": sample.text,
                    "duration": sample.duration,
                    "model": model_name,
                    "hypothesis": result.get("hypothesis"),
                    "processing_time": result.get("processing_time"),
                    "error": result.get("error"),
                    **sample.meta,
                }
                row.update(metrics_dict)
                rows.append(row)

        if metric is not None:
            rows.sort(key=lambda row: row.get(metric, float("-inf")), reverse=True)
        elif include_failures:
            rows.sort(key=lambda row: (not bool(row.get("error")), row.get("processing_time") or 0.0))

        if limit is not None:
            return rows[:limit]
        return rows

    def export_error_cases(
        self,
        output_path: str,
        model: Optional[str] = None,
        metric: Optional[str] = None,
        min_value: float = 0.0,
        include_failures: bool = True,
        limit: Optional[int] = None,
    ) -> str:
        rows = self.error_cases(
            model=model,
            metric=metric,
            min_value=min_value,
            include_failures=include_failures,
            limit=limit,
        )
        return self._rows_to_csv(rows, output_path)

    def save_summary_csv(self, output_path: str, metrics: Optional[List[str]] = None) -> str:
        rows = self.summary(metrics=metrics)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not rows:
            path.write_text("", encoding="utf-8")
            return str(path)

        fieldnames = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)

        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return str(path)

    def compare_on_corpus(
        self,
        metrics: Optional[Union[str, List[str]]] = None,
        primary_metric: Optional[str] = None,
        force_recompute: bool = False,
        metric_force: bool = False,
        batch_size: Optional[int] = None,
        save_step: Optional[int] = None,
    ) -> dict:
        selected_metrics = self._normalise_metric_names(metrics)
        self.evaluate(
            force_recompute=force_recompute,
            metric_force=metric_force,
            batch_size=batch_size,
            save_step=save_step,
        )
        return {
            "dataset": self.analysis_dataset,
            "results": self.results_table(),
            "summary": self.summary(metrics=selected_metrics),
            "comparison": self.compare_models(metrics=selected_metrics),
            "leaderboard": self.leaderboard(
                primary_metric=primary_metric,
                metrics=selected_metrics,
            ),
        }

    def benchmark_models(
        self,
        devices: Optional[List[str]] = None,
        sample_limit: int = 20,
        warmup_samples: int = 1,
        batch_size: Optional[int] = None,
        output_path: Optional[str] = None,
        models: Optional[Sequence[Union[ExperimentModelSpec, ModelLike]]] = None,
    ) -> dict:
        report = self.benchmark(
            devices=devices,
            sample_limit=sample_limit,
            warmup_samples=warmup_samples,
            batch_size=batch_size,
            models=models,
        )
        csv_path = report.save_csv(output_path) if output_path is not None else None
        return {
            "report": report,
            "rows": report.to_rows(),
            "csv_path": csv_path,
        }

    def prepare_thesis_tables(
        self,
        output_dir: str,
        metrics: Optional[Union[str, List[str]]] = None,
        primary_metric: Optional[str] = None,
        force_recompute: bool = False,
        metric_force: bool = False,
        batch_size: Optional[int] = None,
        save_step: Optional[int] = None,
    ) -> dict:
        selected_metrics = self._normalise_metric_names(metrics)
        corpus_report = self.compare_on_corpus(
            metrics=selected_metrics,
            primary_metric=primary_metric,
            force_recompute=force_recompute,
            metric_force=metric_force,
            batch_size=batch_size,
            save_step=save_step,
        )

        base_dir = Path(output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        results_csv = self.save_csv(str(base_dir / "results.csv"))
        summary_csv = self.save_summary_csv(str(base_dir / "summary.csv"), metrics=selected_metrics)
        leaderboard_csv = self.save_leaderboard_csv(
            str(base_dir / "leaderboard.csv"),
            primary_metric=primary_metric,
            metrics=selected_metrics,
        )
        error_cases_csv = self.export_error_cases(
            str(base_dir / "error_cases.csv"),
            metric=primary_metric,
            min_value=0.0,
            include_failures=True,
        )

        return {
            **corpus_report,
            "results_csv": results_csv,
            "summary_csv": summary_csv,
            "leaderboard_csv": leaderboard_csv,
            "error_cases_csv": error_cases_csv,
        }

    def export_appendix_bundle(
        self,
        output_dir: str,
        metrics: Optional[Union[str, List[str]]] = None,
        primary_metric: Optional[str] = None,
        include_benchmark: bool = False,
        benchmark_devices: Optional[List[str]] = None,
        benchmark_sample_limit: int = 20,
        benchmark_warmup_samples: int = 1,
        benchmark_batch_size: Optional[int] = None,
        include_static_report: bool = True,
        force_recompute: bool = False,
        metric_force: bool = False,
        batch_size: Optional[int] = None,
        save_step: Optional[int] = None,
    ) -> dict:
        base_dir = Path(output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        table_paths = self.prepare_thesis_tables(
            output_dir=str(base_dir),
            metrics=metrics,
            primary_metric=primary_metric,
            force_recompute=force_recompute,
            metric_force=metric_force,
            batch_size=batch_size,
            save_step=save_step,
        )

        bundle = dict(table_paths)

        if include_static_report:
            bundle["report_html"] = self.save_report_html(str(base_dir / "report.html"))

        if include_benchmark:
            benchmark_payload = self.benchmark_models(
                devices=benchmark_devices,
                sample_limit=benchmark_sample_limit,
                warmup_samples=benchmark_warmup_samples,
                batch_size=benchmark_batch_size,
                output_path=str(base_dir / "benchmark.csv"),
            )
            bundle["benchmark_report"] = benchmark_payload["report"]
            bundle["benchmark_rows"] = benchmark_payload["rows"]
            bundle["benchmark_csv"] = benchmark_payload["csv_path"]

        return bundle

    def save_results(self, output_path: str) -> str:
        self.analysis_dataset.save_results(output_path)
        return output_path

    def save_csv(self, output_path: str) -> str:
        return self.analysis_dataset.save_csv(output_path)

    def save_excel(self, output_path: str) -> str:
        return self.analysis_dataset.save_excel(output_path)

    def to_pandas(self):
        return self.analysis_dataset.to_pandas()

    def evaluate(
        self,
        force_recompute: bool = False,
        metric_force: bool = False,
        batch_size: Optional[int] = None,
        save_step: Optional[int] = None,
    ) -> BaseASRDataset:
        return self.run(
            force_recompute=force_recompute,
            metric_force=metric_force,
            batch_size=batch_size,
            save_step=save_step,
        )

    def report(
        self,
        audio_dir: Optional[str] = None,
        sections: Optional[Iterable] = None,
        port: int = 8765,
        open_browser: bool = True,
    ) -> ReportServer:
        self.report_server = ReportServer(
            dataset=self.analysis_dataset,
            audio_dir=audio_dir,
            sections=list(sections) if sections is not None else None,
            port=port,
            open_browser=open_browser,
        )
        return self.report_server

    def save_report_html(
        self,
        output_path: str,
        sections: Optional[Iterable] = None,
    ) -> str:
        builder = ReportBuilder(
            dataset=self.analysis_dataset,
            sections=list(sections) if sections is not None else None,
        )
        return builder.save_static_html(output_path)
