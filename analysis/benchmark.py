from __future__ import annotations

import copy
import csv
import inspect
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional


@dataclass
class BenchmarkResult:
    model_name: str
    requested_device: str
    actual_device: str
    samples: int
    successes: int
    failures: int
    total_wall_time: float
    total_audio_duration: float
    avg_latency: float
    median_latency: float
    p95_latency: float
    throughput_samples_per_sec: float
    audio_seconds_per_sec: float
    rtf: Optional[float]
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "requested_device": self.requested_device,
            "actual_device": self.actual_device,
            "samples": self.samples,
            "successes": self.successes,
            "failures": self.failures,
            "total_wall_time": round(self.total_wall_time, 4),
            "total_audio_duration": round(self.total_audio_duration, 4),
            "avg_latency": round(self.avg_latency, 4),
            "median_latency": round(self.median_latency, 4),
            "p95_latency": round(self.p95_latency, 4),
            "throughput_samples_per_sec": round(self.throughput_samples_per_sec, 4),
            "audio_seconds_per_sec": round(self.audio_seconds_per_sec, 4),
            "rtf": None if self.rtf is None else round(self.rtf, 4),
            "error": self.error,
        }


@dataclass
class BenchmarkReport:
    results: List[BenchmarkResult]

    def to_rows(self) -> List[dict]:
        return [result.to_dict() for result in self.results]

    def to_pandas(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "BenchmarkReport.to_pandas() requires pandas. Install plantain2asr[analysis] or pandas manually."
            ) from exc
        return pd.DataFrame(self.to_rows())

    def save_csv(self, output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.to_rows()
        if not rows:
            path.write_text("", encoding="utf-8")
            return str(path)

        fieldnames = list(rows[0].keys())
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return str(path)

    def save_excel(self, output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_pandas().to_excel(path, index=False)
        return str(path)

    def print(self):
        if not self.results:
            print("⚠️ No benchmark results.")
            return

        headers = [
            "Model",
            "Requested",
            "Actual",
            "Samples",
            "OK",
            "Fail",
            "Avg Latency",
            "P95",
            "Samples/s",
            "Audio s/s",
            "RTF",
        ]
        rows = [headers]

        for result in self.results:
            rows.append(
                [
                    result.model_name,
                    result.requested_device,
                    result.actual_device,
                    str(result.samples),
                    str(result.successes),
                    str(result.failures),
                    f"{result.avg_latency:.3f}s",
                    f"{result.p95_latency:.3f}s",
                    f"{result.throughput_samples_per_sec:.2f}",
                    f"{result.audio_seconds_per_sec:.2f}",
                    "-" if result.rtf is None else f"{result.rtf:.3f}",
                ]
            )

        widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
        print("\n⏱️ Model Benchmark")
        print("-" * (sum(widths) + len(widths) * 3))
        for idx, row in enumerate(rows):
            print(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
            if idx == 0:
                print("-" * (sum(widths) + len(widths) * 3))


class ModelBenchmark:
    """
    Универсальный бенчмарк для ASR-моделей.

    Пример:
        bench = ModelBenchmark(sample_limit=20, warmup_samples=2, batch_size=4)
        report = bench.run(
            dataset,
            model_factory=lambda device: Models.Whisper(device=device),
        )
        report.print()
    """

    def __init__(
        self,
        sample_limit: int = 20,
        warmup_samples: int = 1,
        batch_size: Optional[int] = None,
        devices: Optional[List[str]] = None,
    ):
        self.sample_limit = sample_limit
        self.warmup_samples = warmup_samples
        self.batch_size = batch_size
        self.devices = devices

    def run(self, dataset, model_factory: Callable, devices: Optional[List[str]] = None) -> BenchmarkReport:
        selected = list(dataset.take(self.sample_limit))
        if not selected:
            return BenchmarkReport([])

        requested_devices = devices or self.devices or self.available_devices(model_factory)
        results: List[BenchmarkResult] = []

        for device in requested_devices:
            try:
                model = self._build_model(model_factory, device)
            except Exception as e:
                results.append(
                    BenchmarkResult(
                        model_name=self._factory_name(model_factory),
                        requested_device=device,
                        actual_device=device,
                        samples=len(selected),
                        successes=0,
                        failures=len(selected),
                        total_wall_time=0.0,
                        total_audio_duration=sum((s.duration or 0.0) for s in selected),
                        avg_latency=0.0,
                        median_latency=0.0,
                        p95_latency=0.0,
                        throughput_samples_per_sec=0.0,
                        audio_seconds_per_sec=0.0,
                        rtf=None,
                        error=str(e),
                    )
                )
                continue

            actual_device = getattr(model, "device", device)
            benchmark_result = self._benchmark_model(model, selected, device, actual_device)
            results.append(benchmark_result)

        report = BenchmarkReport(results)
        self.report = report
        return report

    @staticmethod
    def available_devices(model_factory: Callable) -> List[str]:
        signature = inspect.signature(model_factory)
        if "device" not in signature.parameters and not any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        ):
            return ["cpu"]

        devices: List[str] = []

        try:
            import torch

            if torch.cuda.is_available():
                devices.append("cuda")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                devices.append("mps")
        except Exception:
            pass

        try:
            import onnxruntime as ort

            providers = set(ort.get_available_providers())
            if "CUDAExecutionProvider" in providers and "cuda" not in devices:
                devices.append("cuda")
            if "CoreMLExecutionProvider" in providers and "mps" not in devices:
                devices.append("mps")
        except Exception:
            pass

        devices.append("cpu")

        seen = set()
        ordered = []
        for device in devices:
            if device not in seen:
                ordered.append(device)
                seen.add(device)
        return ordered

    def _benchmark_model(self, model, samples, requested_device: str, actual_device: str) -> BenchmarkResult:
        working_samples = [copy.copy(sample) for sample in samples]
        total_audio_duration = sum((sample.duration or 0.0) for sample in working_samples)
        batch_size = self.batch_size or getattr(model, "batch_size", 1) or 1

        warmup = working_samples[: min(self.warmup_samples, len(working_samples))]
        if warmup:
            try:
                model.process_samples(warmup, inplace=False)
            except Exception:
                pass

        latencies: List[float] = []
        successes = 0
        failures = 0
        total_wall_time = 0.0

        for batch in self._chunks(working_samples, batch_size):
            start = time.perf_counter()
            processed = model.process_samples(batch, inplace=False)
            total_wall_time += time.perf_counter() - start

            for sample in processed:
                result = sample.asr_results.get(model.name, {})
                latency = float(result.get("processing_time", 0.0) or 0.0)
                latencies.append(latency)
                if result.get("error"):
                    failures += 1
                else:
                    successes += 1

        avg_latency = statistics.fmean(latencies) if latencies else 0.0
        median_latency = statistics.median(latencies) if latencies else 0.0
        p95_latency = self._percentile(latencies, 95)
        throughput = successes / total_wall_time if total_wall_time > 0 else 0.0
        audio_per_sec = total_audio_duration / total_wall_time if total_wall_time > 0 else 0.0
        rtf = (total_wall_time / total_audio_duration) if total_audio_duration > 0 else None

        return BenchmarkResult(
            model_name=model.name,
            requested_device=requested_device,
            actual_device=actual_device,
            samples=len(working_samples),
            successes=successes,
            failures=failures,
            total_wall_time=total_wall_time,
            total_audio_duration=total_audio_duration,
            avg_latency=avg_latency,
            median_latency=median_latency,
            p95_latency=p95_latency,
            throughput_samples_per_sec=throughput,
            audio_seconds_per_sec=audio_per_sec,
            rtf=rtf,
        )

    @staticmethod
    def _build_model(model_factory: Callable, device: str):
        signature = inspect.signature(model_factory)
        if "device" in signature.parameters or any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        ):
            return model_factory(device=device)
        return model_factory()

    @staticmethod
    def _chunks(items: List, chunk_size: int) -> Iterable[List]:
        for idx in range(0, len(items), chunk_size):
            yield items[idx : idx + chunk_size]

    @staticmethod
    def _percentile(values: List[float], percentile: int) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        ordered = sorted(values)
        position = (len(ordered) - 1) * (percentile / 100)
        left = int(position)
        right = min(left + 1, len(ordered) - 1)
        weight = position - left
        return ordered[left] * (1 - weight) + ordered[right] * weight

    @staticmethod
    def _factory_name(model_factory: Callable) -> str:
        return getattr(model_factory, "__name__", type(model_factory).__name__)
