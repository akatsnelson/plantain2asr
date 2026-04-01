from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..utils.logging import get_logger


logger = get_logger(__name__)


def save_unified_results(dataset, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving unified results to %s", output_path)
    with open(path, "w", encoding="utf-8") as handle:
        for sample in dataset._samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")


def save_legacy_results(dataset, output_path: str, model_name: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving legacy results for %s to %s", model_name, output_path)

    saved_count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for sample in dataset._samples:
            if model_name not in sample.asr_results:
                continue

            res = sample.asr_results[model_name]
            data = {
                "id": sample.id,
                "audio_path": sample.audio_path,
                "reference": sample.text,
                "duration": sample.duration,
                **sample.meta,
                "hypothesis": res.get("hypothesis", ""),
                "time": res.get("processing_time", 0.0),
                "model": model_name,
            }
            if res.get("error"):
                data["error"] = res["error"]
            if "metrics" in res:
                for key, value in res["metrics"].items():
                    data[key] = value

            handle.write(json.dumps(data, ensure_ascii=False) + "\n")
            saved_count += 1

    logger.info("Saved %s legacy result rows", saved_count)


def load_model_results(dataset, model_name: str, jsonl_path: str) -> int:
    matched = 0
    skipped = 0

    with open(jsonl_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            audio = data.get("audio_path") or data.get("audio_filepath", "")
            if not audio:
                continue

            sid = Path(audio).name
            sample = dataset._id_map.get(sid)
            if sample is None:
                skipped += 1
                continue

            sample.add_result(
                model_name=model_name,
                hypothesis=data.get("hypothesis", ""),
                duration=(data.get("processing_time") or data.get("time", 0.0)),
            )
            matched += 1

    logger.info(
        "[%s] '%s': matched %s/%s samples%s",
        dataset.name,
        model_name,
        matched,
        len(dataset._samples),
        f" ({skipped} unmatched)" if skipped else "",
    )
    return matched


def load_results(dataset, paths: Union[str, List[str]]) -> None:
    if isinstance(paths, str):
        paths = [paths]

    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            logger.warning("Results file not found: %s", path)
            continue

        logger.info("Loading results from %s", path.name)
        loaded_count = 0

        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                sample_id = data.get("id")
                if sample_id not in dataset._id_map:
                    continue

                sample = dataset._id_map[sample_id]

                if "asr_results" in data:
                    sample.asr_results.update(data["asr_results"])
                    loaded_count += 1
                    continue

                if "results" in data:
                    sample.asr_results.update(data["results"])
                    loaded_count += 1
                    continue

                model_name = data.get("name", data.get("model"))
                if not model_name and "model_info" in data:
                    model_name = data["model_info"].get("name")

                if model_name:
                    hyp = data.get("hypothesis", "")
                    duration = data.get("processing_time", 0.0)
                    err = data.get("error")
                    metrics = {}
                    for key in ["wer", "cer", "mer", "wil", "wip", "accuracy"]:
                        if key in data:
                            metrics[key] = data[key]

                    sample.add_result(
                        model_name,
                        hyp,
                        duration,
                        err,
                        metrics=metrics if metrics else None,
                    )
                    loaded_count += 1

        logger.info("Matched %s/%s samples", loaded_count, len(dataset))


def to_pandas(dataset):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Pandas export requires pandas. Install plantain2asr[analysis] or pandas manually."
        ) from exc

    return pd.DataFrame(iter_results_rows(dataset))


def iter_results_rows(dataset) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample in dataset._samples:
        base_info = {
            "id": sample.id,
            "audio_path": sample.audio_path,
            "duration": sample.duration,
            "reference": sample.text,
            **sample.meta,
        }

        if not sample.asr_results:
            rows.append(base_info.copy())
            continue

        for model_name, res in sample.asr_results.items():
            row = base_info.copy()
            row["model"] = model_name
            row["hypothesis"] = res.get("hypothesis")
            row["processing_time"] = res.get("processing_time")
            row["error"] = res.get("error")
            if "metrics" in res and res["metrics"]:
                row.update(res["metrics"])
            rows.append(row)

    return rows


def save_csv(dataset, output_path: str) -> str:
    rows = iter_results_rows(dataset)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return str(path)

    fieldnames: List[str] = []
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

    logger.info("Saved CSV export to %s", path)
    return str(path)


def save_excel(dataset, output_path: str) -> str:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Excel export requires pandas. Install plantain2asr[analysis] or pandas manually."
        ) from exc

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(iter_results_rows(dataset)).to_excel(path, index=False)
    logger.info("Saved Excel export to %s", path)
    return str(path)


def summarize_by_model(dataset, metrics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    rows = iter_results_rows(dataset)
    grouped: Dict[str, Dict[str, Any]] = {}
    excluded = {
        "id",
        "audio_path",
        "duration",
        "reference",
        "model",
        "hypothesis",
        "processing_time",
        "error",
    }

    for row in rows:
        model_name = row.get("model")
        if not model_name:
            continue

        bucket = grouped.setdefault(
            model_name,
            {
                "model": model_name,
                "samples": 0,
                "errors": 0,
                "_processing_times": [],
                "_metrics": {},
            },
        )
        bucket["samples"] += 1
        if row.get("error"):
            bucket["errors"] += 1

        processing_time = row.get("processing_time")
        if isinstance(processing_time, (int, float)):
            bucket["_processing_times"].append(float(processing_time))

        for key, value in row.items():
            if key in excluded:
                continue
            if metrics is not None and key not in metrics:
                continue
            if isinstance(value, (int, float)):
                bucket["_metrics"].setdefault(key, []).append(float(value))

    summary = []
    for model_name in sorted(grouped):
        bucket = grouped[model_name]
        entry = {
            "model": bucket["model"],
            "samples": bucket["samples"],
            "errors": bucket["errors"],
            "error_rate": (bucket["errors"] / bucket["samples"]) if bucket["samples"] else 0.0,
        }
        timings = bucket["_processing_times"]
        if timings:
            entry["avg_processing_time"] = sum(timings) / len(timings)
        for metric_name, values in sorted(bucket["_metrics"].items()):
            if values:
                entry[f"avg_{metric_name}"] = sum(values) / len(values)
        summary.append(entry)

    return summary
