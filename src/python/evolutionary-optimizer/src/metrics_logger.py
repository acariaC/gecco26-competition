from __future__ import annotations

import csv
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .events import Event

LOGGER = logging.getLogger(__name__)

METRIC_KEYS = [
    "average-waiting-time",
    "co2-emission",
    "driven-distance",
    "maximum-passenger-journey-time",
    "maximum-passenger-non-driving-time",
    "minimum-passenger-pickup-wait-time",
    "relative-throughput",
    "total-cost",
]

CSV_COLUMNS = ["time"] + METRIC_KEYS


class MetricsLogger:


    def __init__(self, output_dir: Path | str | None = None) -> None:
        self._output_dir = Path(output_dir) if output_dir else Path.cwd()
        self._lock = threading.Lock()
        self._rows: list[dict[str, Any]] = []
        self._csv_path: Path | None = None

    def record(self, event: Event) -> None:
        metrics = event.data.get("metrics")
        if not isinstance(metrics, dict):
            return
        sim_time = event.data.get("time", 0)
        row = {"time": sim_time}
        for key in METRIC_KEYS:
            row[key] = metrics.get(key, 0.0)
        with self._lock:
            self._rows.append(row)

    def flush(self) -> Path:
        with self._lock:
            if not self._rows:
                LOGGER.info("No metrics collected, skipping CSV write.")
                return self._ensure_csv_path()
            rows = list(self._rows)
            csv_path = self._ensure_csv_path()

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Wrote %d metric rows to %s", len(rows), csv_path)
        return csv_path

    def reset(self) -> None:
        with self._lock:
            self._rows.clear()
            self._csv_path = None
        LOGGER.info("Metrics logger reset.")

    def _ensure_csv_path(self) -> Path:
        if self._csv_path is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._csv_path = self._output_dir / f"metrics_{timestamp}.csv"
        return self._csv_path
