"""Logging utilities — pretty stdout, CSV, and an in-memory metric tracker.

Optional integrations (wandb, tensorboard) are imported lazily so the base lib
stays light.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str = "pyjepa", level: Optional[int] = None) -> logging.Logger:
    """Get a configured logger. Idempotent — repeat calls return the same instance."""
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level or os.environ.get("PYJEPA_LOG", "INFO"))
    _LOGGERS[name] = logger
    return logger


class AverageMeter:
    """Tracks moving average + min/max of a scalar. Used in training loops."""

    def __init__(self, window: int = 0) -> None:
        self.window = window
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.min = float("inf")
        self.max = float("-inf")
        self._buf: deque[float] = deque(maxlen=self.window) if self.window > 0 else deque()

    def update(self, val: float, n: int = 1) -> None:
        val = float(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.min = min(self.min, val)
        self.max = max(self.max, val)
        if self.window > 0:
            self._buf.append(val)

    @property
    def avg(self) -> float:
        if self.window > 0 and self._buf:
            return sum(self._buf) / len(self._buf)
        return self.sum / max(1, self.count)


class CSVLogger:
    """Tiny header-aware CSV writer. Pass a list of ``(fmt, name)`` tuples once,
    then call :meth:`log` with positional args matching ``fmt``."""

    def __init__(self, path: str | os.PathLike, *cols: tuple[str, str]) -> None:
        self.path = Path(path)
        self.cols = cols
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        self.fh = self.path.open("a", newline="")
        self.writer = csv.writer(self.fh)
        if new:
            self.writer.writerow([n for _, n in cols])
            self.fh.flush()

    def log(self, *vals: Any) -> None:
        row = []
        for (fmt, _), v in zip(self.cols, vals):
            try:
                row.append(fmt % v)
            except (TypeError, ValueError):
                row.append(str(v))
        self.writer.writerow(row)
        self.fh.flush()

    def close(self) -> None:
        try:
            self.fh.close()
        except Exception:
            pass


class MetricTracker:
    """A defaultdict-of-AverageMeter that also remembers ordering and step counts.

    Use it to collect per-iteration losses/metrics then snapshot at the end of an
    epoch. Optional sinks: stdout, CSV, wandb, tensorboard.
    """

    def __init__(
        self,
        csv_path: Optional[str | os.PathLike] = None,
        log_every: int = 50,
        prefix: str = "",
        wandb_run: Any = None,
        tb_writer: Any = None,
    ) -> None:
        self.meters: dict[str, AverageMeter] = defaultdict(AverageMeter)
        self.global_step = 0
        self.start_time = time.time()
        self.log_every = log_every
        self.prefix = prefix
        self._csv_path = Path(csv_path) if csv_path else None
        self._csv_writer: Optional[csv.writer] = None
        self._csv_fh = None
        self._csv_keys: list[str] = []
        self.wandb_run = wandb_run
        self.tb_writer = tb_writer
        self.logger = get_logger()

    def update(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            self.meters[k].update(float(v))

    def step(self, **kwargs: float) -> None:
        self.update(**kwargs)
        self.global_step += 1
        if self.wandb_run is not None and kwargs:
            self.wandb_run.log({f"{self.prefix}{k}": v for k, v in kwargs.items()}, step=self.global_step)
        if self.tb_writer is not None:
            for k, v in kwargs.items():
                self.tb_writer.add_scalar(f"{self.prefix}{k}", v, self.global_step)
        if self._csv_path is not None:
            self._csv_append(kwargs)
        if self.global_step % max(1, self.log_every) == 0:
            self.log_line(stage="step")

    def _csv_append(self, row: dict[str, float]) -> None:
        keys = list(row.keys())
        if self._csv_writer is None:
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_fh = self._csv_path.open("a", newline="")
            self._csv_writer = csv.writer(self._csv_fh)
            self._csv_keys = ["step", "wall"] + keys
            if self._csv_fh.tell() == 0:
                self._csv_writer.writerow(self._csv_keys)
        self._csv_writer.writerow(
            [self.global_step, f"{time.time() - self.start_time:.2f}"]
            + [f"{row.get(k, ''):.6g}" if isinstance(row.get(k), float) else str(row.get(k, "")) for k in self._csv_keys[2:]]
        )
        self._csv_fh.flush()

    def log_line(self, stage: str = "") -> None:
        parts = [f"step={self.global_step}", f"t={time.time() - self.start_time:.1f}s"]
        for k, m in self.meters.items():
            parts.append(f"{k}={m.avg:.4g}")
        tag = f"[{stage}] " if stage else ""
        self.logger.info(f"{tag}{' '.join(parts)}")

    def summary(self) -> dict[str, float]:
        return {k: m.avg for k, m in self.meters.items()}

    def close(self) -> None:
        if self._csv_fh is not None:
            try:
                self._csv_fh.close()
            except Exception:
                pass
