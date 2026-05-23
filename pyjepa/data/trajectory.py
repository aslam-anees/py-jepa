"""Trajectory datasets for world-model / robot learning (LeWM-style).

Each sample is a window of ``window`` consecutive timesteps drawn from an
episode. We support:
- :class:`TrajectoryDataset`: dict-of-numpy-arrays loaded into memory.
- :class:`HDF5TrajectoryDataset`: lazy reader over an HDF5 file, with optional
  caching of frequently-accessed columns.

Each sample is a dict with keys like ``pixels``, ``action``, ``state``, ``goal``.
This matches the API the LeWM trainer expects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """In-memory windowed trajectory dataset.

    Args:
        episodes: list of dicts ``{key: np.ndarray of shape (T_ep, ...)}``.
            All episodes must have the same set of keys.
        window: number of consecutive timesteps per sample (the JEPA context+pred length).
        frameskip: stride between consecutive sampled timesteps (1 = every frame).
        transform: callable applied to the sample dict before returning.
        keys: subset of keys to return; default all.
    """

    def __init__(
        self,
        episodes: Sequence[dict[str, np.ndarray]],
        window: int,
        frameskip: int = 1,
        transform: Optional[Callable] = None,
        keys: Optional[Sequence[str]] = None,
    ) -> None:
        if not episodes:
            raise ValueError("episodes must be non-empty")
        self.episodes = list(episodes)
        self.window = window
        self.frameskip = frameskip
        self.span = window * frameskip
        self.transform = transform
        self.keys = list(keys) if keys is not None else list(self.episodes[0].keys())

        # Pre-compute (episode_idx, start_idx) for each valid window
        self._index: list[tuple[int, int]] = []
        for ep_i, ep in enumerate(self.episodes):
            T = len(next(iter(ep.values())))
            for start in range(0, T - self.span + 1):
                self._index.append((ep_i, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_i, start = self._index[idx]
        ep = self.episodes[ep_i]
        sample: dict[str, torch.Tensor] = {}
        for k in self.keys:
            arr = ep[k]
            sl = arr[start : start + self.span : self.frameskip]
            sample[k] = torch.from_numpy(np.ascontiguousarray(sl))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class HDF5TrajectoryDataset(Dataset):
    """Lazy windowed HDF5 trajectory dataset.

    Assumes the HDF5 file has top-level datasets like ``pixels`` ``(N, C, H, W)``,
    ``action`` ``(N, A)``, ``episode_idx`` ``(N,)``, ``step_idx`` ``(N,)``.

    Set ``keys_to_cache`` to keep small columns (actions, states) in memory while
    streaming heavy ones (pixels) from disk on every ``__getitem__``.
    """

    def __init__(
        self,
        path: str | Path,
        window: int,
        frameskip: int = 1,
        keys_to_cache: Sequence[str] = ("action", "state", "episode_idx", "step_idx"),
        transform: Optional[Callable] = None,
        keys: Optional[Sequence[str]] = None,
    ) -> None:
        try:
            import h5py
        except ImportError as e:
            raise ImportError("h5py required: pip install pyjepa[data]") from e

        self.path = Path(path)
        self.window = window
        self.frameskip = frameskip
        self.span = window * frameskip
        self.transform = transform

        with h5py.File(self.path, "r") as f:
            self.column_names = list(f.keys())
            self.keys = list(keys) if keys is not None else list(self.column_names)
            self._cache: dict[str, np.ndarray] = {}
            for k in keys_to_cache:
                if k in f:
                    self._cache[k] = f[k][:]

        # Build valid (start_idx,) index using episode_idx if present
        if "episode_idx" in self._cache or "ep_idx" in self._cache:
            ep_key = "episode_idx" if "episode_idx" in self._cache else "ep_idx"
            ep_arr = self._cache[ep_key]
            self._valid_starts: list[int] = []
            n = len(ep_arr)
            for start in range(n - self.span + 1):
                if ep_arr[start] == ep_arr[start + self.span - 1]:
                    self._valid_starts.append(start)
        else:
            with h5py.File(self.path, "r") as f:
                first_key = self.column_names[0]
                n = f[first_key].shape[0]
            self._valid_starts = list(range(0, n - self.span + 1))

        self._h5: Optional["h5py.File"] = None

    def _file(self):
        # Open lazily per worker — h5py file handles cannot cross process boundaries
        if self._h5 is None:
            import h5py

            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self._valid_starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self._valid_starts[idx]
        end = start + self.span
        sample: dict[str, torch.Tensor] = {}
        for k in self.keys:
            if k in self._cache:
                arr = self._cache[k][start:end:self.frameskip]
            else:
                arr = self._file()[k][start:end:self.frameskip]
            sample[k] = torch.from_numpy(np.ascontiguousarray(arr))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_col_data(self, name: str) -> np.ndarray:
        """Return an entire column as a numpy array. Used by the eval pipeline."""
        if name in self._cache:
            return self._cache[name]
        return self._file()[name][:]

    def __del__(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
