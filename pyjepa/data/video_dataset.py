"""Video datasets — handles a folder of clips or pre-extracted frame dirs.

For video files we try decord first (fast, no ffmpeg dependency once installed)
then fall back to torchvision/av.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif"}
_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_video_clip(path: Path, num_frames: int, frame_stride: int = 1) -> torch.Tensor:
    """Load ``num_frames`` frames from a video, evenly spaced.

    Returns ``(C, T, H, W)`` float tensor in [0, 1].
    """
    try:
        import decord

        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(str(path))
        total = len(vr)
        span = num_frames * frame_stride
        start = max(0, np.random.randint(0, max(1, total - span + 1)))
        indices = [min(total - 1, start + i * frame_stride) for i in range(num_frames)]
        frames = vr.get_batch(indices)  # (T, H, W, C)
        frames = frames.float() / 255.0
        return frames.permute(3, 0, 1, 2).contiguous()  # (C, T, H, W)
    except ImportError:
        pass

    try:
        from torchvision.io import read_video

        v, _, _ = read_video(str(path), output_format="TCHW", pts_unit="sec")
        v = v.float() / 255.0  # (T, C, H, W)
        total = v.size(0)
        span = num_frames * frame_stride
        start = max(0, np.random.randint(0, max(1, total - span + 1)))
        indices = [min(total - 1, start + i * frame_stride) for i in range(num_frames)]
        v = v[indices]
        return v.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)
    except ImportError as e:
        raise ImportError(
            "Need either decord or torchvision[av] to read videos. "
            "Try: pip install pyjepa[video]"
        ) from e


class VideoDataset(Dataset):
    """Loads short clips from video files. Returns ``transform(clip)`` of shape ``(C, T, H, W)``."""

    def __init__(
        self,
        paths: Union[str, Path, List[Union[str, Path]]],
        num_frames: int = 16,
        frame_stride: int = 4,
        transform: Optional[Callable] = None,
        recursive: bool = True,
    ) -> None:
        if isinstance(paths, (str, Path)):
            root = Path(paths)
            if root.is_dir():
                glob = root.rglob if recursive else root.glob
                self.paths = sorted(p for p in glob("*") if p.suffix.lower() in _VIDEO_EXTENSIONS)
            else:
                self.paths = [Path(paths)]
        else:
            self.paths = [Path(p) for p in paths]
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        clip = _load_video_clip(self.paths[idx], self.num_frames, self.frame_stride)
        if self.transform is not None:
            clip = self.transform(clip)
        return clip


class VideoFrameDataset(Dataset):
    """Pre-extracted frame dataset: ``root/clip_id/frame_xxxxx.jpg``.

    Each subdirectory is one clip. We sample ``num_frames`` frames at stride
    ``frame_stride`` from each clip.
    """

    def __init__(
        self,
        root: Union[str, Path],
        num_frames: int = 16,
        frame_stride: int = 4,
        transform: Optional[Callable] = None,
        return_label: bool = False,
    ) -> None:
        self.root = Path(root)
        self.clips: list[list[Path]] = []
        self.labels: list[int] = []
        self.classes: list[str] = []

        if any(p.is_dir() for p in self.root.iterdir()):
            # Either ImageFolder-style classes or just clip dirs at root
            subdirs = sorted(p for p in self.root.iterdir() if p.is_dir())
            for clip_dir in subdirs:
                frames = sorted(p for p in clip_dir.iterdir() if p.suffix.lower() in _IMG_EXTENSIONS)
                if not frames:
                    # Maybe class dirs containing clip dirs
                    for inner in sorted(clip_dir.iterdir()):
                        if not inner.is_dir():
                            continue
                        inner_frames = sorted(
                            p for p in inner.iterdir() if p.suffix.lower() in _IMG_EXTENSIONS
                        )
                        if inner_frames:
                            self.clips.append(inner_frames)
                            cls = clip_dir.name
                            if cls not in self.classes:
                                self.classes.append(cls)
                            self.labels.append(self.classes.index(cls))
                else:
                    self.clips.append(frames)
                    self.labels.append(-1)

        if not self.clips:
            raise ValueError(f"No clips found under {self.root}")
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.transform = transform
        self.return_label = return_label

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        frames = self.clips[idx]
        total = len(frames)
        span = self.num_frames * self.frame_stride
        start = max(0, np.random.randint(0, max(1, total - span + 1)))
        indices = [min(total - 1, start + i * self.frame_stride) for i in range(self.num_frames)]

        try:
            from PIL import Image

            imgs = [Image.open(frames[i]).convert("RGB") for i in indices]
        except ImportError as e:
            raise ImportError("Pillow required: pip install pyjepa[vision]") from e

        arrays = [np.asarray(img) for img in imgs]
        clip = torch.from_numpy(np.stack(arrays)).float() / 255.0  # (T, H, W, C)
        clip = clip.permute(3, 0, 1, 2).contiguous()  # (C, T, H, W)

        if self.transform is not None:
            clip = self.transform(clip)
        if self.return_label:
            return clip, self.labels[idx]
        return clip
