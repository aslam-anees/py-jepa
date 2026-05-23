"""Simple image datasets — supports a flat folder of images or an ImageFolder layout."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class ImageDataset(Dataset):
    """Loads a list of image files. Returns ``transform(image)`` tensors.

    Args:
        paths: list of file paths or a directory (scanned non-recursively).
        transform: callable applied to each loaded image.
    """

    def __init__(
        self,
        paths: Union[str, Path, List[Union[str, Path]]],
        transform: Optional[Callable] = None,
        recursive: bool = True,
    ) -> None:
        if isinstance(paths, (str, Path)):
            root = Path(paths)
            if root.is_dir():
                glob = root.rglob if recursive else root.glob
                self.paths = sorted(p for p in glob("*") if p.suffix.lower() in _IMG_EXTENSIONS)
            else:
                self.paths = [Path(paths)]
        else:
            self.paths = [Path(p) for p in paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        try:
            from PIL import Image

            img = Image.open(self.paths[idx]).convert("RGB")
        except ImportError as e:
            raise ImportError("Pillow required: `pip install pyjepa[vision]`") from e

        if self.transform is not None:
            img = self.transform(img)
        return img


class ImageFolderDataset(Dataset):
    """ImageNet-style ``root/class_x/img.jpg`` layout.

    Returns ``(image, class_id)``. Class IDs are sorted lexically.
    """

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise ValueError(f"ImageFolderDataset root must be a directory: {self.root}")
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: list[Tuple[Path, int]] = []
        for cls in self.classes:
            cls_idx = self.class_to_idx[cls]
            for p in sorted((self.root / cls).iterdir()):
                if p.suffix.lower() in _IMG_EXTENSIONS:
                    self.samples.append((p, cls_idx))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        try:
            from PIL import Image

            img = Image.open(path).convert("RGB")
        except ImportError as e:
            raise ImportError("Pillow required: `pip install pyjepa[vision]`") from e
        if self.transform is not None:
            img = self.transform(img)
        return img, label
