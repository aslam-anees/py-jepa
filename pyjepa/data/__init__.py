"""Datasets, transforms, and loaders for JEPA training.

Keep the surface lean: most users only need ``ImageDataset``, ``VideoDataset``,
or ``TrajectoryDataset``. Optional dependencies (h5py, torchvision, decord) are
imported lazily so the base lib runs without them.
"""

from .image_dataset import ImageDataset, ImageFolderDataset
from .normalize import IMAGENET_MEAN, IMAGENET_STD, RunningStandardScaler, ZScoreNormalizer
from .trajectory import HDF5TrajectoryDataset, TrajectoryDataset
from .transforms import (
    ImageNormalize,
    Resize,
    ToImageTensor,
    build_image_transform,
    build_video_transform,
)
from .video_dataset import VideoDataset, VideoFrameDataset

__all__ = [
    "HDF5TrajectoryDataset",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "ImageDataset",
    "ImageFolderDataset",
    "ImageNormalize",
    "Resize",
    "RunningStandardScaler",
    "ToImageTensor",
    "TrajectoryDataset",
    "VideoDataset",
    "VideoFrameDataset",
    "ZScoreNormalizer",
    "build_image_transform",
    "build_video_transform",
]
