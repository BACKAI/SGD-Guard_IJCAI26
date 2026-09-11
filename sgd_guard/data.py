"""Dataset and image I/O helpers used by gallery, direction, and protection jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image, ImageFile
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize, to_pil_image

ImageFile.LOAD_TRUNCATED_IMAGES = False
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    relative_path: str


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def scan_images(root: str | Path, one_per_identity: bool = False, max_images: int | None = None) -> list[ImageRecord]:
    """Scan ImageFolder-style data; flat folders get one unique label per image.

    Gallery construction uses ``one_per_identity=True``.  If the input has
    identity subdirectories, the first lexicographically sorted image in each
    identity directory is selected, exactly as required by the paper.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {root}")
    paths = _files(root)
    if not paths:
        raise FileNotFoundError(f"No images found under: {root}")

    grouped: dict[str, list[Path]] = {}
    for path in paths:
        rel = path.relative_to(root)
        key = rel.parts[0] if len(rel.parts) > 1 else path.stem
        grouped.setdefault(key, []).append(path)

    selected: list[tuple[str, Path]] = []
    if one_per_identity:
        for key in sorted(grouped):
            selected.append((key, sorted(grouped[key])[0]))
    else:
        for key in sorted(grouped):
            selected.extend((key, path) for path in sorted(grouped[key]))

    labels = {key: i for i, key in enumerate(sorted({key for key, _ in selected}))}
    records = [ImageRecord(path, labels[key], str(path.relative_to(root))) for key, path in selected]
    if max_images is not None:
        records = records[:max_images]
    if not records:
        raise ValueError("The selected image set is empty")
    return records


def load_image(path: str | Path, size: int | tuple[int, int] | None = None) -> Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None:
            target = (size, size) if isinstance(size, int) else size
            image = image.resize((target[1], target[0]), Image.Resampling.BICUBIC)
        return pil_to_tensor(image).float() / 255.0


def save_image(tensor: Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().float().cpu().clamp(0, 1)
    if image.ndim == 4:
        image = image[0]
    to_pil_image(image).save(path)


class ImageRecordDataset(Dataset):
    def __init__(self, records: Iterable[ImageRecord], size: int = 224):
        self.records = list(records)
        self.size = size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        return load_image(record.path, self.size), record.label, record

