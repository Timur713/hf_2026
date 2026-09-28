"""Controlled plate-area masking for the robustness run required by §2.3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torchvision.io import ImageReadMode, read_image

from reid.data.preprocess import clip_bbox, preprocess_frame
from reid.infer.extractor import Extractor, Item as ExtractorItem
from reid.post.verifier import Item as VerifierItem
from reid.post.verifier import PairVerifier


@dataclass(frozen=True)
class PlateMaskConfig:
    """Geometry relative to the clipped vehicle BBox, not to the full frame."""

    width_ratio: float = 0.26
    aspect_ratio: float = 4.0
    center_x: float = 0.50
    center_y: float = 0.72
    fill: int = 127

    def __post_init__(self) -> None:
        if not 0.0 < self.width_ratio <= 1.0:
            raise ValueError("plate mask width_ratio must be in (0, 1]")
        if self.aspect_ratio <= 0.0:
            raise ValueError("plate mask aspect_ratio must be positive")
        if not 0.0 <= self.center_x <= 1.0 or not 0.0 <= self.center_y <= 1.0:
            raise ValueError("plate mask center coordinates must be in [0, 1]")
        if not 0 <= self.fill <= 255:
            raise ValueError("plate mask fill must be in [0, 255]")

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


DEFAULT_PLATE_MASK = PlateMaskConfig()


def plate_mask_box(
    bbox: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
    config: PlateMaskConfig = DEFAULT_PLATE_MASK,
) -> tuple[int, int, int, int]:
    """Return an absolute, clipped solid-fill rectangle inside the vehicle BBox."""

    vehicle_x0, vehicle_y0, vehicle_x1, vehicle_y1 = clip_bbox(
        *bbox, frame_width, frame_height
    )
    vehicle_width = vehicle_x1 - vehicle_x0
    vehicle_height = vehicle_y1 - vehicle_y0
    mask_width = min(vehicle_width, max(1, round(vehicle_width * config.width_ratio)))
    mask_height = min(vehicle_height, max(1, round(mask_width / config.aspect_ratio)))

    center_x = vehicle_x0 + config.center_x * vehicle_width
    center_y = vehicle_y0 + config.center_y * vehicle_height
    x0 = min(max(round(center_x - mask_width / 2), vehicle_x0), vehicle_x1 - mask_width)
    y0 = min(max(round(center_y - mask_height / 2), vehicle_y0), vehicle_y1 - mask_height)
    return int(x0), int(y0), int(x0 + mask_width), int(y0 + mask_height)


def apply_plate_mask(
    frame: torch.Tensor,
    bbox: tuple[float, float, float, float],
    config: PlateMaskConfig = DEFAULT_PLATE_MASK,
) -> torch.Tensor:
    """Clone a CHW frame and fill a deterministic plate-shaped region."""

    if frame.ndim != 3 or frame.shape[0] != 3 or min(frame.shape[-2:]) <= 0:
        raise ValueError(f"plate masking expects a non-empty CHW RGB frame, got {tuple(frame.shape)}")
    masked = frame.clone()
    height, width = masked.shape[-2:]
    x0, y0, x1, y1 = plate_mask_box(bbox, width, height, config)
    masked[:, y0:y1, x0:x1] = config.fill
    return masked


def _decode_mask_preprocess(
    item: ExtractorItem | VerifierItem,
    cfg: dict,
    mask: PlateMaskConfig,
) -> torch.Tensor:
    image_path, bbox = item
    try:
        frame = read_image(str(image_path), mode=ImageReadMode.RGB)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"failed to decode image {image_path}: {exc}") from exc
    masked = apply_plate_mask(frame, bbox, mask)
    return preprocess_frame(masked, bbox, cfg)


class PlateMaskedExtractor(Extractor):
    """Production extractor with an in-memory solid plate-area fill before C3."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cuda",
        preprocess_workers: int | None = None,
        mask: PlateMaskConfig = DEFAULT_PLATE_MASK,
    ) -> None:
        self.plate_mask = mask
        super().__init__(model_dir, device, preprocess_workers)

    def _decode_frame(self, item: ExtractorItem) -> torch.Tensor:
        # The base extractor reuses this frame for every fusion member, so the
        # control mask is applied exactly once and no model sees unmasked pixels.
        return apply_plate_mask(super()._decode_frame(item), item[1], self.plate_mask)


class PlateMaskedPairVerifier(PairVerifier):
    """Pair verifier that sees the same masked pixels as the embedding model."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda",
        preprocess_workers: int | None = None,
        mask: PlateMaskConfig = DEFAULT_PLATE_MASK,
    ) -> None:
        self.plate_mask = mask
        super().__init__(checkpoint, device, preprocess_workers)

    def _decode_preprocess(self, item: VerifierItem) -> torch.Tensor:
        return _decode_mask_preprocess(item, self.cfg, self.plate_mask)
