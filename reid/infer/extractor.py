"""Model-backed image extractor implementing contract C5."""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torchvision.io import ImageReadMode, read_image

from reid.data.preprocess import preprocess_frame
from reid.infer.fusion import load_fusion_config
from reid.model import load, load_config


Item = tuple[str | Path, tuple[float, float, float, float]]


def _precast_projection_parameters(model: torch.nn.Module, dtype: torch.dtype) -> dict[str, int]:
    """Experimental AMP cast hoisting; never cast norms, embeddings or buffers.

    Called only for explicitly enabled CUDA AMP components. This helper can be
    unit-tested on CPU tensors; numerical/performance acceptance still needs GPU.
    Shared parameters crossing the allowlist are rejected instead of mutating a
    protected parameter indirectly. The on-disk checkpoint is never changed.
    """
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("projection precast requires fp16 or bf16 AMP")
    if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError("projection precast requires an eval, frozen model")
    selected, protected, modules = {}, set(), set()
    for module in model.modules():
        protected.update(id(buffer) for buffer in module.buffers(recurse=False))
        projection = isinstance(module, (torch.nn.Linear, torch.nn.Conv2d))
        for name, parameter in module.named_parameters(recurse=False):
            if projection and name in {"weight", "bias"}:
                if parameter.dtype not in (torch.float32, dtype):
                    raise ValueError("projection precast supports only FP32 or matching AMP parameters")
                if parameter.dtype == torch.float32:
                    selected[id(parameter)] = parameter
                    modules.add(id(module))
            else:
                protected.add(id(parameter))
    if set(selected) & protected:
        raise ValueError("projection parameter is shared with a protected norm/embedding/parameter")
    before, after = 0, 0
    with torch.no_grad():
        for parameter in selected.values():
            before += parameter.numel() * parameter.element_size()
            parameter.data = parameter.data.to(dtype=dtype)
            after += parameter.numel() * parameter.element_size()
    return {"modules_cast": len(modules), "parameters_cast": len(selected),
            "parameter_bytes_before": before, "parameter_bytes_after": after}


class Extractor:
    """Decode, crop and embed with one C3 model or a fixed-scale fusion artifact."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cuda",
        preprocess_workers: int | None = None,
        precast_projections: bool = False,
    ) -> None:
        if type(precast_projections) is not bool:
            raise ValueError("precast_projections must be a bool")
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        self._fusion_components = None
        if (self.model_dir / "fusion_config.json").exists():
            self._load_fusion()
        else:
            self.cfg = load_config(str(self.model_dir))
            self.model = load(str(self.model_dir), str(self.device))
            self.amp_dtype = self._amp_dtype_for(self.cfg)
        self.precast_projection_stats = {
            "requested": precast_projections, "applied": False, "eligible_components": 0,
            "modules_cast": 0, "parameters_cast": 0,
            "parameter_bytes_before": 0, "parameter_bytes_after": 0,
        }
        if precast_projections and self.device.type == "cuda":
            members = ([(self.model, self.amp_dtype)] if self._fusion_components is None else
                       [(model, dtype) for model, _, dtype, _ in self._fusion_components])
            for model, dtype in members:
                if dtype not in (torch.float16, torch.bfloat16):
                    continue
                model.eval().requires_grad_(False)
                stats = _precast_projection_parameters(model, dtype)
                self.precast_projection_stats["eligible_components"] += 1
                for name, value in stats.items():
                    self.precast_projection_stats[name] += value
            self.precast_projection_stats["applied"] = self.precast_projection_stats["parameters_cast"] > 0
        if preprocess_workers is None:
            # JPEG decode and antialiased resize are CPU-bound and release the
            # GIL. Eight workers saturate the 8-vCPU A10 host while the GPU is
            # idle between batches; batch=1 stays on the direct path below.
            default_workers = min(8, os.cpu_count() or 1) if self.device.type == "cuda" else 1
            preprocess_workers = int(os.environ.get("REID_PREPROCESS_WORKERS", default_workers))
        if preprocess_workers <= 0:
            raise ValueError("preprocess_workers must be positive")
        self.preprocess_workers = preprocess_workers
        self._preprocess_pool = (
            ThreadPoolExecutor(
                max_workers=preprocess_workers,
                thread_name_prefix="reid-preprocess",
            )
            if preprocess_workers > 1
            else None
        )

    def _amp_dtype_for(self, cfg: dict) -> torch.dtype | None:
        if self.device.type != "cuda":
            return None
        return {
            "fp16": torch.float16, "float16": torch.float16,
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        }.get(str(cfg.get("infer_dtype", "fp32")).lower())

    def _load_fusion(self) -> None:
        # An old preprocessing override might add a safety transform (e.g. a
        # plate mask). Never silently bypass it when switching to fusion.
        if (type(self)._decode_preprocess is not Extractor._decode_preprocess
                and type(self)._decode_frame is Extractor._decode_frame):
            raise ValueError("fusion preprocessing overrides must use the _decode_frame hook")
        manifest = load_fusion_config(self.model_dir)
        scale_reference = math.sqrt(min(entry["similarity_std"] for entry in manifest["components"]))
        for entry in manifest["components"]:
            # Reject an unrepresentable spread instead of silently dropping an
            # entire member when its coefficient underflows to zero in float32.
            if scale_reference / math.sqrt(entry["similarity_std"]) < np.finfo(np.float32).tiny:
                raise ValueError("similarity_std dynamic range is too wide for float32 fusion")
        components = []
        for entry in manifest["components"]:
            path = (self.model_dir / entry["path"]).resolve()
            cfg = load_config(str(path))
            dim = cfg.get("embed_dim")
            if type(dim) is not int or dim <= 0:
                raise ValueError(f"component {path}: embed_dim must be a positive integer")
            if cfg.get("tta_flip", False):
                raise ValueError(f"component {path}: tta_flip is not supported by fusion")
            if str(cfg.get("infer_dtype", "fp32")).lower() not in {
                "fp16", "float16", "bf16", "bfloat16", "fp32", "float32",
            }:
                raise ValueError(f"component {path}: unsupported infer_dtype")
            model = load(str(path), str(self.device)).eval().requires_grad_(False)
            components.append((model, cfg, self._amp_dtype_for(cfg), entry["similarity_std"]))
        self._fusion_components = components
        self.model = torch.nn.ModuleList([component[0] for component in components]).eval()
        self.amp_dtype = None  # Autocast is selected separately for each member.
        self.cfg = {
            **manifest,
            "arch": "fusion",
            "embed_dim": sum(component[1]["embed_dim"] for component in components),
            "input_size": [component[1]["input_size"] for component in components],
            "infer_dtype": "per-component",
        }

    def _decode_frame(self, item: Item) -> torch.Tensor:
        """Decode one RGB frame; shared hook for transforms before every model."""
        image_path, _ = item
        try:
            frame = read_image(str(image_path), mode=ImageReadMode.RGB)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"failed to decode image {image_path}: {exc}") from exc
        if frame.ndim != 3 or frame.shape[0] != 3 or min(frame.shape[-2:]) <= 0:
            raise ValueError(
                f"decoded image {image_path} has invalid RGB shape {tuple(frame.shape)}"
            )
        return frame

    def _decode_preprocess(self, item: Item) -> torch.Tensor:
        return preprocess_frame(self._decode_frame(item), item[1], self.cfg)

    def _decode_preprocess_fusion(self, item: Item) -> tuple[torch.Tensor, ...]:
        frame = self._decode_frame(item)
        return tuple(preprocess_frame(frame, item[1], cfg)
                     for _, cfg, _, _ in self._fusion_components)

    def _extract_fusion(self, items: Sequence[Item]) -> np.ndarray:
        if self._preprocess_pool is not None and len(items) > 1:
            tensors = list(self._preprocess_pool.map(self._decode_preprocess_fusion, items))
        else:
            tensors = [self._decode_preprocess_fusion(item) for item in items]
        blocks = []
        # Removing a common factor before concatenation is algebraically exact
        # after the final L2 and avoids overflow for very small fixed std values.
        reference = math.sqrt(min(component[3] for component in self._fusion_components))
        for index, (model, cfg, amp_dtype, similarity_std) in enumerate(self._fusion_components):
            batch = torch.stack([row[index] for row in tensors]).to(self.device, non_blocking=True)
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type, dtype=amp_dtype, enabled=amp_dtype is not None,
            ):
                features = model.features(batch)  # Before BNNeck, matching OOF (A-053).
            features = features.float()
            if features.shape != (len(items), cfg["embed_dim"]):
                raise ValueError(f"fusion component {index}: unexpected embedding shape {tuple(features.shape)}")
            norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
            valid = torch.isfinite(features).all() & torch.isfinite(norms).all() & (norms > 0).all()
            if not valid:
                raise ValueError(f"fusion component {index}: embeddings must be finite and nonzero")
            blocks.append((features / norms) * (reference / math.sqrt(similarity_std)))
        embeddings = torch.nn.functional.normalize(torch.cat(blocks, dim=1), dim=1)
        return embeddings.cpu().numpy().astype(np.float32, copy=False)

    def extract(self, image_path: str | Path, bbox: Sequence[float]) -> np.ndarray:
        result = self.extract_batch([(image_path, tuple(float(x) for x in bbox))])
        return result[0]

    def extract_batch(self, items: Sequence[Item]) -> np.ndarray:
        if not items:
            dim = int(self.cfg["embed_dim"])
            return np.empty((0, dim), dtype=np.float32)

        if getattr(self, "_fusion_components", None) is not None:
            return self._extract_fusion(items)

        if self._preprocess_pool is not None and len(items) > 1:
            tensors = list(self._preprocess_pool.map(self._decode_preprocess, items))
        else:
            tensors = [self._decode_preprocess(item) for item in items]
        batch = torch.stack(tensors).to(self.device, non_blocking=True)

        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_dtype is not None,
        ):
            # A-053: OOF retrieval/calibration uses pooled features BEFORE BNNeck.
            # Embedder.forward applies the training neck and produces a different
            # vector. Keep that training API unchanged; align the B inference path.
            embeddings = self.model.features(batch)
        embeddings = torch.nn.functional.normalize(embeddings.float(), dim=1)
        return embeddings.cpu().numpy().astype(np.float32, copy=False)
