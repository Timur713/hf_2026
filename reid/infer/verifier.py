"""Image I/O and neural-network adapter for the pair verifier (track B).

Ranking uses the PairScorer protocol; it does not load models or decode images.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torchvision.io import ImageReadMode, read_image

from reid.data.preprocess import preprocess_frame
from reid.model.verifier import PairVerifier as VerifierNetwork
from reid.post.verifier import Item


class PairVerifier:
    """Load A3.3 ``verifier.pt`` and score one query against its top-k."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda",
        preprocess_workers: int | None = None,
    ) -> None:
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "verifier.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"verifier checkpoint not found: {checkpoint_path}")

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for verifier but is unavailable")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        arch = str(payload["arch"])
        input_side = int(payload["input"])
        head = payload.get("head", "concat")
        if head == "concat":
            network = VerifierNetwork
        elif head == "corr":
            from reid.model.corr_verifier import CorrVerifier

            network = CorrVerifier
        else:
            raise ValueError(
                f"unsupported verifier head: {head!r}; expected 'concat' or 'corr'"
            )
        # Use the training-side architecture directly so a future model change
        # cannot silently make the inference copy incompatible with its state.
        self.model = network(arch, (input_side, input_side), pretrained=False)
        self.model.load_state_dict(payload["state"], strict=True)
        self.model.to(self.device).eval()
        self.cfg = {
            "input_size": [input_side, input_side],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "resize": "stretch",
            "bbox_pad": 0.0,
        }
        if preprocess_workers is None:
            default_workers = (
                min(8, os.cpu_count() or 1) if self.device.type == "cuda" else 1
            )
            preprocess_workers = int(
                os.environ.get("REID_VERIFIER_WORKERS", default_workers)
            )
        if preprocess_workers <= 0:
            raise ValueError("preprocess_workers must be positive")
        self._pool = (
            ThreadPoolExecutor(
                max_workers=preprocess_workers, thread_name_prefix="reid-verifier"
            )
            if preprocess_workers > 1
            else None
        )

    def _decode_preprocess(self, item: Item) -> torch.Tensor:
        image_path, bbox = item
        frame = read_image(str(image_path), mode=ImageReadMode.RGB)
        return preprocess_frame(frame, bbox, self.cfg)

    def score_pairs(self, query: Item, candidates: Sequence[Item]) -> np.ndarray:
        if not candidates:
            return np.empty(0, dtype=np.float32)
        items = [query, *candidates]
        if self._pool is None:
            prepared = [self._decode_preprocess(item) for item in items]
        else:
            prepared = list(self._pool.map(self._decode_preprocess, items))
        query_tensor = prepared[0].unsqueeze(0).expand(len(candidates), -1, -1, -1)
        candidate_tensor = torch.stack(prepared[1:])
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ),
        ):
            logits = self.model(
                query_tensor.to(self.device), candidate_tensor.to(self.device)
            )
        result = logits.float().cpu().numpy().astype(np.float32, copy=False)
        if result.shape != (len(candidates),) or not np.isfinite(result).all():
            raise RuntimeError("verifier returned invalid logits")
        return result
