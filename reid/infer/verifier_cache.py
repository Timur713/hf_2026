"""Experimental, bounded static-gallery token cache; never caches query state.

The backbone uses a fixed batch of four. This can differ numerically from the
legacy joint 2*K backbone forward. Keep opt-in until exact output/quality and
CUDA speed gates have been measured; mathematical equivalence is not a receipt.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from reid.infer.verifier import PairVerifier
from reid.model.verifier import PairVerifier as ConcatNetwork
from reid.post.verifier import Item


def _key(item: Item) -> tuple:
    path, bbox = item
    coords = tuple(float(x) for x in bbox)
    if len(coords) != 4 or not np.isfinite(coords).all():
        raise ValueError("cache item requires a finite four-coordinate BBox")
    return (str(Path(path).resolve()), coords)


class StaticGalleryTokenVerifier:
    """A cache bound to one immutable model, preprocessing config and gallery.

    Existing adapter decoding is reused, including masked-control subclasses.
    No camera/vehicle labels or other queries are accepted. Unknown candidates
    fail closed, rather than populating a request-dependent cache. Cache tensors
    stay on the model device; the byte cap excludes model and temporary tensors.
    """

    INTERNAL_BATCH = 4

    def __init__(
        self,
        verifier: PairVerifier,
        gallery: Sequence[Item],
        max_cache_bytes: int = 1024 * 1024 * 1024,
    ) -> None:
        if type(verifier.model) is not ConcatNetwork:
            raise ValueError(
                "token cache supports only the unchanged concat PairVerifier"
            )
        if not gallery or max_cache_bytes < 1:
            raise ValueError(
                "nonempty static gallery and positive cache budget required"
            )
        self.verifier = verifier
        self.device = verifier.device
        self._cfg = deepcopy(verifier.cfg)
        self._mask = deepcopy(getattr(verifier, "plate_mask", None))
        self._signature = self._model_signature()
        keys = [_key(item) for item in gallery]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate static gallery path/BBox")
        self._index = {key: index for index, key in enumerate(keys)}
        self._tokens = None
        for start in range(0, len(gallery), self.INTERNAL_BATCH):
            block = gallery[start : start + self.INTERNAL_BATCH]
            tokens = self._encode(block)
            if self._tokens is None:
                required = len(gallery) * tokens[0].numel() * tokens.element_size()
                if required > max_cache_bytes:
                    raise ValueError(
                        f"gallery token cache needs {required} bytes; cap={max_cache_bytes}"
                    )
                self._tokens = torch.empty(
                    (len(gallery), *tokens.shape[1:]),
                    dtype=tokens.dtype,
                    device=self.device,
                )
            self._tokens[start : start + len(block)].copy_(tokens)
        self.cache_bytes = self._tokens.numel() * self._tokens.element_size()

    def _model_signature(self) -> tuple:
        model = self.verifier.model
        if any(module.training for module in model.modules()):
            raise ValueError("token cache requires an immutable eval-mode model")
        return tuple(
            (name, id(tensor), tensor._version, str(tensor.device), tensor.dtype)
            for name, tensor in [*model.named_parameters(), *model.named_buffers()]
        )

    def _check_immutable(self) -> None:
        if (
            self._model_signature() != self._signature
            or self.verifier.cfg != self._cfg
            or getattr(self.verifier, "plate_mask", None) != self._mask
            or self.verifier.device != self.device
        ):
            raise ValueError(
                "model/preprocessing changed; rebuild the static gallery cache"
            )

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        )

    def _encode(self, items: Sequence[Item]) -> torch.Tensor:
        # Query calls contain exactly one item, repeated only within that same
        # query's fixed-shape forward. Gallery preparation never receives queries.
        pool = self.verifier._pool
        decode = self.verifier._decode_preprocess
        prepared = (
            [decode(item) for item in items]
            if pool is None
            else list(pool.map(decode, items))
        )
        prepared += [prepared[-1]] * (self.INTERNAL_BATCH - len(prepared))
        with torch.inference_mode(), self._autocast():
            tokens = self.verifier.model.backbone.forward_features(
                torch.stack(prepared).to(self.device)
            )[: len(items)]
        if tokens.ndim != 3 or not torch.isfinite(tokens).all():
            raise RuntimeError("verifier backbone returned invalid tokens")
        return tokens

    def score_pairs(self, query: Item, candidates: Sequence[Item]) -> np.ndarray:
        self._check_immutable()
        if not candidates:
            return np.empty(0, dtype=np.float32)
        try:
            indices = [self._index[_key(item)] for item in candidates]
        except KeyError as exc:
            raise ValueError("candidate outside prepared static gallery") from exc
        n = len(candidates)
        # Neither query tokens nor logits survive this call.
        query_tokens = self._encode([query]).expand(n, -1, -1)
        model = self.verifier.model
        with torch.inference_mode(), self._autocast():
            gallery_tokens = self._tokens[indices]
            a = model.proj(query_tokens) + model.segment[0]
            b = model.proj(gallery_tokens) + model.segment[1]
            joined = torch.cat([model.cls.expand(n, -1, -1), a, b], dim=1)
            logits = model.head(model.encoder(joined)[:, 0]).squeeze(-1)
        result = logits.float().cpu().numpy().astype(np.float32, copy=False)
        if result.shape != (n,) or not np.isfinite(result).all():
            raise RuntimeError("cached verifier returned invalid logits")
        return result
