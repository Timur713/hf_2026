"""D082 extraction shared by the application, singleton API and speed benchmark."""

from __future__ import annotations

from collections import deque
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch

from reid.infer.d082 import extract_component_rows
from reid.infer.extractor import Extractor
from reid.post.ranker import PreparedRankerGallery


class GraphFeatures:
    """Own fixed-shape graph buffers; callers serialize complete extraction calls."""

    def __init__(self, model, amp_dtype):
        self.model = model
        self.amp_dtype = amp_dtype
        self.graph = None
        self.capture_seconds = None

    def _context(self):
        return torch.autocast(
            device_type="cuda",
            dtype=self.amp_dtype,
            enabled=self.amp_dtype is not None,
            cache_enabled=False,
        )

    def _capture(self, batch):
        started = time.perf_counter()
        self.input = torch.empty_like(batch)
        self.input.copy_(batch)
        stream = torch.cuda.Stream(device=batch.device)
        current = torch.cuda.current_stream(batch.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream), torch.inference_mode(), self._context():
            for _ in range(3):
                self.model.features(self.input)
        current.wait_stream(stream)
        torch.cuda.synchronize(batch.device)
        graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.graph(graph, stream=stream),
            torch.inference_mode(),
            self._context(),
        ):
            output = self.model.features(self.input)
        current.wait_stream(stream)
        self.output = output
        self.graph = graph
        self.capture_seconds = time.perf_counter() - started

    def features(self, batch):
        if batch.device.type != "cuda" or batch.shape[0] != 4:
            raise ValueError("CUDA Graph requires a CUDA fixed-four batch")
        if self.graph is None:
            self._capture(batch)
        if (batch.shape, batch.dtype, batch.device) != (
            self.input.shape,
            self.input.dtype,
            self.input.device,
        ):
            raise ValueError("CUDA Graph input signature changed")
        self.input.copy_(batch)
        self.graph.replay()
        return self.output


class D082Extractor(Extractor):
    """Production D082: bounded JPEG pipeline, member graphs, singleton H2D.

    CUDA forwards retain fixed-four shape and original FP16 arithmetic. CPU is
    the reference fixed-four path. Graphs own their buffers; the call lock also
    protects capture and recovery from decoding or invalid-feature failures.
    """

    def __init__(
        self,
        model_dir,
        device="cuda",
        preprocess_workers=None,
        precast_projections=False,
        fusion_internal_batch_size=4,
    ):
        if precast_projections or fusion_internal_batch_size != 4:
            raise ValueError("D082 requires original precision and fixed-four batches")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        super().__init__(
            model_dir,
            device,
            preprocess_workers=preprocess_workers,
            precast_projections=False,
            fusion_internal_batch_size=4,
        )
        if self._fusion_components is None or len(self._fusion_components) != 2:
            raise ValueError("D082 requires its two original fusion components")
        self._call_lock = threading.RLock()
        if self.device.type == "cuda":
            self._fusion_components = [
                (GraphFeatures(model, dtype), cfg, dtype, std)
                for model, cfg, dtype, std in self._fusion_components
            ]
        scales = [c["similarity_std"] for c in self.cfg["components"]]
        reference = math.sqrt(min(scales))
        self._fusion = SimpleNamespace(
            coefficients=tuple(reference / math.sqrt(s) for s in scales)
        )

    def extract_batch(self, items):
        """Expose the same component normalization/fusion used by the pipeline.

        Keeping this in the production class makes the scoring extract() path
        and the application share arithmetic, without a benchmark-only adapter.
        """
        if not items:
            return np.empty((0, self.cfg["embed_dim"]), dtype=np.float32)
        rows = [
            SimpleNamespace(image_path=Path(path), bbox=tuple(bbox))
            for path, bbox in items
        ]
        components = extract_component_rows(self, rows, len(rows))
        return PreparedRankerGallery._fuse(self._fusion, components)

    def _prepared_chunks(self, items):
        pool = self._preprocess_pool
        if pool is None or len(items) == 1:
            prepared = (
                list(pool.map(self._decode_preprocess_fusion, items))
                if pool is not None and len(items) > 1
                else [self._decode_preprocess_fusion(item) for item in items]
            )
            for start in range(0, len(prepared), 4):
                yield prepared[start : start + 4]
            return
        # At most two fixed-four chunks are submitted ahead. Futures preserve
        # input order even when JPEGs finish out of order. CPU workers continue
        # decoding the next chunk while the caller runs this chunk on CUDA.
        pending = deque()
        iterator = iter(items)

        def refill():
            while len(pending) < 8:
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                pending.append(pool.submit(self._decode_preprocess_fusion, item))

        refill()
        try:
            while pending:
                chunk = [
                    pending.popleft().result() for _ in range(min(4, len(pending)))
                ]
                refill()
                yield chunk
        finally:
            # Drain running tasks before releasing the call lock, including on
            # failure; an old decode must not leak into a later masked call.
            for future in pending:
                future.cancel()
            for future in pending:
                if not future.cancelled():
                    try:
                        future.result()
                    except Exception:
                        pass

    def _forward_device(self, tensors, *, components_only):
        singleton = len(tensors) == 4 and all(row is tensors[0] for row in tensors[1:])
        if singleton:
            # All four references are this query's own padding. Model kernels
            # still consume the original contiguous four-image graph buffer.
            batches = tuple(
                tensor.unsqueeze(0)
                .to(self.device, non_blocking=True)
                .expand(4, -1, -1, -1)
                for tensor in tensors[0]
            )
        else:
            batches = tuple(
                torch.stack([row[index] for row in tensors]).to(
                    self.device, non_blocking=True
                )
                for index in range(len(self._fusion_components))
            )
        return self._forward_batches(batches, components_only=components_only)

    def _forward_batches(self, batches, *, components_only):
        blocks, checks = [], []
        reference = math.sqrt(min(c[3] for c in self._fusion_components))
        for index, (model, cfg, dtype, std) in enumerate(self._fusion_components):
            batch = batches[index]
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=self.device.type,
                    dtype=dtype,
                    enabled=dtype is not None,
                ),
            ):
                features = model.features(batch)
            features = features.float()
            if features.shape != (len(batch), cfg["embed_dim"]):
                raise ValueError(
                    f"fusion component {index}: unexpected embedding shape {tuple(features.shape)}"
                )
            norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
            checks.append(
                torch.isfinite(features).all()
                & torch.isfinite(norms).all()
                & (norms > 0).all()
            )
            if components_only:
                blocks.append(torch.nn.functional.normalize(features, dim=1))
            else:
                blocks.append((features / norms) * (reference / math.sqrt(std)))
        result = torch.cat(blocks, dim=1)
        if not components_only:
            result = torch.nn.functional.normalize(result, dim=1)
        return result, torch.stack(checks)

    def _extract_fusion(self, items, *, components_only=False):
        if self.device.type != "cuda":
            return super()._extract_fusion(items, components_only=components_only)
        with self._call_lock:
            outputs, checks = [], []
            prepared = self._prepared_chunks(items)
            try:
                for real in prepared:
                    padded = real + [real[-1]] * (4 - len(real))
                    result, valid = self._forward_device(
                        padded, components_only=components_only
                    )
                    outputs.append(result[: len(real)])
                    checks.append(valid)
            finally:
                prepared.close()
            valid = torch.stack(checks).cpu().numpy()
            if not valid.all():
                index = int(np.where(~valid)[1][0])
                raise ValueError(
                    f"fusion component {index}: embeddings must be finite and nonzero"
                )
            output = (
                torch.cat(outputs, dim=0).cpu().numpy().astype(np.float32, copy=False)
            )
            if components_only:
                boundaries = np.cumsum(
                    [c[1]["embed_dim"] for c in self._fusion_components]
                )[:-1]
                return tuple(np.split(output, boundaries, axis=1))
            return output
