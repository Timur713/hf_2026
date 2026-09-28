# Bounded Inductor probe — not enabled, 27.09.2026

Code: `41992f2` (`experiment_compile_fusion.py`), isolated toolchain Dockerfile
`64db058`. Actual `model.features()` callables are wrapped with
`torch.compile(backend="inductor", mode="default", fullgraph=True, dynamic=True)`.
No model, shared C3, production Extractor, threshold or default image changes.
Train fold0 inputs only; all-data weights permit this numeric/speed probe,
**not a validation mAP claim**. No verifier/search is timed.

## Result

Same weights, host, eight CPU threads/workers, serial eager → compiled runs:

| Full C5 | Eager | Compiled |
|---|---:|---:|
| batch1 p50, ms | 32.794 | 30.054 |
| batch1 p90, ms | 35.291 | 32.471 |
| best FPS among1/8/16/32 (batch32) | 80.768 | 87.537 |
| peak allocated, MiB | 1667.671 | 1603.090 |

50 warmups/300 latency samples and at least10s per throughput batch. Compilation
is excluded from these warmed C5 timings, but its startup cost is recorded:
the first32-image batch1 probe took35.03s and batch8 took43.87s including
compilation; subsequent batch16/32 probes took~0.37s. Whole experiment192.54s.

All32 fixed, evenly spaced train images were compared at each batch size.
Every batch-size comparison has32 differing rows. Maximum absolute embedding
drift across comparisons is **0.000676796**, minimum cosine **0.999997947**.
These small differences are not bitwise equivalence and do not prove unchanged
ranking, candidates or mAP. Source model SHA remained unchanged.
Status `completed_with_drift` / exit2 is deliberately retained, not hidden.

Throughput improved~8.4% in this serial pair, but still stayed below100FPS.
It does not remove the pair/trio tradeoff. Do not enable by default: any later
promotion requires matched OOF quality, query-independence, full offline
cold-start/timeout and deterministic-output acceptance of that exact backend.
The successful numeric/speed execution is not final competition acceptance.

## Environment and failed first attempt

A5000/R535.54.03, Torch2.5.1+cu121, Triton3.1.0. Runtime has no network.
Parent image ID `sha256:fd37c6b13ce3ef4e3ba33e92ea7fe9e6433219f1f6e469e92783572bcd97677e`.
Experimental image with headers:
`sha256:712f7eff03df3150dc995eb4064cd48dec7a89576075e2f806e11c5cb5aed15d`.
These are local image IDs, not uploaded registry digests. `headers/packages.tsv`
records installed system versions; meta-packages g++ and python3-dev are pinned
in Dockerfile, but a future production toolchain needs its full dependency lock.

The first extra image installed g++ only and failed because `Python.h` was
absent. `initial/` preserves that70.77s run, raw error and its eager baseline
(81.639FPS); it is not counted as a compiled measurement. One corrected retry
added Python development headers. Production image was never modified.

```bash
docker build -f scripts/Dockerfile.compile-probe \
  -t falcon-compile-probe:20260927-headers .
# In this isolated container, source at/app, data/model mounts read-only,
# output at/out, --gpus all --network none --shm-size2g:
PYTHONPATH=/app python3 /app/scripts/experiment_compile_fusion.py \
  --model /artifacts/fusion_all_d063 --data-root /data --splits /app/splits \
  --out /out/compile_fusion_headers_20260927 --device cuda:0 \
  --workers 8 --torch-threads 8 --max-seconds 840
```

Output must not already exist. Supervisor isolates its worker process group
and terminates it at the840s budget; build was separately bounded to300s.
Raw reports/logs and package receipt are committed here. Inductor/Triton caches
are regenerable and are not part of the Git payload or the production bundle.
