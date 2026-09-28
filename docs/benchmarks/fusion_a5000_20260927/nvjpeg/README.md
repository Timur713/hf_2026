# Batched nvJPEG probe — rejected, 27.09.2026

Code `a94239d`, script SHA in `report.json`; same A5000/R535 host and
all-data D-063 trio as the technical acceptance. No test images, quality
measurement, verifier, projection precast or production change.

```bash
PYTHONPATH=. python scripts/experiment_nvjpeg_fusion.py \
  --model /artifacts/fusion_all_d063 --data-root /data \
  --mode bench --device cuda:0 --workers 8 --torch-threads 8 \
  --out /out/nvjpeg_bench
```

The experimental path reads CPU encoded bytes, calls torchvision's batched
CUDA `decode_jpeg`, copies decoded full frames back to CPU, then uses the
unchanged shared C3 preprocessing. Temporary decoded frames exist only for
the current extraction call. The runtime container had no network; report
`commit:null` is expected because its source mount contains no Git metadata.

| Measurement | Batched nvJPEG | Production self-contained image |
|---|---:|---:|
| batch1 p50, ms | 28.849 | 31.599 |
| best throughput, FPS (batch32) | 49.287 | 81.445 |
| peak allocated, MiB | 1667.7 | 1667.7 |

Full C5: 50 warmups, 300 latency samples and at least 10 seconds each at
batch1/8/16/32. The small latency improvement does not compensate for much
worse throughput. This particular GPU-decode → CPU-preprocess implementation
is rejected. No OOF quality run was needed to reject it on speed, and no pixel
or mAP equivalence is claimed. This does not disprove GPU-resident ROI/C3
pipelines, which would be a different implementation requiring quality checks.

Sources are unchanged according to before/after SHA. Raw JSON and run log are
committed here; the earlier `results_20260927_0013.tar.gz` predates this probe.
