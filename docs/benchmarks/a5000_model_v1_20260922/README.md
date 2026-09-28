# A5000 model-v1 acceptance — 22.09.2026

Paid session `falcon-a5000-day3` on Sesterce: 2× NVIDIA RTX A5000 24 GB,
16 vCPU, 90 GB RAM, Ubuntu 22.04. The exact repository commit is stored in
`git_commit.txt`; machine-readable C5 results are in `bench_final.json`.

Command:

```bash
REID_PREPROCESS_WORKERS=8 ./scripts/a5000_session.sh \
  --models artifacts/models/final \
  --verifier artifacts/models/verifier_f0/verifier.pt \
  --output outputs/a5000_session_20260922_retry \
  --label a5000-model-v1-20260922
```

The script completed in 135 seconds after its Docker layers were cached. It
built the pinned image, ran the full test query/gallery pipeline with
`network_mode: none` and the real pair verifier, validated all three output
files, then ran the C5 benchmark.

Key result: p50 **16.848 ms**, p90 **18.675 ms**, best throughput
**241.087 FPS** at batch 32, peak allocated VRAM **834.9 MiB**, model load
**1.091 s**. The container reported PyTorch `2.5.1+cu121`, CUDA runtime 12.1
and two visible A5000 GPUs. Host driver: `580.178.04`.

Offline output hashes:

```text
c58f26191f5796cec3e5a4d8e8962f1c67d568ce8950d83dc5640ddd82df079d  submission.csv
41e8881ee92a14c9b8573c627ff46b4be553eb5b874ad7aa494d0a1ac24a12ee  embeddings.npy
a1fe50603839c0f6a2dca20393637b3a3cc7526bf3010df8d305eed86c332218  candidates.csv
```

The provider's current Ubuntu package named `nvidia-driver-535-server` is a
transitional package that installs R580, so this machine could not provide a
literal R535 runtime. CUDA 12.1 requires a Linux driver at least 525.60.13;
therefore R535 satisfies the published CUDA compatibility boundary, but a
strict direct R535 smoke remains separate from this hardware run. Source:
[NVIDIA CUDA 12.1 release notes](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/).
