# A5000: direct measurements, 27.09.2026

Session host: one NVIDIA RTX A5000 24 GB, 8 vCPU, 45 GB RAM, 250 GB disk,
NVIDIA driver 535.54.03 (CUDA 12.2), Ubuntu 22.04. Container: PyTorch
2.5.1+cu121; runtime network disabled. This is the correct GPU architecture,
but **not the organizer's 128-logical-CPU host**. Measurements are indicative,
not awarded competition points or a measurement of the whole pipeline.

## Initial complete C5 comparison

Inputs are full train fold0 JPEG frames and their BBoxes. Each timing includes
read/decode/crop/preprocess/forward/L2. Serial runs, 50 warmups, 300 batch1
measurements, >=10 seconds per batch. No test-based tuning.

| Configuration | p50 batch1, ms | Best FPS among 1/8/16/32 | Estimated speed points /20 |
|---|---:|---:|---:|
| D-063 trio, default 8 CPU threads /8 workers | 31.457 | 82.158 | 16.432 |
| DINO392+CLIP288 pair, same settings | 24.884 | 102.674 | 20.000 |
| Trio repeat | 30.583 | 79.594 | 15.919 |
| Trio, 1 thread /8 workers | 40.337 | 86.057 | 17.127 |
| Trio, 2 threads /8 workers | 35.804 | 85.516 | 17.103 |
| Trio, 1 thread /4 workers | 40.265 | 70.967 | 14.127 |

The extrapolated 112 FPS was not confirmed. The earlier A10 ratio cannot be
transferred to this larger fusion. The two best thread settings differ by
0.024 projected points, below a useful claim of a robust win; do not cherry-pick
one 10-second sample. Batch64/128 in the diagnostic sweep are **not eligible**
for the specified score and were excluded above.

Speed points use the continuous formulas in MainTechSpec §7.1, not a binary
100-FPS pass/fail. The throughput part is worth 0.2 points per FPS between 50
and 100. Architecture choice requires its own quality/candidate evidence too.

## Quality evidence for composition (not a final decision)

CPU replay of the released, SHA-verified fold0 C4 dumps gives the pair:
fusion **0.839196**, then hub **0.861191**, without verifier. The corresponding
trio +hub is **0.862852**. This is a single-fold comparison, not evidence of
equal full-stack two-fold quality; no threshold was selected in this probe.
The pair's trial 6.3 must not be described as calibrated for that composition.

The all-data trio contains the new frozen input392 verifier with SHA
`aa254e396ac0be78d29f32641e83f8d18e165d76eb3345c18c1054d105b9abf5`, rerank
weight 0.15 and D-063 threshold 6.3. All seven source assets were independently
checked against GitHub API SHA digests (the old upstream SHA256SUMS file is
not sufficient for the new verifier). Total inference weights:
**1,160,606,944 bytes**, below the decimal 2 GB limit. All-data metadata gate
passes; metadata alone does not prove historical training provenance.

## Reproduction

```bash
SESSION_ROOT=/path/to/isolated-session DATA_ROOT=/path/to/dataset \
  FUSION_BASE_IMAGE=falcon-runtime:20260927 \
  bash scripts/fusion_a5000_final_session.sh
SESSION_ROOT=/path/to/isolated-session DATA_ROOT=/path/to/dataset \
  bash scripts/fusion_thread_sweep.sh
```

The session expects verified release assets already downloaded by
`fetch_release_remote.py`; no GitHub token enters the server or image.
Raw reports are in `raw/`. These first measurements do not by themselves
close final Docker acceptance, output identity, OOF quality or plate robustness.

## Projection-precast experiment (not enabled in production)

`c8b7711` adds an opt-in `precast_projections=True`: frozen Linear/Conv2d
parameters are converted once to the existing AMP dtype. Norms, embeddings and
buffers remain FP32; source checkpoints are unchanged. The actual GPU cast
covered 152 modules /303 parameters, 1,055,081,472 →527,540,736 parameter bytes.

`check_precast_equivalence.py` at `5cf97c9`, with explicit `--device cuda:0`,
compared identical all-data weights on all **1774 train fold0 inputs at batch32**
and 32 evenly spaced batch1 inputs. Both paths were **bitwise identical**,
maximum absolute difference zero. This is numeric equivalence on those inputs,
not an OOF quality estimate and not a proof for every possible input/backend.
The first call using implicit `cuda` failed before computation; `1535992`
fixes that argument resolution. The successful raw report retains the actual
older script SHA, rather than claiming it ran the later correction.

| Opt-in precast setting | p50 batch1, ms | Best FPS, batch32 | Peak allocated MiB |
|---|---:|---:|---:|
| 8 CPU threads /8 workers | 29.689 | 81.426 | 1198.0 |
| 2 CPU threads /8 workers | 33.404 | 86.430 | 1198.0 |

The corresponding non-precast peak was 1667.7 MiB. Memory improves by about
470 MiB, but throughput remains below 100 and within the spread of the earlier
thread runs. This experiment **does not solve the speed tradeoff**, so the
production default remains false. No timing overlap with the subsequent Docker
build occurred. See `raw/precast_*.json`; reproduction is
`scripts/fusion_precast_benchmark.sh` after the successful parity report.
