# Final D-065 pair: self-contained A10 acceptance, 27 September 2026

Selected stack: **DINO392 + CLIP288, verifier392, D3072, weight0.15,
threshold6.6, no SigLIP**. No model, preprocessing, threshold or ranking
algorithm was changed during this acceptance. Test inputs were used only for
inference, format and repeatability, not calibration or error analysis.

## Identity and scope

- Build source: `6a496647898f091c32f2d40f63822547f62550c3`, clean `git archive`
  plus verified private release cache. Ordinary `docker build` without `--target`.
- Image ID / local OCI index digest:
  `sha256:cf90a88d6841b169d2c8d543162ce7da710464cd3c31c965daf5759133f0fa89`.
  Docker29 reports it in `RepoDigests`; **no registry image was uploaded**.
  Reproduction uses the pinned Dockerfile and weight release, not a claimed
  pullable image. [Raw image inspection](image_inspect.json).
- Manifest SHA:
  `4e3eacd2431ae634369b8e34689cb48d579f034207d1fd72bdff29661ad71e22`.
- Archive SHA:
  `9eeb99fae8bdcb228265ee83ac51da60da9896199d4746d843b30f3976fa49a8`.
- [Complete identity](identity.json), [package and release](../final_pair_release_20260927/README.md).
- Host: NVIDIA A10, driver615.71.09, 8vCPU Intel Icelake, 32GB RAM;
  Linux/amd64, PyTorch2.5.1+cu121/CUDA12.1. [Hardware/version receipt](host.txt).
  This is **not** another A5000/R535 measurement.

All inference containers used `--network none`. Only the dataset (read-only)
and output directory were mounted: **no host source or weight mounts**.
The minimal image has no Git metadata/executable, so raw reports record a null
commit; the externally recorded build source above supplies that identity.

## Results

| Check | Result |
|---|---|
| Image `make check` | **482 passed, 3 host-Git skips**, C6 self-test and actual pair/verifier smoke50 PASS |
| Required smoke | Additional `--require-runner` smoke50 PASS; no model skip |
| Local `make check` | **493 passed**, C6 and actual pair/verifier smoke50 PASS; [log](make_check_macos.log) |
| All-data metadata/integrity/whole-image weight audit | **PASS**, exactly **788647632 bytes**; [report](all_data_weights.json) |
| Two complete deterministic runs | **PASS**, all three files byte-identical; [report](determinism.json) |
| Two fresh containers, default ENTRYPOINT/CMD | **PASS**, also byte-identical to both harness runs; [comparison](comparison.json), [container1](container_1.json), [container2](container_2.json) |
| Full C6 | **PASS**, 1110 query/750 gallery, `(1860,3072)` float32; [report](c6.json) |
| Fixed-shape query content/permutation control | **PASS**, exact anchor embedding bytes, top10 and candidate rows; [report](query_content.json) |
| Singleton versus batched strict numeric check | **FAIL retained**; confidence-only differences, details below; [report](query_independence.json) |
| Train-only approximate plate-area control | **PASS technically**,1138 query/636 gallery; [report](mask_control.json). Real plate coverage is not certified |

The three image test skips are exclusively host-side Git-history checks:
[re-run with reasons](host_only_skips.log). They are covered by the local run.
Eight additional local tests belong to the historical full-frame replay under
`docs/benchmarks/fusion_replay_20260925/`, outside the image's source allowlist.
Thus493 local tests are not presented as493 executed tests inside the image.

Full pipeline time, excluding model load: **127.768s /121.619s**.
Embedder load1.852/1.870s, verifier0.287/0.298s. Peak Torch-allocated memory
for the entire stack: **1436.692MiB**; not total process VRAM.

Fresh default containers ran17:26:09–17:28:08UTC and17:28:10–17:30:07UTC.
Their independent C6 receipts are [run1](c6_1.json) and [run2](c6_2.json).
Container inspection confirms exit0, `NetworkMode=none`, `/data` read-only
and `/out` writable, with no other host mounts. Thus all **four complete runs**
produce the exact canonical bytes below, including the ordinary product command.

## C5 speed, not end-to-end FPS

[Raw benchmark](benchmark.json): full JPEG→crop→resize→forward→L2,
50 warmups,300 batch1 measurements, each throughput measurement ≥10s,
8 preprocessing workers,8 Torch threads; experimental projection precast off.

| Batch | FPS |
|---:|---:|
| 1 | 41.231 |
| 8 | 100.235 |
| 16 | 106.278 |
| 32 | **107.288** |

Batch1 p50 **24.415ms**, p90 26.780ms; load2.672s; extractor peak1312.959MiB.
Verifier/search are outside this timer, but inside the full pipeline above.
The earlier actual A5000/R535 pair result remains **102.674FPS/24.884ms**;
do not relabel this A10 run or extrapolate it into a new A5000 score.

## Query independence versus numerical equality

The same fixed gallery is reused without mutation. In the fixed-shape GPU
probe,8 anchor queries retain **exact embeddings and outputs** when all24
neighbours are replaced, then the replacement batch is reversed. The CPU
control also passed. The [code audit](../final_pair_query_code_20260927/README.md)
found no query-to-query statistics/attention/normalization or result reuse.

The separate strict probe compares24 train queries at batch1 with batched,
reversed and extra-query variants. All top10 IDs, candidate IDs and acceptance
decisions agree.17 queries have confidence differences; maximum absolute
confidence delta **0.000392910000000413**, embedding delta
**0.0001313658431172371**. This is consistent with FP16/kernel batch-shape
rounding, not evidence by itself of using another query's contents.

Comparing the saved batched baseline directly with reverse/extra-query variants
gives exact aligned embeddings and output fields for all24 fixed queries:
[additional comparison](query_batch_controls.json). Reproduce without inference:
`PYTHONPATH=. python docs/benchmarks/final_pair_acceptance_20260927/compare_query_controls.py --inputs <raw>/query_independence --out <new-report.json>`.

**The original `passed:false` and wrapper's nonzero result are preserved.**
No tolerance, confidence rounding, parameter change or weakened assertion was
introduced to get a green result. The information-isolation evidence is
distinct from universal cross-shape bitwise equality, which is **not claimed**.
The repeated full runs use the same configuration and are byte-identical.
Neither finite probe proves equality for all possible near-threshold inputs.

## Canonical output identity

| File | Bytes | SHA-256 |
|---|---:|---|
| `submission.csv` |404040|`726ccb2b4a4f77d1f0e47d32baa365f4ea631ce1cf186586f66a1100a1914d79`|
| `embeddings.npy` |22855808|`33dec6f02f7ac94a1b9e37a65ef0fb156a566123b02877c07f8181b95038626c`|
| `candidates.csv` |242465|`90e1b19bba4e067bd7bb9867a1be3421eaea11e89fe3a3fe7be5172877ac064c`|

Input CSV SHA are in C6. These are integrity/format records, not test quality
statistics. The old D1536 files remain recoverable in Git history.

Full raw acceptance backup: `falcon_pair_acceptance_20260927.tar.gz`,
**133350514 bytes**, SHA
`8c6f0f387ba0de295de5de353e2e13b01696095d54ad23184cfff8791ce58d1b`.
Server and downloaded local archive hashes match. It contains complete
deterministic/plain/masked/query-control outputs and the main log, not source
images or credentials. Fresh default outputs are duplicates of the saved
canonical bytes; their separate execution/inspection receipts are committed here.
The archive was uploaded to the private pair release; [GitHub asset receipt](github_backup_asset.json)
confirms the same size/SHA. Download: [raw acceptance archive](https://github.com/Timur713/hack_falcon_2026/releases/download/fusion-pair-d065-20260927-rc1/falcon_pair_acceptance_20260927.tar.gz).

## GPU lifecycle

User limit:2hours/85₽ for the existing A10. Provider history records
UNSHELVE20:04:08,STOP20:33:28,SHELVE20:35:31MSK. Around20:36 the UI confirmed
**SHELVED_OFFLOADED**, tariff108₽/month for retained resources. GPU released;
VM/storage retained. About32minutes/22₽ estimated, not a reconciled invoice.
Essential outputs were backed up beforeSTOP; subsequent Git/doc work was local.

## Reproduce

Use the release-cache steps in [fusion_docker.md](../../fusion_docker.md), then:

```bash
BUILD_CONTEXT="$PWD" DATA_ROOT=/absolute/dataset \
  OUTPUT_ROOT=/absolute/new-pair-acceptance \
  FUSION_IMAGE=falcon-fusion:pair-d065-rc1 \
  BENCH_LABEL=final-pair-self-contained \
  bash scripts/fusion_rc_acceptance_session.sh
```

Expect the strict cross-shape numeric diagnostic to be reported separately,
not converted into an unconditional overall PASS. Additional fresh-process
default ENTRYPOINT/CMD checks use [repeat_default.sh](repeat_default.sh).

## Remaining boundaries, not missing A handoff

- [Independent Linux C4 replay](../final_pair_replay_b_20260927/README.md)
  reproduced both folds exactly at the metric level: mean mAP0.852541031330,
  CandidateScore0.960470231512. No missing logit was invented. This is C4 OOF2,
  not full-frame Docker quality or all-data train evaluation.
- Cache/full-frame quality parity and real plate coverage remain limitations;
  neither is certified by these technical checks.
- Public anonymous download/build can be checked only after authorized
  repository publication and credential/license review. No visibility changes
  or competition submission were performed by this session.
- [Current delivery audit](delivery_readiness.json) passes default runtime,
  canonicalC6/D3072/acceptedSHA and both provenance checks. It still flags
  historical credentials before public publication and the missing official
  `extractor.py`; its `final_acceptance:false` is preserved by design. These are
  not missing weights or an old SigLIP blocker.
