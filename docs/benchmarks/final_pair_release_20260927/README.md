# Final pair D-065 package — 27.09.2026

## Completed locally

- A handoff verified from `a4adb87`: DINO392 + CLIP288, verifier392, weight0.15, threshold6.6; no SigLIP.
- Source tensor SHA checked against D-066. YAML changes are metadata-only.
- `prepare_fusion_artifact.py` extracted **only 636 train fold0 gallery images**, CPU/PyTorch2.2.2, batch16. Fixed settings; no query or threshold search.
- Strict USTAR archive installation passed; every member/size/SHA and verifier sidecar/checkpoint binding checked.
- `audit_release.py weights --require-all-data` PASS: **788647632 bytes**. This verifies checkpoint metadata, not historical training membership from tensors.
- `make check`: **466 tests PASS**, C6 self-test PASS, actual full-stack smoke50 PASS on macOS/CPU. No skipped smoke. [Log](make_check_macos.log).
- After the supplementary fixed-shape query checker and updated readiness
  audit: **493 tests PASS**, C6 self-test and real pair/verifier smoke50 PASS.
  [Final local log](make_check_macos_final.log). GPU acceptance remains separate.
- Default Docker/Compose select fusion; explicit legacy remains available. No new runtime model/preprocessing/postprocessing algorithm is introduced.
- Supplementary **real CPU content-isolation PASS**: fixed gallery636, three
  batches32, eight anchor queries unchanged when all24 neighbours are replaced
  and when the replacement batch is reversed. Anchor embedding bytes, top10
  and candidate rows match exactly. [Machine report](query_content_cpu.json),
  [log](query_content_cpu.log). This does not replace GPU checks or the separate
  singleton-vs-batch strict numerical diagnostic.

## Immutable package

Release: [fusion-pair-d065-20260927-rc1](https://github.com/Timur713/hack_falcon_2026/releases/tag/fusion-pair-d065-20260927-rc1).

Published **2026-09-27 16:58:48 UTC**, private prerelease, not a draft.
[GitHub API receipt](github_release.json) verifies all four uploaded asset
sizes/digests against local files. The previous model-v1 and trio RC were not
overwritten. Release tag was created from pushed commit `0e9de2a8eec338d6273966af1de322c452bb8589`.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `fusion-pair-d065-bundle.tar.gz` | 732536437 | `9eeb99fae8bdcb228265ee83ac51da60da9896199d4746d843b30f3976fa49a8` |
| `manifest.json` | see file | `4e3eacd2431ae634369b8e34689cb48d579f034207d1fd72bdff29661ad71e22` |
| `fusion_config.json` | 379 | `eace9e6e59699b168e2d2ae6cef5d8d063f0cbe01241e4e8f311d21255e90fe6` |

[Manifest](manifest.json) contains trained weight URLs, all member hashes, preparation environment and gallery CSV hash. [Config](fusion_config.json) pins D=1536+1536, scales, weight and threshold. [All-data audit](weights_audit.json) validates the installed archive's actual bytes.

Source preparation commit `3769e4f` includes sidecar support `97aa71b`; code is based on A `a4adb87`. Final build commit and image digest must be recorded after the real build; neither is invented from the package hash.

## GPU acceptance completed in a separate bounded session

The local package preparation above incurred no GPU cost. Subsequently the user
authorized A10≤2h/85₽. The final image from`6a49664` was built and exercised
offline; four full runs (including two fresh default containers) have identical
outputs,C6,all-data weight audit and trainmasked-control passed. Canonical
`submission/` is nowD3072. Image identity,three outputSHA and rawresults:
[final pair acceptance](../final_pair_acceptance_20260927/README.md).

GPU fixed-shape neighbour replacement/permutation passed exactly. The separate
singleton/batched strict numerical check remainsFAIL with confidence-only
delta≤0.000393; original receipts are retained, not weakened or labelledPASS.
A10 was returned toSHELVED_OFFLOADED around20:36MSK after verifiedbackup,
about32min/22₽estimated,withinbudget. Storage remains billable.

Independent [Linux/x86 C4 replay](../final_pair_replay_b_20260927/README.md)
now exactly reproduces bothfolds' metrics with the original logits. The earlier
ARM cutoff failure remains recorded; no missing logit was invented. This still
does not turn OOF C4 quality into full-frame all-data Docker quality.

## Commands

See [fusion_docker.md](../../fusion_docker.md) for pinned private-cache download, public build and offline acceptance commands. Archive verification on this host:

```bash
PYTHONPATH=. python scripts/stage_fusion_release.py --cache-root . \
  --bundle artifacts/absent_release_directory \
  --archive-cache artifacts/fusion-pair-d065-bundle.tar.gz \
  --destination artifacts/archive_install_verified \
  --manifest-sha256 4e3eacd2431ae634369b8e34689cb48d579f034207d1fd72bdff29661ad71e22 \
  --source-release-url https://github.com/Timur713/hack_falcon_2026/releases/tag/fusion-pair-d065-20260927-rc1 \
  --archive-url https://github.com/Timur713/hack_falcon_2026/releases/download/fusion-pair-d065-20260927-rc1/fusion-pair-d065-bundle.tar.gz \
  --archive-sha256 9eeb99fae8bdcb228265ee83ac51da60da9896199d4746d843b30f3976fa49a8 \
  --require-verifier
PYTHONPATH=. python scripts/audit_release.py weights \
  --root artifacts/archive_install_verified --model artifacts/archive_install_verified \
  --verifier artifacts/archive_install_verified/verifier.pt --require-all-data \
  --report outputs/pair_weights_audit.json
make check PYTHON=.venv/bin/python
```

Use new destination/report paths for a repeat run; the installer refuses overwrites. The package was archived with `COPYFILE_DISABLE=1 tar --format=ustar`; manifest URLs/environment annotations were added before hashing without changing model/config bytes or scales.
