# D-063 integration candidate — 27.09.2026

Private prerelease, not the accepted competition submission. This immutable
bundle is intended to let the team reproduce the current all-data stack while
the remaining quality/provenance/masking gates are resolved.

- Three embedders: DINOv2-B/392 cls_top16, CLIP-B/288 cls_top16,
  SigLIP2-B/288 top16; concatenated D=3840.
- Working frozen all-data verifier, input392, SHA-256
  `aa254e396ac0be78d29f32641e83f8d18e165d76eb3345c18c1054d105b9abf5`.
- D-063 policy: verifier weight0.15, candidate threshold6.3, gallery-only hub.
- Total inference weights 1,160,606,944 bytes, below 2,000,000,000.
- Archive `fusion-d063-trio-bundle.tar.gz`: 1,077,315,785 bytes, SHA-256
  `ee364eb7bb965ff308b50975fab3bd74827d348a5659f49d1c44297a4be9d2f4`.
- Manifest SHA-256
  `b47761be7a7ab03163e9d9f76f029abe9a4a9c0ac9a1f3c1634f57086c4eaf7a`.

Source assets are the independently SHA-verified files in `fusion-all-v1`.
The original SHA256SUMS there predates the working input392 verifier; this
bundle's manifest includes the actual accepted file. Model/config bytes have
not been modified. Fusion scales were computed from train fold0 gallery only;
no query statistics or test-based calibration were used.

Use the explicit Docker target `fusion`, with the fixed pins in Dockerfile.
The default target remains the older `legacy` image until full acceptance.
Build can access package repositories; runtime uses `--network none`. Do not
pass GitHub credentials into the build: download this private asset beforehand
with `gh release download` and use the checked local archive cache.

Measured A5000 C5 baseline: trio 79.6–82.2 FPS, pair speed probe102.7 FPS.
Thread/precast diagnostics reached about86 FPS for the trio, not100. The host
has8vCPU rather than the organizer's128logicalCPUs. No new all-data mAP is
claimed. Independent OOF controls found that the raw-frame/cache gap comes
mainly from input preparation, notFP16. OOF verifier assets and matched fold1
pooling/provenance still need confirmation by A. A geometric plate mask is not
proof that actual plates are covered. See the dated reports and PLAN for live
technical acceptance status; this prerelease label must not be read as closure
of B7.1/B7.2.

## Completed technical acceptance

Clean tracked source `f47f8a7669c924616ee699fbbcd2f2cbbe887049`, cached verified
bundle, image ID `sha256:8d2a53783e391b5c3f98fca375db7a0bb56bc3d5a5d52ab5e52da745e81caaa9`.
This is a local image ID, not an uploaded registry digest. Actual R535 driver;
self-contained runtime without source/model mounts and with `--network none`.
397 tests passed, two host Git-history fixtures skipped (Git absent), smoke50
passed. Full test runs took132.216/132.305s with the working input392 verifier;
all three files are byte-identical and C6-valid. C5:31.599ms/81.445FPS.

Result asset `results_20260927_0013.tar.gz`:440760477bytes, SHA-256
`ac6846dd8ce41a8218ad433a17b0e628f9476bc0a773adc766c8c1bf44893ee0`.
Server/local/GitHub digests match. It includes111 outputs/log files and no
source images. The candidate C6 files are under
`outputs/offline_d063/determinism/run_1/`; canonical `submission/` is unchanged.

Query isolation test uses train only: top10, candidate IDs and acceptance
unchanged on24 queries; reverse/extra-neighbor batches are bitwise equal.
Singleton versus batch has confidence drift up to0.001421, so the strict gate
remains FAIL. The overall acceptance queue correctly exits nonzero.
Train geometric masked runs passed C6 but do not establish actual plate coverage.
All-data outputs are not used to claim OOF mAP or recalibrate thresholds.

The later negative nvJPEG experiment is committed under
`docs/benchmarks/fusion_a5000_20260927/nvjpeg/`, not in this earlier result asset.
