# Final pair source provenance — 27.09.2026

This handoff closes the source-weight metadata required by rule 5 for the final
pair selected in D-065. It does not change model tensors, preprocessing,
ranking, calibration, or reported quality.

## Result

| Final component | Public initialization | Immutable revision | Source SHA-256 |
|---|---|---|---|
| `hires392_all` | Meta AI DINOv2 ViT-B/14, timm conversion | `4685c99dabffe5affac90bd99dbffd25801ae58d` | `55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b` |
| `div_clip_all` | LAION/OpenCLIP ViT-B/16 LAION-2B, timm conversion | `fbadfdd1c0e903c58855b70ae04740b22da9d305` | `35fcf20089626ce1534aa3d4d8d97a02ca79e171a1735d0c8d58e150eded9ad5` |
| `v_fus_all_frz_input392` | Meta AI DINOv2 ViT-S/14, timm conversion | `4610ca143709d58a633b6397a74412c2c3842454` | `04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081` |

The exact public URLs, byte sizes, licenses, upstream projects, final artifact
hashes, and pinned Python package versions are recorded in
`source_models.json`. The architecture-to-timm mapping is unchanged from the
training commit `b2823b0` in `reid/model/__init__.py`; that code requests
`pretrained=True` for the exact names listed in the manifest.

## Verification

The Hugging Face model API was queried with blob metadata enabled. All three
repositories reported the revision, `model.safetensors` byte size, and LFS
SHA-256 stored in the manifest. The DINOv2-B and DINOv2-S files were also
hashed from the local Hugging Face cache. The CLIP file was streamed from its
immutable `resolve/<revision>/model.safetensors` URL and hashed locally.

Run the metadata check without downloading 779 MB of weights:

```bash
PYTHONPATH=. python scripts/analysis/verify_model_provenance.py \
  --manifest docs/benchmarks/final_pair_provenance_20260927/source_models.json
```

Use `--download` to hash the actual immutable source bytes as well. The command
fails closed on a changed revision, missing file, byte-size mismatch, or digest
mismatch.

## Handoff to B

Use `hires392_all.yaml` and `div_clip_all.yaml` instead of the metadata-poor
YAML files currently attached to `fusion-all-v1`. Copy the verifier sidecar
next to its checkpoint, include all three source rows in README, and recalculate
the final bundle/manifest SHA. Existing weight SHA-256 values remain unchanged.
