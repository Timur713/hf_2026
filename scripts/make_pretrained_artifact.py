"""Упаковывает публичные веса backbone в артефакт модели без дообучения (zero-shot baseline).

Записывает источник весов (URL на Hugging Face, ревизию, sha256) в model_config.yaml.
    PYTHONPATH=. python scripts/make_pretrained_artifact.py --arch dinov2_vitb14 --input 224 --pool cls
"""
import argparse
import hashlib

from huggingface_hub import HfApi, hf_hub_download

from reid.model import REGISTRY, Embedder, build_backbone, save

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="dinov2_vitb14", choices=sorted(REGISTRY))
    ap.add_argument("--input", type=int, default=224)
    ap.add_argument("--pool", default="cls")
    ap.add_argument("--name")
    args = ap.parse_args()

    timm_name, tag, _takes_img_size = REGISTRY[args.arch]
    repo = f"timm/{timm_name}.{tag}"
    revision = HfApi().model_info(repo).sha
    weights_path = hf_hub_download(repo, "model.safetensors", revision=revision)
    backbone = build_backbone(args.arch, [args.input, args.input], pretrained=True)
    name = args.name or f"{args.arch}_zeroshot_{args.input}_{args.pool}"
    cfg = {
        "name": name,
        "arch": args.arch,
        "pretrained": {"url": f"https://huggingface.co/{repo}", "revision": revision,
                       "sha256": sha256(weights_path), "author": "Meta AI (DINOv2), timm port"},
        "input_size": [args.input, args.input],
        "bbox_pad": 0.0,
        "resize": "stretch",
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "pool": args.pool,
        "embed_dim": backbone.num_features * (2 if args.pool == "cls_avg" else 1),
        "infer_dtype": "fp16",
        "tta_flip": False,
        "train": {"commit": None, "config": None, "folds": "none (zero-shot)"},
        "val": {},
    }
    save(Embedder(backbone, args.pool), cfg, f"artifacts/models/{name}")
    print(f"saved artifacts/models/{name}  ({repo}@{revision[:10]})")


if __name__ == "__main__":
    main()
