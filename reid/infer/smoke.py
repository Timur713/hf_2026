"""Run the C5 pipeline on 50 inputs when its runner and model are available."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

from reid.eval.validate_outputs import ValidationError, validate_outputs


CSV_HEADER = ["image_id", "x", "y", "w", "h"]


def _read_rows(path: Path, limit: int) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_HEADER:
            raise ValueError(
                "{}: expected header {!r}, got {!r}".format(
                    path, CSV_HEADER, reader.fieldnames
                )
            )
        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append(row)
            if len(rows) == limit:
                break
    if len(rows) != limit:
        raise ValueError(
            "{}: need {} rows for smoke check, found {}".format(path, limit, len(rows))
        )
    return rows


def _write_rows(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def _preflight_images(images_dir: Path, rows: Sequence[Dict[str, str]]) -> None:
    missing: List[str] = []
    for row in rows:
        image_id = row["image_id"]
        image_path = images_dir / "{}.jpg".format(image_id)
        if not image_path.is_file():
            missing.append(str(image_path))
    if missing:
        preview = ", ".join(missing[:3])
        suffix = " ..." if len(missing) > 3 else ""
        raise ValueError("missing smoke input images: {}{}".format(preview, suffix))


def _resolve_model(model_dir: Path | None) -> Path | None:
    production = Path("artifacts/models/d082")
    if model_dir is None and (production.exists() or production.is_symlink()):
        model_dir = production / "bundle"
        from reid.infer.d082_production import inspect_bundle

        pin = (production / "manifest.sha256").read_text().strip()
        inspect_bundle(model_dir, pin)
        return model_dir
    # Prefer the shipped fusion image. A partially installed default bundle is
    # an error, never a reason to smoke a different, historical model instead.
    fusion = Path("artifacts/models/fusion")
    if model_dir is None and fusion.exists():
        model_dir = fusion
    if model_dir is not None and (
        model_dir == fusion or (model_dir / "fusion_config.json").exists()
    ):
        from reid.infer.fusion import load_fusion_config

        load_fusion_config(model_dir)
        return model_dir
    candidates = (
        [model_dir]
        if model_dir is not None
        else [
            Path("artifacts/models/final"),
            Path("artifacts/models/dinov2_vitb14_zeroshot_224_cls"),
        ]
    )
    for candidate in candidates:
        if (
            candidate is not None
            and (candidate / "model_config.yaml").is_file()
            and (candidate / "weights.safetensors").is_file()
        ):
            return candidate
    return None


def _resolve_verifier(model_dir: Path, verifier_path: Path | None) -> Path | None:
    if verifier_path is None and (model_dir / "fusion_config.json").is_file():
        verifier_path = model_dir / "verifier.pt"
    if verifier_path is not None and not verifier_path.is_file():
        raise ValueError(f"missing smoke verifier: {verifier_path}")
    return verifier_path


def run_smoke(
    data_root: Path,
    n_images: int,
    require_runner: bool,
    model_dir: Path | None = None,
    verifier_path: Path | None = None,
) -> int:
    if n_images < 20 or n_images % 2:
        raise ValueError("--n-images must be an even integer of at least 20")

    per_split = n_images // 2
    query_rows = _read_rows(data_root / "test_query.csv", per_split)
    gallery_rows = _read_rows(data_root / "test_gallery.csv", per_split)
    _preflight_images(data_root / "images", query_rows + gallery_rows)

    if importlib.util.find_spec("reid.infer.run") is None:
        message = (
            "B2.1 runner reid.infer.run is not implemented; "
            "preflight passed for {} images, end-to-end inference was not run"
        ).format(n_images)
        if require_runner:
            print("FAIL: {}".format(message), file=sys.stderr)
            return 2
        print("SKIP: {}".format(message))
        return 0

    resolved_model = _resolve_model(model_dir)
    if resolved_model is None:
        message = (
            "B2.1 runner is present, but no complete smoke model artifact was found"
        )
        if require_runner:
            print("FAIL: {}".format(message), file=sys.stderr)
            return 2
        print("SKIP: {}".format(message))
        return 0

    resolved_verifier = _resolve_verifier(resolved_model, verifier_path)
    with tempfile.TemporaryDirectory(prefix="reid-e2e-smoke-") as temp_dir:
        root = Path(temp_dir)
        query_csv = root / "query.csv"
        gallery_csv = root / "gallery.csv"
        out_dir = root / "outputs"
        out_dir.mkdir()
        _write_rows(query_csv, query_rows)
        _write_rows(gallery_csv, gallery_rows)

        command = [
            sys.executable,
            "-m",
            "reid.infer.run",
            "--images",
            str(data_root / "images"),
            "--query",
            str(query_csv),
            "--gallery",
            str(gallery_csv),
            "--out",
            str(out_dir),
            "--model",
            str(resolved_model),
        ]
        if (resolved_model / "d082_production.json").exists():
            from reid.infer.d082_production import digest

            pin = digest(
                (resolved_model.parent / "manifest.sha256").read_text().strip()
            )
            command.extend(("--d082-production-manifest-sha256", pin))
        if resolved_verifier is not None:
            command.extend(("--verifier", str(resolved_verifier)))
        subprocess.run(command, check=True)
        validate_outputs(
            query_csv,
            gallery_csv,
            out_dir / "submission.csv",
            out_dir / "embeddings.npy",
            out_dir / "candidates.csv",
        )

    print(
        "PASS: end-to-end inference and C6 validation passed for {} images".format(
            n_images
        )
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--n-images", type=int, default=50)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--verifier",
        type=Path,
        help="override verifier; fusion defaults to its bundled verifier.pt",
    )
    parser.add_argument(
        "--require-runner",
        action="store_true",
        help="fail rather than skip while the B2.1 runner is absent",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return run_smoke(
            args.data_root,
            args.n_images,
            args.require_runner,
            args.model,
            args.verifier,
        )
    except (OSError, ValueError, ValidationError, subprocess.CalledProcessError) as exc:
        print("FAIL: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
