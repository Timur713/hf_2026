"""CLI-only logging configuration; library imports never change root logging."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import logging
from pathlib import Path
import sys
from time import perf_counter


LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(level: str) -> None:
    """One stderr handler for this package, without disturbing host handlers."""
    logger = logging.getLogger("reid")
    for handler in logger.handlers[:]:
        if getattr(handler, "_reid_cli", False):
            logger.removeHandler(handler)
            handler.close()
    handler = logging.StreamHandler(sys.stderr)
    handler._reid_cli = True
    handler.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_identity(model: Path, verifier: Path | None) -> dict:
    """Identify metadata bytes, not attest unverified manifest weight hashes.

    Heavy model verification remains the release audit's responsibility. An
    external verifier has no enclosing manifest, so hash its checkpoint here.
    """
    records = {
        name: file_sha256(model / name)
        for name in ("manifest.json", "fusion_config.json", "model_config.yaml")
        if (model / name).is_file()
    }
    result = {
        "model_path": str(model),
        "metadata_sha256": records,
        "weight_integrity": "not_checked_by_logger",
    }
    if verifier is not None:
        checkpoint = verifier / "verifier.pt" if verifier.is_dir() else verifier
        result["verifier_path"] = str(checkpoint)
        result["verifier_sha256"] = file_sha256(checkpoint)
    return result


@contextmanager
def stage(logger: logging.Logger, name: str, **context):
    start = perf_counter()
    logger.debug("stage_start stage=%s context=%s", name, context)
    try:
        yield
    except Exception as exc:
        logger.error(
            "stage_failed stage=%s elapsed_s=%.6f context=%s error=%s: %s",
            name,
            perf_counter() - start,
            context,
            type(exc).__name__,
            exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        raise
    else:
        logger.info("stage_done stage=%s elapsed_s=%.6f", name, perf_counter() - start)
