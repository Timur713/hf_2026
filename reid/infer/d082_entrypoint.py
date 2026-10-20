"""Docker-only D082 entrypoint: immutable baked pin, cache off, no legacy fallback."""

import logging
from pathlib import Path
import sys

from reid.infer import run
from reid.infer.d082_production import digest

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2] / "artifacts/models/d082"
FORBIDDEN = {
    "--model",
    "--verifier",
    "--d082-manifest-sha256",
    "--d082-production-manifest-sha256",
    "--verifier-gallery-cache",
    "--candidate-threshold",
    "--rerank-weight",
    "--rerank-mode",
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(arg.split("=", 1)[0] in FORBIDDEN for arg in argv):
        LOGGER.error("D082 image forbids model/pin/recipe/cache overrides")
        return 2
    try:
        pin = digest((ROOT / "manifest.sha256").read_text().strip())
    except (OSError, ValueError) as exc:
        LOGGER.error("D082 baked manifest pin unavailable: %s", exc)
        return 1
    return run.main(
        [
            *argv,
            "--model",
            str(ROOT / "bundle"),
            "--verifier",
            str(ROOT / "bundle/verifier.pt"),
            "--d082-production-manifest-sha256",
            pin,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
