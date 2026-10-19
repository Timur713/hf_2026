"""Docker-only D082 entrypoint: immutable baked pin, cache off, no legacy fallback."""

from pathlib import Path
import sys

from reid.infer import run
from reid.infer.d082_production import digest

ROOT = Path("/app/artifacts/models/d082")
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
        print("D082 image forbids model/pin/recipe/cache overrides", file=sys.stderr)
        return 2
    try:
        pin = digest((ROOT / "manifest.sha256").read_text().strip())
    except (OSError, ValueError) as exc:
        print(f"D082 baked manifest pin unavailable: {exc}", file=sys.stderr)
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
