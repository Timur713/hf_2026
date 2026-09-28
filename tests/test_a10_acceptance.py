import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "a10_acceptance.sh"


def test_a10_script_is_valid_bash():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_a10_dry_run_passes_verifier_to_offline_container():
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--model",
            "artifacts/models/final",
            "--verifier",
            "artifacts/models/verifier_f0/verifier.pt",
            "--label",
            "test-session",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout
    assert "docker compose build reid" in output
    assert "MODEL_DIR=/app/artifacts/models/final" in output
    assert "VERIFIER_PATH=/app/artifacts/models/verifier_f0/verifier.pt" in output
    assert "artifacts/models/final:/app/artifacts/models/final:ro" in output
    assert "DRY RUN" in output
