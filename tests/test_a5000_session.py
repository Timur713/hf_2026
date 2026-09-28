import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "a5000_session.sh"


def test_a5000_script_is_valid_bash():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_a5000_dry_run_prints_full_protocol():
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--models",
            "artifacts/models/vits,artifacts/models/vitb",
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
    assert "VERIFIER_PATH=/app/artifacts/models/verifier_f0/verifier.pt" in output
    assert output.count("scripts/bench_extract.py") == 2
    assert "artifacts/models/vits:/app/artifacts/models/vits:ro" in output
    assert "artifacts/models/vitb:/app/artifacts/models/vitb:ro" in output
    assert "--warmup 50" in output
    assert "--iterations 300" in output
    assert "--batch-sizes 1\\,8\\,16\\,32" in output
    assert "--throughput-seconds 10" in output
    assert "--require-cuda" in output
    assert "DRY RUN" in output
