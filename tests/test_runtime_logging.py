import logging
import os
import subprocess
import sys

import numpy as np
import pytest

from reid.infer import run
from reid.infer.observability import configure_logging, stage
from tests.test_infer_pipeline import FakeExtractor, FakeVerifier, _write_input


@pytest.fixture(autouse=True)
def restore_logger():
    logger = logging.getLogger("reid")
    handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate
    yield
    for handler in logger.handlers[:]:
        if handler not in handlers:
            logger.removeHandler(handler)
            handler.close()
    logger.handlers = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def test_rank_module_does_not_import_torch_or_image_io():
    code = """
import sys
from reid.post.verifier import PairScorer, rerank_topk
assert 'torch' not in sys.modules
assert 'torchvision' not in sys.modules
assert 'reid.infer.verifier' not in sys.modules
from reid.post.verifier import PairVerifier
from reid.infer.verifier import PairVerifier as Adapter
assert PairVerifier is Adapter
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=dict(os.environ, PYTHONPATH=".")
    )


def test_cli_logging_preserves_outputs_and_effective_config(
    tmp_path, monkeypatch, capsys
):
    q, g = ["q0", "q1"], [f"g{i}" for i in range(12)]
    for name in q + g:
        (tmp_path / f"{name}.jpg").touch()
    _write_input(tmp_path / "q.csv", q)
    _write_input(tmp_path / "g.csv", g)
    rng = np.random.default_rng(7)
    vectors = {name: rng.normal(size=6) for name in q + g}
    model = FakeExtractor(vectors)
    model.cfg = {"candidate_threshold": 0.7, "rerank_weight": 0.15}
    monkeypatch.setattr(run, "Extractor", lambda *args: model)
    monkeypatch.setattr(run, "PairVerifier", lambda *args: FakeVerifier())
    verifier = tmp_path / "verifier.pt"
    verifier.write_bytes(b"test-only-injected-verifier")
    args = [
        "--images",
        str(tmp_path),
        "--query",
        str(tmp_path / "q.csv"),
        "--gallery",
        str(tmp_path / "g.csv"),
        "--model",
        str(tmp_path / "unused"),
        "--verifier",
        str(verifier),
        "--candidate-threshold",
        "0.5",
        "--device",
        "cpu",
    ]
    for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
        assert (
            run.main([*args, "--out", str(tmp_path / level), "--log-level", level]) == 0
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        if level in ("DEBUG", "INFO"):
            assert '"candidate_threshold": 0.5' in captured.err
            assert '"threshold_source": "CLI"' in captured.err
            assert "stage=extract_query" in captured.err
            assert "stage=validate_c6" in captured.err
            assert "verifier_sha256" in captured.err
            assert "query_count=2 gallery_count=12" in captured.err
        else:
            assert captured.err == ""
        assert ("DEBUG" in captured.err) == (level == "DEBUG")
    for name in ("submission.csv", "embeddings.npy", "candidates.csv"):
        assert (
            len(
                {
                    (tmp_path / level / name).read_bytes()
                    for level in ("DEBUG", "INFO", "WARNING", "ERROR")
                }
            )
            == 1
        )


def test_cli_errors_and_env_precedence(tmp_path, monkeypatch, capsys):
    args = [
        "--images",
        str(tmp_path),
        "--query",
        "missing.csv",
        "--gallery",
        "g.csv",
        "--out",
        str(tmp_path / "out"),
    ]
    monkeypatch.setenv("REID_LOG_LEVEL", "debug")
    assert run._parser().parse_args(args).log_level == "DEBUG"
    assert (
        run._parser().parse_args([*args, "--log-level", "error"]).log_level == "ERROR"
    )
    assert run.main([*args, "--twins", "on"]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "WARNING" in captured.err and "ERROR" in captured.err
    assert "stage_failed stage=read_inputs" in captured.err
    assert "missing.csv" in captured.err and "Traceback" in captured.err
    assert not (tmp_path / "out").exists()
    assert run.main([*args, "--batch-size", "0"]) == 2
    monkeypatch.setenv("REID_LOG_LEVEL", "INVALID")
    with pytest.raises(SystemExit) as exc:
        run.main(args)
    assert exc.value.code == 2


def test_configuration_is_idempotent_and_does_not_change_root(capsys):
    handlers = logging.getLogger().handlers[:]
    configure_logging("INFO")
    configure_logging("INFO")
    with stage(logging.getLogger("reid.test"), "test"):
        pass
    assert capsys.readouterr().err.count("stage_done") == 1
    assert logging.getLogger().handlers == handlers


def test_real_module_entrypoint_logs_info_to_stderr(tmp_path):
    cmd = [
        sys.executable,
        "-m",
        "reid.infer.run",
        "--images",
        str(tmp_path),
        "--query",
        "missing.csv",
        "--gallery",
        "g.csv",
        "--out",
        str(tmp_path / "out"),
        "--log-level",
        "DEBUG",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 1 and result.stdout == ""
    assert "DEBUG reid.infer.run stage_start" in result.stderr
    assert "ERROR reid.infer.run stage_failed" in result.stderr
