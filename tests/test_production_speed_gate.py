"""The performance gate follows the published median/measurement protocol."""

from copy import deepcopy

import pytest

from scripts.bench_final_candidate import target_gate


def report():
    return dict(
        latency_batch1=dict(p50_ms=39.95, mean_ms=40.14, warmup=50, iterations=300),
        throughput=[
            dict(batch_size=b, fps=120.0, seconds=10.1) for b in (1, 8, 16, 32)
        ],
        device=dict(name="NVIDIA RTX A5000"),
    )


def test_gate_uses_median_even_if_mean_exceeds_limit():
    assert target_gate(report())["actual_a5000_target_pass"]


def test_mean_does_not_override_failing_median():
    data = report()
    data["latency_batch1"].update(p50_ms=41.0, mean_ms=39.0)
    assert not target_gate(data)["actual_a5000_target_pass"]


def test_h100_result_does_not_claim_a5000_pass():
    data = report()
    data["device"]["name"] = "NVIDIA H100 80GB HBM3"
    result = target_gate(data)
    assert result["measured_numeric_targets_pass"]
    assert not result["actual_a5000_target_pass"]


@pytest.mark.parametrize(
    "case", ["warmup", "iterations", "duration", "batches", "nonfinite"]
)
def test_incomplete_protocol_cannot_pass(case):
    data = deepcopy(report())
    if case in ("warmup", "iterations"):
        data["latency_batch1"][case] = 1
    elif case == "duration":
        data["throughput"][0]["seconds"] = 1
    elif case == "batches":
        data["throughput"].pop()
    else:
        data["latency_batch1"]["p50_ms"] = float("nan")
    with pytest.raises(ValueError):
        target_gate(data)
