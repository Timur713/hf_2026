# CPU protocol audit — 25.09.2026

Source commit `45106dc`, clean tracked worktree; seed `20260925`.
Saved report: [protocol_probes.json](protocol_probes.json).

```bash
PYTHONPATH=. .venv/bin/python scripts/audit_research_protocol.py \
  --out outputs/protocol-audit-new.json
make check PYTHON=.venv/bin/python
```

242 tests, C6 self-test and the standard 50-image smoke passed. The smoke uses
legacy model-v1; the audit helper itself uses only synthetic arrays and committed
train-fold metadata. No new real-model mAP, final weights or GPU measurements.
The initial plain `make check` selected system Python without NumPy; the command
above explicitly selects the existing project environment and passed.

The JSON records exact source hashes and scope. Fusion formulas differed on
17/40 synthetic top-1 choices; top-30 vs top-10 verifier normalization alone
differed on 15/100. These are counterexamples, not estimated real-data degradation.
Fold0 odd/even query halves share 306 vehicle identities. Each of five train
galleries has at most one frame per (vehicle, camera). Research Bench matches
canonical eval_local on the synthetic ranking control (absolute delta 0).

Interpretation and primary-source research:
[map10_growth_review_20260925.md](../../map10_growth_review_20260925.md).
