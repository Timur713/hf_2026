from pathlib import Path


def test_image_only_copies_selected_weights_and_audits_complete_app():
    dockerfile = Path("Dockerfile").read_text()
    assert "COPY . ." not in dockerfile
    assert "COPY artifacts/models ./" not in dockerfile
    assert "scripts/stage_release.py --cache-root /build-context" in dockerfile
    assert '--model-url "${MODEL_URL}" --model-sha256 "${MODEL_SHA256}"' in dockerfile
    from scripts.stage_release import ASSETS
    assert set(ASSETS.values()) == {
        "artifacts/models/final/weights.safetensors",
        "artifacts/models/final/model_config.yaml",
        "artifacts/models/verifier_f0/verifier.pt",
    }
    assert "audit_release.py weights --root /app" in dockerfile


def test_offline_runtime_is_enforced_in_compose():
    import yaml

    config = yaml.safe_load(Path("docker-compose.yml").read_text())
    service = config["services"]["reid"]
    assert service["network_mode"] == "none"
    assert service["build"]["target"] == "fusion"
    assert any(v.get("target") == "/data" and v.get("read_only") for v in service["volumes"])


def test_compose_fusion_defaults_preserve_explicit_environment_overrides():
    import yaml

    service = yaml.safe_load(Path("docker-compose.yml").read_text())["services"]["reid"]
    args = service["command"]
    assert args[args.index("--model") + 1] == "${MODEL_DIR:-/app/artifacts/models/fusion}"
    assert args[args.index("--verifier") + 1] == "${VERIFIER_PATH:-/app/artifacts/models/fusion/verifier.pt}"
    for flag, value in {
        "--images": "${IMAGES_DIR:-/data/images}",
        "--query": "${QUERY_CSV:-/data/test_query.csv}",
        "--gallery": "${GALLERY_CSV:-/data/test_gallery.csv}",
        "--out": "${OUT_DIR:-/out}",
        "--twins": "${TWINS:-off}",
    }.items():
        assert args[args.index(flag) + 1] == value
    # Final threshold/weight belong to the SHA-pinned artifact. Hard-coded
    # legacy CLI values would silently override its calibrated policy.
    assert "--candidate-threshold" not in args
    assert "--rerank-weight" not in args


def test_delivery_inspector_recognizes_actual_compose_fusion_default():
    from scripts.check_delivery_readiness import docker_defaults

    defaults = docker_defaults(Path("."))
    assert defaults["uses_fusion"] is True
    assert defaults["offline"] is True


def test_cuda_constraints_pin_all_direct_dependencies():
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    lock = {}
    for line in Path("requirements-lock-cu121.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        req = Requirement(line)
        spec, = req.specifier
        assert spec.operator == "==" and "*" not in spec.version
        lock[canonicalize_name(req.name)] = spec.version
    for line in Path("requirements.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        req = Requirement(line)
        assert req.specifier.contains(lock[canonicalize_name(req.name)])
    dockerfile = Path("Dockerfile").read_text()
    assert dockerfile.count("--constraint requirements-lock-cu121.txt") == 2
